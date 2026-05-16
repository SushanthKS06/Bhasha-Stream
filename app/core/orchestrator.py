"""
Asynchronous pipeline state machine (The Brain) for Bhasha-Stream.
Manages agent states: LISTENING, THINKING, SPEAKING, INTERRUPTED.
"""
import asyncio
import logging
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.services.vad import VADService
from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Explicit agent states for the orchestration state machine."""
    LISTENING = auto()  # Waiting for user speech
    THINKING = auto()   # Processing transcription, generating response
    SPEAKING = auto()   # Synthesizing and playing audio response
    INTERRUPTED = auto()  # User interrupted, cleaning up


class Orchestrator:
    """
    Asynchronous orchestration state machine managing the voice agent pipeline.
    
    Coordinates VAD, STT, LLM, and TTS services with proper state transitions
    and interruption handling.
    """
    
    def __init__(
        self,
        inbound_queue: asyncio.Queue[bytes],
        outbound_queue: asyncio.Queue[bytes],
        interrupt_queue: asyncio.Queue[bool],
    ) -> None:
        self.inbound_queue = inbound_queue
        self.outbound_queue = outbound_queue
        self.interrupt_queue = interrupt_queue
        
        # State management
        self.state: AgentState = AgentState.LISTENING
        self.state_lock = asyncio.Lock()
        
        # Audio buffer for VAD processing
        self.audio_buffer: bytearray = bytearray()
        self.speech_buffer: bytearray = bytearray()  # Accumulated speech for STT
        
        # Service instances
        self.vad_service = VADService()
        self.stt_service = STTService()
        self.llm_service = LLMService()
        self.tts_service = TTSService()
        
        # Task tracking for cancellation
        self.current_llm_task: Optional[asyncio.Task] = None
        self.current_tts_task: Optional[asyncio.Task] = None
        self.current_stt_task: Optional[asyncio.Task] = None
        
        # Text accumulation for interleaved LLM->TTS streaming
        self.text_buffer: str = ""
        self.word_count: int = 0
        
        # Regex for semantic breaks (punctuation that ends a phrase)
        self.semantic_break_pattern = re.compile(rf"[{re.escape(settings.llm.chunk_delimiters)}]")
        
        # Interruption flag
        self._interrupt_requested = False
        
    async def _transition_state(self, new_state: AgentState) -> None:
        """Thread-safe state transition with logging."""
        async with self.state_lock:
            old_state = self.state
            self.state = new_state
            logger.debug(f"State transition: {old_state.name} -> {new_state.name}")
    
    async def _handle_interruption(self) -> None:
        """
        Handle user interruption during SPEAKING state.
        Cancels LLM/TTS tasks, flushes queues, and signals client.
        """
        logger.info("Handling user interruption")
        
        # Set interrupt flag
        self._interrupt_requested = True
        
        # Cancel current LLM task if running
        if self.current_llm_task and not self.current_llm_task.done():
            self.current_llm_task.cancel()
            try:
                await self.current_llm_task
            except asyncio.CancelledError:
                logger.debug("LLM task cancelled successfully")
        
        # Cancel current TTS task if running
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
            try:
                await self.current_tts_task
            except asyncio.CancelledError:
                logger.debug("TTS task cancelled successfully")
        
        # Flush outbound audio queue
        while not self.outbound_queue.empty():
            try:
                self.outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Send interruption signal to client
        await self.interrupt_queue.put(True)
        
        # Reset text buffer
        self.text_buffer = ""
        self.word_count = 0
        
        # Transition to INTERRUPTED briefly, then back to LISTENING
        await self._transition_state(AgentState.INTERRUPTED)
        await asyncio.sleep(0.01)  # Brief pause for cleanup
        await self._transition_state(AgentState.LISTENING)
        
        self._interrupt_requested = False
    
    async def _process_vad(self, audio_frame: bytes) -> bool:
        """
        Process audio frame through VAD and return speech detection result.
        Accumulates speech frames for STT processing.
        """
        is_speech = await self.vad_service.is_speech(audio_frame)
        
        if is_speech:
            # Accumulate speech frames
            self.speech_buffer.extend(audio_frame)
            
            # If we were in LISTENING, transition to THINKING once we have enough speech
            if self.state == AgentState.LISTENING:
                min_speech_bytes = (
                    settings.audio.vad_min_speech_duration_ms 
                    * settings.audio.bytes_per_frame 
                    // settings.audio.frame_duration_ms
                )
                if len(self.speech_buffer) >= min_speech_bytes:
                    await self._transition_state(AgentState.THINKING)
        else:
            # Check if we should trigger endpoint (hangover time exceeded)
            if self.state == AgentState.THINKING and len(self.speech_buffer) > 0:
                # VAD detected silence after speech - process the accumulated speech
                hangover_bytes = (
                    settings.audio.vad_hangover_time_ms
                    * settings.audio.bytes_per_frame
                    // settings.audio.frame_duration_ms
                )
                
                # In a real implementation, we'd track silence duration
                # For now, if we have speech buffer and VAD says silence, process it
                if len(self.speech_buffer) >= hangover_bytes:
                    # Trigger STT processing
                    asyncio.create_task(self._process_speech_to_text())
                    self.speech_buffer = bytearray()
        
        return is_speech
    
    async def _process_speech_to_text(self) -> None:
        """Process accumulated speech buffer through STT service."""
        if len(self.speech_buffer) == 0:
            return
        
        logger.debug(f"Processing STT on {len(self.speech_buffer)} bytes")
        
        try:
            # Run STT in background task
            self.current_stt_task = asyncio.create_task(
                self.stt_service.transcribe(bytes(self.speech_buffer)),
                name="stt_task",
            )
            
            transcription = await self.current_stt_task
            
            if transcription and transcription.strip():
                logger.info(f"STT Result: {transcription}")
                # Trigger LLM generation
                asyncio.create_task(self._generate_and_stream_response(transcription))
            else:
                # No valid transcription, return to listening
                await self._transition_state(AgentState.LISTENING)
                
        except Exception as e:
            logger.error(f"STT processing error: {e}", exc_info=True)
            await self._transition_state(AgentState.LISTENING)
        finally:
            self.current_stt_task = None
    
    async def _accumulate_and_flush_tokens(self, token: str) -> None:
        """
        Accumulate LLM tokens and flush to TTS on semantic breaks.
        Implements phrase/clause-splitting for low-latency interleaving.
        """
        self.text_buffer += token
        self.word_count = len(self.text_buffer.split())
        
        # Check for semantic break (punctuation)
        match = self.semantic_break_pattern.search(self.text_buffer)
        
        should_flush = False
        flush_text = ""
        
        if match:
            # Split at the punctuation mark
            split_idx = match.end()
            flush_text = self.text_buffer[:split_idx].strip()
            self.text_buffer = self.text_buffer[split_idx:].lstrip()
            self.word_count = len(self.text_buffer.split())
            should_flush = len(flush_text) > 0
        elif self.word_count >= settings.llm.max_words_before_flush:
            # Flush on word count threshold
            flush_text = self.text_buffer
            self.text_buffer = ""
            self.word_count = 0
            should_flush = True
        
        if should_flush and flush_text:
            logger.debug(f"Flushing text to TTS: {flush_text[:50]}...")
            # Create TTS task
            self.current_tts_task = asyncio.create_task(
                self._synthesize_and_stream(flush_text),
                name="tts_task",
            )
    
    async def _synthesize_and_stream(self, text: str) -> None:
        """Synthesize text to audio and stream chunks to outbound queue."""
        try:
            async for audio_chunk in self.tts_service.synthesize(text):
                if self._interrupt_requested:
                    break
                await self.outbound_queue.put(audio_chunk)
        except asyncio.CancelledError:
            logger.debug("TTS synthesis cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
    
    async def _generate_and_stream_response(self, prompt: str) -> None:
        """
        Generate LLM response and stream tokens to TTS incrementally.
        Handles interruption during generation.
        """
        logger.info(f"Generating response for: {prompt[:50]}...")
        
        # Transition to SPEAKING (we'll start producing audio soon)
        await self._transition_state(AgentState.SPEAKING)
        
        try:
            # Stream tokens from LLM
            self.current_llm_task = asyncio.create_task(
                self._llm_token_streamer(prompt),
                name="llm_streamer",
            )
            
            await self.current_llm_task
            
            # Flush any remaining text in buffer
            if self.text_buffer.strip():
                await self._synthesize_and_stream(self.text_buffer)
                self.text_buffer = ""
                self.word_count = 0
                
        except asyncio.CancelledError:
            logger.debug("LLM generation cancelled by interruption")
            raise
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
        finally:
            self.current_llm_task = None
            # Return to listening after response complete
            if not self._interrupt_requested:
                await self._transition_state(AgentState.LISTENING)
    
    async def _llm_token_streamer(self, prompt: str) -> None:
        """Stream individual tokens from LLM and accumulate for TTS."""
        try:
            async for token in self.llm_service.generate(prompt):
                if self._interrupt_requested:
                    break
                await self._accumulate_and_flush_tokens(token)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Token streaming error: {e}", exc_info=True)
    
    async def run(self) -> None:
        """
        Main orchestrator loop.
        Continuously processes inbound audio and manages state transitions.
        """
        logger.info("Orchestrator started in LISTENING state")
        
        try:
            while True:
                # Get audio frame from inbound queue
                try:
                    audio_frame = await asyncio.wait_for(
                        self.inbound_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Skip empty frames
                if len(audio_frame) == 0:
                    continue
                
                # Process based on current state
                if self.state == AgentState.LISTENING:
                    # Process VAD to detect speech onset
                    await self._process_vad(audio_frame)
                    
                elif self.state == AgentState.THINKING:
                    # Continue accumulating speech, check for interruptions
                    is_speech = await self._process_vad(audio_frame)
                    
                    # If user speaks while thinking, that's normal (continue accumulating)
                    # The VAD handler will trigger STT when silence is detected
                    
                elif self.state == AgentState.SPEAKING:
                    # Check if user is speaking (interruption detection)
                    is_speech = await self.vad_service.is_speech(audio_frame)
                    
                    if is_speech:
                        # User interrupted! Handle immediately
                        await self._handle_interruption()
                        
                elif self.state == AgentState.INTERRUPTED:
                    # Drain audio buffer during interruption recovery
                    continue
                    
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
            raise
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            raise
        finally:
            # Cleanup on exit
            await self._cleanup()
    
    async def _cleanup(self) -> None:
        """Clean up all pending tasks and resources."""
        logger.info("Cleaning up orchestrator resources")
        
        # Cancel all active tasks
        for task in [self.current_llm_task, self.current_tts_task, self.current_stt_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear buffers
        self.audio_buffer.clear()
        self.speech_buffer.clear()
        self.text_buffer = ""
        self.word_count = 0
        
        # Final state reset
        await self._transition_state(AgentState.LISTENING)
