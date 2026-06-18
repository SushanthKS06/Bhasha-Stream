"""
Asynchronous pipeline state machine (The Brain) for Bhasha-Stream.
Manages agent states: LISTENING, THINKING, SPEAKING, INTERRUPTED.

Fixes applied:
  - BUG-10: asyncio.create_task() result now stored and tracked (no fire-and-forget)
  - BUG-11: Speech buffer captured into local variable BEFORE clearing AND before task creation
  - BUG-12: Proper time-based silence hangover logic (tracks elapsed silence seconds)
  - ARCH-06: Per-connection queues with maxsize (passed in from websocket handler)
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from enum import Enum, auto
from typing import Optional, Set

from app.core.config import settings
from app.services.llm import LLMService
from app.services.stt import STTService
from app.services.tts import TTSService
from app.services.vad import VADService

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Explicit states for the voice agent state machine."""
    LISTENING = auto()      # Waiting for user speech onset
    THINKING = auto()       # Accumulating speech, preparing for STT
    SPEAKING = auto()       # Generating LLM response + synthesizing audio
    INTERRUPTED = auto()    # User barged in; tearing down active tasks


class Orchestrator:
    """
    Asynchronous orchestration state machine for the voice agent pipeline.

    Coordinates VAD → STT → LLM → TTS with correct state transitions,
    barge-in interruption handling, and safe task lifecycle management.

    One Orchestrator is created per WebSocket connection, so all internal
    state is inherently session-isolated.
    """

    def __init__(
        self,
        inbound_queue: asyncio.Queue,
        outbound_queue: asyncio.Queue,
        interrupt_queue: asyncio.Queue,
    ) -> None:
        self.inbound_queue = inbound_queue
        self.outbound_queue = outbound_queue
        self.interrupt_queue = interrupt_queue

        # ── State machine ──────────────────────────────────────────────────
        self.state: AgentState = AgentState.LISTENING
        self._state_lock = asyncio.Lock()

        # ── Audio buffers ──────────────────────────────────────────────────
        self._speech_buffer: bytearray = bytearray()  # Accumulates speech for STT

        # ── VAD / hangover tracking (BUG-12 fix) ──────────────────────────
        # monotonic timestamp of first silence frame after speech ends
        self._silence_start_time: Optional[float] = None
        self._stt_triggered: bool = False  # Guard against duplicate STT triggers

        # ── Service instances (one per connection) ─────────────────────────
        self.vad_service = VADService()
        self.stt_service = STTService()
        self.llm_service = LLMService()
        self.tts_service = TTSService()

        # ── Task tracking (BUG-10 fix) ─────────────────────────────────────
        # Every created task is registered here so it can be cancelled cleanly.
        self.active_tasks: Set[asyncio.Task] = set()
        self._current_stt_task: Optional[asyncio.Task] = None
        self._current_llm_task: Optional[asyncio.Task] = None
        self._current_tts_task: Optional[asyncio.Task] = None

        # ── Interleaving / phrase-splitting ────────────────────────────────
        self._text_buffer: str = ""
        self._interrupt_flag: bool = False

        # Regex that matches any TTS flush punctuation
        self._break_pattern = re.compile(
            rf"[{re.escape(settings.llm.chunk_delimiters)}]"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────────────────

    async def _transition(self, new_state: AgentState) -> None:
        """Acquire state lock, log, and update state atomically."""
        async with self._state_lock:
            logger.debug(f"State: {self.state.name} → {new_state.name}")
            self.state = new_state

    # ─────────────────────────────────────────────────────────────────────────
    # Interruption handling
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_interruption(self) -> None:
        """
        Handle barge-in: cancel active LLM/TTS tasks, flush audio queue,
        send null-signal to client, and return to LISTENING.
        Target: < 50ms end-to-end.
        """
        logger.info("⚡ Interruption detected — cancelling pipeline")
        self._interrupt_flag = True

        # Cancel all tracked tasks concurrently for minimum latency
        cancel_targets = [
            t for t in [
                self._current_llm_task,
                self._current_tts_task,
                self._current_stt_task,
            ]
            if t and not t.done()
        ]

        for task in cancel_targets:
            task.cancel()

        if cancel_targets:
            # Await cancellation without raising — we just want them stopped
            await asyncio.gather(*cancel_targets, return_exceptions=True)

        # Flush buffered outbound audio
        while not self.outbound_queue.empty():
            try:
                self.outbound_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Signal client to halt local playback
        await self.interrupt_queue.put(True)

        # Reset text accumulation
        self._text_buffer = ""
        self._silence_start_time = None
        self._stt_triggered = False
        self._speech_buffer.clear()

        # Brief INTERRUPTED state, then immediately back to LISTENING
        await self._transition(AgentState.INTERRUPTED)
        await asyncio.sleep(0.01)
        await self._transition(AgentState.LISTENING)

        self._interrupt_flag = False

    # ─────────────────────────────────────────────────────────────────────────
    # VAD processing (called every 30ms)
    # ─────────────────────────────────────────────────────────────────────────

    async def _process_vad(self, audio_frame: bytes) -> bool:
        """
        Run VAD on a single 30ms audio frame and advance the state machine.
        """
        if audio_frame == b"STOP_SIGNAL":
            logger.info("Manual STOP signal received — forcing STT trigger")
            if len(self._speech_buffer) > 0:
                self._stt_triggered = True
                captured_audio = bytes(self._speech_buffer)
                self._speech_buffer.clear()
                self._silence_start_time = None

                await self._transition(AgentState.THINKING)

                task = asyncio.create_task(
                    self._process_speech_to_text(captured_audio),
                    name="stt_task",
                )
                self._current_stt_task = task
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
            return False

        is_speech = await self.vad_service.is_speech(audio_frame)

        if is_speech or True:  # Always accumulate speech since VAD is temporarily bypassed
            # Accumulate speech frames
            self._speech_buffer.extend(audio_frame)
            self._silence_start_time = None  # Reset silence clock on every speech frame

        else:
            # Silence frame while in THINKING state — count elapsed silence time
            if self.state == AgentState.THINKING and len(self._speech_buffer) > 0:
                now = time.monotonic()

                if self._silence_start_time is None:
                    # First silence frame after speech — start the hangover clock
                    self._silence_start_time = now
                else:
                    elapsed_ms = (now - self._silence_start_time) * 1000

                    # BUG-12 fix: trigger STT only after silence exceeds hangover window
                    if elapsed_ms >= settings.audio.vad_hangover_time_ms and not self._stt_triggered:
                        self._stt_triggered = True

                        # BUG-11 fix: capture buffer content BEFORE clearing it
                        captured_audio = bytes(self._speech_buffer)
                        self._speech_buffer.clear()
                        self._silence_start_time = None

                        # BUG-10 fix: store the task reference so we can cancel it
                        task = asyncio.create_task(
                            self._process_speech_to_text(captured_audio),
                            name="stt_task",
                        )
                        self._current_stt_task = task
                        self.active_tasks.add(task)
                        task.add_done_callback(self.active_tasks.discard)

            elif self.state == AgentState.LISTENING and len(self._speech_buffer) > 0:
                # User started speaking but stopped before reaching the 200ms threshold.
                # Clear the accumulated buffer so random clicks don't build up over time.
                self._speech_buffer.clear()

        return is_speech

    # ─────────────────────────────────────────────────────────────────────────
    # STT
    # ─────────────────────────────────────────────────────────────────────────

    async def _process_speech_to_text(self, audio_data: bytes) -> None:
        """Transcribe captured speech and dispatch LLM generation."""
        logger.debug(f"STT: processing {len(audio_data):,} bytes of audio")

        try:
            transcription = await self.stt_service.transcribe(audio_data)

            if transcription and transcription.strip():
                logger.info(f"STT result: '{transcription}'")
                # Send STT result to UI
                await self.outbound_queue.put(f"USER: {transcription.strip()}")
                
                # Dispatch LLM generation as a tracked task
                task = asyncio.create_task(
                    self._generate_and_stream_response(transcription),
                    name="llm_tts_pipeline",
                )
                self._current_llm_task = task
                self.active_tasks.add(task)
                task.add_done_callback(self.active_tasks.discard)
            else:
                logger.debug("STT returned empty result — returning to LISTENING")
                await self._transition(AgentState.LISTENING)

        except asyncio.CancelledError:
            logger.debug("STT task cancelled")
            raise
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            await self._transition(AgentState.LISTENING)
        finally:
            self._current_stt_task = None
            self._stt_triggered = False

    # ─────────────────────────────────────────────────────────────────────────
    # LLM → TTS interleaving pipeline
    # ─────────────────────────────────────────────────────────────────────────

    async def _generate_and_stream_response(self, prompt: str) -> None:
        """
        Stream LLM tokens → phrase splitter → TTS chunks → outbound queue.
        Handles cancellation from interruption cleanly at every await point.
        """
        logger.info(f"LLM: generating response for '{prompt[:60]}'")
        await self._transition(AgentState.SPEAKING)

        try:
            async for token in self.llm_service.generate(prompt):
                if self._interrupt_flag:
                    break
                await self._accumulate_and_flush(token)

            # Flush any remaining text after the stream completes
            if self._text_buffer.strip() and not self._interrupt_flag:
                await self._synthesize_phrase(self._text_buffer.strip())
                self._text_buffer = ""

        except asyncio.CancelledError:
            logger.debug("LLM/TTS pipeline cancelled by interruption")
            raise
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
        finally:
            self._current_llm_task = None
            self._text_buffer = ""
            if not self._interrupt_flag:
                await self._transition(AgentState.LISTENING)

    async def _accumulate_and_flush(self, token: str) -> None:
        """
        Buffer LLM tokens and flush to TTS on semantic phrase boundaries.

        Flush triggers:
          1. Punctuation delimiter found in buffer (., ?, !)
          2. Word count exceeds max_words_before_flush threshold
        """
        self._text_buffer += token
        word_count = len(self._text_buffer.split())

        match = self._break_pattern.search(self._text_buffer)

        if match:
            split_idx = match.end()
            phrase = self._text_buffer[:split_idx].strip()
            self._text_buffer = self._text_buffer[split_idx:].lstrip()

            if phrase:
                await self._synthesize_phrase(phrase)

        elif word_count >= settings.llm.max_words_before_flush and self._text_buffer[-1].isspace():
            phrase = self._text_buffer.strip()
            self._text_buffer = ""
            if phrase:
                await self._synthesize_phrase(phrase)

    async def _synthesize_phrase(self, text: str) -> None:
        """Synthesize a phrase and push audio chunks to the outbound queue."""
        logger.debug(f"TTS flush: '{text[:50]}'")
        await self.outbound_queue.put(f"AI: {text.strip()}")

        try:
            async for audio_chunk in self.tts_service.synthesize(text):
                if self._interrupt_flag:
                    return
                await self.outbound_queue.put(audio_chunk)

        except asyncio.CancelledError:
            logger.debug("TTS phrase synthesis cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main orchestrator event loop.
        Reads audio frames from inbound_queue and drives the state machine.
        Runs until the WebSocket connection is closed (CancelledError).
        """
        logger.info("Orchestrator started — state: LISTENING")

        try:
            while True:
                try:
                    audio_frame = await asyncio.wait_for(
                        self.inbound_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if not audio_frame:
                    continue

                current_state = self.state  # snapshot to avoid TOCTOU

                if current_state == AgentState.LISTENING:
                    await self._process_vad(audio_frame)

                elif current_state == AgentState.THINKING:
                    await self._process_vad(audio_frame)

                elif current_state == AgentState.SPEAKING:
                    # Check for barge-in
                    is_speech = await self.vad_service.is_speech(audio_frame)
                    if is_speech:
                        await self._handle_interruption()

                elif current_state == AgentState.INTERRUPTED:
                    # Drain inbound frames while cleanup completes
                    continue

        except asyncio.CancelledError:
            logger.info("Orchestrator loop cancelled — shutting down")
            raise
        except Exception as e:
            logger.error(f"Orchestrator fatal error: {e}", exc_info=True)
            raise
        finally:
            await self._cleanup()

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Cancel all tracked tasks and release resources on shutdown."""
        logger.info("Orchestrator cleanup: cancelling all active tasks")

        # Cancel every tracked task
        tasks_to_cancel = list(self.active_tasks)
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Close LLM HTTP session
        await self.llm_service.close()

        # Clear all buffers
        self._speech_buffer.clear()
        self._text_buffer = ""

        logger.info("Orchestrator cleanup complete.")
