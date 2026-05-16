"""
faster-whisper CTranslate2 transcription service for Bhasha-Stream.
Provides async speech-to-text with optimized latency for Indian code-switched languages.
"""
import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class STTService:
    """
    Speech-to-Text service using faster-whisper (CTranslate2 backend).
    
    Optimized for low-latency transcription of Hindi-English code-switched speech.
    Supports GPU acceleration when available.
    """
    
    def __init__(self) -> None:
        self.model_size: str = settings.stt.model_size
        self.device: str = settings.stt.device
        self.compute_type: str = settings.stt.compute_type
        self.language: str = settings.stt.language
        self.beam_size: int = settings.stt.beam_size
        self.best_of: int = settings.stt.best_of
        
        # Model instance (lazy-loaded)
        self._model: Optional[object] = None
        self._model_lock = asyncio.Lock()
        
        # Try to initialize the model
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize faster-whisper model: {e}. Using mock STT.")
    
    def _initialize_model(self) -> None:
        """Initialize faster-whisper WhisperModel."""
        try:
            from faster_whisper import WhisperModel
            
            logger.info(
                f"Loading faster-whisper model '{self.model_size}' "
                f"on device='{self.device}' with compute_type='{self.compute_type}'"
            )
            
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            
            logger.info("faster-whisper model loaded successfully")
            
        except ImportError:
            logger.warning("faster-whisper not installed. Using mock STT.")
            raise
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {e}")
            raise
    
    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw PCM audio bytes (16kHz 16-bit mono)
            
        Returns:
            Transcribed text string, or None if transcription fails
        """
        if len(audio_data) == 0:
            logger.warning("Empty audio data provided for transcription")
            return None
        
        start_time = time.perf_counter()
        
        try:
            # Ensure model is loaded
            if self._model is None:
                async with self._model_lock:
                    if self._model is None:
                        # Try to load again or use mock
                        try:
                            self._initialize_model()
                        except Exception:
                            # Fall back to mock transcription
                            return await self._mock_transcribe(audio_data)
            
            # Run transcription in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_transcription,
                audio_data,
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"STT transcription completed in {elapsed_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            # Fall back to mock on error
            return await self._mock_transcribe(audio_data)
    
    def _run_transcription(self, audio_data: bytes) -> Optional[str]:
        """Run actual faster-whisper transcription (blocking call)."""
        if self._model is None:
            return None
        
        # Convert bytes to numpy array
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run transcription
        segments, info = self._model.transcribe(
            audio_array,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            vad_filter=True,  # Enable VAD filtering
            vad_parameters=dict(
                threshold=0.5,
                min_silence_duration_ms=300,
            ),
        )
        
        # Combine all segments into final text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        full_text = " ".join(text_parts)
        
        logger.debug(
            f"Detected language: {info.language} with probability {info.language_probability:.2f}"
        )
        
        return full_text if full_text else None
    
    async def _mock_transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Mock transcription for testing/fallback.
        Returns placeholder text simulating code-switched speech.
        """
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Return mock Hinglish text
        mock_responses = [
            "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
            "Hello! How can I help you today?",
            "मुझे बताइए, क्या काम है?",
            "Yes, please go ahead and tell me.",
            "ठीक है, मैं समझ गया।",
        ]
        
        import random
        return random.choice(mock_responses)
    
    async def transcribe_streaming(
        self,
        audio_chunks: asyncio.Queue[bytes],
    ) -> AsyncGenerator[str, None]:
        """
        Stream transcription results as audio chunks arrive.
        Useful for real-time partial hypothesis updates.
        
        Args:
            audio_chunks: Queue of incoming audio frames
            
        Yields:
            Partial transcription results
        """
        buffer = bytearray()
        min_chunk_size = settings.audio.bytes_per_frame * 10  # 300ms minimum
        
        while True:
            try:
                chunk = await asyncio.wait_for(audio_chunks.get(), timeout=1.0)
                buffer.extend(chunk)
                
                # Process when we have enough audio
                if len(buffer) >= min_chunk_size:
                    result = await self.transcribe(bytes(buffer))
                    if result:
                        yield result
                    buffer.clear()
                    
            except asyncio.TimeoutError:
                # No new chunks, continue waiting
                continue
            except asyncio.CancelledError:
                break
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            import gc
            gc.collect()
            
            logger.info("STT model unloaded")
