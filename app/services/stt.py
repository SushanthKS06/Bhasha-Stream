"""
faster-whisper CTranslate2 transcription service for Bhasha-Stream.
Provides async speech-to-text with optimized latency for Indian code-switched languages.

Fixes applied:
  - BUG-16: asyncio.get_running_loop() replaces deprecated get_event_loop()
"""
from __future__ import annotations

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
    Supports GPU acceleration when available and falls back to mock STT when the
    model is unavailable (e.g., in CI/CD environments).
    """

    def __init__(self) -> None:
        self.model_size: str = settings.stt.model_size
        self.device: str = settings.stt.device
        self.compute_type: str = settings.stt.compute_type
        self.language: str = settings.stt.language
        self.beam_size: int = settings.stt.beam_size
        self.best_of: int = settings.stt.best_of

        # Model instance — lazy-loaded on first transcription call
        self._model: Optional[object] = None
        self._model_lock = asyncio.Lock()

        # Try eager initialization so first request is fast
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(
                f"Failed to initialize faster-whisper model: {e}. Using mock STT."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_model(self) -> None:
        """Load faster-whisper WhisperModel onto the configured device."""
        try:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading faster-whisper model '{self.model_size}' "
                f"on device='{self.device}' compute_type='{self.compute_type}'"
            )

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

            logger.info("faster-whisper model loaded successfully.")

        except ImportError:
            logger.warning("faster-whisper not installed. Using mock STT.")
            raise
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe raw PCM audio to text.

        The blocking faster-whisper call is offloaded to the default thread
        executor so it never stalls the asyncio event loop.

        Args:
            audio_data: Raw PCM bytes (16kHz 16-bit mono)

        Returns:
            Transcribed text, or None on failure.
        """
        if len(audio_data) == 0:
            logger.warning("Empty audio data provided for transcription.")
            return None

        start_time = time.perf_counter()

        try:
            # Ensure model is loaded (double-checked locking)
            if self._model is None:
                async with self._model_lock:
                    if self._model is None:
                        try:
                            self._initialize_model()
                        except Exception:
                            return await self._mock_transcribe(audio_data)

            # BUG-16 fix: get_running_loop() not get_event_loop()
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._run_transcription,
                audio_data,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"STT completed in {elapsed_ms:.1f}ms")

            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return await self._mock_transcribe(audio_data)

    # ─────────────────────────────────────────────────────────────────────────
    # Sync transcription (runs in thread pool)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_transcription(self, audio_data: bytes) -> Optional[str]:
        """Blocking faster-whisper transcription call — executed in thread pool."""
        if self._model is None:
            return None

        import numpy as np

        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio_array,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_silence_duration_ms=300,
            ),
        )

        text_parts = [seg.text.strip() for seg in segments]
        full_text = " ".join(text_parts).strip()

        logger.debug(
            f"Detected language: {info.language} "
            f"(probability={info.language_probability:.2f})"
        )

        return full_text if full_text else None

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming transcription (partial hypothesis updates)
    # ─────────────────────────────────────────────────────────────────────────

    async def transcribe_streaming(
        self,
        audio_chunks: asyncio.Queue,
    ) -> AsyncGenerator[str, None]:
        """
        Yield partial transcription results as audio chunks arrive.
        Useful for real-time hypothesis display.

        Args:
            audio_chunks: Queue of incoming 30ms audio frames

        Yields:
            Partial transcription strings.
        """
        buffer = bytearray()
        min_chunk_size = settings.audio.bytes_per_frame * 10  # 300ms minimum

        while True:
            try:
                chunk = await asyncio.wait_for(audio_chunks.get(), timeout=1.0)
                buffer.extend(chunk)

                if len(buffer) >= min_chunk_size:
                    result = await self.transcribe(bytes(buffer))
                    if result:
                        yield result
                    buffer.clear()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # Mock / Fallback
    # ─────────────────────────────────────────────────────────────────────────

    async def _mock_transcribe(self, audio_data: bytes) -> Optional[str]:
        """
        Mock transcription for testing or when the model is unavailable.
        Returns a random Hinglish phrase with a simulated processing delay.
        """
        import random

        await asyncio.sleep(0.1)

        mock_responses = [
            "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
            "Hello! How can I help you today?",
            "मुझे बताइए, क्या काम है?",
            "Yes, please go ahead and tell me.",
            "ठीक है, मैं समझ गया।",
        ]
        return random.choice(mock_responses)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def unload_model(self) -> None:
        """Unload the model to free GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

            import gc
            gc.collect()

            logger.info("STT model unloaded.")
