"""
MeloTTS/VITS chunked text synthesizer for Bhasha-Stream.
Provides streaming text-to-speech audio generation optimized for Indian languages.

Fixes applied:
  - BUG-08: 'from typing import List' moved to top of file (was at bottom, used before import)
  - BUG-17: asyncio.get_running_loop() replaces deprecated get_event_loop()
  - BUG-18: MeloTTS API fixed — uses integer speaker_id, output to temp file, correct resampling
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import AsyncGenerator, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service using MeloTTS or VITS backend.

    Converts text phrases into raw PCM audio bytes and yields them
    incrementally for ultra-low latency streaming to the client.
    Optimised for Hindi-English code-switched speech synthesis.
    """

    def __init__(self) -> None:
        self.engine: str = settings.tts.engine
        self.language: str = settings.tts.language
        self.speaker_id: int = settings.tts.speaker_id       # BUG-18: integer, not string
        self.model_sample_rate: int = settings.tts.sample_rate
        self.output_sample_rate: int = settings.tts.output_sample_rate
        self.chunk_size_samples: int = settings.tts.chunk_size_samples

        # Model instance — lazy-loaded on first synthesis call
        self._model: Optional[object] = None
        self._model_lock = asyncio.Lock()

        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize TTS model: {e}. Using mock TTS.")

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_model(self) -> None:
        """Initialize TTS model according to configured engine."""
        if self.engine == "melotts":
            try:
                from melo.api import TTS

                logger.info(
                    f"Initializing MeloTTS: language={self.language}, "
                    f"speaker_id={self.speaker_id}"
                )
                self._model = TTS(language=self.language, device="auto")
                logger.info("MeloTTS initialized successfully.")

            except ImportError:
                logger.warning("MeloTTS not installed. Falling back to mock TTS.")
                raise
            except Exception as e:
                logger.error(f"MeloTTS initialization error: {e}")
                raise

        elif self.engine == "vits":
            raise NotImplementedError(
                "VITS engine is not yet implemented. Use 'melotts' or contribute a VITS adapter."
            )
        else:
            raise ValueError(
                f"Unknown TTS engine '{self.engine}'. Supported: melotts, vits."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to PCM audio and yield chunks incrementally.

        Args:
            text: Input text to synthesize (one phrase/clause at a time for low latency)

        Yields:
            16-bit mono PCM audio chunks at output_sample_rate.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS synthesis.")
            return

        # BUG-17 fix: get_running_loop() not get_event_loop()
        start_time = asyncio.get_running_loop().time()

        try:
            # Lazy model initialization with double-checked locking
            if self._model is None:
                async with self._model_lock:
                    if self._model is None:
                        try:
                            self._initialize_model()
                        except Exception:
                            async for chunk in self._mock_synthesize(text):
                                yield chunk
                            return

            if self.engine == "melotts" and self._model is not None:
                async for chunk in self._synthesize_melo(text):
                    yield chunk
            else:
                async for chunk in self._mock_synthesize(text):
                    yield chunk

            elapsed_ms = (asyncio.get_running_loop().time() - start_time) * 1000
            logger.debug(f"TTS synthesis completed in {elapsed_ms:.1f}ms for '{text[:40]}'")

        except asyncio.CancelledError:
            logger.debug("TTS synthesis cancelled by interruption.")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            async for chunk in self._mock_synthesize(text):
                yield chunk

    # ─────────────────────────────────────────────────────────────────────────
    # MeloTTS backend
    # ─────────────────────────────────────────────────────────────────────────

    async def _synthesize_melo(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize via MeloTTS, resampling output to the pipeline sample rate."""
        if self._model is None:
            raise RuntimeError("MeloTTS model not initialized.")

        loop = asyncio.get_running_loop()

        # Run blocking synthesis in thread pool (never block the event loop)
        audio_bytes = await loop.run_in_executor(
            None,
            self._run_melo_synthesis,
            text,
        )

        if not audio_bytes:
            return

        # Stream audio back in fixed-size chunks
        chunk_bytes = self.chunk_size_samples * 2  # 16-bit = 2 bytes/sample

        for i in range(0, len(audio_bytes), chunk_bytes):
            chunk = audio_bytes[i : i + chunk_bytes]
            if chunk:
                yield chunk

    def _run_melo_synthesis(self, text: str) -> Optional[bytes]:
        """
        Blocking MeloTTS call — executed in a thread pool executor.

        MeloTTS requires writing audio to a temp file; we read it back and
        optionally resample from the model's native rate to the pipeline rate.

        BUG-18 fix: uses integer speaker_id (not string "female") and follows
        the correct MeloTTS.tts_to_file() API.
        """
        if self._model is None:
            return None

        try:
            import numpy as np
            import soundfile as sf

            # MeloTTS writes to a file; use a temp file to capture output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Correct MeloTTS API: tts_to_file(text, speaker_id, output_path, speed)
                self._model.tts_to_file(
                    text,
                    self.speaker_id,          # BUG-18: must be int
                    tmp_path,
                    speed=1.0,
                )

                audio_array, file_sr = sf.read(tmp_path, dtype="float32")

            finally:
                # Always clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            # Resample if the model's native rate differs from the pipeline rate
            if file_sr != self.output_sample_rate:
                audio_array = self._resample(audio_array, file_sr, self.output_sample_rate)

            # Convert float32 → int16 PCM bytes
            audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
            return audio_int16.tobytes()

        except Exception as e:
            logger.error(f"MeloTTS runtime error: {e}")
            return None

    @staticmethod
    def _resample(audio: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
        """Resample audio array using scipy (no librosa dependency required)."""
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(target_sr, orig_sr)
            up, down = target_sr // g, orig_sr // g
            return resample_poly(audio, up, down).astype("float32")
        except Exception as e:
            logger.warning(f"Resampling failed ({e}); returning original audio.")
            return audio

    # ─────────────────────────────────────────────────────────────────────────
    # Mock / Fallback
    # ─────────────────────────────────────────────────────────────────────────

    async def _mock_synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Mock TTS for testing or when model is unavailable.
        Generates low-amplitude white noise scaled to estimated speech duration.
        """
        import numpy as np

        # Rough estimate: ~15 chars/sec of speech
        estimated_sec = max(0.3, len(text) / 15.0)
        total_samples = int(self.output_sample_rate * estimated_sec)
        samples_per_chunk = self.chunk_size_samples

        for i in range(0, total_samples, samples_per_chunk):
            chunk_size = min(samples_per_chunk, total_samples - i)
            # Low-amplitude noise so it's audible but obviously synthetic
            noise = np.random.randint(-200, 200, size=chunk_size, dtype=np.int16)
            yield noise.tobytes()
            # Pace the output to simulate real synthesis throughput
            await asyncio.sleep(chunk_size / self.output_sample_rate)

        logger.debug(f"Mock TTS: generated ~{estimated_sec:.1f}s for '{text[:30]}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods
    # ─────────────────────────────────────────────────────────────────────────

    async def synthesize_complete(self, text: str) -> Optional[bytes]:
        """
        Synthesize text and return all audio as a single bytes object.
        For non-streaming use cases (e.g., caching short responses).
        """
        chunks: List[bytes] = []

        async for chunk in self.synthesize(text):
            chunks.append(chunk)

        return b"".join(chunks) if chunks else None

    async def health_check(self) -> bool:
        """Return True if the TTS model is loaded and ready."""
        if self._model is not None:
            return True
        try:
            self._initialize_model()
            return self._model is not None
        except Exception:
            return False

    def unload_model(self) -> None:
        """Unload model to free GPU/CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

            import gc
            gc.collect()

            logger.info("TTS model unloaded.")
