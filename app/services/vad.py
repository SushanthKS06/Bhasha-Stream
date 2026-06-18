"""
Silero VAD speech detection service for Bhasha-Stream.
Uses ONNX Runtime for efficient voice activity detection on 30ms windows.

Fixes applied:
  - BUG-13: Model path from config + auto-download on first run
  - BUG-14: ONNX inference offloaded to executor (non-blocking)
  - BUG-15: asyncio.get_running_loop() replaces deprecated get_event_loop()
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class VADService:
    """
    Voice Activity Detection service using Silero VAD (ONNX Runtime).

    Analyzes 30ms audio windows (480 samples at 16kHz) and returns boolean
    speech / silence detection with hysteresis and hangover time tracking.
    Falls back to an energy-based heuristic when the ONNX model is unavailable.
    """

    def __init__(self) -> None:
        self.sample_rate: int = settings.audio.sample_rate
        self.speech_threshold: float = settings.audio.vad_speech_threshold
        self.silence_threshold: float = settings.audio.vad_silence_threshold
        self.hangover_time_ms: int = settings.audio.vad_hangover_time_ms

        # Hysteresis state
        self._is_speaking: bool = False
        self._silence_start_time: Optional[float] = None
        self._speech_start_time: Optional[float] = None

        # ONNX session and Silero hidden state (v5 uses a single 'state' tensor)
        self._session: Optional[object] = None
        self._state: Optional[np.ndarray] = None

        # Executor for blocking ONNX calls (BUG-14 fix)
        self._executor = None  # None → uses default ThreadPoolExecutor

        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(
                f"Failed to initialize Silero VAD: {e}. Falling back to energy VAD."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_model(self) -> None:
        """Load Silero VAD ONNX model, downloading it if missing."""
        model_path = settings.vad.model_path

        if not os.path.exists(model_path):
            logger.info(
                f"Silero VAD model not found at '{model_path}'. Attempting download..."
            )
            self._download_model(model_path)

        try:
            import onnxruntime as ort

            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1

            self._session = ort.InferenceSession(
                model_path,
                sess_opts=session_options,
                providers=["CPUExecutionProvider"],
            )

            self._reset_hidden_state()
            logger.info("Silero VAD ONNX model initialized successfully.")

        except ImportError:
            logger.warning("onnxruntime not installed. Falling back to energy VAD.")
            raise
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise

    def _download_model(self, dest_path: str) -> None:
        """Download silero_vad.onnx from GitHub releases."""
        import urllib.request

        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        url = settings.vad.download_url
        logger.info(f"Downloading Silero VAD from {url} → {dest_path}")
        try:
            urllib.request.urlretrieve(url, dest_path)
            logger.info("Silero VAD model downloaded successfully.")
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            raise

    def _reset_hidden_state(self) -> None:
        """Reset LSTM hidden state to zeros for a new conversation (v5 uses 128-dim state)."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def is_speech(self, audio_frame: bytes) -> bool:
        """
        Analyse a 30ms PCM audio frame and return True if speech is detected.

        Args:
            audio_frame: Raw PCM bytes (960 bytes = 480 samples × 2 bytes, 16kHz 16-bit mono)

        Returns:
            True if speech is active, False for silence.
        """
        if len(audio_frame) == 0:
            return False

        # Decode PCM int16 → float32 in [-1, 1]
        audio_data = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0

        # Pad / truncate to exactly one VAD window
        expected = settings.audio.samples_per_frame
        if len(audio_data) < expected:
            audio_data = np.pad(audio_data, (0, expected - len(audio_data)))
        elif len(audio_data) > expected:
            audio_data = audio_data[:expected]

        try:
            if self._session is not None:
                probability = await self._run_vad_inference_async(audio_data)
            else:
                probability = self._energy_vad(audio_data)

            return self._apply_hysteresis(probability)

        except Exception as e:
            logger.error(f"VAD inference error: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Inference (blocking → offloaded to thread executor, BUG-14 fix)
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_vad_inference_async(self, audio_data: np.ndarray) -> float:
        """
        Offload the blocking ONNX session.run() to a thread executor so it
        never stalls the asyncio event loop (BUG-14 fix).
        """
        loop = asyncio.get_running_loop()  # BUG-15 fix: get_running_loop() not get_event_loop()
        probability = await loop.run_in_executor(
            self._executor,
            self._run_vad_inference_sync,
            audio_data,
        )
        return probability

    def _run_vad_inference_sync(self, audio_data: np.ndarray) -> float:
        """Synchronous Silero VAD inference — runs in thread pool."""
        if self._session is None or self._state is None:
            raise RuntimeError("VAD session not initialised.")

        inputs = {
            "input": audio_data.reshape(1, -1),
            "sr": np.array([self.sample_rate], dtype=np.int64),
            "state": self._state,
        }

        outputs = self._session.run(None, inputs)

        # Persist recurrent hidden state for the next frame
        self._state = outputs[1]

        return float(outputs[0][0])

    def _energy_vad(self, audio_data: np.ndarray) -> float:
        """
        Fallback energy-based VAD (no ML model required).
        Returns a probability-like value in [0, 1].
        """
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        # Normalise against a typical speech energy level
        speech_threshold = 0.02
        return min(1.0, rms / speech_threshold)

    # ─────────────────────────────────────────────────────────────────────────
    # Hysteresis
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_hysteresis(self, probability: float) -> bool:
        """
        Prevent rapid state flipping by using separate speech/silence thresholds
        and a hangover timer.
        """
        # BUG-15 fix: asyncio.get_running_loop() — safe to call from sync context
        # but we need monotonic time; use time.monotonic() here since we're sync.
        import time
        current_time = time.monotonic()

        if probability >= self.speech_threshold:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_time = current_time
                self._silence_start_time = None
            return True

        elif probability <= self.silence_threshold:
            if self._is_speaking:
                if self._silence_start_time is None:
                    self._silence_start_time = current_time
                else:
                    silence_ms = (current_time - self._silence_start_time) * 1000
                    if silence_ms >= self.hangover_time_ms:
                        self._is_speaking = False
                        self._speech_start_time = None
                        self._silence_start_time = None
            return False

        else:
            # Uncertain region — maintain current state (hysteresis plateau)
            return self._is_speaking

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset VAD state for a new conversation turn."""
        self._is_speaking = False
        self._silence_start_time = None
        self._speech_start_time = None
        if self._session is not None:
            self._reset_hidden_state()
        logger.debug("VAD state reset.")
