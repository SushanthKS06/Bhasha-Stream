"""
Silero VAD speech detection state for Bhasha-Stream.
Uses ONNX Runtime for efficient voice activity detection on 30ms windows.
"""
import asyncio
import logging
from typing import Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class VADService:
    """
    Voice Activity Detection service using Silero VAD model.
    
    Analyzes 30ms audio windows (480 samples at 16kHz) and returns
    boolean speech detection states with hangover time tracking.
    """
    
    def __init__(self) -> None:
        self.sample_rate: int = settings.audio.sample_rate
        self.speech_threshold: float = settings.audio.vad_speech_threshold
        self.silence_threshold: float = settings.audio.vad_silence_threshold
        self.hangover_time_ms: int = settings.audio.vad_hangover_time_ms
        
        # State tracking
        self._is_speaking: bool = False
        self._silence_start_time: Optional[float] = None
        self._speech_start_time: Optional[float] = None
        
        # Try to load the Silero VAD model
        self._model: Optional[object] = None
        self._session: Optional[object] = None
        self._h: Optional[np.ndarray] = None  # Hidden state
        self._c: Optional[np.ndarray] = None  # Cell state
        
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize Silero VAD model: {e}. Using mock VAD.")
    
    def _initialize_model(self) -> None:
        """Initialize Silero VAD model with ONNX Runtime."""
        try:
            import onnxruntime as ort
            
            # Load Silero VAD model (would be downloaded from GitHub in production)
            # For now, we'll use a placeholder path
            model_path = "silero_vad.onnx"
            
            # Initialize ONNX session
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            
            self._session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            
            # Initialize hidden states
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
            
            logger.info("Silero VAD model initialized successfully")
            
        except FileNotFoundError:
            # Model file not found - will use mock implementation
            logger.warning("Silero VAD model file not found. Using mock VAD.")
            raise
        except ImportError:
            logger.warning("onnxruntime not installed. Using mock VAD.")
            raise
    
    async def is_speech(self, audio_frame: bytes) -> bool:
        """
        Analyze audio frame and return speech detection result.
        
        Args:
            audio_frame: Raw PCM audio bytes (expecting 960 bytes for 30ms at 16kHz 16-bit mono)
            
        Returns:
            True if speech is detected, False otherwise
        """
        if len(audio_frame) == 0:
            return False
        
        # Convert bytes to numpy array (int16 -> float32 normalized)
        audio_data = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Validate frame size
        expected_samples = settings.audio.samples_per_frame
        if len(audio_data) != expected_samples:
            # Resample or pad if necessary
            if len(audio_data) < expected_samples:
                audio_data = np.pad(audio_data, (0, expected_samples - len(audio_data)))
            else:
                audio_data = audio_data[:expected_samples]
        
        try:
            if self._session is not None:
                # Run Silero VAD inference
                probability = await self._run_vad_inference(audio_data)
            else:
                # Mock VAD - simulate speech detection based on energy
                probability = await self._mock_vad_inference(audio_data)
            
            # Apply hysteresis to prevent rapid state flipping
            current_time = asyncio.get_event_loop().time()
            
            if probability >= self.speech_threshold:
                # Speech detected
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_start_time = current_time
                    self._silence_start_time = None
                return True
                
            elif probability <= self.silence_threshold:
                # Silence detected
                if self._is_speaking:
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time
                    else:
                        # Check if hangover time exceeded
                        silence_duration_ms = (current_time - self._silence_start_time) * 1000
                        if silence_duration_ms >= self.hangover_time_ms:
                            self._is_speaking = False
                            self._speech_start_time = None
                            self._silence_start_time = None
                return False
            else:
                # In uncertain region - maintain current state
                return self._is_speaking
                
        except Exception as e:
            logger.error(f"VAD inference error: {e}")
            return False
    
    async def _run_vad_inference(self, audio_data: np.ndarray) -> float:
        """Run actual Silero VAD inference via ONNX Runtime."""
        if self._session is None or self._h is None or self._c is None:
            raise RuntimeError("VAD model not initialized")
        
        # Prepare inputs for Silero VAD
        # Input: audio (batch_size, samples), sr (sample rate), h, c (hidden states)
        inputs = {
            'input': audio_data.reshape(1, -1),
            'sr': np.array([self.sample_rate], dtype=np.int64),
            'h': self._h,
            'c': self._c,
        }
        
        # Run inference
        outputs = self._session.run(None, inputs)
        
        # Update hidden states for next frame
        self._h = outputs[1]
        self._c = outputs[2]
        
        # Get probability (first output)
        probability = float(outputs[0][0])
        
        return probability
    
    async def _mock_vad_inference(self, audio_data: np.ndarray) -> float:
        """
        Mock VAD inference based on audio energy.
        Used when Silero model is not available.
        """
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Simple energy-based threshold (calibrated for typical speech levels)
        # This is a placeholder - real VAD needs proper ML model
        speech_energy_threshold = 0.02  # Adjust based on testing
        
        # Normalize to probability-like value (0-1 range)
        probability = min(1.0, rms_energy / speech_energy_threshold)
        
        # Add some noise to make it more realistic
        noise = np.random.uniform(-0.1, 0.1)
        probability = max(0.0, min(1.0, probability + noise))
        
        return probability
    
    def reset(self) -> None:
        """Reset VAD state (useful for new conversation turns)."""
        self._is_speaking = False
        self._silence_start_time = None
        self._speech_start_time = None
        
        # Reset hidden states if using Silero
        if self._h is not None:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
        if self._c is not None:
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        
        logger.debug("VAD state reset")
