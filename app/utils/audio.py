"""
Raw 16kHz PCM audio manipulation utilities for Bhasha-Stream.
Provides low-level audio processing functions for frame handling and conversion.
"""
import logging
from typing import Generator, List, Tuple

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


def bytes_to_audio_array(
    audio_bytes: bytes,
    sample_rate: int = settings.audio.sample_rate,
    dtype: np.dtype = np.int16,
) -> np.ndarray:
    """
    Convert raw PCM bytes to numpy array.
    
    Args:
        audio_bytes: Raw PCM audio bytes
        sample_rate: Audio sample rate (default 16000)
        dtype: Data type of audio samples (default int16)
        
    Returns:
        Numpy array of audio samples normalized to float32 [-1.0, 1.0]
    """
    if len(audio_bytes) == 0:
        return np.array([], dtype=np.float32)
    
    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_bytes, dtype=dtype)
    
    # Normalize to float32 range [-1.0, 1.0]
    if dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif dtype == np.int32:
        audio_array = audio_array.astype(np.float32) / 2147483648.0
    else:
        audio_array = audio_array.astype(np.float32)
    
    return audio_array


def audio_array_to_bytes(
    audio_array: np.ndarray,
    dtype: np.dtype = np.int16,
) -> bytes:
    """
    Convert numpy audio array back to raw PCM bytes.
    
    Args:
        audio_array: Numpy array of audio samples (assumed float32 [-1.0, 1.0])
        dtype: Target data type for output (default int16)
        
    Returns:
        Raw PCM audio bytes
    """
    if len(audio_array) == 0:
        return b''
    
    # Convert from float to integer format
    if dtype == np.int16:
        audio_int = (audio_array * 32767).astype(np.int16)
    elif dtype == np.int32:
        audio_int = (audio_array * 2147483647).astype(np.int32)
    else:
        audio_int = audio_array.astype(dtype)
    
    return audio_int.tobytes()


def split_audio_into_frames(
    audio_data: bytes,
    frame_size_bytes: int = settings.audio.bytes_per_frame,
) -> Generator[bytes, None, None]:
    """
    Split continuous audio data into fixed-size frames.
    
    Args:
        audio_data: Continuous PCM audio bytes
        frame_size_bytes: Size of each frame in bytes (default 960 for 30ms at 16kHz 16-bit)
        
    Yields:
        Individual audio frames
    """
    offset = 0
    total_length = len(audio_data)
    
    while offset < total_length:
        end_offset = min(offset + frame_size_bytes, total_length)
        frame = audio_data[offset:end_offset]
        
        # Only yield complete frames (optional - can be configured)
        if len(frame) == frame_size_bytes:
            yield frame
        elif len(frame) > 0:
            # Yield partial frame with padding
            padding_needed = frame_size_bytes - len(frame)
            padded_frame = frame + (b'\x00' * padding_needed)
            yield padded_frame
        
        offset = end_offset


def merge_frames_to_audio(frames: List[bytes]) -> bytes:
    """
    Merge multiple audio frames into continuous audio data.
    
    Args:
        frames: List of audio frame bytes
        
    Returns:
        Merged continuous PCM audio bytes
    """
    return b''.join(frames)


def calculate_audio_duration_ms(
    audio_bytes: bytes,
    sample_rate: int = settings.audio.sample_rate,
    bits_per_sample: int = settings.audio.bits_per_sample,
    channels: int = settings.audio.channels,
) -> float:
    """
    Calculate duration of audio data in milliseconds.
    
    Args:
        audio_bytes: Raw PCM audio bytes
        sample_rate: Audio sample rate
        bits_per_sample: Bits per sample (e.g., 16)
        channels: Number of audio channels
        
    Returns:
        Duration in milliseconds
    """
    bytes_per_second = (sample_rate * bits_per_sample * channels) // 8
    duration_sec = len(audio_bytes) / bytes_per_second
    return duration_sec * 1000


def resample_audio(
    audio_array: np.ndarray,
    original_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio array from one sample rate to another.
    
    Args:
        audio_array: Input audio samples (float32)
        original_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if original_sr == target_sr:
        return audio_array
    
    try:
        import librosa
        resampled = librosa.resample(
            audio_array,
            orig_sr=original_sr,
            target_sr=target_sr,
        )
        return resampled
    except ImportError:
        logger.warning("librosa not installed. Using naive resampling.")
        # Naive resampling (not recommended for production)
        ratio = target_sr / original_sr
        new_length = int(len(audio_array) * ratio)
        indices = np.linspace(0, len(audio_array) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_array)), audio_array)


def normalize_audio_level(
    audio_array: np.ndarray,
    target_dbfs: float = -20.0,
) -> np.ndarray:
    """
    Normalize audio to target dBFS level.
    
    Args:
        audio_array: Input audio samples (float32 [-1.0, 1.0])
        target_dbfs: Target level in dBFS (default -20)
        
    Returns:
        Normalized audio array
    """
    if len(audio_array) == 0:
        return audio_array
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio_array ** 2))
    
    if rms < 1e-10:
        # Signal too quiet, return as-is
        return audio_array
    
    # Calculate current dBFS
    current_dbfs = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain with clipping prevention
    normalized = audio_array * gain_linear
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def detect_clipping(audio_array: np.ndarray, threshold: float = 0.99) -> bool:
    """
    Detect if audio signal has clipping (samples at max amplitude).
    
    Args:
        audio_array: Input audio samples (float32 [-1.0, 1.0])
        threshold: Amplitude threshold for clipping detection
        
    Returns:
        True if clipping detected, False otherwise
    """
    if len(audio_array) == 0:
        return False
    
    max_amplitude = np.max(np.abs(audio_array))
    return max_amplitude >= threshold


def apply_gain(audio_array: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain adjustment to audio.
    
    Args:
        audio_array: Input audio samples (float32 [-1.0, 1.0])
        gain_db: Gain in decibels (positive = louder, negative = quieter)
        
    Returns:
        Gain-adjusted audio array (clipped to [-1.0, 1.0])
    """
    gain_linear = 10 ** (gain_db / 20)
    adjusted = audio_array * gain_linear
    return np.clip(adjusted, -1.0, 1.0)


def create_silence_frame(
    frame_size_bytes: int = settings.audio.bytes_per_frame,
) -> bytes:
    """
    Create a silence frame (all zeros).
    
    Args:
        frame_size_bytes: Size of frame in bytes
        
    Returns:
        Silence frame bytes
    """
    return b'\x00' * frame_size_bytes


def validate_audio_frame(
    frame: bytes,
    expected_size: int = settings.audio.bytes_per_frame,
) -> Tuple[bool, str]:
    """
    Validate an audio frame meets requirements.
    
    Args:
        frame: Audio frame bytes to validate
        expected_size: Expected frame size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(frame) == 0:
        return False, "Empty frame"
    
    if len(frame) != expected_size:
        return False, f"Frame size mismatch: got {len(frame)}, expected {expected_size}"
    
    # Check for all zeros (silence)
    if frame == b'\x00' * len(frame):
        return True, "Valid silence frame"
    
    # Check for valid PCM data (basic sanity check)
    audio_array = np.frombuffer(frame, dtype=np.int16)
    if np.all(np.abs(audio_array) > 32000):
        return False, "Suspicious: all samples near maximum amplitude"
    
    return True, "Valid audio frame"


def extract_features(
    audio_array: np.ndarray,
    sample_rate: int = settings.audio.sample_rate,
) -> dict:
    """
    Extract basic audio features for analysis.
    
    Args:
        audio_array: Input audio samples (float32)
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary of audio features
    """
    if len(audio_array) == 0:
        return {}
    
    features = {
        'rms_energy': float(np.sqrt(np.mean(audio_array ** 2))),
        'peak_amplitude': float(np.max(np.abs(audio_array))),
        'zero_crossing_rate': float(
            np.sum(np.diff(np.sign(audio_array)) != 0) / len(audio_array)
        ),
        'duration_ms': calculate_audio_duration_ms(
            audio_array.tobytes(),
            sample_rate,
        ),
    }
    
    # Try to get spectral features if librosa is available
    try:
        import librosa
        
        # Zero-padding for FFT
        fft_size = 2048
        spectrum = np.abs(librosa.stft(audio_array, n_fft=fft_size))
        
        features['spectral_centroid'] = float(
            np.mean(librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0])
        )
        
    except ImportError:
        pass
    
    return features
