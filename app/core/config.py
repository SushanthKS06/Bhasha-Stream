"""
Application configurations & thresholds for Bhasha-Stream.
Centralized configuration management with environment variable support.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_duration_ms: int = 30
    samples_per_frame: int = 480  # 16000 * 0.030
    bytes_per_frame: int = 960  # 480 * 2 (16-bit)
    
    # VAD Configuration
    vad_speech_threshold: float = 0.5
    vad_silence_threshold: float = 0.35
    vad_hangover_time_ms: int = 300  # Silence duration before triggering endpoint
    vad_min_speech_duration_ms: int = 200  # Minimum speech to consider valid


@dataclass(frozen=True)
class STTConfig:
    """Speech-to-Text configuration."""
    model_size: str = "tiny"  # Options: tiny, base, small, medium, large
    device: str = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    compute_type: str = "float16"
    language: str = "hi"  # Default to Hindi for code-switched
    beam_size: int = 5
    best_of: int = 5


@dataclass(frozen=True)
class LLMConfig:
    """Large Language Model configuration."""
    endpoint_url: str = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
    model_name: str = os.getenv("LLM_MODEL", "sarvam-m")
    api_key: str = os.getenv("LLM_API_KEY", "dummy-key")
    max_tokens: int = 256
    temperature: float = 0.7
    timeout_seconds: float = 30.0
    
    # Streaming configuration
    chunk_delimiters: str = ".?!"  # Punctuation that triggers TTS synthesis
    max_words_before_flush: int = 7  # Max words to buffer before flushing to TTS


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-Speech configuration."""
    engine: str = "melotts"  # Options: melotts, vits
    language: str = "hi-en"  # Code-switched Hindi-English
    speaker: str = "female"
    sample_rate: int = 16000
    chunk_size_samples: int = 480  # Match audio frame size


@dataclass(frozen=True)
class NetworkConfig:
    """Network and WebSocket configuration."""
    websocket_timeout_seconds: float = 30.0
    connection_timeout_seconds: float = 10.0
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    max_queue_size: int = 100  # Max frames in queue before dropping


@dataclass(frozen=True)
class LatencyConfig:
    """Latency targets and monitoring configuration."""
    target_rtt_ms: int = 800  # Round-trip time target
    stt_latency_budget_ms: int = 200
    llm_first_token_latency_ms: int = 300
    tts_first_chunk_latency_ms: int = 200
    interruption_response_ms: int = 50  # Must respond to interrupt within this time


@dataclass(frozen=True)
class Settings:
    """Main settings container combining all configurations."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    
    # Convenience properties for direct access
    @property
    def websocket_timeout_seconds(self) -> float:
        return self.network.websocket_timeout_seconds
    
    @property
    def allowed_origins(self) -> List[str]:
        return self.network.allowed_origins


# Global settings instance
settings = Settings()
