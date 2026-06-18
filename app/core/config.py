"""
Application configurations & thresholds for Bhasha-Stream.

Centralized configuration management using pydantic-settings so that
every field can be overridden cleanly via environment variables or a
.env file — not evaluated at class-definition import time (BUG-19 fix).

Usage:
    from app.core.config import settings
    settings.llm.endpoint_url  # reads VLLM_ENDPOINT at runtime
"""
from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# Shared base config dict — all sub-configs read from the same .env file
# and suppress the model_ namespace warning
# ─────────────────────────────────────────────────────────────────────────────

def _base_cfg(**kwargs) -> SettingsConfigDict:
    """Factory for a consistent SettingsConfigDict across all sub-configs."""
    return SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),   # suppresses model_ namespace UserWarnings
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sub-config models
# ─────────────────────────────────────────────────────────────────────────────

class AudioConfig(BaseSettings):
    """Audio processing configuration."""

    model_config = _base_cfg(env_prefix="AUDIO_")

    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    frame_duration_ms: int = 30
    samples_per_frame: int = 480    # 16000 * 0.030
    bytes_per_frame: int = 960      # 480 * 2 (16-bit = 2 bytes/sample)

    # VAD thresholds
    vad_speech_threshold: float = 0.5
    vad_silence_threshold: float = 0.35
    vad_hangover_time_ms: int = 300
    vad_min_speech_duration_ms: int = 200


class STTConfig(BaseSettings):
    """Speech-to-Text configuration."""

    model_config = _base_cfg(env_prefix="STT_")

    model_size: str = "tiny"
    device: str = "cpu"
    compute_type: str = "float32"
    language: str = "hi"
    beam_size: int = 5
    best_of: int = 5

    @field_validator("device", mode="before")
    @classmethod
    def coerce_device(cls, v: str) -> str:
        if v == "auto":
            import os
            return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        return v


class LLMConfig(BaseSettings):
    """Large Language Model configuration."""

    # No env_prefix — uses full env var names (VLLM_ENDPOINT, LLM_MODEL, etc.)
    model_config = _base_cfg()

    endpoint_url: str = Field(
        default="http://localhost:8001/v1",
        alias="VLLM_ENDPOINT",
        validation_alias="VLLM_ENDPOINT",
    )
    model_name: str = Field(
        default="sarvamai/sarvam-m",
        alias="LLM_MODEL",
        validation_alias="LLM_MODEL",
    )
    api_key: str = Field(
        default="bhasha-local-key",
        alias="LLM_API_KEY",
        validation_alias="LLM_API_KEY",
    )
    max_tokens: int = 256
    temperature: float = 0.7
    timeout_seconds: float = 30.0

    chunk_delimiters: str = ".?!"
    max_words_before_flush: int = 7


class TTSConfig(BaseSettings):
    """Text-to-Speech configuration."""

    model_config = _base_cfg(env_prefix="TTS_")

    engine: str = "melotts"
    language: str = "EN"
    speaker_id: int = 0
    sample_rate: int = 24000
    output_sample_rate: int = 16000
    chunk_size_samples: int = 480


class NetworkConfig(BaseSettings):
    """Network and WebSocket configuration."""

    model_config = _base_cfg()

    websocket_timeout_seconds: float = 30.0
    connection_timeout_seconds: float = 10.0
    # Stored as plain str to avoid pydantic-settings JSON pre-parsing List[str].
    # Use the .allowed_origins property (or Settings.allowed_origins) for a list.
    allowed_origins_str: str = Field(
        default="*",
        alias="ALLOWED_ORIGINS",
        validation_alias="ALLOWED_ORIGINS",
    )
    max_queue_size: int = 100

    @property
    def allowed_origins(self) -> List[str]:
        """Split comma-separated origins string into a list."""
        return [o.strip() for o in self.allowed_origins_str.split(",") if o.strip()]


class VADConfig(BaseSettings):
    """VAD model configuration."""

    model_config = _base_cfg(env_prefix="VAD_")

    model_path: str = "models/silero_vad.onnx"
    # Current ONNX file location in the silero-vad repo (updated from old /files/ path)
    download_url: str = (
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    )


class LatencyConfig(BaseSettings):
    """Latency targets and monitoring configuration."""

    model_config = _base_cfg(env_prefix="LATENCY_")

    target_rtt_ms: int = 800
    stt_latency_budget_ms: int = 200
    llm_first_token_latency_ms: int = 300
    tts_first_chunk_latency_ms: int = 200
    interruption_response_ms: int = 50


# ─────────────────────────────────────────────────────────────────────────────
# Root settings container
# ─────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Root settings — composes all sub-configs."""

    model_config = _base_cfg()

    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)

    @property
    def websocket_timeout_seconds(self) -> float:
        return self.network.websocket_timeout_seconds

    @property
    def allowed_origins(self) -> List[str]:
        return self.network.allowed_origins


# Singleton — import and use everywhere
settings = Settings()
