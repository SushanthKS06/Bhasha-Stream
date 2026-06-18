"""
vLLM async streaming token client for Bhasha-Stream.
Provides low-latency token-by-token LLM response streaming via OpenAI-compatible SSE.

Fixes applied:
  - ARCH-05: Conversation history maintained across turns within a session
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from app.core.config import settings

logger = logging.getLogger(__name__)

# Maximum number of conversation turns to retain in context
MAX_HISTORY_TURNS = 10


class LLMService:
    """
    Large Language Model service backed by a vLLM OpenAI-compatible endpoint.

    Streams tokens incrementally for ultra-low latency text generation,
    optimised for Indian code-switched languages (Hinglish, Tanglish, etc.).

    Maintains a rolling conversation history per service instance so the agent
    has contextual memory across turns within a single WebSocket session.
    """

    def __init__(self) -> None:
        self.endpoint_url: str = settings.llm.endpoint_url
        self.model_name: str = settings.llm.model_name
        self.api_key: str = settings.llm.api_key
        self.max_tokens: int = settings.llm.max_tokens
        self.temperature: float = settings.llm.temperature
        self.timeout_seconds: float = settings.llm.timeout_seconds

        # Lazy HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # ── Conversation history (ARCH-05 fix) ────────────────────────────
        # Each entry: {"role": "user"|"assistant", "content": "..."}
        self._history: List[Dict[str, str]] = []

        # System prompt sets the agent's persona and language behaviour
        self._system_prompt: str = (
            "You are a helpful AI assistant fluent in Hindi, English, and Hinglish "
            "(Hindi-English code-switched language). "
            "Respond naturally in the same language variety that the user uses. "
            "Keep responses concise and conversational — aim for 1-3 sentences. "
            "Do not use markdown, bullet points, or formatting symbols in your responses "
            "as your output will be converted directly to speech."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HTTP session management
    # ─────────────────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp session."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
        return self._session

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Append user prompt to history, stream assistant tokens, then persist
        the full response back into history for the next turn.

        Args:
            prompt: Raw user utterance (transcribed speech)

        Yields:
            Individual UTF-8 characters/tokens as they arrive from vLLM.
        """
        # Append user turn to history
        self._history.append({"role": "user", "content": prompt})
        self._trim_history()

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0
        full_response_chars: List[str] = []

        try:
            session = await self._get_session()

            messages = [{"role": "system", "content": self._system_prompt}]
            messages.extend(self._history)

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                "stream_options": {"include_usage": False},
            }

            url = f"{self.endpoint_url.rstrip('/')}/chat/completions"
            logger.debug(f"LLM request → {url} | history_turns={len(self._history)}")

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API {response.status}: {error_text}")
                    async for token in self._mock_generate(prompt):
                        full_response_chars.append(token)
                        yield token
                    return

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()

                    if not line:
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # strip "data: "

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE data: {e}")
                        continue

                    logger.debug(f"Raw chunk: {data_str}")

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    content = choices[0].get("delta", {}).get("content")

                    if not content:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft_ms = (first_token_time - start_time) * 1000
                        logger.debug(f"LLM TTFT: {ttft_ms:.1f}ms")

                    for char in content:
                        token_count += 1
                        full_response_chars.append(char)
                        yield char

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"LLM done: {token_count} chars in {elapsed_ms:.1f}ms "
                f"({token_count / max(elapsed_ms / 1000, 0.001):.0f} chars/sec)"
            )

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during LLM generation: {e}")
            async for token in self._mock_generate(prompt):
                full_response_chars.append(token)
                yield token

        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            async for token in self._mock_generate(prompt):
                full_response_chars.append(token)
                yield token

        finally:
            # Persist assistant response into history regardless of success/failure
            if full_response_chars:
                assistant_text = "".join(full_response_chars)
                self._history.append({"role": "assistant", "content": assistant_text})
                self._trim_history()

    async def generate_complete(self, prompt: str) -> Optional[str]:
        """
        Generate a complete response (non-streaming). Useful for pre-fetching
        or non-latency-critical code paths.
        """
        tokens: List[str] = []
        async for token in self.generate(prompt):
            tokens.append(token)
        return "".join(tokens) if tokens else None

    def clear_history(self) -> None:
        """Clear conversation history (e.g., on session end or topic reset)."""
        self._history.clear()
        logger.debug("LLM conversation history cleared.")

    async def health_check(self) -> bool:
        """Ping the vLLM /health endpoint."""
        try:
            session = await self._get_session()
            url = f"{self.endpoint_url.rstrip('/')}/health"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the aiohttp session and release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("LLM HTTP session closed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _trim_history(self) -> None:
        """Keep only the most recent MAX_HISTORY_TURNS user+assistant pairs."""
        # Each "turn" is 2 messages (user + assistant). Keep the last N pairs.
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    async def _mock_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Mock streaming response for CI/CD or when vLLM is unavailable.
        Simulates realistic per-token delays.
        """
        import random

        mock_responses = [
            "नमस्ते! मैं आपकी मदद करने के लिए यहाँ हूँ। बताइए, क्या सेवा कर सकता हूँ?",
            "Hello! I'm here to help you. Please tell me what you need assistance with.",
            "जी बिल्कुल! यह तो बहुत अच्छा सवाल है। देखिए, मैं आपको समझाता हूँ।",
            "Yes, absolutely! That's a great question. Let me explain this to you.",
            "ठीक है, सुनिए। यह काम काफी simple है बस थोड़ा ध्यान देना होगा।",
        ]

        response_text = random.choice(mock_responses)
        base_delay = 0.02  # 20ms per character → ~50 chars/sec

        for char in response_text:
            yield char
            await asyncio.sleep(base_delay + random.uniform(0, 0.005))
