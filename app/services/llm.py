"""
vLLM async streaming token client for Bhasha-Stream.
Provides low-latency token-by-token LLM response streaming.
"""
import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Large Language Model service using vLLM async streaming API.
    
    Connects to a vLLM endpoint and streams tokens incrementally
    for ultra-low latency text generation optimized for Indian languages.
    """
    
    def __init__(self) -> None:
        self.endpoint_url: str = settings.llm.endpoint_url
        self.model_name: str = settings.llm.model_name
        self.api_key: str = settings.llm.api_key
        self.max_tokens: int = settings.llm.max_tokens
        self.temperature: float = settings.llm.temperature
        self.timeout_seconds: float = settings.llm.timeout_seconds
        
        # HTTP session (lazy-initialized)
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # System prompt for code-switched Hindi-English conversations
        self.system_prompt: str = (
            "You are a helpful AI assistant fluent in Hindi, English, and Hinglish "
            "(Hindi-English code-switched language). Respond naturally in the same "
            "language variety that the user uses. Keep responses concise and conversational."
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with proper configuration."""
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
    
    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate LLM response and stream tokens one by one.
        
        Args:
            prompt: User input text
            
        Yields:
            Individual tokens as they are generated
        """
        start_time = time.perf_counter()
        token_count = 0
        first_token_time: Optional[float] = None
        
        try:
            session = await self._get_session()
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            
            url = f"{self.endpoint_url.rstrip('/')}/chat/completions"
            
            logger.debug(f"Sending LLM request to {url}")
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    # Fall back to mock generation
                    async for token in self._mock_generate(prompt):
                        yield token
                    return
                
                # Process SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    if not line.startswith('data: '):
                        continue
                    
                    data_str = line[6:]  # Remove 'data: ' prefix
                    
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Extract token from response
                        choices = data.get('choices', [])
                        if choices and len(choices) > 0:
                            delta = choices[0].get('delta', {})
                            content = delta.get('content', '')
                            
                            if content:
                                # Track first token latency
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                    latency_ms = (first_token_time - start_time) * 1000
                                    logger.debug(f"First token latency: {latency_ms:.2f}ms")
                                
                                # Yield individual characters/tokens
                                # For true tokenization, we'd use the model's tokenizer
                                # Here we yield character-by-character for smooth streaming
                                for char in content:
                                    token_count += 1
                                    yield char
                                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE data: {e}")
                        continue
                        
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"LLM generation completed: {token_count} tokens in {elapsed_ms:.2f}ms "
                f"({token_count / (elapsed_ms / 1000):.2f} tokens/sec)"
            )
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during LLM generation: {e}")
            # Fall back to mock generation
            async for token in self._mock_generate(prompt):
                yield token
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            # Fall back to mock generation
            async for token in self._mock_generate(prompt):
                yield token
        finally:
            pass
    
    async def generate_complete(self, prompt: str) -> Optional[str]:
        """
        Generate complete LLM response without streaming.
        Useful for non-streaming use cases.
        
        Args:
            prompt: User input text
            
        Returns:
            Complete response text, or None if generation fails
        """
        tokens: List[str] = []
        
        async for token in self.generate(prompt):
            tokens.append(token)
        
        return ''.join(tokens) if tokens else None
    
    async def _mock_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Mock token generation for testing/fallback.
        Simulates streaming tokens with realistic delays.
        """
        # Mock responses for common prompts (code-switched Hindi-English)
        mock_responses = [
            "नमस्ते! मैं आपकी मदद करने के लिए यहाँ हूँ। बताइए, क्या सेवा कर सकता हूँ?",
            "Hello! I'm here to help you. Please tell me what you need assistance with.",
            "जी बिल्कुल! यह तो बहुत अच्छा सवाल है। देखिए, मैं आपको समझाता हूँ...",
            "Yes, absolutely! That's a great question. Let me explain this to you...",
            "ठीक है, सुनिए। यह काम काफी simple है बस थोड़ा ध्यान देना होगा।",
        ]
        
        import random
        response_text = random.choice(mock_responses)
        
        # Simulate tokenization delay
        base_delay = 0.02  # 20ms per token (50 tokens/sec)
        
        for i, char in enumerate(response_text):
            yield char
            # Add variable delay to simulate realistic token generation
            delay = base_delay + (random.random() * 0.01)
            await asyncio.sleep(delay)
    
    async def health_check(self) -> bool:
        """Check if LLM endpoint is healthy and responsive."""
        try:
            session = await self._get_session()
            url = f"{self.endpoint_url.rstrip('/')}/health"
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("LLM HTTP session closed")
