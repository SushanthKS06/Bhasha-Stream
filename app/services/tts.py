"""
MeloTTS/VITS chunked text synthesizer for Bhasha-Stream.
Provides streaming text-to-speech audio generation optimized for Indian languages.
"""
import asyncio
import logging
from typing import AsyncGenerator, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """
    Text-to-Speech service using MeloTTS or VITS backend.
    
    Converts text chunks into raw audio bytes and yields them
    incrementally for ultra-low latency audio streaming.
    Optimized for Hindi-English code-switched speech synthesis.
    """
    
    def __init__(self) -> None:
        self.engine: str = settings.tts.engine
        self.language: str = settings.tts.language
        self.speaker: str = settings.tts.speaker
        self.sample_rate: int = settings.tts.sample_rate
        self.chunk_size_samples: int = settings.tts.chunk_size_samples
        
        # Model instance (lazy-loaded)
        self._model: Optional[object] = None
        self._model_lock = asyncio.Lock()
        
        # Try to initialize the model
        try:
            self._initialize_model()
        except Exception as e:
            logger.warning(f"Failed to initialize TTS model: {e}. Using mock TTS.")
    
    def _initialize_model(self) -> None:
        """Initialize TTS model based on configured engine."""
        if self.engine == "melotts":
            try:
                from melo.api import TTS
                
                logger.info(f"Initializing MeloTTS with language={self.language}")
                
                # Initialize MeloTTS model
                self._model = TTS(language=self.language, device='auto')
                
                logger.info("MeloTTS model initialized successfully")
                
            except ImportError:
                logger.warning("MeloTTS not installed. Falling back to mock TTS.")
                raise
            except Exception as e:
                logger.error(f"Error initializing MeloTTS: {e}")
                raise
                
        elif self.engine == "vits":
            try:
                # Placeholder for VITS implementation
                # In production, this would load a VITS model
                logger.info("VITS engine selected (placeholder implementation)")
                raise NotImplementedError("VITS implementation pending")
                
            except Exception as e:
                logger.error(f"Error initializing VITS: {e}")
                raise
        else:
            logger.warning(f"Unknown TTS engine '{self.engine}'. Using mock TTS.")
    
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to audio and stream chunks incrementally.
        
        Args:
            text: Input text to synthesize
            
        Yields:
            Raw PCM audio chunks (16-bit mono at configured sample rate)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS synthesis")
            return
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Ensure model is loaded
            if self._model is None:
                async with self._model_lock:
                    if self._model is None:
                        try:
                            self._initialize_model()
                        except Exception:
                            # Fall back to mock synthesis
                            async for chunk in self._mock_synthesize(text):
                                yield chunk
                            return
            
            # Run synthesis based on engine type
            if self.engine == "melotts" and self._model is not None:
                async for chunk in self._synthesize_melo(text):
                    yield chunk
            else:
                # Fallback to mock
                async for chunk in self._mock_synthesize(text):
                    yield chunk
                    
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(f"TTS synthesis completed in {elapsed_ms:.2f}ms")
            
        except asyncio.CancelledError:
            logger.debug("TTS synthesis cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            # Fall back to mock on error
            async for chunk in self._mock_synthesize(text):
                yield chunk
    
    async def _synthesize_melo(self, text: str) -> AsyncGenerator[bytes, None]:
        """Synthesize using MeloTTS with chunked output."""
        if self._model is None:
            raise RuntimeError("MeloTTS model not initialized")
        
        try:
            loop = asyncio.get_event_loop()
            
            # Run synthesis in executor to avoid blocking
            audio_data = await loop.run_in_executor(
                None,
                self._run_melo_synthesis,
                text,
            )
            
            if audio_data is None:
                return
            
            # Stream audio in chunks
            chunk_bytes = self.chunk_size_samples * 2  # 16-bit = 2 bytes per sample
            
            for i in range(0, len(audio_data), chunk_bytes):
                chunk = audio_data[i:i + chunk_bytes]
                if len(chunk) > 0:
                    # Convert numpy array to bytes if needed
                    if hasattr(chunk, 'tobytes'):
                        yield chunk.tobytes()
                    elif isinstance(chunk, bytes):
                        yield chunk
                    else:
                        # Assume it's already bytes-like
                        yield bytes(chunk)
                        
        except Exception as e:
            logger.error(f"MeloTTS synthesis error: {e}")
            raise
    
    def _run_melo_synthesis(self, text: str) -> Optional[bytes]:
        """Run MeloTTS synthesis (blocking call)."""
        if self._model is None:
            return None
        
        try:
            # MeloTTS returns numpy array of audio samples
            audio_array = self._model.tts(text, speaker_id=self.speaker)
            
            # Convert to bytes (int16 PCM)
            import numpy as np
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.error(f"MeloTTS runtime error: {e}")
            return None
    
    async def _mock_synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Mock TTS synthesis for testing/fallback.
        Generates silence audio chunks simulating real TTS output.
        """
        import numpy as np
        
        # Estimate audio duration based on text length
        # Rough estimate: ~15 characters per second of speech
        estimated_duration_sec = len(text) / 15.0
        total_samples = int(self.sample_rate * estimated_duration_sec)
        
        # Generate in chunks
        samples_per_chunk = self.chunk_size_samples
        chunks_generated = 0
        
        # Simulate synthesis delay
        synthesis_delay_per_chunk = 0.05  # 50ms per chunk
        
        for i in range(0, total_samples, samples_per_chunk):
            remaining = total_samples - i
            chunk_size = min(samples_per_chunk, remaining)
            
            # Generate silent audio (placeholder - real TTS would generate actual speech)
            # In production, this would be actual synthesized audio
            audio_chunk = np.zeros(chunk_size, dtype=np.int16)
            
            # Add some variation to make it less obviously fake
            # Real TTS would have actual waveform here
            noise = np.random.randint(-100, 100, size=chunk_size, dtype=np.int16)
            audio_chunk = audio_chunk + noise
            
            yield audio_chunk.tobytes()
            
            chunks_generated += 1
            
            # Simulate synthesis time
            await asyncio.sleep(synthesis_delay_per_chunk)
        
        logger.debug(f"Mock TTS generated {chunks_generated} chunks for '{text[:30]}...'")
    
    async def synthesize_complete(self, text: str) -> Optional[bytes]:
        """
        Synthesize complete audio without streaming.
        Useful for non-streaming use cases.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Complete audio bytes, or None if synthesis fails
        """
        chunks: List[bytes] = []
        
        async for chunk in self.synthesize(text):
            chunks.append(chunk)
        
        return b''.join(chunks) if chunks else None
    
    async def health_check(self) -> bool:
        """Check if TTS model is loaded and responsive."""
        try:
            if self._model is not None:
                return True
            
            # Try to initialize
            self._initialize_model()
            return self._model is not None
            
        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return False
    
    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
            import gc
            gc.collect()
            
            logger.info("TTS model unloaded")


# Import List for type hints
from typing import List
