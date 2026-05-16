"""
Bhasha-Stream: Ultra-low latency duplex voice-to-voice AI agent.
Application entry point & FastAPI setup.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.websocket import router as websocket_router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle and shared resources."""
    logger.info("Starting Bhasha-Stream backend...")
    
    # Initialize shared queues for the application
    app.state.inbound_audio_queue = asyncio.Queue()
    app.state.outbound_audio_queue = asyncio.Queue()
    app.state.interrupt_queue = asyncio.Queue()
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Bhasha-Stream backend...")
    # Clear queues
    while not app.state.inbound_audio_queue.empty():
        try:
            app.state.inbound_audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    while not app.state.outbound_audio_queue.empty():
        try:
            app.state.outbound_audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="Bhasha-Stream",
        description="Ultra-low latency duplex voice-to-voice AI agent for Indian code-switched languages",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(websocket_router)
    
    return app


app = create_app()
