"""
Bhasha-Stream: Ultra-low latency duplex voice-to-voice AI agent.
Application entry point & FastAPI setup.

Fixes applied:
  - BUG-09: Removed shared queue initialization from app.state (queues are now per-connection)
  - ARCH-06: max_queue_size enforced at queue creation time in websocket.py
  - ARCH-09: loguru integrated as a drop-in stdlib logging sink
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

# ── Windows event loop fix ────────────────────────────────────────────────────
# aiohttp (used by LLMService) requires SelectorEventLoop on Windows.
# Python 3.11+ defaults to ProactorEventLoop which breaks async HTTP.
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ──────────────────────────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger as loguru_logger

from app.api.websocket import router as websocket_router
from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup (ARCH-09 fix: use loguru as advertised in requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    Route stdlib logging through loguru for structured, leveled output.
    This makes all logger.info() calls from every module go through loguru
    with consistent formatting and future OpenTelemetry compatibility.
    """
    # Remove loguru's default sink
    loguru_logger.remove()

    # Add a pretty console sink
    loguru_logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level="DEBUG",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # Intercept stdlib logging and redirect through loguru
    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno  # type: ignore[assignment]

            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back  # type: ignore[assignment]
                depth += 1

            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.DEBUG, force=True)

    # Silence extremely noisy third-party debug logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


_configure_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and graceful shutdown.

    Note: Shared queues have been removed (BUG-09 fix). All queue creation
    happens per-connection inside the WebSocket handler.
    """
    loguru_logger.info("━━━ Bhasha-Stream starting ━━━")
    loguru_logger.info(f"LLM endpoint : {settings.llm.endpoint_url}")
    loguru_logger.info(f"STT model    : {settings.stt.model_size} ({settings.stt.device})")
    loguru_logger.info(f"TTS engine   : {settings.tts.engine}")

    yield

    loguru_logger.info("━━━ Bhasha-Stream shutting down ━━━")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory function for creating and configuring the FastAPI application."""
    app = FastAPI(
        title="Bhasha-Stream",
        description=(
            "Ultra-low latency duplex voice-to-voice AI agent "
            "for Indian code-switched languages (Hinglish, Tanglish, etc.)"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — restrict origins in production via ALLOWED_ORIGINS env var
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers — websocket.py now includes both /health and /v1/stream
    app.include_router(websocket_router)

    # ── Test UI ───────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    async def get_test_page():
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()

    return app


app = create_app()
