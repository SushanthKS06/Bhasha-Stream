"""
WebSocket protocol layer & network IO for Bhasha-Stream.
Handles binary PCM 16kHz mono audio frames for bidirectional duplex streaming.

Fixes applied:
  - BUG-09: Per-connection queues (not shared app-state queues)
  - ARCH-07: /health HTTP endpoint added
  - ARCH-08: Graceful WebSocket close handling
  - ARCH-06: Queue maxsize from config applied at creation
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from app.core.config import settings
from app.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Health check endpoint (ARCH-07 fix)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["ops"])
async def health_check() -> JSONResponse:
    """
    HTTP health probe used by Docker, Kubernetes, and load balancers.
    Returns 200 OK when the application is ready to serve requests.
    """
    return JSONResponse({"status": "ok", "service": "bhasha-stream"})


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket I/O coroutines
# ─────────────────────────────────────────────────────────────────────────────

async def _websocket_reader(
    websocket: WebSocket,
    inbound_queue: asyncio.Queue,
) -> None:
    """
    Continuously read raw binary PCM frames from the WebSocket client
    and push them onto the per-connection inbound queue.

    Terminates cleanly on WebSocketDisconnect or CancelledError.
    """
    frames_received = 0
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.websocket_timeout_seconds,
                )

                if message["type"] == "websocket.receive":
                    if "text" in message and message["text"] == "STOP":
                        logger.info("Received manual STOP signal from client")
                        if not inbound_queue.full():
                            await inbound_queue.put(b"STOP_SIGNAL")
                    elif "bytes" in message:
                        data = message["bytes"]
                        if not data:
                            continue

                        frames_received += 1
                        if frames_received % 100 == 0:
                            logger.debug(f"Server received {frames_received} audio frames")

                        # Drop frame if queue is at capacity (backpressure, ARCH-06 fix)
                        if inbound_queue.full():
                            logger.warning("Inbound queue full — dropping oldest frame")
                            try:
                                inbound_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass

                        await inbound_queue.put(data)
                        
                elif message["type"] == "websocket.disconnect":
                    logger.info("Client disconnected (reader)")
                    return

            except asyncio.TimeoutError:
                # Send WebSocket ping to keep connection alive
                try:
                    await websocket.send_bytes(b"")  # keepalive
                except Exception:
                    break
                continue

            except WebSocketDisconnect:
                logger.info("Client disconnected (reader)")
                return

    except asyncio.CancelledError:
        logger.debug("WebSocket reader cancelled")
        raise
    except Exception as e:
        logger.error(f"WebSocket reader error: {e}", exc_info=True)
        raise


async def _websocket_writer(
    websocket: WebSocket,
    outbound_queue: asyncio.Queue,
    interrupt_queue: asyncio.Queue,
) -> None:
    """
    Continuously read synthesized audio chunks from the outbound queue
    and send them to the client. Handles interruption signals by flushing
    the queue and sending a 4-byte null stop-signal to the client.

    Terminates cleanly on WebSocketDisconnect or CancelledError.
    """
    try:
        while True:
            # ── Priority: check for interrupt signal first ─────────────────
            try:
                interrupt_signal = interrupt_queue.get_nowait()
                if interrupt_signal:
                    # 4-byte null sentinel tells the client to halt playback
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_bytes(b"\x00\x00\x00\x00")
                    logger.debug("Sent interruption signal to client")

                    # Discard queued audio that is now stale
                    while not outbound_queue.empty():
                        try:
                            outbound_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    continue

            except asyncio.QueueEmpty:
                pass  # No interrupt pending

            # ── Normal path: send next audio chunk ────────────────────────
            try:
                item = await asyncio.wait_for(
                    outbound_queue.get(),
                    timeout=0.05,  # 50ms — short enough to re-check interrupt
                )

                if websocket.client_state == WebSocketState.CONNECTED:
                    if isinstance(item, bytes):
                        await websocket.send_bytes(item)
                    elif isinstance(item, str):
                        await websocket.send_text(item)

            except asyncio.TimeoutError:
                continue  # Loop back to check interrupt queue

            except WebSocketDisconnect:
                logger.info("Client disconnected (writer)")
                return

    except asyncio.CancelledError:
        logger.debug("WebSocket writer cancelled")
        raise
    except Exception as e:
        logger.error(f"WebSocket writer error: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Main WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/v1/stream")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Bidirectional audio streaming endpoint.

    Accepts binary PCM 16kHz mono audio frames from the client and streams
    synthesized speech audio back.

    BUG-09 fix: Each connection gets its own isolated queues — no shared state
    between concurrent clients.

    ARCH-06 fix: Queues are bounded by NetworkConfig.max_queue_size to prevent
    unbounded memory growth under load.
    """
    await websocket.accept()
    client_info = websocket.client
    logger.info(f"WebSocket connected: {client_info}")

    max_q = settings.network.max_queue_size

    # ── Per-connection isolated queues (BUG-09 fix) ───────────────────────
    inbound_queue: asyncio.Queue = asyncio.Queue(maxsize=max_q)
    outbound_queue: asyncio.Queue = asyncio.Queue(maxsize=max_q)
    interrupt_queue: asyncio.Queue = asyncio.Queue(maxsize=16)

    # ── One Orchestrator per connection ───────────────────────────────────
    orchestrator = Orchestrator(
        inbound_queue=inbound_queue,
        outbound_queue=outbound_queue,
        interrupt_queue=interrupt_queue,
    )

    reader_task: asyncio.Task | None = None
    writer_task: asyncio.Task | None = None
    orchestrator_task: asyncio.Task | None = None

    try:
        reader_task = asyncio.create_task(
            _websocket_reader(websocket, inbound_queue),
            name=f"ws_reader_{id(websocket)}",
        )
        writer_task = asyncio.create_task(
            _websocket_writer(websocket, outbound_queue, interrupt_queue),
            name=f"ws_writer_{id(websocket)}",
        )
        orchestrator_task = asyncio.create_task(
            orchestrator.run(),
            name=f"orchestrator_{id(websocket)}",
        )

        # Run all three concurrently; if any exits (error or disconnect),
        # return_exceptions=True prevents gather from cancelling siblings
        # prematurely — we handle cleanup in finally.
        results = await asyncio.gather(
            reader_task,
            writer_task,
            orchestrator_task,
            return_exceptions=True,
        )

        # Log any unexpected errors from gather results
        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result, (asyncio.CancelledError, WebSocketDisconnect)
            ):
                logger.error(f"Pipeline task raised: {result}", exc_info=result)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {client_info}")
    except Exception as e:
        logger.error(f"WebSocket endpoint error: {e}", exc_info=True)
    finally:
        # ── Graceful teardown (ARCH-08 fix) ───────────────────────────────
        # Cancel all tasks and await them so no orphaned coroutines remain
        all_tasks = [reader_task, writer_task, orchestrator_task]
        for task in all_tasks:
            if task and not task.done():
                task.cancel()

        # Gather with return_exceptions to suppress CancelledError propagation
        await asyncio.gather(
            *[t for t in all_tasks if t is not None],
            return_exceptions=True,
        )

        # Attempt a clean WebSocket close frame (ARCH-08 fix)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000)
            except Exception:
                pass  # Already closed or network error — safe to ignore

        logger.info(f"WebSocket session fully cleaned up: {client_info}")
