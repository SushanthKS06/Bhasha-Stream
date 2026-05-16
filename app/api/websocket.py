"""
WebSocket protocol layer & network IO for Bhasha-Stream.
Handles binary PCM 16kHz mono audio frames.
"""
import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.config import settings
from app.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


async def websocket_reader(
    websocket: WebSocket,
    inbound_queue: asyncio.Queue[bytes],
) -> None:
    """
    Continuously read raw binary audio from WebSocket and push to inbound queue.
    Runs as a background task until connection closes.
    """
    try:
        while True:
            # Receive binary data with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=settings.websocket_timeout_seconds,
                )
                
                # Validate frame size (expecting 30ms of 16kHz 16-bit mono = 960 bytes)
                if len(data) == 0:
                    logger.warning("Received empty audio frame")
                    continue
                
                await inbound_queue.put(data)
                
            except asyncio.TimeoutError:
                # Send keepalive or continue
                continue
            except WebSocketDisconnect:
                logger.info("Client disconnected during read")
                raise
            except Exception as e:
                logger.error(f"Error reading from WebSocket: {e}")
                raise
                
    except Exception as e:
        logger.error(f"WebSocket reader error: {e}")
        raise


async def websocket_writer(
    websocket: WebSocket,
    outbound_queue: asyncio.Queue[bytes],
    interrupt_queue: asyncio.Queue[bool],
) -> None:
    """
    Continuously read synthesized audio chunks from outbound queue and flush to WebSocket.
    Handles interruption signals to stop playback.
    Runs as a background task until connection closes.
    """
    try:
        while True:
            # Check for interrupt signal first
            try:
                interrupt_signal = interrupt_queue.get_nowait()
                if interrupt_signal:
                    # Send interruption packet to client
                    interrupt_packet = b"\x00\x00\x00\x00"  # 4-byte null signal
                    await websocket.send_bytes(interrupt_packet)
                    logger.debug("Sent interruption signal to client")
                    # Flush remaining audio in queue
                    while not outbound_queue.empty():
                        try:
                            outbound_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    continue
            except asyncio.QueueEmpty:
                pass
            
            # Wait for audio chunk with timeout
            try:
                audio_chunk = await asyncio.wait_for(
                    outbound_queue.get(),
                    timeout=0.1,  # Short timeout to check for interrupts frequently
                )
                
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_bytes(audio_chunk)
                    
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info("Client disconnected during write")
                raise
            except Exception as e:
                logger.error(f"Error writing to WebSocket: {e}")
                raise
                
    except Exception as e:
        logger.error(f"WebSocket writer error: {e}")
        raise


@router.websocket("/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for bidirectional audio streaming.
    Accepts binary PCM 16kHz mono audio frames and returns synthesized audio.
    """
    await websocket.accept()
    logger.info(f"New WebSocket connection from {websocket.client}")
    
    # Get shared queues from app state
    inbound_queue: asyncio.Queue[bytes] = websocket.app.state.inbound_audio_queue
    outbound_queue: asyncio.Queue[bytes] = websocket.app.state.outbound_audio_queue
    interrupt_queue: asyncio.Queue[bool] = websocket.app.state.interrupt_queue
    
    # Create orchestrator for this session
    orchestrator = Orchestrator(
        inbound_queue=inbound_queue,
        outbound_queue=outbound_queue,
        interrupt_queue=interrupt_queue,
    )
    
    # Track tasks for cleanup
    reader_task = None
    writer_task = None
    orchestrator_task = None
    
    try:
        # Run concurrent loops using asyncio.gather
        reader_task = asyncio.create_task(
            websocket_reader(websocket, inbound_queue),
            name="websocket_reader",
        )
        writer_task = asyncio.create_task(
            websocket_writer(websocket, outbound_queue, interrupt_queue),
            name="websocket_writer",
        )
        orchestrator_task = asyncio.create_task(
            orchestrator.run(),
            name="orchestrator",
        )
        
        # Wait for all tasks - if one fails, cancel the others
        await asyncio.gather(
            reader_task,
            writer_task,
            orchestrator_task,
            return_exceptions=True,
        )
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        # Cleanup: cancel all tasks
        for task in [reader_task, writer_task, orchestrator_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("WebSocket connection closed and resources cleaned up")
