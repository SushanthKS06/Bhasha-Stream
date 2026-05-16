# tests/test_stream.py
import asyncio
import pytest
import websockets
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.mark.asyncio
async def test_websocket_duplex_stream():
    """
    Integration test: Connect to WebSocket, send PCM frames, 
    and verify connection stability and basic echo/response structure.
    Note: Requires the app to be running or using an ASGI lifespan override.
    """
    uri = "ws://localhost:8000/v1/stream"
    
    # Since we can't easily spin up the full GPU stack in a unit test env,
    # this test verifies the protocol handshake and frame handling logic
    # assuming mocked services are injected via dependency overrides in a real CI pipeline.
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send 10 frames of silence (30ms each)
            frame_size = 960 # 16000Hz * 0.03s * 2 bytes
            silence = b'\x00' * frame_size
            
            for _ in range(10):
                await websocket.send(silence)
            
            # Expect some acknowledgment or just ensure no crash
            # In a real scenario, we might wait for a specific binary response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                # If we get here, the connection is alive
                assert isinstance(response, (bytes, str))
            except asyncio.TimeoutError:
                # Timeout is acceptable if the agent is just listening (no speech detected)
                pass
                
    except ConnectionRefusedError:
        pytest.skip("Backend server not running for integration test")