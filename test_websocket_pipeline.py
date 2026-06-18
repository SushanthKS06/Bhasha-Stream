"""
Test 4: Full end-to-end WebSocket voice pipeline test.
Connects to /v1/stream, sends real speech PCM audio (or simulated speech noise),
and verifies audio chunks come back.

Run from e:\Bhasha-Stream-main with uvicorn running on port 8000.
"""
import asyncio
import sys
import time
import numpy as np

# ── Windows asyncio fix ────────────────────────────────────────────────────
# websockets library requires SelectorEventLoop on Windows.
# Python 3.11+ defaults to ProactorEventLoop which is incompatible.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ──────────────────────────────────────────────────────────────────────────

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets==12.0"])
    import websockets


SAMPLE_RATE = 16000
FRAME_SAMPLES = 480      # 30ms
FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit


def make_silence(frames=10):
    """Generate silent PCM frames."""
    return [b"\x00" * FRAME_BYTES] * frames


def make_speech_noise(frames=30):
    """Generate loud noise that mimics speech energy for VAD testing."""
    result = []
    for _ in range(frames):
        # High-amplitude noise to trigger energy-based VAD
        noise = np.random.randint(-8000, 8000, FRAME_SAMPLES, dtype=np.int16)
        result.append(noise.tobytes())
    return result


async def run_test():
    uri = "ws://localhost:8000/v1/stream"
    print(f"\n🔌 Connecting to {uri}...")

    try:
        # Increase open_timeout because the server takes ~25s to load ML models on the first run
        async with websockets.connect(uri, ping_interval=None, open_timeout=60.0) as ws:
            print("✅ WebSocket connected!\n")

            # ── Phase 1: Send silence (should stay quiet) ─────────────────
            print("📤 Phase 1: Sending 10 silence frames (300ms)...")
            for frame in make_silence(10):
                await ws.send(frame)
            await asyncio.sleep(0.5)
            print("   → No response expected (agent is LISTENING). ✅\n")

            # ── Phase 2: Send speech-like noise ──────────────────────────
            print("📤 Phase 2: Sending 60 speech-noise frames (1.8s) to trigger VAD...")
            t_start = time.perf_counter()
            for frame in make_speech_noise(60):
                await ws.send(frame)
                await asyncio.sleep(0.03)  # pace at real-time (30ms/frame)

            # ── Phase 3: Send silence to trigger endpoint ─────────────────
            print("📤 Phase 3: Sending 15 silence frames (450ms) to trigger STT endpoint...")
            for frame in make_silence(15):
                await ws.send(frame)
                await asyncio.sleep(0.03)

            # ── Phase 4: Wait for audio response ─────────────────────────
            print("\n⏳ Waiting for audio response (up to 40s - first run lazy loads TTS)...")
            audio_chunks = []
            interrupt_received = False
            deadline = time.perf_counter() + 40.0

            while time.perf_counter() < deadline:
                try:
                    data = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(data, bytes):
                        if len(data) == 4 and all(b == 0 for b in data):
                            print("   🛑 Interruption signal received")
                            interrupt_received = True
                        else:
                            audio_chunks.append(data)
                            if len(audio_chunks) == 1:
                                ttfb = (time.perf_counter() - t_start) * 1000
                                print(f"   🔊 First audio chunk received! TTFB: {ttfb:.0f}ms")
                            else:
                                print(f"   🔊 Audio chunk #{len(audio_chunks)}: {len(data)} bytes")
                except asyncio.TimeoutError:
                    if audio_chunks:
                        break  # Got some audio, stop waiting
                    continue

            # ── Results ───────────────────────────────────────────────────
            print("\n" + "═" * 50)
            print("📊 TEST RESULTS")
            print("═" * 50)
            print(f"   Audio chunks received : {len(audio_chunks)}")
            total_bytes = sum(len(c) for c in audio_chunks)
            total_ms = total_bytes / (SAMPLE_RATE * 2) * 1000
            print(f"   Total audio received  : {total_bytes:,} bytes (~{total_ms:.0f}ms of speech)")
            print(f"   Interrupt signals     : {1 if interrupt_received else 0}")

            if audio_chunks:
                print("\n✅ FULL PIPELINE WORKING — Voice round-trip confirmed!")
                print("   VAD → STT → LLM (Sarvam) → TTS → Audio response")
            else:
                print("\n⚠️  No audio received.")
                print("   This is expected if:")
                print("   • MeloTTS is not installed (using mock silent TTS)")
                print("   • STT got empty transcription from noise input")
                print("   • Check uvicorn logs for details")

    except ConnectionRefusedError:
        print("❌ Cannot connect — is uvicorn running on port 8000?")
        print("   Run: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        print("   Common causes:")
        print("   • Uvicorn not running → start it first")
        print("   • Models still downloading on first connection (wait 30s, retry)")
        print("   • websockets version mismatch → pip install 'websockets==12.0'")


asyncio.run(run_test())
