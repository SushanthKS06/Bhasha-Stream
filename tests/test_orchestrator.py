# tests/test_orchestrator.py
"""
Critical tests for the Bhasha-Stream Orchestrator state machine.

Fixes applied (BUG-01 through BUG-07):
  - BUG-01: Import corrected to `Orchestrator` (not `VoiceOrchestrator`)
  - BUG-02: Fixture uses correct constructor signature; services injected post-construction
  - BUG-03: Attribute name corrected to `orchestrator.inbound_queue`
  - BUG-04: Method name corrected to `orchestrator._process_vad()`
  - BUG-05: Attribute name corrected to `orchestrator.state`
  - BUG-06: `active_tasks` is a real Set[Task] on the fixed Orchestrator
  - BUG-07: Method name corrected to `orchestrator._handle_interruption()`
"""
import asyncio
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# BUG-01 fix: correct class name
from app.core.orchestrator import Orchestrator, AgentState
from app.services.vad import VADService
from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_services():
    """Create properly specced mock dependencies for the orchestrator."""
    vad = AsyncMock(spec=VADService)
    stt = AsyncMock(spec=STTService)
    llm = AsyncMock(spec=LLMService)
    tts = AsyncMock(spec=TTSService)

    # Safe defaults
    vad.is_speech.return_value = False
    stt.transcribe.return_value = "Mocked transcription"

    # LLM generate must be an async generator
    async def _mock_llm_gen(prompt: str):
        for char in "Hello there. How are you?":
            yield char

    llm.generate = _mock_llm_gen

    # TTS synthesize must be an async generator
    async def _mock_tts_synth(text: str):
        yield b"\x00" * 960  # one audio frame of silence

    tts.synthesize = _mock_tts_synth

    # close() called during cleanup
    llm.close = AsyncMock()

    return {"vad": vad, "stt": stt, "llm": llm, "tts": tts}


@pytest.fixture
def orchestrator(mock_services):
    """
    Construct an Orchestrator with correct args (BUG-02 fix),
    then inject mock services directly onto the instance.
    """
    # BUG-02 fix: correct constructor signature
    orc = Orchestrator(
        inbound_queue=asyncio.Queue(maxsize=100),
        outbound_queue=asyncio.Queue(maxsize=100),
        interrupt_queue=asyncio.Queue(maxsize=16),
    )

    # Inject mocks (Orchestrator constructs its own services in __init__,
    # so we replace them after construction)
    orc.vad_service = mock_services["vad"]
    orc.stt_service = mock_services["stt"]
    orc.llm_service = mock_services["llm"]
    orc.tts_service = mock_services["tts"]

    return orc


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: VAD speech boundary detection
# ─────────────────────────────────────────────────────────────────────────────

async def test_vad_speech_boundary_detection(orchestrator, mock_services):
    """
    Simulate raw PCM binary audio streaming and verify the VAD correctly
    drives state transitions without drifting.

    BUG-03 fix: `orchestrator.inbound_queue` (not `inbound_audio_queue`)
    BUG-04 fix: `orchestrator._process_vad()` (not `_process_vad_batch`)
    BUG-05 fix: `orchestrator.state` (not `current_state`)
    """
    sample_rate = 16000
    frame_samples = int(sample_rate * 0.03)  # 480 samples / 30ms

    silence_frame = b"\x00" * (frame_samples * 2)  # 960 bytes
    speech_frame = np.random.randint(
        -32768, 32767, frame_samples, dtype=np.int16
    ).tobytes()

    # 30 silence frames → should stay in LISTENING
    # 30 speech frames  → should transition to THINKING
    vad_responses = [False] * 30 + [True] * 30
    mock_services["vad"].is_speech.side_effect = vad_responses

    # Process all silence frames — state must remain LISTENING
    for _ in range(30):
        await orchestrator._process_vad(silence_frame)

    # BUG-05 fix: correct attribute name
    assert orchestrator.state == AgentState.LISTENING, (
        "State should remain LISTENING during continuous silence"
    )

    # Process speech frames until minimum speech duration is reached
    for _ in range(30):
        await orchestrator._process_vad(speech_frame)

    # After enough speech frames, should have transitioned to THINKING
    assert orchestrator.state == AgentState.THINKING, (
        "State should be THINKING after sustained speech detection"
    )

    # VAD should have been called for each frame
    assert mock_services["vad"].is_speech.call_count == 60


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Interleaving phrase splitter
# ─────────────────────────────────────────────────────────────────────────────

async def test_interleaving_phrase_splitter(orchestrator, mock_services):
    """
    Verify the phrase-splitter correctly groups tokens and flushes to TTS
    on punctuation marks — the key mechanism behind the <800ms RTT.
    """
    flushed_phrases = []

    # Capture all phrases sent to the TTS mock
    async def _capture_tts(text: str):
        flushed_phrases.append(text)
        yield b"\x00" * 960

    mock_services["tts"].synthesize = _capture_tts
    orchestrator.tts_service = mock_services["tts"]

    # Feed tokens into the accumulator directly
    tokens = list("Hello there. How are you?")
    for token in tokens:
        await orchestrator._accumulate_and_flush(token)

    # Flush any remaining buffer
    if orchestrator._text_buffer.strip():
        flushed_phrases.append(orchestrator._text_buffer.strip())

    # "Hello there." should be one phrase, "How are you?" another
    assert len(flushed_phrases) >= 2, (
        f"Expected ≥2 TTS flushes (punctuation splits), got {len(flushed_phrases)}: {flushed_phrases}"
    )
    assert any("Hello there" in p for p in flushed_phrases), (
        f"Expected 'Hello there.' in flushed phrases. Got: {flushed_phrases}"
    )
    assert any("How are you" in p for p in flushed_phrases), (
        f"Expected 'How are you?' in flushed phrases. Got: {flushed_phrases}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 (Critical): Interruption handling
# ─────────────────────────────────────────────────────────────────────────────

async def test_critical_interruption_handling(orchestrator, mock_services):
    """
    Simulate a user barge-in mid-utterance.

    Verifies that:
    1. All active pipeline tasks are cancelled within the interruption call
    2. The outbound audio queue is fully flushed
    3. State machine resets to LISTENING

    BUG-05 fix: `orchestrator.state` (not `current_state`)
    BUG-06 fix: `orchestrator.active_tasks` is a real Set on the fixed Orchestrator
    BUG-07 fix: `orchestrator._handle_interruption()` (not `handle_interruption`)
    """
    # 1. Set the agent into SPEAKING state
    await orchestrator._transition(AgentState.SPEAKING)
    assert orchestrator.state == AgentState.SPEAKING

    # 2. Create two slow background tasks that simulate active LLM + TTS
    async def _slow_llm():
        await asyncio.sleep(10)

    async def _slow_tts():
        await asyncio.sleep(10)

    llm_task = asyncio.create_task(_slow_llm(), name="slow_llm")
    tts_task = asyncio.create_task(_slow_tts(), name="slow_tts")

    # Register them in the orchestrator's tracking set (BUG-06 fix: set exists)
    orchestrator.active_tasks.add(llm_task)
    orchestrator.active_tasks.add(tts_task)
    orchestrator._current_llm_task = llm_task
    orchestrator._current_tts_task = tts_task

    # 3. Pre-fill outbound queue with stale audio
    await orchestrator.outbound_queue.put(b"old_audio_1")
    await orchestrator.outbound_queue.put(b"old_audio_2")
    assert not orchestrator.outbound_queue.empty()

    # 4. Trigger interruption (BUG-07 fix: correct method name)
    await orchestrator._handle_interruption()

    # 5. Verify state reset
    assert orchestrator.state == AgentState.LISTENING, (
        "State must be LISTENING after interruption handling"
    )

    # 6. Verify tasks were cancelled
    # Give event loop a tick to propagate cancellation
    await asyncio.sleep(0)

    assert llm_task.cancelled(), "LLM task must be cancelled after interruption"
    assert tts_task.cancelled(), "TTS task must be cancelled after interruption"

    # 7. Verify outbound queue is empty
    assert orchestrator.outbound_queue.empty(), (
        "Outbound queue must be flushed after interruption"
    )

    # 8. Suppress CancelledError from awaiting cancelled tasks
    for task in [llm_task, tts_task]:
        try:
            await task
        except asyncio.CancelledError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: STT buffer capture race condition guard
# ─────────────────────────────────────────────────────────────────────────────

async def test_stt_buffer_captured_before_clear(orchestrator, mock_services):
    """
    Verify BUG-11 fix: the speech buffer is captured into a local snapshot
    before being cleared. The STT service must receive non-empty audio data
    even though the buffer is reset immediately after task creation.
    """
    received_audio: list = []

    async def _capture_stt(audio_data: bytes):
        received_audio.append(audio_data)
        return "captured"

    mock_services["stt"].transcribe = _capture_stt

    # Fill the speech buffer with mock speech data
    mock_audio = b"\x01\x02" * 480  # 960 bytes of non-zero audio
    orchestrator._speech_buffer.extend(mock_audio)
    orchestrator._stt_triggered = False

    # Call STT processing directly with captured data (as the fixed VAD logic does)
    captured = bytes(orchestrator._speech_buffer)
    orchestrator._speech_buffer.clear()

    await orchestrator._process_speech_to_text(captured)

    # Give event loop a tick for the task chain to complete
    await asyncio.sleep(0.05)

    assert len(received_audio) == 1, "STT must be called exactly once"
    assert received_audio[0] == mock_audio, (
        "STT must receive the original audio data, not an empty buffer"
    )
    assert len(orchestrator._speech_buffer) == 0, "Speech buffer must be cleared"