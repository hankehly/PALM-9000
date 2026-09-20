from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from palm_9000.processors import AudioRecordingControlProcessor, WakeWordGate


@pytest.fixture
def audio_buffer():
    buffer = AsyncMock()
    buffer.start_recording = AsyncMock()
    buffer.stop_recording = AsyncMock()
    return buffer


@pytest.fixture
def processor(audio_buffer, monkeypatch):
    proc = AudioRecordingControlProcessor(audio_buffer)
    # Detach from a real pipeline: capture pushes instead of forwarding.
    pushed = []
    monkeypatch.setattr(
        proc, "push_frame", AsyncMock(side_effect=lambda f, d: pushed.append((f, d)))
    )
    proc.pushed = pushed
    return proc


async def test_bot_started_speaking_starts_recording(processor, audio_buffer):
    await processor.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    audio_buffer.start_recording.assert_awaited_once()
    audio_buffer.stop_recording.assert_not_awaited()


@pytest.mark.parametrize(
    "frame",
    [
        BotStoppedSpeakingFrame(),
        CancelFrame(),
        EndFrame(),
        ErrorFrame(error="boom"),
    ],
    ids=["bot_stopped", "cancel", "end", "error"],
)
async def test_terminal_frames_stop_recording(processor, audio_buffer, frame):
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    audio_buffer.stop_recording.assert_awaited_once()
    audio_buffer.start_recording.assert_not_awaited()


async def test_unrelated_frames_touch_neither(processor, audio_buffer):
    await processor.process_frame(TextFrame(text="hi"), FrameDirection.DOWNSTREAM)
    audio_buffer.start_recording.assert_not_awaited()
    audio_buffer.stop_recording.assert_not_awaited()


@pytest.mark.parametrize(
    "frame",
    [
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        TextFrame(text="passthrough"),
    ],
    ids=["started", "stopped", "text"],
)
async def test_every_frame_is_forwarded(processor, frame):
    """The processor is a tap, not a filter - nothing may be swallowed."""
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert [f for f, _ in processor.pushed] == [frame]


async def test_direction_is_preserved(processor):
    await processor.process_frame(TextFrame(text="up"), FrameDirection.UPSTREAM)
    assert processor.pushed[0][1] is FrameDirection.UPSTREAM


async def test_full_speaking_cycle(processor, audio_buffer):
    await processor.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    audio_buffer.start_recording.assert_awaited_once()
    audio_buffer.stop_recording.assert_awaited_once()
    assert len(processor.pushed) == 2


def test_holds_the_audio_buffer_it_controls(audio_buffer):
    proc = AudioRecordingControlProcessor(audio_buffer)
    assert proc._audio_buffer is audio_buffer


class FakeDetector:
    def __init__(self, scores=None):
        self.scores = list(scores or [])
        self.resets = 0
        self.seen = 0

    def process(self, audio):
        self.seen += 1
        return self.scores.pop(0) if self.scores else 0.0

    def reset(self):
        self.resets += 1


class FakeLLM:
    def __init__(self):
        self.paused_calls = []

    def set_audio_input_paused(self, paused):
        self.paused_calls.append(paused)


def audio_frame(n=8):
    return InputAudioRawFrame(audio=b"\x00\x00" * n, sample_rate=16000, num_channels=1)


class Clock:
    """Controllable monotonic clock, so tests never sleep."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return Clock()


def make_gate(detector, llm, clock, **kw):
    gate = WakeWordGate(
        detector=detector, llm=llm, silence_timeout_secs=30.0, now=clock, **kw
    )
    gate.pushed = []
    gate.push_frame = AsyncMock(side_effect=lambda f, d: gate.pushed.append((f, d)))
    return gate


class TestWakeWordGate:
    async def test_stays_asleep_below_threshold(self, clock):
        detector, llm = FakeDetector([0.1, 0.2]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == []

    async def test_score_above_threshold_unpauses(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False]
        assert detector.resets == 1

    async def test_does_not_detect_while_awake(self, clock):
        detector, llm = FakeDetector([0.9, 0.9, 0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        seen_after_wake = detector.seen
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert detector.seen == seen_after_wake, "detector ran while awake"
        assert llm.paused_calls == [False]

    async def test_silence_timeout_re_arms(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False, True]

    @pytest.mark.parametrize(
        "frame",
        [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
        ],
        ids=["user_start", "user_stop", "bot_start", "bot_stop"],
    )
    async def test_activity_frames_reset_the_deadline(self, clock, frame):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(20.0)
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        clock.advance(20.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False], "deadline was not reset by activity"

    async def test_every_frame_is_forwarded(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        frames = [audio_frame(), TextFrame(text="hi"), BotStoppedSpeakingFrame()]

        for f in frames:
            await gate.process_frame(f, FrameDirection.DOWNSTREAM)

        assert [f for f, _ in gate.pushed] == frames

    async def test_detector_failure_keeps_it_asleep(self, clock):
        class Exploding(FakeDetector):
            def process(self, audio):
                raise RuntimeError("onnx blew up")

        llm = FakeLLM()
        gate = make_gate(Exploding(), llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [], "must fail closed"
        assert len(gate.pushed) == 1, "frame must still be forwarded"

    async def test_activity_frames_are_ignored_while_asleep(self, clock):
        """A bot frame must not extend a deadline that is not running."""
        detector, llm = FakeDetector([0.0]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == []
        assert len(gate.pushed) == 1
