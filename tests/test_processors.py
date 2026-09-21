from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    TextFrame,
    UserSpeakingFrame,
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

    async def test_re_arms_exactly_on_the_deadline(self, clock):
        """The deadline is inclusive: at exactly T+timeout the gate sleeps."""
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(30.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False, True]

    async def test_a_failing_unpause_leaves_the_gate_asleep(self, clock):
        """_awake must never be True while the service is still paused."""

        class ExplodingLLM(FakeLLM):
            def set_audio_input_paused(self, paused):
                super().set_audio_input_paused(paused)
                raise RuntimeError("websocket is gone")

        detector = FakeDetector([0.9])
        gate = make_gate(detector, ExplodingLLM(), clock)

        with pytest.raises(RuntimeError):
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate._awake is False
        assert gate._deadline == 0.0

    async def test_a_failing_pause_keeps_the_gate_awake_to_retry(self, clock):
        """Mirror of the unpause case, and the more dangerous direction.

        If the pause fails, audio is still flowing to Google. Staying awake
        means the next frame retries it; flipping to asleep would leave the
        microphone open while the gate believed it was gating.
        """

        class ExplodingLLM(FakeLLM):
            def set_audio_input_paused(self, paused):
                super().set_audio_input_paused(paused)
                if paused:
                    raise RuntimeError("websocket is gone")

        detector = FakeDetector([0.9])
        gate = make_gate(detector, ExplodingLLM(), clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(31.0)
        with pytest.raises(RuntimeError):
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate._awake is True, "must stay awake so the pause is retried"

    @pytest.mark.parametrize(
        "frame",
        [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            UserSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
            BotSpeakingFrame(),
        ],
        ids=[
            "user_start",
            "user_stop",
            "user_speaking",
            "bot_start",
            "bot_stop",
            "bot_speaking",
        ],
    )
    async def test_activity_frames_reset_the_deadline(self, clock, frame):
        """Activity frames reset the silence deadline.

        These frames are emitted downstream of the gate and reach it only as
        the upstream copy of a broadcast. The gate must track them to extend
        the silence timeout when the user/bot is actively speaking.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        # half_duplex=False keeps paused_calls an exact detector of sleep.
        # With half-duplex on, BotStartedSpeakingFrame pauses deliberately,
        # and a sleep that happened while the bot spoke would call no setter
        # at all - so the assertion below would pass vacuously. The
        # half-duplex behaviour these frames now also drive is pinned
        # separately in TestHalfDuplex.
        gate = make_gate(detector, llm, clock, half_duplex=False)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(20.0)
        await gate.process_frame(frame, FrameDirection.UPSTREAM)
        clock.advance(20.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False], "deadline was not reset by activity"

    async def test_a_long_bot_reply_does_not_trip_the_timeout(self, clock):
        """A 45s answer must not put the gate to sleep mid-sentence.

        BotStartedSpeakingFrame alone only sets the deadline once, at t=0.
        Without the periodic BotSpeakingFrame also resetting it, a reply
        longer than silence_timeout_secs trips _sleep() while the bot is
        still audibly talking. Confirms both that a long reply survives and
        that the timeout still fires once activity genuinely stops - so this
        cannot pass by simply breaking the timeout.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        # half_duplex=False for the same reason as the test above: this one
        # is about the deadline, and under half-duplex a mid-reply sleep
        # calls no setter (the service is already paused), which would make
        # the first assertion vacuous. TestHalfDuplex has the twin of this
        # test for the default configuration.
        gate = make_gate(detector, llm, clock, half_duplex=False)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)  # wake

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        for _ in range(4):  # BotSpeakingFrame every 10s out to 40s
            clock.advance(10.0)
            await gate.process_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        clock.advance(5.0)  # 45s total
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False], "woke once and must not have slept"

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False, True], "must still sleep once truly idle"

    async def test_every_frame_is_forwarded(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        sent = [
            (audio_frame(), FrameDirection.DOWNSTREAM),
            (TextFrame(text="hi"), FrameDirection.DOWNSTREAM),
            (BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM),
        ]

        for frame, direction in sent:
            await gate.process_frame(frame, direction)

        assert gate.pushed == sent, (
            "frames must be forwarded in the direction they arrived"
        )

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
        """A bot frame must not extend a deadline that is not running.

        Activity frames arrive as upstream broadcasts, even though they may be
        emitted downstream. The gate must not extend a deadline that does not
        exist (i.e., when not yet awake).
        """
        detector, llm = FakeDetector([0.0]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == []
        assert len(gate.pushed) == 1
        assert gate._deadline == 0.0


class TestHalfDuplex:
    """The microphone is closed for as long as the bot is speaking.

    Mic and speaker run on independent clocks here (I2S off the SoC, USB
    out), so PulseAudio's echo canceller drifts ~1.8s out of alignment and
    residual echo reaches Gemini, whose server-side VAD reads it as a
    barge-in and cuts the reply off mid-sentence. Sending no microphone
    audio at all while the bot speaks removes that failure instead of
    making it less likely; the cost is that barge-in no longer works.

    _awake therefore no longer implies the pause state: pause is a function
    of _awake AND _bot_speaking.
    """

    async def test_starts_paused_like_the_service(self, clock):
        """The service is built with start_audio_paused=True."""
        gate = make_gate(FakeDetector(), FakeLLM(), clock)

        assert gate._paused is True

    async def test_bot_start_pauses_while_awake(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False, True]
        assert gate._awake is True, "pausing for the reply is not falling asleep"

    async def test_bot_stop_unpauses_while_awake(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False, True, False]
        assert gate._awake is True

    async def test_a_repeated_bot_frame_does_not_call_the_setter_again(self, clock):
        """Reconciling means the setter is only called on a real change."""
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        for _ in range(3):
            await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False, True]

    async def test_bot_stop_while_asleep_does_not_unpause(self, clock):
        """The gate can re-arm mid-reply; the stop frame must not open it."""
        detector, llm = FakeDetector([0.0]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [], "asleep is paused, whatever the bot does"
        assert gate._awake is False
        assert gate._paused is True

    async def test_waking_mid_reply_leaves_the_audio_paused(self, clock):
        """Awake while the bot talks still means nothing reaches Gemini."""
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [], "no mic audio may reach Gemini mid-reply"
        assert gate._awake is True, "the wake word was still heard"
        assert gate._deadline == 30.0, "and the silence timer still started"

        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False], "the reply ending opens the mic"

    async def test_sleeping_while_the_bot_speaks_stays_paused(self, clock):
        """A lost BotSpeakingFrame can still let the deadline expire.

        Sleeping mid-reply must leave the service paused - it already is -
        and the BotStoppedSpeakingFrame that arrives afterwards must not
        undo it.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate._awake is False, "the silence timeout still fires"
        assert gate._paused is True
        assert llm.paused_calls == [False, True], "already paused; no second call"

        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False, True], "must not unpause while asleep"

    async def test_a_long_reply_pauses_once_and_unpauses_once(self, clock):
        """The half-duplex twin of test_a_long_bot_reply_does_not_trip_the_timeout."""
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)  # wake

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        for _ in range(4):  # BotSpeakingFrame every 10s out to 40s
            clock.advance(10.0)
            await gate.process_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        clock.advance(5.0)  # 45s total
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False, True, False], (
            "one pause for the whole reply, one unpause when it ends"
        )
        assert gate._awake is True

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False, True, False, True], (
            "and it still sleeps once truly idle"
        )

    async def test_disabled_reproduces_the_old_behaviour(self, clock):
        """half_duplex=False: bot frames only ever touch the deadline.

        Hardware with echo cancellation that works keeps barge-in this way.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock, half_duplex=False)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False], "the mic stays open through the reply"

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False, True], "the silence timeout is unchanged"

    async def test_a_failing_mid_reply_pause_is_retried(self, clock):
        """The mirrored state must not record a pause the service refused."""

        class ExplodingLLM(FakeLLM):
            def __init__(self):
                super().__init__()
                self.explode = False

            def set_audio_input_paused(self, paused):
                super().set_audio_input_paused(paused)
                if self.explode:
                    raise RuntimeError("websocket is gone")

        llm = ExplodingLLM()
        gate = make_gate(FakeDetector([0.9]), llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        llm.explode = True
        with pytest.raises(RuntimeError):
            await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert gate._paused is False, "the service never took the pause"
        assert gate._awake is True

        llm.explode = False
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)

        assert llm.paused_calls == [False, True, True], "the next frame retries it"
