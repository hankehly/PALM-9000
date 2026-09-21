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

from palm_9000 import processors as processors_module
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

    @pytest.mark.parametrize("wake_first", [False, True], ids=["asleep", "awake"])
    async def test_non_audio_frames_are_always_forwarded(self, clock, wake_first):
        """Only user audio is gated; every other frame is a pass-through.

        Half of the old test_every_frame_is_forwarded. It keeps the part
        that killed the hardcoded-direction mutant: each frame is asserted
        to arrive downstream paired with the direction it came in on, so
        pushing everything DOWNSTREAM (or dropping the direction argument
        altogether) fails on the UPSTREAM entries. The audio half of that
        test is now test_audio_is_dropped_while_asleep and
        test_audio_is_forwarded_once_awake, which assert the opposite of
        each other and so could not have stayed in one list.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        if wake_first:
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
            assert gate._awake is True
            gate.pushed.clear()

        sent = [
            (TextFrame(text="down"), FrameDirection.DOWNSTREAM),
            (TextFrame(text="up"), FrameDirection.UPSTREAM),
            (BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM),
            (BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM),
            (UserStartedSpeakingFrame(), FrameDirection.UPSTREAM),
            (EndFrame(), FrameDirection.DOWNSTREAM),
        ]
        for frame, direction in sent:
            await gate.process_frame(frame, direction)

        assert gate.pushed == sent, (
            "non-audio frames must be forwarded in the direction they arrived"
        )

    async def test_audio_is_dropped_while_asleep(self, clock):
        """The core promise: nothing downstream ever sees pre-wake audio.

        Pausing the service is not enough on its own - a paused flag lives
        on a processor further down the pipeline, and audio already in
        flight is judged by that flag when it *arrives*. A frame the gate
        never pushes cannot leak however long it sits in a queue.
        """
        detector, llm = FakeDetector([0.1, 0.1]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate.pushed == [], "no user audio may leave the gate while asleep"
        assert detector.seen == 2, "but the detector must still hear it"

    async def test_audio_is_forwarded_once_awake(self, clock):
        """The other half: gating off means the audio actually flows.

        Without this, dropping unconditionally would pass every other test
        in the file.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        waking, after = audio_frame(), audio_frame()

        await gate.process_frame(waking, FrameDirection.DOWNSTREAM)
        await gate.process_frame(after, FrameDirection.DOWNSTREAM)

        assert gate.pushed == [
            (waking, FrameDirection.DOWNSTREAM),
            (after, FrameDirection.DOWNSTREAM),
        ], "the waking frame and everything after it must go through"

    async def test_the_frame_that_trips_the_timeout_is_dropped(self, clock):
        """Falling asleep takes effect on the very frame that caused it."""
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        waking = audio_frame()
        await gate.process_frame(waking, FrameDirection.DOWNSTREAM)

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate.pushed == [(waking, FrameDirection.DOWNSTREAM)]

    async def test_detector_failure_keeps_it_asleep(self, clock):
        class Exploding(FakeDetector):
            def process(self, audio):
                raise RuntimeError("onnx blew up")

        llm = FakeLLM()
        gate = make_gate(Exploding(), llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [], "must fail closed"
        assert gate.pushed == [], "and closed means the audio is dropped too"

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

    async def test_mic_audio_is_dropped_while_the_bot_speaks(self, clock):
        """Half-duplex drops the audio as well as pausing the service.

        Echo cannot be read as a barge-in by Gemini's server-side VAD if the
        echo never leaves this processor.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)  # wake
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        gate.pushed.clear()

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate.pushed == [], "no mic audio may leave the gate mid-reply"
        assert gate._awake is True, "dropping is not falling asleep"

        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
        resumed = audio_frame()
        await gate.process_frame(resumed, FrameDirection.DOWNSTREAM)

        assert gate.pushed[-1] == (resumed, FrameDirection.DOWNSTREAM), (
            "and the reply ending resumes it"
        )

    async def test_disabled_keeps_forwarding_audio_through_the_reply(self, clock):
        """half_duplex=False: the bot speaking gates nothing, as before.

        Pins the drop to the same condition as the pause rather than to
        _bot_speaking on its own - barge-in hardware must keep its audio.
        """
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock, half_duplex=False)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)  # wake
        await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        gate.pushed.clear()

        during = audio_frame()
        await gate.process_frame(during, FrameDirection.DOWNSTREAM)

        assert gate.pushed == [(during, FrameDirection.DOWNSTREAM)]

    async def test_a_refused_pause_still_drops_the_audio(self, clock):
        """Defence in depth: the drop must not depend on the pause landing.

        _paused records only what the service accepted. If the pause raised,
        the gate is awake with _paused still False - and that is exactly the
        moment the microphone must not be open. Deciding on the gate's own
        state (awake, bot_speaking) rather than on _paused is what makes the
        two mechanisms independent.
        """

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
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)  # wake

        llm.explode = True
        with pytest.raises(RuntimeError):
            await gate.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        assert gate._paused is False, "the service never took the pause"
        gate.pushed.clear()

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert gate.pushed == [], "the gate drops it anyway"


class TestThroughputDiagnostic:
    """Every 250 audio frames - exactly 5.0s of 20ms audio - report the wait.

    The gate is marginal by construction on a Zero 2W: the detector costs
    ~400-600ms per scoring pass and runs every 500ms. Nothing else in the
    log says whether it is keeping up, and falling behind means audio
    queueing somewhere upstream.
    """

    @staticmethod
    def _capture(monkeypatch):
        """Collect logger.debug lines about throughput (pause lines share it)."""
        messages = []
        monkeypatch.setattr(
            processors_module.logger,
            "debug",
            lambda msg, *a, **kw: messages.append(msg),
        )
        return messages

    @staticmethod
    def _throughput(messages):
        return [m for m in messages if "audio frames" in m]

    async def test_logs_once_every_250_frames(self, clock, monkeypatch):
        """249 frames say nothing; the 250th reports the elapsed 5.00s.

        The clock is the injected one, so a diagnostic reading
        time.monotonic directly would report ~0.00s here and fail.
        """
        messages = self._capture(monkeypatch)
        gate = make_gate(FakeDetector(), FakeLLM(), clock)

        for _ in range(249):
            clock.advance(0.02)
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert self._throughput(messages) == [], "nothing before the 250th frame"

        clock.advance(0.02)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert self._throughput(messages) == [
            "250 audio frames in 5.00s (5.00s = realtime)"
        ]
        assert gate.pushed == [], "dropped frames still count towards the total"

    async def test_the_next_interval_starts_at_the_last_log(self, clock, monkeypatch):
        """The second reading is since the first, not since startup.

        A diagnostic that never moved its mark would print 15.00s here,
        which is the difference between "the gate stalled" and "the gate
        stalled once, ten seconds ago".
        """
        messages = self._capture(monkeypatch)
        gate = make_gate(FakeDetector(), FakeLLM(), clock)

        for _ in range(250):
            clock.advance(0.02)
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        for _ in range(250):  # the same 5.0s of audio, taking 10.0s to arrive
            clock.advance(0.04)
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert self._throughput(messages) == [
            "250 audio frames in 5.00s (5.00s = realtime)",
            "250 audio frames in 10.00s (5.00s = realtime)",
        ]

    async def test_non_audio_frames_are_not_counted(self, clock, monkeypatch):
        """It measures the audio path, not general frame traffic."""
        messages = self._capture(monkeypatch)
        gate = make_gate(FakeDetector(), FakeLLM(), clock)

        for _ in range(249):
            clock.advance(0.02)
            await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        for _ in range(10):
            await gate.process_frame(TextFrame(text="noise"), FrameDirection.DOWNSTREAM)

        assert self._throughput(messages) == []
