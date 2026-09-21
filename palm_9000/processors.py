import time

from loguru import logger
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    UserSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AudioRecordingControlProcessor(FrameProcessor):
    """
    Starts/stops AudioBufferProcessor recording based on bot speaking frames
    and stops the heart display on cancel/end/error.
    """

    def __init__(self, audio_buffer: AudioBufferProcessor) -> None:
        super().__init__()
        self._audio_buffer = audio_buffer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            await self._audio_buffer.start_recording()
        elif isinstance(
            frame, (BotStoppedSpeakingFrame, CancelFrame, EndFrame, ErrorFrame)
        ):
            await self._audio_buffer.stop_recording()

        await self.push_frame(frame, direction)


class WakeWordGate(FrameProcessor):
    """Keeps user audio off the wire until a wake word is heard.

    The service is constructed with start_audio_paused=True, so the gate only
    ever has to open it. Every frame is forwarded unchanged; pausing and
    unpausing the service is the sole side effect.

    With `half_duplex` on (the default) the input is paused again for as long
    as the bot is speaking, and restored when it stops. That is a measured
    workaround, not a preference: the microphone is I2S off the SoC and the
    speaker is USB, so they run on independent clocks and PulseAudio's echo
    canceller drifts ~1.8s out of alignment ("Playback too far ahead (88396),
    drop source 33944", 124 resyncs in 30 minutes). An AEC works on echo
    tails of 100-500ms; at that offset it cannot function, drift_compensation
    included, so residual echo reaches Gemini and its server-side VAD reads
    it as the user interrupting. Half-duplex removes the failure instead of
    reducing its probability: no microphone audio reaches Gemini while the
    bot speaks, so echo cannot interrupt it whatever the clocks, the volume
    or the room. The cost is no barge-in. Hardware with an echo canceller
    that actually works can set half_duplex=False and keep it.
    """

    # Start/stop frames alone leave a gap: nothing arrives between them, so a
    # long turn never refreshes the deadline. BotSpeakingFrame and
    # UserSpeakingFrame are broadcast periodically *during* speech (every
    # ~0.2s - see pipecat's base_output.py and vad_controller.py) and close
    # that gap.
    _ACTIVITY_FRAMES = (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        UserSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        BotSpeakingFrame,
    )

    def __init__(
        self,
        detector,
        llm,
        threshold: float = 0.5,
        silence_timeout_secs: float = 30.0,
        half_duplex: bool = True,
        now=time.monotonic,
    ) -> None:
        super().__init__()
        self._detector = detector
        self._llm = llm
        self._threshold = threshold
        self._silence_timeout_secs = silence_timeout_secs
        self._half_duplex = half_duplex
        self._now = now
        self._awake = False
        self._bot_speaking = False
        # The service is constructed with start_audio_paused=True, so the
        # mirror of its state starts paused too.
        self._paused = True
        self._deadline = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            # Tracked whether awake or asleep, so the flag is already right
            # if the wake word lands in the middle of a reply.
            self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
            self._apply_pause(self._awake)

        if self._awake and isinstance(frame, self._ACTIVITY_FRAMES):
            self._deadline = self._now() + self._silence_timeout_secs
        elif isinstance(frame, InputAudioRawFrame):
            if self._awake:
                if self._now() >= self._deadline:
                    self._sleep()
            else:
                self._maybe_wake(frame.audio)

        await self.push_frame(frame, direction)

    def _maybe_wake(self, audio: bytes) -> None:
        try:
            score = self._detector.process(audio)
        except Exception:
            # Fail closed: a broken detector must never open the microphone.
            logger.exception("Wake-word detection failed; staying asleep")
            return
        if score > self._threshold:
            logger.info(f"Wake word detected (score {score:.2f})")
            self._detector.reset()
            self._apply_pause(True)
            self._deadline = self._now() + self._silence_timeout_secs

    def _sleep(self) -> None:
        logger.info(
            f"No speech for {self._silence_timeout_secs}s - going back to sleep"
        )
        self._apply_pause(False)

    def _apply_pause(self, awake: bool) -> None:
        """Reconcile the service's pause state, then record the new state.

        Pause is a function of both `awake` and whether the bot is speaking,
        so it cannot be read off `_awake` any more. Doing the setter first
        and the state flip only once it returns preserves both fail-safe
        orderings:

        * a failing unpause leaves the gate asleep, so the next frame retries
          rather than believing it is awake while the service is still paused;
        * a failing pause leaves the gate awake, so the next frame retries
          rather than believing it is gating while audio still flows to
          Google - the one failure the feature exists to prevent.
        """
        want = (not awake) or (self._half_duplex and self._bot_speaking)
        if want != self._paused:
            self._llm.set_audio_input_paused(want)
            self._paused = want
        self._awake = awake
