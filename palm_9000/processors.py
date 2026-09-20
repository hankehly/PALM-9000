import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
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
    """

    _ACTIVITY_FRAMES = (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    )

    def __init__(
        self,
        detector,
        llm,
        threshold: float = 0.5,
        silence_timeout_secs: float = 30.0,
        now=time.monotonic,
    ) -> None:
        super().__init__()
        self._detector = detector
        self._llm = llm
        self._threshold = threshold
        self._silence_timeout_secs = silence_timeout_secs
        self._now = now
        self._awake = False
        self._deadline = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

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
            self._llm.set_audio_input_paused(False)
            self._awake = True
            self._deadline = self._now() + self._silence_timeout_secs

    def _sleep(self) -> None:
        logger.info(
            f"No speech for {self._silence_timeout_secs}s - going back to sleep"
        )
        self._awake = False
        self._llm.set_audio_input_paused(True)
