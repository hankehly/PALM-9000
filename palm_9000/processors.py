from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
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
        elif isinstance(frame, (BotStoppedSpeakingFrame, CancelFrame, EndFrame, ErrorFrame)):
            await self._audio_buffer.stop_recording()

        await self.push_frame(frame, direction)
