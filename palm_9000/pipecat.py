from gpiozero import LED
from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from palm_9000.settings import settings


class LEDSyncProcessor(FrameProcessor):
    def __init__(self, led_pin: int):
        super().__init__()
        self.led = LED(led_pin)
        self.speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self.led.blink(on_time=0.25, off_time=0.25)
            print("LED ON - Bot started speaking")
            self.speaking = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.led.off()
            self.speaking = False
            print("LED OFF - Bot stopped speaking")

        elif isinstance(frame, (EndFrame, CancelFrame, ErrorFrame)):
            logger.info("Cleaning up LEDSyncProcessor")
            self.speaking = False
            self.led.close()

        await self.push_frame(frame)
