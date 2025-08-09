import asyncio
import math
import time

from gpiozero import LED
from loguru import logger
from luma.core.interface.serial import noop, spi
from luma.core.render import canvas
from luma.led_matrix.device import max7219
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
            logger.info("LED ON - Bot started speaking")
            self.speaking = True

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.led.off()
            self.speaking = False
            logger.info("LED OFF - Bot stopped speaking")

        elif isinstance(frame, (EndFrame, CancelFrame, ErrorFrame)):
            logger.info("Cleaning up LEDSyncProcessor")
            self.speaking = False
            self.led.close()

        await self.push_frame(frame)


class MAX7219PulseHeartProcessor(FrameProcessor):
    """
    Shows a heart pulse animation.
    
    Follow the instructions below to hook up the hardware and install the libraries.
    https://docs.sunfounder.com/projects/raphael-kit/en/latest/python_pi5/pi5_1.1.6_led_dot_matrix_python.html
    """

    def __init__(self, fps=30, cycle_time=1.0, min_brightness=4, max_brightness=255):
        super().__init__()
        self.serial = spi(port=0, device=0, gpio=noop())
        self.device = max7219(self.serial, cascaded=1)

        # Pulse tunables
        self.fps = fps
        self.cycle_time = cycle_time
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

        # Runtime state
        self._pulse_task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._speaking = False

    def draw_heart(self):
        pixels = [
            # fmt: off
                    (1, 1),                                 (6, 1),
            (0, 2), (1, 2), (2, 2),                 (5, 2), (6, 2), (7, 2),
            (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
                    (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
                            (2, 5), (3, 5), (4, 5), (5, 5),
                                    (3, 6), (4, 6)
            # fmt: on
        ]
        with canvas(self.device) as draw:
            for x, y in pixels:
                draw.point((x, y), fill="white")

    def _brightness_from_t(self, t: float) -> int:
        """
        t is time within cycle [0, cycle_time).
        Instant attack (first 10%), exponential decay afterwards.
        """
        norm_t = (t % self.cycle_time) / self.cycle_time
        if norm_t < 0.1:
            brightness_factor = 1.0  # instant pulse
        else:
            brightness_factor = max(0.0, math.exp(-5 * (norm_t - 0.1)))
        return self.min_brightness + int(
            brightness_factor * (self.max_brightness - self.min_brightness)
        )

    async def _pulse_loop(self):
        """Runs until _stop_evt is set."""
        start = time.monotonic()
        period = 1.0 / float(self.fps)
        try:
            while not self._stop_evt.is_set():
                t = time.monotonic() - start
                b = self._brightness_from_t(t)
                self.device.contrast(b)
                self.draw_heart()
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            # Task cancelled: fall through to cleanup
            pass
        finally:
            # Turn off/clear when stopping
            try:
                self.device.contrast(0)
                # max7219 device has clear()
                if hasattr(self.device, "clear"):
                    self.device.clear()
            except Exception:
                pass

    def start_pulse(self):
        if self._pulse_task and not self._pulse_task.done():
            return  # already running
        self._stop_evt.clear()
        self._pulse_task = asyncio.create_task(self._pulse_loop())

    async def stop_pulse(self):
        if not self._pulse_task:
            return
        self._stop_evt.set()
        # Give it a moment to finish the current iteration
        try:
            await asyncio.wait_for(self._pulse_task, timeout=0.5)
        except asyncio.TimeoutError:
            self._pulse_task.cancel()
            try:
                await self._pulse_task
            except asyncio.CancelledError:
                pass
        finally:
            self._pulse_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self.start_pulse()
            logger.info("PulseHeart START - Bot started speaking")

        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.stop_pulse()
            logger.info("PulseHeart STOP - Bot stopped speaking")

        elif isinstance(frame, (EndFrame, CancelFrame, ErrorFrame)):
            logger.info("Cleaning up MAX7219PulseHeartProcessor")
            await self.stop_pulse()

        await self.push_frame(frame)
