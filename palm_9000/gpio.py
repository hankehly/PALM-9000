import asyncio
import threading
import time

import numpy as np
from loguru import logger
from luma.core.interface.serial import noop, spi
from luma.core.render import canvas
from luma.led_matrix.device import max7219

_INT16_MAX = 32768.0


class Max7219AmplitudeHeart:
    """
    Drive a heart icon on an 8x8 MAX7219, brightness = audio amplitude.
    Call `start()` once. Feed audio via `heart.process_audio(audio_bytes)`.

    Sample Usage:

        import asyncio
        from palm_9000.gpio import Max7219AmplitudeHeart

        async def main():
            heart = Max7219AmplitudeHeart(fps=60, ema=0.4, gamma=2.0, channels=1)
            await heart.start()

            # In a real app you'll be pulling 16‑bit mono (or interleaved) PCM
            # audio frames from a microphone / audio callback. Below we just
            # simulate a few seconds of varying amplitude.
            import math, struct, time
            sample_rate = 16000
            frame_ms = 40  # 40 ms frames
            frame_samples = int(sample_rate * frame_ms / 1000)
            t = 0.0
            dt = frame_samples / sample_rate
            try:
                for _ in range(int(5 * 1000 / frame_ms)):  # ~5 seconds
                    # Create a synthetic sine wave whose amplitude slowly pulses
                    amp = 0.2 + 0.75 * (0.5 * (1 + math.sin(2 * math.pi * 0.6 * t)))
                    freq = 440
                    frame = [
                        int(amp * 0.8 * 32767
                            * math.sin(2 * math.pi * freq * (t + i / sample_rate)))
                        for i in range(frame_samples)
                    ]
                    audio_bytes = struct.pack('<' + 'h'*len(frame), *frame)
                    heart.process_audio(audio_bytes)
                    await asyncio.sleep(frame_ms / 1000.0)
                    t += dt
            finally:
                await heart.stop()

        if __name__ == "__main__":
            asyncio.run(main())

    Notes:
    - `process_audio` is thread-safe; you can call it from an audio callback thread.
    - Audio must be 16-bit little-endian PCM. For multi-channel audio set `channels`.
    - Brightness curve is smoothed (EMA) and gamma-corrected for perceptual response.
    """

    def __init__(
        self,
        fps: int = 90,
        min_brightness: int = 4,
        max_brightness: int = 255,
        ema: float = 0.35,
        gamma: float = 2.2,
        channels: int = 1,
        refresh_secs: float = 5.0,
    ) -> None:
        # Hardware is opened by start(), not here, so this object can be
        # constructed on a machine without SPI (spidev is Linux-only).
        self.serial = None
        self.device = None

        # Tuning
        self.fps = fps
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.ema = ema  # smoothing (0..1), higher = snappier
        self.gamma = gamma  # perceptual correction
        self.channels = max(1, int(channels))
        # How often to redraw the (static) heart as a guard against the
        # display losing state. 0 disables the periodic redraw entirely.
        self.refresh_secs = refresh_secs

        # State
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._env = 0.0  # smoothed envelope 0..1
        self._level = 0.0  # latest raw level 0..1 (thread-safe)
        self._lock = threading.Lock()

    def _open_device(self) -> None:
        """Open the SPI bus and MAX7219. Idempotent; needs real hardware."""
        if self.device is None:
            self.serial = spi(port=0, device=0, gpio=noop())
            self.device = max7219(self.serial, cascaded=1)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._open_device()
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the render loop and blank the display.

        Never raises on account of the render task. Awaiting a task that
        already failed -- an unplugged SPI bus, say -- would re-raise its
        exception here, and since stop() runs from __aexit__ that would
        replace whatever error the caller was already unwinding. The render
        failure is logged with its traceback instead.
        """
        if not self._task:
            return
        self._stop_evt.set()
        try:
            await asyncio.wait_for(self._task, timeout=0.5)
        except TimeoutError:
            # Two very different things land here, because TimeoutError
            # subclasses OSError: the render loop overran its grace period,
            # or the loop itself raised TimeoutError (an SPI or socket read
            # timing out). task.cancelled() tells them apart -- wait_for
            # cancels the task on a real timeout.
            #
            # Never re-await the task in either case. It is already finished,
            # and re-awaiting re-raises its exception past these handlers and
            # out of stop(), which would mask whatever the caller was
            # unwinding.
            if self._task.cancelled():
                logger.debug("Render task did not stop within the grace period")
            else:
                logger.exception("Render task failed")
        # CancelledError derives from BaseException, not Exception, so a
        # cancellation of stop() itself still propagates past this handler.
        except Exception:
            logger.exception("Render task failed")
        finally:
            self._task = None
            self._blank()

        # _run() swallows CancelledError so it can blank the display on its
        # way out. That means a cancellation aimed at *this* coroutine gets
        # absorbed: wait_for cancels _run, _run returns normally anyway, and
        # wait_for hands back that result instead of propagating. Without
        # this check the caller's cancellation would vanish silently.
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError()

    async def __aenter__(self) -> "Max7219AmplitudeHeart":
        """Start the display, guaranteeing stop() on the way out.

        Using the context manager makes it structurally impossible to leave
        the matrix lit when something later in startup fails.
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    def _blank(self) -> None:
        """Turn the display off. Never raises: this runs during shutdown."""
        if self.device is None:
            return
        try:
            self.device.contrast(0)
            if hasattr(self.device, "clear"):
                self.device.clear()
        except Exception as exc:
            # Losing the bus while shutting down is not worth crashing over,
            # but it should not vanish silently either.
            logger.debug(f"Could not blank the display: {exc!r}")

    def process_audio(self, audio_bytes: bytes) -> None:
        """
        Feed audio bytes (int16 interleaved). Uses self.channels to downmix
        if needed. Safe to call from any thread.
        """
        if not audio_bytes:
            self._set_level(0.0)
            return
        x = np.frombuffer(audio_bytes, dtype=np.int16)
        ch = self.channels
        if ch > 1:
            frames = (x.size // ch) * ch
            if frames == 0:
                self._set_level(0.0)
                return
            x = x[:frames].reshape(-1, ch).mean(axis=1)
        xf = x.astype(np.float32) / _INT16_MAX
        level = float(np.sqrt(np.mean(xf * xf)))
        level = max(0.0, min(1.0, level * 1.6))  # small headroom
        self._set_level(level)

    def _set_level(self, v: float) -> None:
        v = 0.0 if v < 0 else (1.0 if v > 1.0 else v)
        with self._lock:
            self._level = v

    def _get_level(self) -> float:
        with self._lock:
            return self._level

    def _brightness_from_level(self, level01: float) -> int:
        # EMA smoothing
        self._env = (1 - self.ema) * self._env + self.ema * level01
        # Perceptual gamma
        perceptual = self._env ** (1.0 / self.gamma)
        return self.min_brightness + int(
            perceptual * (self.max_brightness - self.min_brightness)
        )

    def _draw_heart(self) -> None:
        pixels = [
            # fmt: off
            (1, 1),
            (6, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 3),
            (7, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5),
            (3, 6),
            (4, 6),
            # fmt: on
        ]
        with canvas(self.device) as draw:
            for x, y in pixels:
                draw.point((x, y), fill="white")

    async def _run(self) -> None:
        """Track the audio envelope with the display's brightness.

        The heart pattern never changes, so the framebuffer is pushed once
        rather than every frame, and the intensity register is only written
        when the computed brightness actually differs from what the device
        already holds. A steady signal therefore produces no SPI traffic at
        all, instead of ~90 full-frame flushes per second.

        The envelope still advances every tick: the EMA is a filter over
        time, so skipping the arithmetic would change the animation.
        """
        period = 1.0 / float(self.fps)
        last_brightness: int | None = None
        last_redraw = 0.0
        try:
            self._draw_heart()
            last_redraw = time.monotonic()

            while not self._stop_evt.is_set():
                brightness = self._brightness_from_level(self._get_level())
                if brightness != last_brightness:
                    self.device.contrast(brightness)
                    last_brightness = brightness

                # Cheap insurance against the display losing state: redraw
                # the static pattern occasionally rather than never.
                now = time.monotonic()
                if self.refresh_secs and (now - last_redraw) >= self.refresh_secs:
                    self._draw_heart()
                    last_redraw = now

                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass
        finally:
            self._blank()
