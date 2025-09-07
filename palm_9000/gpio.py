import asyncio
import threading

import numpy as np
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
                    frame = [int(amp * 0.8 * 32767 * math.sin(2 * math.pi * freq * (t + i / sample_rate))) for i in range(frame_samples)]
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
    ) -> None:
        # Display init
        self.serial = spi(port=0, device=0, gpio=noop())
        self.device = max7219(self.serial, cascaded=1)

        # Tuning
        self.fps = fps
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.ema = ema  # smoothing (0..1), higher = snappier
        self.gamma = gamma  # perceptual correction
        self.channels = max(1, int(channels))

        # State
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._env = 0.0  # smoothed envelope 0..1
        self._level = 0.0  # latest raw level 0..1 (thread-safe)
        self._lock = threading.Lock()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if not self._task:
            return
        self._stop_evt.set()
        try:
            await asyncio.wait_for(self._task, timeout=0.5)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        finally:
            self._task = None
            # turn off & clear
            try:
                self.device.contrast(0)
                if hasattr(self.device, "clear"):
                    self.device.clear()
            except Exception:
                pass

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

    async def _run(self) -> None:
        period = 1.0 / float(self.fps)
        try:
            while not self._stop_evt.is_set():
                lvl = self._get_level()
                b = self._brightness_from_level(lvl)
                self.device.contrast(b)
                self._draw_heart()
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass
        finally:
            try:
                self.device.contrast(0)
                if hasattr(self.device, "clear"):
                    self.device.clear()
            except Exception:
                pass
