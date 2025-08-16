import asyncio
import threading

import numpy as np
import RPi.GPIO as GPIO
from loguru import logger
from luma.core.interface.serial import noop, spi
from luma.core.render import canvas
from luma.led_matrix.device import max7219

from palm_9000.settings import settings

_INT16_MAX = 32768.0


class LedAmplitudeGPIO:
    def __init__(self, gpio_pin=18, pwm_hz=800, fps=120, ema=0.35, gamma=2.2):
        self.pin = gpio_pin
        self.fps = fps
        self.ema = ema
        self.gamma = gamma

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, pwm_hz)
        self.pwm.start(0.0)

        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._level_raw = 0.0
        self._env = 0.0
        self._lock = threading.Lock()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
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
            self.pwm.ChangeDutyCycle(0.0)
            self.pwm.stop()
            GPIO.cleanup(self.pin)
            self._task = None

    def make_callback(
        self,
        channels: int,
        *,
        gain: float = 1.0,
        noise_floor: float = 0.015,
        hysteresis: float = 0.005,
    ):
        """
        Returns a function to feed audio bytes (int16 interleaved).
        - gain: linear boost after gating (1.0 = no boost)
        - noise_floor: gate threshold (RMS, 0..1). Below this → LED off.
        - hysteresis: extra margin to prevent flicker when hovering near floor.
        """
        # Keep a tiny bit of state for hysteresis (thread-safe enough here)
        gate_open = False

        def _cb(audio_bytes: bytes):
            nonlocal gate_open
            if not audio_bytes:
                self._set_level(0.0)
                return

            x = np.frombuffer(audio_bytes, dtype=np.int16)
            if channels > 1:
                frames = (x.size // channels) * channels
                if frames == 0:
                    self._set_level(0.0)
                    return
                x = x[:frames].reshape(-1, channels).mean(axis=1)

            xf = x.astype(np.float32) / _INT16_MAX
            rms = float(np.sqrt(np.mean(xf * xf)))  # 0..1

            # Noise gate with hysteresis
            if gate_open:
                # stay open until we drop below (floor - hysteresis)
                if rms < max(0.0, noise_floor - hysteresis):
                    gate_open = False
            else:
                # stay closed until we exceed (floor + hysteresis)
                if rms > min(1.0, noise_floor + hysteresis):
                    gate_open = True

            if not gate_open:
                self._set_level(0.0)
                return

            # Map [noise_floor..1] -> [0..1] to use full LED range
            if rms <= noise_floor:
                level = 0.0
            else:
                level = (rms - noise_floor) / (1.0 - noise_floor)

            # Apply user gain, clamp
            level = max(0.0, min(1.0, level * gain))
            self._set_level(level)

        return _cb

    def _set_level(self, v: float):
        with self._lock:
            self._level_raw = v

    def _get_level(self) -> float:
        with self._lock:
            return self._level_raw

    async def _run(self):
        period = 1.0 / float(self.fps)
        try:
            while not self._stop_evt.is_set():
                tgt = self._get_level()
                self._env = (1 - self.ema) * self._env + self.ema * tgt
                perceptual = self._env ** (1.0 / self.gamma)
                duty_pct = max(0.0, min(100.0, perceptual * 100.0))
                self.pwm.ChangeDutyCycle(duty_pct)
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass
        finally:
            self.pwm.ChangeDutyCycle(0.0)


import asyncio
import math
import threading
import time
from collections import deque

import numpy as np
from luma.core.interface.serial import noop, spi
from luma.core.render import canvas
from luma.led_matrix.device import max7219

_INT16_MAX = 32768.0


class Max7219AmplitudeHeart:
    """
    Drive a heart icon on an 8x8 MAX7219, brightness = audio amplitude.
    Call `start()` once. Feed audio via the callback from `make_callback(...)`.
    """

    def __init__(
        self, fps=90, min_brightness=4, max_brightness=255, ema=0.35, gamma=2.2
    ):
        # Display init
        self.serial = spi(port=0, device=0, gpio=noop())
        self.device = max7219(self.serial, cascaded=1)

        # Tuning
        self.fps = fps
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.ema = ema  # smoothing (0..1), higher = snappier
        self.gamma = gamma  # perceptual correction

        # State
        self._task: asyncio.Task | None = None
        self._stop_evt = asyncio.Event()
        self._env = 0.0  # smoothed envelope 0..1
        self._level = 0.0  # latest raw level 0..1 (thread-safe)
        self._lock = threading.Lock()

    # ---------- public control ----------
    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop_evt.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self):
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

    def make_callback(self, channels: int):
        """
        Returns a function you can call with audio bytes (int16 interleaved).
        Safe to call from any thread (blocking write, PyAudio callback, etc.).
        """

        def _cb(audio_bytes: bytes):
            print(len(audio_bytes))
            if not audio_bytes:
                self._set_level(0.0)
                return
            x = np.frombuffer(audio_bytes, dtype=np.int16)
            if channels > 1:
                frames = (x.size // channels) * channels
                if frames == 0:
                    self._set_level(0.0)
                    return
                x = x[:frames].reshape(-1, channels).mean(axis=1)
            xf = x.astype(np.float32) / _INT16_MAX
            # RMS; you could blend with peak if you want more punch
            level = float(np.sqrt(np.mean(xf * xf)))
            # gentle headroom so small signals still show
            level = max(0.0, min(1.0, level * 1.6))
            self._set_level(level)

        return _cb

    # ---------- internals ----------
    def _set_level(self, v: float):
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

    def _draw_heart(self):
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

    async def _run(self):
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
