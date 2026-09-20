"""Tests for the MAX7219 heart display.

The luma SPI/device classes are replaced with fakes so the suite runs without
a Raspberry Pi. ``canvas`` is a context manager in luma, so the fake mirrors
that shape.
"""

import asyncio
import math
import struct

import numpy as np
import pytest

from palm_9000 import gpio as gpio_module
from palm_9000.gpio import Max7219AmplitudeHeart


class FakeDevice:
    # luma's canvas builds a framebuffer image from these.
    mode = "1"
    size = (8, 8)

    def __init__(self, *, with_clear=True):
        self.contrasts: list[int] = []
        self.cleared = 0
        self.displays = 0
        if not with_clear:
            # Exercise the hasattr(device, "clear") guard.
            del type(self).clear

    def contrast(self, value):
        self.contrasts.append(value)

    def display(self, image):
        self.displays += 1

    def clear(self):
        self.cleared += 1


class FakeDraw:
    def __init__(self):
        self.points: list[tuple] = []

    def point(self, xy, fill=None):
        self.points.append((xy, fill))


class FakeCanvas:
    """Mimics luma.core.render.canvas' context-manager protocol."""

    last_draw: FakeDraw | None = None

    def __init__(self, device):
        self.device = device
        self.draw = FakeDraw()

    def __enter__(self):
        FakeCanvas.last_draw = self.draw
        return self.draw

    def __exit__(self, *exc):
        # luma pushes the framebuffer to the device when the block exits.
        if hasattr(self.device, "displays"):
            self.device.displays += 1
        return False


@pytest.fixture
def fake_hardware(monkeypatch):
    """Patch the luma entry points used by palm_9000.gpio."""
    devices = []

    def fake_max7219(serial, cascaded=1):
        device = FakeDevice()
        device.cascaded = cascaded
        devices.append(device)
        return device

    monkeypatch.setattr(gpio_module, "spi", lambda **kw: ("spi", kw))
    monkeypatch.setattr(gpio_module, "noop", lambda: "noop")
    monkeypatch.setattr(gpio_module, "max7219", fake_max7219)
    monkeypatch.setattr(gpio_module, "canvas", FakeCanvas)
    return devices


@pytest.fixture
def heart(fake_hardware):
    return Max7219AmplitudeHeart(fps=1000, min_brightness=0)


def pcm(samples) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


class TestConstruction:
    def test_construction_does_not_touch_hardware(self, monkeypatch):
        """No SPI at __init__ time, so this constructs on a dev machine.

        spidev is Linux-only; opening the bus in __init__ made the class
        unusable off a Pi. Note the *module* always imported fine -- it was
        construction that failed.
        """
        opened = []
        monkeypatch.setattr(
            gpio_module,
            "spi",
            lambda **kw: opened.append(kw) or ("spi", kw),
        )
        monkeypatch.setattr(
            gpio_module, "max7219", lambda serial, cascaded=1: FakeDevice()
        )

        heart = Max7219AmplitudeHeart()

        assert opened == [], "SPI must not be opened until start()"
        assert heart.device is None
        assert heart.serial is None

    async def test_start_opens_spi0_ce0_with_one_cascaded_module(
        self, fake_hardware, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(
            gpio_module, "spi", lambda **kw: captured.update(kw) or ("spi", kw)
        )

        heart = Max7219AmplitudeHeart(fps=1000)
        await heart.start()
        try:
            assert captured["port"] == 0
            assert captured["device"] == 0
            assert heart.device.cascaded == 1
        finally:
            await heart.stop()

    def test_open_device_is_idempotent(self, fake_hardware):
        heart = Max7219AmplitudeHeart()
        heart._open_device()
        first = heart.device
        heart._open_device()
        assert heart.device is first

    async def test_stop_before_start_does_not_touch_a_missing_device(self):
        """stop() must not blow up when the device was never opened."""
        heart = Max7219AmplitudeHeart()
        await heart.stop()
        assert heart.device is None

    def test_defaults(self, fake_hardware):
        h = Max7219AmplitudeHeart()
        assert h.fps == 90
        assert h.min_brightness == 4
        assert h.max_brightness == 255
        assert h.ema == 0.35
        assert h.gamma == 2.2
        assert h.channels == 1

    @pytest.mark.parametrize("channels,expected", [(0, 1), (-5, 1), (1, 1), (2, 2)])
    def test_channels_is_floored_at_one(self, fake_hardware, channels, expected):
        assert Max7219AmplitudeHeart(channels=channels).channels == expected

    def test_starts_with_no_task(self, heart):
        assert heart._task is None


class TestProcessAudio:
    def test_empty_audio_sets_level_to_zero(self, heart):
        heart._set_level(0.9)
        heart.process_audio(b"")
        assert heart._get_level() == 0.0

    def test_silence_gives_zero_level(self, heart):
        heart.process_audio(pcm([0] * 100))
        assert heart._get_level() == pytest.approx(0.0)

    def test_louder_audio_gives_a_higher_level(self, heart):
        heart.process_audio(pcm([1000] * 100))
        quiet = heart._get_level()
        heart.process_audio(pcm([20000] * 100))
        loud = heart._get_level()
        assert loud > quiet

    def test_level_is_rms_with_headroom(self, heart):
        """level = rms/32768 * 1.6, clamped to 1.0."""
        heart.process_audio(pcm([8192] * 64))
        assert heart._get_level() == pytest.approx(8192 / 32768 * 1.6, rel=1e-3)

    def test_level_is_clamped_to_one(self, heart):
        heart.process_audio(pcm([32767] * 64))
        assert heart._get_level() == 1.0

    def test_downmixes_multichannel_audio(self, fake_hardware):
        h = Max7219AmplitudeHeart(channels=2)
        # Left loud, right silent -> mean halves the amplitude.
        interleaved = []
        for _ in range(64):
            interleaved += [10000, 0]
        h.process_audio(pcm(interleaved))
        assert h._get_level() == pytest.approx(5000 / 32768 * 1.6, rel=1e-3)

    def test_multichannel_buffer_too_short_yields_zero(self, fake_hardware):
        """A single int16 cannot form one 4-channel frame."""
        h = Max7219AmplitudeHeart(channels=4)
        h._set_level(0.7)
        h.process_audio(pcm([123]))
        assert h._get_level() == 0.0

    def test_is_safe_from_another_thread(self, heart):
        import threading

        def worker():
            for _ in range(200):
                heart.process_audio(pcm([5000] * 16))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert 0.0 <= heart._get_level() <= 1.0


class TestLevelClamping:
    @pytest.mark.parametrize(
        "value,expected", [(-1.0, 0.0), (-0.001, 0.0), (0.5, 0.5), (1.5, 1.0)]
    )
    def test_set_level_clamps(self, heart, value, expected):
        heart._set_level(value)
        assert heart._get_level() == expected


class TestBrightness:
    def test_silence_maps_to_min_brightness(self, fake_hardware):
        h = Max7219AmplitudeHeart(min_brightness=7, ema=1.0)
        assert h._advance_envelope(0.0) == 7

    def test_full_level_maps_to_max_brightness(self, fake_hardware):
        h = Max7219AmplitudeHeart(min_brightness=0, max_brightness=255, ema=1.0)
        assert h._advance_envelope(1.0) == 255

    def test_applies_gamma_correction(self, fake_hardware):
        h = Max7219AmplitudeHeart(
            min_brightness=0, max_brightness=255, ema=1.0, gamma=2.0
        )
        # perceptual = level ** (1/gamma)
        assert h._advance_envelope(0.25) == int(math.sqrt(0.25) * 255)

    def test_ema_smooths_toward_the_target(self, fake_hardware):
        h = Max7219AmplitudeHeart(
            min_brightness=0, max_brightness=255, ema=0.5, gamma=1.0
        )
        first = h._advance_envelope(1.0)
        second = h._advance_envelope(1.0)
        assert first == int(0.5 * 255)
        assert second > first  # converging upward

    def test_result_stays_within_bounds(self, fake_hardware):
        h = Max7219AmplitudeHeart(min_brightness=10, max_brightness=200)
        for level in (0.0, 0.1, 0.5, 0.9, 1.0):
            assert 10 <= h._advance_envelope(level) <= 200


class TestDrawHeart:
    def test_draws_the_expected_pixel_count(self, heart):
        heart._draw_heart()
        assert len(FakeCanvas.last_draw.points) == 28

    def test_pixels_fit_an_eight_by_eight_grid(self, heart):
        heart._draw_heart()
        for (x, y), _ in FakeCanvas.last_draw.points:
            assert 0 <= x <= 7
            assert 0 <= y <= 7

    def test_shape_is_vertically_symmetric(self, heart):
        """A heart mirrors across the x axis."""
        heart._draw_heart()
        pts = {xy for xy, _ in FakeCanvas.last_draw.points}
        assert {(7 - x, y) for x, y in pts} == pts

    def test_pixels_are_lit_white(self, heart):
        heart._draw_heart()
        assert all(fill == "white" for _, fill in FakeCanvas.last_draw.points)


class TestStartStop:
    async def test_start_creates_a_running_task(self, heart):
        await heart.start()
        try:
            assert heart._task is not None
            assert not heart._task.done()
        finally:
            await heart.stop()

    async def test_start_is_idempotent(self, heart):
        await heart.start()
        first = heart._task
        await heart.start()
        try:
            assert heart._task is first
        finally:
            await heart.stop()

    async def test_start_replaces_a_finished_task(self, heart):
        await heart.start()
        heart._stop_evt.set()
        await asyncio.sleep(0.05)
        finished = heart._task
        heart._stop_evt.clear()
        await heart.start()
        try:
            assert heart._task is not finished
        finally:
            await heart.stop()

    async def test_stop_without_start_is_a_noop(self, heart):
        await heart.stop()  # must not raise
        assert heart._task is None

    async def test_stop_clears_the_task_and_display(self, heart):
        await heart.start()
        await asyncio.sleep(0.02)
        await heart.stop()

        assert heart._task is None
        assert heart.device.contrasts[-1] == 0
        assert heart.device.cleared >= 1

    async def test_stop_cancels_a_task_that_ignores_the_event(self, heart):
        async def never_finishes():
            await asyncio.sleep(30)

        heart._task = asyncio.create_task(never_finishes())
        await heart.stop()

        assert heart._task is None

    async def test_stop_survives_a_failing_display(self, heart, monkeypatch):
        await heart.start()
        await asyncio.sleep(0.02)

        def boom(_value):
            raise OSError("SPI went away")

        monkeypatch.setattr(heart.device, "contrast", boom)
        await heart.stop()  # the except Exception guard must swallow it

        assert heart._task is None

    async def test_stop_handles_a_device_without_clear(
        self, fake_hardware, monkeypatch
    ):
        class NoClearDevice:
            def __init__(self):
                self.contrasts = []

            def contrast(self, v):
                self.contrasts.append(v)

        monkeypatch.setattr(
            gpio_module, "max7219", lambda serial, cascaded=1: NoClearDevice()
        )
        h = Max7219AmplitudeHeart(fps=1000)
        await h.start()
        await asyncio.sleep(0.02)
        await h.stop()

        assert h.device.contrasts[-1] == 0


class TestRenderLoop:
    async def test_updates_brightness_continuously(self, heart):
        await heart.start()
        heart.process_audio(pcm([20000] * 64))
        await asyncio.sleep(0.05)
        await heart.stop()

        assert len(heart.device.contrasts) > 1

    async def test_tracks_audio_amplitude(self, heart):
        await heart.start()
        heart.process_audio(pcm([0] * 64))
        await asyncio.sleep(0.03)
        quiet = heart.device.contrasts[-1]

        heart.process_audio(pcm([32000] * 64))
        await asyncio.sleep(0.05)
        loud = heart.device.contrasts[-1]
        await heart.stop()

        assert loud > quiet

    async def test_redraws_the_heart_every_frame(self, heart):
        FakeCanvas.last_draw = None
        await heart.start()
        await asyncio.sleep(0.03)
        await heart.stop()

        assert FakeCanvas.last_draw is not None

    async def test_cancelling_the_task_shuts_the_display_down(self, heart):
        await heart.start()
        await asyncio.sleep(0.02)

        task = heart._task
        task.cancel()
        await task  # _run swallows CancelledError

        assert task.done()
        assert heart.device.contrasts[-1] == 0

    async def test_loop_period_follows_fps(self, fake_hardware, monkeypatch):
        slept = []
        real_sleep = asyncio.sleep

        async def recording_sleep(delay, *a, **kw):
            slept.append(delay)
            return await real_sleep(0, *a, **kw)

        monkeypatch.setattr(gpio_module.asyncio, "sleep", recording_sleep)

        h = Max7219AmplitudeHeart(fps=50)
        await h.start()
        await real_sleep(0.02)
        h._stop_evt.set()
        await real_sleep(0.01)

        assert slept and slept[0] == pytest.approx(1 / 50)


def test_int16_max_constant():
    assert gpio_module._INT16_MAX == 32768.0


def test_process_audio_accepts_numpy_backed_bytes(heart):
    data = (np.ones(64, dtype=np.int16) * 4096).tobytes()
    heart.process_audio(data)
    assert heart._get_level() > 0


class TestRedrawIsNotRepeated:
    """The heart pattern is static, so it must not be re-flushed every frame."""

    async def test_pattern_is_flushed_once_not_per_frame(self, heart):
        await heart.start()
        heart.process_audio(pcm([8000] * 64))
        await asyncio.sleep(0.08)  # ~80 frames at fps=1000
        await heart.stop()

        # One initial draw. The periodic refresh defaults to 5s, far longer
        # than this test runs, so nothing else should have been pushed.
        assert heart.device.displays == 1

    async def test_steady_level_stops_writing_contrast(self, heart):
        await heart.start()
        heart.process_audio(pcm([9000] * 64))
        await asyncio.sleep(0.05)  # let the EMA converge
        settled = len(heart.device.contrasts)

        await asyncio.sleep(0.05)  # another ~50 frames at the same level
        after = len(heart.device.contrasts)
        await heart.stop()

        assert after == settled, (
            "a constant audio level must produce no further SPI writes; "
            f"{after - settled} extra contrast writes were made"
        )

    async def test_changing_level_still_writes(self, heart):
        await heart.start()
        heart.process_audio(pcm([0] * 64))
        await asyncio.sleep(0.05)
        before = len(heart.device.contrasts)

        heart.process_audio(pcm([32000] * 64))
        await asyncio.sleep(0.05)
        after = len(heart.device.contrasts)
        await heart.stop()

        assert after > before, "brightness changes must still reach the device"

    async def test_no_duplicate_consecutive_contrast_values(self, heart):
        await heart.start()
        heart.process_audio(pcm([12000] * 64))
        await asyncio.sleep(0.1)
        await heart.stop()

        # The final 0 written by stop() may legitimately repeat a prior value,
        # so compare only the values written by the render loop.
        written = heart.device.contrasts[:-1]
        duplicates = [
            (a, b) for a, b in zip(written, written[1:], strict=False) if a == b
        ]
        assert duplicates == [], f"redundant repeated writes: {duplicates[:5]}"

    async def test_periodic_refresh_redraws_the_pattern(self, fake_hardware):
        """A slow redraw guards against the display losing state."""
        heart = Max7219AmplitudeHeart(fps=1000, refresh_secs=0.02)
        await heart.start()
        await asyncio.sleep(0.09)
        await heart.stop()

        assert heart.device.displays > 1

    async def test_refresh_can_be_disabled(self, fake_hardware):
        heart = Max7219AmplitudeHeart(fps=1000, refresh_secs=0)
        await heart.start()
        await asyncio.sleep(0.06)
        await heart.stop()

        assert heart.device.displays == 1


class TestContextManager:
    async def test_aenter_starts_and_aexit_stops(self, fake_hardware):
        heart = Max7219AmplitudeHeart(fps=1000)
        async with heart as entered:
            assert entered is heart
            assert heart._task is not None
        assert heart._task is None

    async def test_exit_runs_even_when_the_body_raises(self, fake_hardware):
        heart = Max7219AmplitudeHeart(fps=1000)
        with pytest.raises(RuntimeError):
            async with heart:
                raise RuntimeError("boom")
        assert heart._task is None
        assert heart.device.contrasts[-1] == 0


class TestBlank:
    def test_is_a_noop_when_no_device_was_opened(self, fake_hardware):
        Max7219AmplitudeHeart()._blank()  # must not raise

    def test_logs_instead_of_raising_when_the_bus_fails(
        self, fake_hardware, monkeypatch
    ):
        heart = Max7219AmplitudeHeart()
        heart._open_device()

        def boom(_value):
            raise OSError("SPI went away")

        monkeypatch.setattr(heart.device, "contrast", boom)
        heart._blank()  # swallowed, not raised

    def test_turns_the_display_off(self, fake_hardware):
        heart = Max7219AmplitudeHeart()
        heart._open_device()
        heart._blank()
        assert heart.device.contrasts[-1] == 0
        assert heart.device.cleared >= 1


class TestStopDoesNotMaskTheRealError:
    """A failed render task must not replace the error being unwound.

    stop() awaits the render task. If that task already raised -- say the SPI
    bus disconnected and _draw_heart() failed -- re-awaiting it re-raises
    inside __aexit__, which would discard whatever the body was failing with.
    """

    async def test_render_failure_does_not_replace_the_body_error(
        self, fake_hardware, monkeypatch
    ):
        heart = Max7219AmplitudeHeart(fps=1000)

        def explode():
            raise OSError("SPI bus disconnected")

        with pytest.raises(RuntimeError, match="the real problem"):
            async with heart:
                # Make the render loop die the way a yanked cable would.
                monkeypatch.setattr(heart, "_draw_heart", explode)
                heart._stop_evt.clear()
                await asyncio.sleep(0.02)
                raise RuntimeError("the real problem")

    async def test_render_failure_alone_does_not_escape_stop(
        self, fake_hardware, monkeypatch
    ):
        """Even with no body error, shutdown should not raise."""
        heart = Max7219AmplitudeHeart(fps=1000)
        await heart.start()

        async def failing_run():
            raise OSError("SPI bus disconnected")

        heart._task = asyncio.create_task(failing_run())
        await asyncio.sleep(0.01)

        await heart.stop()  # must not raise
        assert heart._task is None

    async def test_the_render_failure_is_still_logged(self, fake_hardware, monkeypatch):
        """Swallowed is not the same as hidden."""
        records = []
        monkeypatch.setattr(
            gpio_module.logger, "exception", lambda msg, *a, **kw: records.append(msg)
        )

        heart = Max7219AmplitudeHeart(fps=1000)
        await heart.start()

        async def failing_run():
            raise OSError("SPI bus disconnected")

        heart._task = asyncio.create_task(failing_run())
        await asyncio.sleep(0.01)
        await heart.stop()

        assert records, "the render task failure vanished without a trace"

    async def test_failure_during_cancellation_is_also_contained(
        self, fake_hardware, monkeypatch
    ):
        """A task that refuses to stop, then fails while being cancelled.

        asyncio.wait_for cancels the task itself on timeout and re-raises
        whatever the task ended with, so this arrives as an OSError rather
        than a TimeoutError.
        """
        records = []
        monkeypatch.setattr(
            gpio_module.logger, "exception", lambda msg, *a, **kw: records.append(msg)
        )

        heart = Max7219AmplitudeHeart(fps=1000)
        await heart.start()

        async def stubborn():
            try:
                await asyncio.sleep(30)  # ignores _stop_evt, so stop() times out
            except asyncio.CancelledError:
                raise OSError("bus died while shutting down") from None

        heart._task = asyncio.create_task(stubborn())
        await asyncio.sleep(0.01)

        await heart.stop()  # must not raise

        assert heart._task is None
        # wait_for re-raises the task's own error instead of TimeoutError,
        # so this lands in stop()'s general handler.
        assert any("render task failed" in m.lower() for m in records), records

    async def test_cancelling_stop_itself_still_propagates(self, fake_hardware):
        """CancelledError is a BaseException, so it must pass through."""
        heart = Max7219AmplitudeHeart(fps=1000)
        await heart.start()

        async def never():
            await asyncio.sleep(30)

        heart._task = asyncio.create_task(never())
        heart._task.cancel()
        await asyncio.sleep(0)

        with pytest.raises(asyncio.CancelledError):
            await heart.stop()
