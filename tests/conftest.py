"""Shared fixtures and hardware fakes.

Several modules import hardware libraries at module scope that cannot be
installed off a Raspberry Pi (``RPi.GPIO``) or that need system packages
(``pyaudio``, ``sounddevice``). Those are injected into ``sys.modules`` here,
before the modules under test are imported, so the suite runs anywhere.

``luma`` is a real dependency and imports fine everywhere, so it is left
alone; individual tests patch ``palm_9000.gpio.spi`` / ``.max7219`` instead.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# palm_9000.settings instantiates Settings() at import time, and google_api_key
# is required. Without this, importing the package would depend on a developer's
# local .env and would fail outright in CI.
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

# --------------------------------------------------------------------------
# Fake RPi.GPIO (not installable off-Pi). Records every call so tests can
# assert on the exact bit-banging sequence ADC0834 performs.
# --------------------------------------------------------------------------


class FakeGPIO:
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls: list[tuple] = []
        self.setups: list[tuple] = []
        self.outputs: list[tuple] = []
        # RPi.GPIO requires a pin-numbering mode before setup(); None means
        # the caller has not chosen one yet.
        self.mode = None
        # Values that successive input() calls return.
        self.input_values: list[int] = []
        self._input_index = 0

    def setmode(self, mode):
        self.mode = mode
        self.calls.append(("setmode", mode))

    def getmode(self):
        return self.mode

    def setup(self, pin, mode):
        self.setups.append((pin, mode))
        self.calls.append(("setup", pin, mode))

    def output(self, pin, value):
        self.outputs.append((pin, value))
        self.calls.append(("output", pin, value))

    def input(self, pin):
        self.calls.append(("input", pin))
        if not self.input_values:
            return 0
        value = self.input_values[self._input_index % len(self.input_values)]
        self._input_index += 1
        return value


_fake_gpio = FakeGPIO()
_rpi_module = types.ModuleType("RPi")
_gpio_module = types.ModuleType("RPi.GPIO")
for _name in ("BCM", "OUT", "IN", "HIGH", "LOW"):
    setattr(_gpio_module, _name, getattr(FakeGPIO, _name))
_gpio_module.setmode = _fake_gpio.setmode
_gpio_module.getmode = _fake_gpio.getmode
_gpio_module.setup = _fake_gpio.setup
_gpio_module.output = _fake_gpio.output
_gpio_module.input = _fake_gpio.input
_rpi_module.GPIO = _gpio_module
sys.modules.setdefault("RPi", _rpi_module)
sys.modules.setdefault("RPi.GPIO", _gpio_module)

# --------------------------------------------------------------------------
# Fake pyaudio / sounddevice (need system libraries; only used by utils.py).
# --------------------------------------------------------------------------

_pyaudio_module = types.ModuleType("pyaudio")
_pyaudio_module.PyAudio = MagicMock(name="PyAudio")
sys.modules.setdefault("pyaudio", _pyaudio_module)

_sd_module = types.ModuleType("sounddevice")
_sd_module.check_input_settings = MagicMock(name="check_input_settings")
sys.modules.setdefault("sounddevice", _sd_module)


@pytest.fixture
def fake_gpio():
    """The shared FakeGPIO, reset before each use."""
    _fake_gpio.reset()
    return _fake_gpio


@pytest.fixture
def env(monkeypatch):
    """Clear every setting-related env var so Settings starts from defaults."""
    for name in (
        "GOOGLE_API_KEY",
        "GOOGLE_MULTIMODAL_LIVE_VOICE_ID",
        "GEMINI_LIVE_MODEL",
        "GOOGLE_CLOUD_PROJECT",
        "PICOVOICE_ACCESS_KEY",
        "PORCUPINE_KEYWORD",
        "PORCUPINE_KEYWORD_PATH",
        "PORCUPINE_MODEL_PATH",
        "PVLEOPARD_MODEL_PATH",
        "WHISPER_MODEL",
        "INPUT_DEVICE",
        "SAMPLE_RATE",
        "SILENCE_TIMEOUT",
        "VAD_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch
