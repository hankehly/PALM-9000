import struct
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

from palm_9000 import utils


class TestResample:
    def test_changes_length_by_the_rate_ratio(self):
        audio = np.zeros(44100, dtype=np.float32)
        out = utils.resample(audio, 44100, 16000)
        assert len(out) == 16000

    def test_preserves_a_sine_wave_frequency(self):
        sr_in, sr_out, freq = 44100, 16000, 440.0
        t = np.arange(sr_in) / sr_in
        audio = np.sin(2 * np.pi * freq * t)

        out = utils.resample(audio, sr_in, sr_out)

        # Dominant FFT bin should still sit at 440 Hz (within a couple of bins).
        peak_bin = int(np.argmax(np.abs(np.fft.rfft(out))))
        assert abs(peak_bin - freq) <= 2

    def test_reduces_the_ratio_by_gcd(self, monkeypatch):
        """44100:16000 must be reduced to 441:160 to save compute."""
        captured = {}

        def fake_resample_poly(audio, up, down):
            captured["up"] = up
            captured["down"] = down
            return audio

        monkeypatch.setattr(utils, "resample_poly", fake_resample_poly)
        utils.resample(np.zeros(10), 44100, 16000)

        assert (captured["up"], captured["down"]) == (160, 441)

    def test_identical_rates_reduce_to_one_to_one(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            utils,
            "resample_poly",
            lambda a, up, down: captured.update(up=up, down=down) or a,
        )
        utils.resample(np.zeros(10), 16000, 16000)
        assert (captured["up"], captured["down"]) == (1, 1)


class TestPlayAudio:
    @staticmethod
    def _pcm(n=4096, amplitude=1000):
        return struct.pack("<%dh" % n, *([amplitude] * n))

    def test_writes_every_frame_to_the_stream(self, monkeypatch):
        pa = MagicMock()
        stream = pa.open.return_value
        monkeypatch.setattr(utils.pyaudio, "PyAudio", MagicMock(return_value=pa))

        utils.play_audio(self._pcm(), sample_rate=16000)

        assert stream.write.call_count > 0
        written = b"".join(call.args[0] for call in stream.write.call_args_list)
        assert len(written) == 4096 * 2
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        pa.terminate.assert_called_once()

    def test_opens_stream_with_the_given_sample_rate(self, monkeypatch):
        pa = MagicMock()
        monkeypatch.setattr(utils.pyaudio, "PyAudio", MagicMock(return_value=pa))

        utils.play_audio(self._pcm(n=64), sample_rate=24000)

        kwargs = pa.open.call_args.kwargs
        assert kwargs["rate"] == 24000
        assert kwargs["channels"] == 1
        assert kwargs["output"] is True

    def test_volume_multiplies_samples(self, monkeypatch):
        pa = MagicMock()
        stream = pa.open.return_value
        monkeypatch.setattr(utils.pyaudio, "PyAudio", MagicMock(return_value=pa))

        utils.play_audio(self._pcm(n=64, amplitude=1000), volume=2.0)

        written = b"".join(call.args[0] for call in stream.write.call_args_list)
        samples = np.frombuffer(written, dtype=np.int16)
        assert samples[0] == 2000

    def test_volume_clips_instead_of_wrapping(self, monkeypatch):
        """Without np.clip, int16 overflow would wrap to a negative value."""
        pa = MagicMock()
        stream = pa.open.return_value
        monkeypatch.setattr(utils.pyaudio, "PyAudio", MagicMock(return_value=pa))

        utils.play_audio(self._pcm(n=64, amplitude=30000), volume=4.0)

        written = b"".join(call.args[0] for call in stream.write.call_args_list)
        samples = np.frombuffer(written, dtype=np.int16)
        assert samples[0] == 32767

    def test_output_is_a_valid_wav_stream(self, monkeypatch):
        """The in-memory WAV header must be well formed (16-bit mono)."""
        captured = {}
        pa = MagicMock()

        def capture_open(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        pa.open.side_effect = capture_open
        pa.get_format_from_width.side_effect = lambda w: f"width{w}"
        monkeypatch.setattr(utils.pyaudio, "PyAudio", MagicMock(return_value=pa))

        utils.play_audio(self._pcm(n=64))

        assert captured["format"] == "width2"


class TestWaitUntilDeviceAvailable:
    def test_returns_true_once_the_device_checks_out(self, monkeypatch):
        monkeypatch.setattr(utils.sd, "check_input_settings", lambda device: None)
        assert utils.wait_until_device_available(1) is True

    def test_retries_until_the_device_appears(self, monkeypatch):
        attempts = {"n": 0}

        def flaky(device):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("device busy")

        monkeypatch.setattr(utils.sd, "check_input_settings", flaky)
        assert utils.wait_until_device_available(1, timeout=5.0) is True
        assert attempts["n"] == 3

    def test_raises_after_the_timeout(self, monkeypatch):
        monkeypatch.setattr(
            utils.sd,
            "check_input_settings",
            MagicMock(side_effect=OSError("never available")),
        )
        with pytest.raises(RuntimeError, match="Mic still unavailable"):
            utils.wait_until_device_available(1, timeout=0.1)


class TestRemoveWhitespace:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a b c", "abc"),
            ("  leading and trailing  ", "leadingandtrailing"),
            ("tabs\tand\nnewlines", "tabsandnewlines"),
            ("", ""),
            ("nospace", "nospace"),
            ("　全角　スペース　", "全角スペース"),
        ],
    )
    def test_strips_all_whitespace(self, text, expected):
        assert utils.remove_whitespace(text) == expected
