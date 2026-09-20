from pathlib import Path

import numpy as np
import pytest

from palm_9000.wakeword import LiveKitWakeWordDetector

MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "wakeword" / "hey_livekit.onnx"
)


class FakeModel:
    """Stands in for livekit.wakeword.WakeWordModel."""

    def __init__(self, scores=None):
        self.scores = scores if scores is not None else {"hey_livekit": 0.0}
        self.calls = []

    def predict(self, frame):
        self.calls.append(frame)
        return self.scores


class TestScoreReduction:
    def test_returns_the_highest_score(self):
        detector = LiveKitWakeWordDetector(MODEL, model=FakeModel({"a": 0.2, "b": 0.9}))
        assert detector.process(b"\x00\x00") == 0.9

    def test_returns_zero_for_an_empty_result(self):
        detector = LiveKitWakeWordDetector(MODEL, model=FakeModel({}))
        assert detector.process(b"\x00\x00") == 0.0

    def test_converts_bytes_to_int16(self):
        model = FakeModel()
        LiveKitWakeWordDetector(MODEL, model=model).process(b"\x01\x00\x02\x00")
        frame = model.calls[0]
        assert frame.dtype == np.int16
        assert frame.tolist() == [1, 2]

    def test_empty_audio_scores_zero_without_calling_the_model(self):
        model = FakeModel()
        assert LiveKitWakeWordDetector(MODEL, model=model).process(b"") == 0.0
        assert model.calls == []

    def test_reset_clears_the_model(self):
        detector = LiveKitWakeWordDetector(MODEL, model=FakeModel())
        detector.reset()
        assert detector._model is None


class TestLoad:
    def test_load_raises_for_a_missing_model(self, tmp_path):
        """Startup must fail loudly, not silently stay asleep forever."""
        detector = LiveKitWakeWordDetector(tmp_path / "nope.onnx")
        with pytest.raises(FileNotFoundError, match="nope.onnx"):
            detector.load()

    def test_load_is_a_noop_when_a_model_is_injected(self):
        detector = LiveKitWakeWordDetector(MODEL, model=FakeModel())
        detector.load()  # must not raise or replace the stub
        assert isinstance(detector._model, FakeModel)


class TestAgainstTheRealModel:
    """Exercises the actual ONNX wiring, not just the stub."""

    def test_silence_does_not_trigger(self):
        detector = LiveKitWakeWordDetector(MODEL)
        silence = np.zeros(1280, dtype=np.int16).tobytes()
        scores = [detector.process(silence) for _ in range(12)]
        assert max(scores) < 0.5, scores

    def test_white_noise_does_not_trigger(self):
        detector = LiveKitWakeWordDetector(MODEL)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 3000, 1280).astype(np.int16).tobytes()
        scores = [detector.process(noise) for _ in range(12)]
        assert max(scores) < 0.5, scores

    def test_load_succeeds_for_the_committed_model(self):
        detector = LiveKitWakeWordDetector(MODEL)
        detector.load()
        assert detector._model is not None
