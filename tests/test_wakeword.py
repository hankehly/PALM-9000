from pathlib import Path

import numpy as np
import pytest

from palm_9000.wakeword import WINDOW_SAMPLES, LiveKitWakeWordDetector

MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "wakeword" / "hey_livekit.onnx"
)

# The frame size LocalAudioTransport actually pushes: 20ms at 16kHz.
# pipecat/transports/local/audio.py:76 -- num_frames = sample_rate // 100 * 2
REAL_FRAME_SAMPLES = 320


class FakeModel:
    """Stands in for livekit.wakeword.WakeWordModel."""

    def __init__(self, scores=None):
        self.scores = scores if scores is not None else {"hey_livekit": 0.0}
        self.calls = []

    def predict(self, frame):
        self.calls.append(frame)
        return self.scores


class TestScoreReduction:
    """Score reduction / conversion logic, isolated from buffering with a
    1-sample window and hop so every process() call scores immediately."""

    def test_returns_the_highest_score(self):
        detector = LiveKitWakeWordDetector(
            MODEL,
            model=FakeModel({"a": 0.2, "b": 0.9}),
            window_samples=1,
            hop_samples=1,
        )
        assert detector.process(b"\x00\x00") == 0.9

    def test_returns_zero_for_an_empty_result(self):
        detector = LiveKitWakeWordDetector(
            MODEL, model=FakeModel({}), window_samples=1, hop_samples=1
        )
        assert detector.process(b"\x00\x00") == 0.0

    def test_converts_bytes_to_int16(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=2, hop_samples=1
        )
        detector.process(b"\x01\x00\x02\x00")
        frame = model.calls[0]
        assert frame.dtype == np.int16
        assert frame.tolist() == [1, 2]

    def test_empty_audio_scores_zero_without_calling_the_model(self):
        model = FakeModel()
        assert LiveKitWakeWordDetector(MODEL, model=model).process(b"") == 0.0
        assert model.calls == []


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


class TestBuffering:
    """Pins the rolling-buffer fix itself: predict() is stateless, so
    LiveKitWakeWordDetector must accumulate audio and rate-limit scoring on
    its own. All use a fake model so they exercise only the buffering, not
    the real ONNX wiring (that's TestAgainstTheRealModel below).
    """

    def test_nothing_scored_before_window_fills(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=100, hop_samples=20
        )
        frame = np.zeros(40, dtype=np.int16).tobytes()

        detector.process(frame)  # 40 samples buffered
        detector.process(frame)  # 80 samples buffered
        assert model.calls == []

        detector.process(frame)  # 120 samples: window is now full
        assert len(model.calls) == 1

    def test_hop_is_honoured(self):
        """predict() should run roughly every hop_samples, not every frame."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=100, hop_samples=20
        )
        frame = np.zeros(10, dtype=np.int16).tobytes()
        num_frames = 40
        for _ in range(num_frames):
            detector.process(frame)

        # Window fills at frame 10 (100 samples), then a fresh score every 2
        # frames (20 hop samples / 10 samples-per-frame): far fewer calls
        # than frames, and more than a lone one-off.
        assert 1 < len(model.calls) < num_frames // 2

    def test_predict_receives_a_full_window(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=100, hop_samples=20
        )
        frame = np.zeros(10, dtype=np.int16).tobytes()
        for _ in range(10):
            detector.process(frame)

        assert len(model.calls) == 1
        received = model.calls[0]
        assert len(received) == 100
        assert received.dtype == np.int16


class TestReset:
    def test_reset_clears_buffer_but_keeps_model(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=100, hop_samples=20
        )
        frame = np.zeros(10, dtype=np.int16).tobytes()
        for _ in range(10):
            detector.process(frame)
        assert len(model.calls) == 1  # window filled once

        detector.reset()
        assert detector._model is model  # kept, not rebuilt

        for _ in range(9):
            detector.process(frame)
        assert len(model.calls) == 1  # buffer hasn't filled again yet

        detector.process(frame)  # the 10th frame refills the window
        assert len(model.calls) == 2


class TestAgainstTheRealModel:
    """Exercises the actual ONNX wiring, not just the stub."""

    def test_real_frame_size_does_not_raise(self):
        """The regression test for the stateless-predict() bug.

        Before the fix, process() handed each 320-sample pipeline frame
        straight to predict(), which raises an onnxruntime
        InvalidArgument shape error for any input that small -- every
        single frame threw. Feeding a full window's worth of the real
        transport frame size must not raise, on any frame, and must
        return a score.
        """
        detector = LiveKitWakeWordDetector(MODEL)
        frame = np.zeros(REAL_FRAME_SAMPLES, dtype=np.int16).tobytes()
        num_frames = WINDOW_SAMPLES // REAL_FRAME_SAMPLES
        for _ in range(num_frames):
            score = detector.process(frame)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_full_window_of_silence_is_nonzero(self):
        """predict() returns a flat 0.0 for anything under ~2s regardless
        of content, so a zero score alone would not prove the classifier
        ran. Once the buffer holds a full window, the real model returns a
        small but genuinely non-zero score for silence (~0.0047 measured
        against the committed model) -- proof the mel/embedding/classifier
        path actually executed rather than hitting predict()'s
        not-enough-data early return. The exact value isn't asserted, only
        that it is present and nowhere near triggering.
        """
        detector = LiveKitWakeWordDetector(MODEL)
        frame = np.zeros(REAL_FRAME_SAMPLES, dtype=np.int16).tobytes()
        num_frames = WINDOW_SAMPLES // REAL_FRAME_SAMPLES
        scores = [detector.process(frame) for _ in range(num_frames)]
        assert 0.0 < scores[-1] < 0.5, scores

    def test_silence_does_not_trigger(self):
        detector = LiveKitWakeWordDetector(MODEL)
        frame = np.zeros(REAL_FRAME_SAMPLES, dtype=np.int16).tobytes()
        num_frames = WINDOW_SAMPLES // REAL_FRAME_SAMPLES
        scores = [detector.process(frame) for _ in range(num_frames)]
        assert max(scores) < 0.5, scores

    def test_white_noise_does_not_trigger(self):
        detector = LiveKitWakeWordDetector(MODEL)
        rng = np.random.default_rng(0)
        num_frames = WINDOW_SAMPLES // REAL_FRAME_SAMPLES
        scores = []
        for _ in range(num_frames):
            frame = rng.normal(0, 3000, REAL_FRAME_SAMPLES).astype(np.int16).tobytes()
            scores.append(detector.process(frame))
        assert max(scores) < 0.5, scores

    def test_load_succeeds_for_the_committed_model(self):
        detector = LiveKitWakeWordDetector(MODEL)
        detector.load()
        assert detector._model is not None

    def test_the_committed_model_is_the_one_actually_loaded(self):
        """A stub or a wrong model file cannot produce this key.

        The other tests here only bound the score from above, so they would
        pass against anything that returns 0.0. This one pins the identity of
        the model that is actually running.
        """
        detector = LiveKitWakeWordDetector(MODEL)
        detector.load()
        scores = detector._model.predict(np.zeros(1280, dtype=np.int16))
        assert "hey_livekit" in scores, scores
