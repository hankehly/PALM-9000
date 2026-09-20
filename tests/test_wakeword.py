from pathlib import Path

import numpy as np
import pytest

from palm_9000.wakeword import (
    EMBEDDING_WINDOW,
    HOP_SAMPLES,
    INT16_SCALE,
    MEL_BINS,
    MEL_HOP_SAMPLES,
    MEL_WINDOW_SAMPLES,
    MIN_EMBEDDINGS,
    MIN_WINDOW_SAMPLES,
    REQUIRED_MODEL_ATTRS,
    WINDOW_SAMPLES,
    LiveKitWakeWordDetector,
)

MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "wakeword" / "hey_livekit.onnx"
)

# The frame size LocalAudioTransport actually pushes: 20ms at 16kHz.
# pipecat/transports/local/audio.py:76 -- num_frames = sample_rate // 100 * 2
REAL_FRAME_SAMPLES = 320

SAMPLE_RATE = 16000
EMBEDDING_DIMS = 96


class FakeMelFrontend:
    """Stands in for livekit's MelSpectrogramFrontend, without ONNX.

    The geometry is not invented: frames-per-sample follows the measured
    MEL_WINDOW_SAMPLES / MEL_HOP_SAMPLES, because the detector's carry-over
    and its embedding cadence are both derived from it. A fake that guessed
    would make the buffering tests pass for the wrong reason.

    Frame values depend on the audio in the frame, so a dropped left
    context or a gap between calls changes what the embeddings see.
    """

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, audio):
        self.calls.append(audio)
        n_frames = (len(audio) - MEL_WINDOW_SAMPLES) // MEL_HOP_SAMPLES + 1
        sums = np.array(
            [
                audio[s * MEL_HOP_SAMPLES : s * MEL_HOP_SAMPLES + MEL_WINDOW_SAMPLES]
                .astype(np.float64)
                .sum()
                for s in range(n_frames)
            ],
            dtype=np.float32,
        )
        frames = sums[:, np.newaxis] + np.arange(MEL_BINS, dtype=np.float32)
        return frames[np.newaxis, :, :]  # (1, frames, 32), like the real one


class FakeSpeechEmbedding:
    """Stands in for livekit's SpeechEmbedding CNN.

    Returns a content-dependent vector so ordering and window contents are
    visible downstream, and pins the (batch, 76, 32) input contract.
    """

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, windows):
        self.calls.append(windows)
        assert windows.shape[1:] == (EMBEDDING_WINDOW, MEL_BINS), windows.shape
        value = np.float32(windows.sum())
        return np.full((windows.shape[0], EMBEDDING_DIMS), value, dtype=np.float32)


class FakeClassifier:
    """Stands in for an onnxruntime.InferenceSession over the embeddings."""

    def __init__(self, score, calls):
        self.score = score
        self.calls = calls

    def run(self, output_names, feeds):
        assert output_names is None, output_names
        (emb_input,) = feeds.values()
        self.calls.append(emb_input)
        return [np.array([[self.score]], dtype=np.float32)]


class FakeModel:
    """Stands in for livekit.wakeword.WakeWordModel's private surface.

    `calls` collects one entry -- the classifier's input -- per classifier
    run, so for the single-classifier fakes used below `len(model.calls)` is
    the number of scored windows, the same thing the old predict()-counting
    tests measured.
    """

    INPUT_NAME = "embeddings"

    def __init__(self, scores=None):
        scores = {"hey_livekit": 0.0} if scores is None else scores
        self.calls = []
        self.mel_calls = []
        self.embedding_calls = []
        self._mel_frontend = FakeMelFrontend(self.mel_calls)
        self._speech_embedding = FakeSpeechEmbedding(self.embedding_calls)
        self._classifiers = {
            name: (FakeClassifier(score, self.calls), self.INPUT_NAME)
            for name, score in scores.items()
        }


def silence(n):
    return np.zeros(n, dtype=np.int16)


def frames_of(samples, size=REAL_FRAME_SAMPLES):
    """Chop int16 audio into the transport's real frame size."""
    return [samples[i : i + size].tobytes() for i in range(0, len(samples), size)]


class TestScoreReduction:
    """Score reduction / conversion logic, isolated from buffering timing:
    hop_samples=1 and a single frame that already fills the window, so
    process() always scores on the first call. (A 1-sample window would do
    this more cheaply, but the constructor no longer allows one -- see
    TestWindowGuard.)
    """

    def test_returns_the_highest_score(self):
        # Powers of two: the real classifier's output is float32, and so is
        # the fake's, so 0.9 would come back as 0.8999999761581421.
        detector = LiveKitWakeWordDetector(
            MODEL,
            model=FakeModel({"a": 0.25, "b": 0.875}),
            window_samples=WINDOW_SAMPLES,
            hop_samples=1,
        )
        assert detector.process(silence(WINDOW_SAMPLES).tobytes()) == 0.875

    def test_returns_zero_when_no_classifier_is_loaded(self):
        detector = LiveKitWakeWordDetector(
            MODEL, model=FakeModel({}), window_samples=WINDOW_SAMPLES, hop_samples=1
        )
        assert detector.process(silence(WINDOW_SAMPLES).tobytes()) == 0.0

    def test_converts_bytes_to_scaled_float32(self):
        """int16 bytes in, float32 /32768 out -- exactly what livekit's own
        predict() feeds the mel frontend, so the two paths stay comparable
        sample for sample."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=1
        )
        detector.process(silence(WINDOW_SAMPLES - 2).tobytes())  # not full yet
        detector.process(b"\x01\x00\x02\x00")

        audio = model.mel_calls[0]
        assert audio.dtype == np.float32
        assert len(audio) == WINDOW_SAMPLES
        assert audio[-2:].tolist() == [1 / INT16_SCALE, 2 / INT16_SCALE]

    def test_empty_audio_scores_zero_without_touching_the_model(self):
        model = FakeModel()
        assert LiveKitWakeWordDetector(MODEL, model=model).process(b"") == 0.0
        assert model.mel_calls == []
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

    @pytest.mark.parametrize("missing", REQUIRED_MODEL_ATTRS)
    def test_load_raises_when_the_library_renames_its_internals(self, missing):
        """The whole reason this class is 27x cheaper is that it drives
        WakeWordModel's private attributes instead of calling predict().
        If an upgrade renames one, that must stop the process at startup,
        naming the attribute -- not fall back to predict(), which would be
        a silent 27x slowdown and a plant that stops answering with no
        error in the log.
        """
        model = FakeModel()
        delattr(model, missing)
        detector = LiveKitWakeWordDetector(MODEL, model=model)
        with pytest.raises(RuntimeError, match=missing):
            detector.load()


class TestWindowGuard:
    """window_samples below the model's real minimum must fail loudly at
    construction, not silently go deaf forever -- see MIN_WINDOW_SAMPLES in
    palm_9000/wakeword.py for how that floor was measured.
    """

    def test_rejects_a_window_below_the_floor(self):
        with pytest.raises(ValueError, match="window_samples"):
            LiveKitWakeWordDetector(MODEL, window_samples=MIN_WINDOW_SAMPLES - 1)

    def test_accepts_a_window_at_exactly_the_floor(self):
        detector = LiveKitWakeWordDetector(MODEL, window_samples=MIN_WINDOW_SAMPLES)
        assert detector.window_samples == MIN_WINDOW_SAMPLES


class TestHopGuard:
    """hop_samples <= 0 would score on every process() call instead of
    throttling, defeating the reason the hop exists."""

    @pytest.mark.parametrize("hop", [0, -1])
    def test_rejects_a_non_positive_hop(self, hop):
        with pytest.raises(ValueError, match="hop_samples"):
            LiveKitWakeWordDetector(MODEL, hop_samples=hop)

    def test_accepts_a_hop_of_one(self):
        detector = LiveKitWakeWordDetector(MODEL, hop_samples=1)
        assert detector.hop_samples == 1


class TestBuffering:
    """Pins the rolling-buffer behaviour itself: the wake-word stack has no
    audio state of its own, so LiveKitWakeWordDetector must accumulate audio
    and rate-limit scoring. All use a fake model so they exercise only the
    buffering, not the real ONNX wiring (that's TestAgainstTheRealModel and
    TestMatchesTheLibrary below).

    window_samples is WINDOW_SAMPLES (32000, comfortably above the real
    MIN_WINDOW_SAMPLES floor) throughout; hop_samples and frame sizes are
    chosen per test and verified against a direct simulation of process()'s
    bookkeeping, not guessed.
    """

    def test_nothing_scored_before_window_fills(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=6400
        )
        frame = silence(12800).tobytes()

        detector.process(frame)  # 12800 samples buffered
        detector.process(frame)  # 25600 samples buffered
        assert model.calls == []

        detector.process(frame)  # 38400 samples: window is now full
        assert len(model.calls) == 1

    def test_hop_is_honoured(self):
        """Scoring should happen roughly every hop_samples, not every frame."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=8000
        )
        frame = silence(2000).tobytes()
        num_frames = 64
        for _ in range(num_frames):
            detector.process(frame)

        # Window fills at frame 16 (32000 samples), then a fresh score every
        # 4 frames (8000 hop samples / 2000 samples-per-frame): far fewer
        # calls than frames, and more than a lone one-off. (16 exactly, by
        # direct simulation -- comfortably inside this range rather than
        # pinned to it, since num_frames // 2 is not what's under test.)
        assert 1 < len(model.calls) < num_frames // 2

    def test_hop_rounding_matches_the_nominal_rate_when_frame_does_not_divide_hop(
        self,
    ):
        """Regression for the pre-fix `_samples_since_score = 0`.

        320 does not divide 1000. Zeroing the backlog instead of
        subtracting the hop rounds the *effective* hop up to 1280 (the next
        multiple of 320 at or above 1000) and under-scores: over 500 frames
        that would be 500*320/1280 = 125 calls instead of the nominal
        500*320/1000 = 160. Subtracting keeps the long-run rate at the
        nominal 160 -- verified by direct simulation before writing this
        assertion, not guessed.
        """
        model = FakeModel()
        hop = 1000
        frame_samples = REAL_FRAME_SAMPLES  # 320; 1000 % 320 != 0
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=hop
        )
        frame = silence(frame_samples).tobytes()

        # Fill the window first, so this measures the steady-state rate.
        # No need to clear _samples_since_score by hand: taking the backlog
        # modulo the hop already leaves it at 0 here, which is what
        # test_filling_the_window_does_not_cause_a_burst_of_scores pins.
        for _ in range(WINDOW_SAMPLES // frame_samples):
            detector.process(frame)
        assert detector._samples_since_score == 0
        model.calls.clear()

        num_frames = 500
        for _ in range(num_frames):
            detector.process(frame)

        assert len(model.calls) == 160  # nominal: 500 * 320 / 1000

    def test_filling_the_window_does_not_cause_a_burst_of_scores(self):
        """The warm-up backlog must not turn into consecutive scored windows.

        _samples_since_score counts every frame, including the ~2s of frames
        that merely fill the buffer. On the frame where the window first
        fills it therefore holds a whole window of backlog. Draining that
        one hop at a time scored on 33 consecutive frames -- measured, at
        the real 320-sample frame size -- which is seconds of solid,
        event-loop-blocking inference on a slow Pi, at startup and again
        after every wake, since reset() empties the buffer.

        Taking the backlog modulo the hop keeps the remainder (so the
        rounding test above still holds) while discarding the backlog.
        """
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
        )
        frame = silence(REAL_FRAME_SAMPLES).tobytes()

        fill_frames = WINDOW_SAMPLES // REAL_FRAME_SAMPLES
        for _ in range(fill_frames):
            detector.process(frame)

        # The fill frame itself scores once; the next frames must not,
        # until a fresh hop has actually elapsed.
        assert len(model.calls) == 1
        hop_frames = HOP_SAMPLES // REAL_FRAME_SAMPLES
        for _ in range(hop_frames - 1):
            detector.process(frame)
        assert len(model.calls) == 1, "backlog drained as a burst of scores"

        detector.process(frame)
        assert len(model.calls) == 2, "scoring did not resume after one hop"

    def test_the_first_score_covers_a_full_window(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=6400
        )
        frame = silence(3200).tobytes()
        for _ in range(10):
            detector.process(frame)

        assert len(model.calls) == 1
        assert len(model.mel_calls[0]) == WINDOW_SAMPLES  # the whole buffer
        # The classifier contract: 16 embeddings of 96 dims, float32.
        assert model.calls[0].shape == (1, MIN_EMBEDDINGS, EMBEDDING_DIMS)
        assert model.calls[0].dtype == np.float32

    def test_only_the_new_audio_is_melled_after_the_first_score(self):
        """The point of the whole class. Once the window is full, a hop must
        cost one mel over the new audio and one embedding -- not a re-mel of
        the 2s window and 16 embeddings, which is the 614ms-per-call Pi
        stall this replaced."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
        )
        for frame in frames_of(silence(WINDOW_SAMPLES)):
            detector.process(frame)
        assert len(model.embedding_calls) == MIN_EMBEDDINGS  # warm-up only
        model.embedding_calls.clear()

        for frame in frames_of(silence(HOP_SAMPLES)):
            detector.process(frame)

        assert len(model.calls) == 2, "the hop did not score"
        assert len(model.embedding_calls) == 1, "recomputed cached embeddings"
        # The hop's own audio plus the carried remainder of the mel frame
        # that was still incomplete: at least MEL_WINDOW - MEL_HOP samples,
        # never a whole frame more. Nowhere near the 2s predict() re-mels.
        melled = len(model.mel_calls[-1])
        assert HOP_SAMPLES + MEL_WINDOW_SAMPLES - MEL_HOP_SAMPLES <= melled
        assert melled < HOP_SAMPLES + MEL_WINDOW_SAMPLES, "re-melled old audio"

    def test_each_mel_call_carries_the_previous_left_context(self):
        """A mel frame needs MEL_WINDOW_SAMPLES of audio, so the samples of
        the frame that is still incomplete have to survive into the next
        call. Drop them and every chunk boundary produces subtly wrong mel
        frames -- a smoke test still passes and detection quietly degrades,
        which is why this is asserted sample for sample rather than by
        eyeballing a score.
        """
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
        )
        ramp = np.linspace(-20000, 20000, WINDOW_SAMPLES + 4 * HOP_SAMPLES)
        for frame in frames_of(ramp.astype(np.int16)):
            detector.process(frame)

        assert len(model.mel_calls) >= 3
        for previous, current in zip(
            model.mel_calls, model.mel_calls[1:], strict=False
        ):
            n_frames = (len(previous) - MEL_WINDOW_SAMPLES) // MEL_HOP_SAMPLES + 1
            carried = previous[n_frames * MEL_HOP_SAMPLES :]
            assert len(carried) >= MEL_WINDOW_SAMPLES - MEL_HOP_SAMPLES
            assert np.array_equal(current[: len(carried)], carried), (
                "the next mel call did not start where the previous one left off"
            )

    def test_embeddings_are_scored_oldest_first(self):
        """Stacking the cache newest-first would still produce a plausible
        score from a plausible-looking (1, 16, 96) tensor, so the order is
        pinned directly: with a rising ramp the fake embedding values rise
        with time, and the classifier must see them in that order."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
        )
        ramp = np.linspace(0, 20000, WINDOW_SAMPLES + 4 * HOP_SAMPLES)
        for frame in frames_of(ramp.astype(np.int16)):
            detector.process(frame)

        sequence = model.calls[-1][0, :, 0]
        assert np.all(np.diff(sequence) > 0), sequence

    def test_a_score_shorter_than_a_mel_frame_adds_no_mel_work(self):
        """A hop smaller than a mel frame is legal (hop_samples >= 1) and
        must still score -- off the cached embeddings, without melling a
        stub of audio too short to produce a frame."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=1
        )
        detector.process(silence(WINDOW_SAMPLES).tobytes())
        assert len(model.mel_calls) == 1

        assert detector.process(silence(2).tobytes()) == 0.0  # the fake scores 0.0
        assert len(model.calls) == 2, "the hop did not score"
        assert len(model.mel_calls) == 1, "melled less than one frame of audio"


class TestReset:
    def test_reset_clears_buffer_but_keeps_model(self):
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=6400
        )
        frame = silence(3200).tobytes()
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

    def test_reset_clears_the_mel_and_embedding_caches(self):
        """Audio alone is not the only state any more. A surviving mel frame
        or embedding would let a re-armed gate score on audio from before
        the wake it just handled."""
        model = FakeModel()
        detector = LiveKitWakeWordDetector(
            MODEL, model=model, window_samples=WINDOW_SAMPLES, hop_samples=HOP_SAMPLES
        )
        for frame in frames_of(silence(WINDOW_SAMPLES + HOP_SAMPLES)):
            detector.process(frame)
        assert len(detector._embeddings) == MIN_EMBEDDINGS

        detector.reset()
        assert len(detector._pending) == 0
        assert len(detector._mel) == 0
        assert len(detector._embeddings) == 0

    def test_a_reset_detector_scores_like_a_fresh_one(self):
        """The behavioural half of the test above, against the real model:
        after a reset, a window of silence must score exactly what a
        never-used detector scores on it. Any embedding that survived the
        reset would carry the earlier noise into the score.
        """
        used = LiveKitWakeWordDetector(MODEL)
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 4000, WINDOW_SAMPLES).astype(np.int16)
        for frame in frames_of(noise):
            used.process(frame)
        used.reset()

        quiet = frames_of(silence(WINDOW_SAMPLES))
        after_reset = [used.process(f) for f in quiet][-1]
        fresh = LiveKitWakeWordDetector(MODEL)
        expected = [fresh.process(f) for f in quiet][-1]
        assert after_reset == expected


class TestAgainstTheRealModel:
    """Exercises the actual ONNX wiring, not just the stub."""

    def test_real_frame_size_does_not_raise(self):
        """The regression test for the stateless-predict() bug.

        Before the fix, process() handed each 320-sample pipeline frame
        straight to the model, which raises an onnxruntime InvalidArgument
        shape error for any input that small -- every single frame threw.
        Feeding a full window's worth of the real transport frame size must
        not raise, on any frame, and must return a score.
        """
        detector = LiveKitWakeWordDetector(MODEL)
        for frame in frames_of(silence(WINDOW_SAMPLES)):
            score = detector.process(frame)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_full_window_of_silence_is_nonzero(self):
        """Nothing scores at all until 16 embeddings exist, so a zero score
        alone would not prove the classifier ran. Once the buffer holds a
        full window, the real model returns a small but genuinely non-zero
        score for silence (~0.0047 measured against the committed model) --
        proof the mel/embedding/classifier path actually executed. The exact
        value isn't asserted, only that it is present and nowhere near
        triggering.
        """
        detector = LiveKitWakeWordDetector(MODEL)
        scores = [detector.process(f) for f in frames_of(silence(WINDOW_SAMPLES))]
        assert 0.0 < scores[-1] < 0.5, scores

    def test_silence_does_not_trigger(self):
        detector = LiveKitWakeWordDetector(MODEL)
        scores = [detector.process(f) for f in frames_of(silence(WINDOW_SAMPLES))]
        assert max(scores) < 0.5, scores

    def test_white_noise_does_not_trigger(self):
        detector = LiveKitWakeWordDetector(MODEL)
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 3000, WINDOW_SAMPLES).astype(np.int16)
        scores = [detector.process(f) for f in frames_of(noise)]
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
        assert "hey_livekit" in detector._model._classifiers

    def test_the_empirical_floor_is_exactly_sufficient(self):
        """MIN_WINDOW_SAMPLES must be the true boundary, not an estimate.

        TestWindowGuard pins that one sample below it is rejected at
        construction. This pins the other half: the floor itself is
        already enough for the real classifier to run (not just for the
        >= comparison to pass). hop_samples == window_samples so a single
        full-window call scores immediately.
        """
        detector = LiveKitWakeWordDetector(
            MODEL,
            window_samples=MIN_WINDOW_SAMPLES,
            hop_samples=MIN_WINDOW_SAMPLES,
        )
        score = detector.process(silence(MIN_WINDOW_SAMPLES).tobytes())
        assert score > 0.0, "the floor is not actually sufficient to score"


# Three extra hops past the warm-up window, so the compared score comes from
# embeddings built incrementally rather than from the one-off warm-up pass.
# A multiple of HOP_SAMPLES keeps the detector's mel/embedding grid aligned
# with the grid predict() lays over the trailing window, so the two really
# do look at the same 16 embedding positions.
AGREEMENT_HOPS = 3
AGREEMENT_SAMPLES = WINDOW_SAMPLES + AGREEMENT_HOPS * HOP_SAMPLES

# Realistic microphone noise floor, ~-47dBFS against these test tones. Real
# rooms and real ADCs always have one; a mathematically pure tone does not,
# and the frontend's top_db clamp (see TestMatchesTheLibrary) makes that
# difference matter.
NOISE_FLOOR = 40


def agreement_signals(n):
    """Varied audio for the agreement test.

    Silence alone would be worthless: it scores ~0.0047 no matter how the
    mel frames are assembled, so a badly broken path would agree with the
    library by accident. These have real dynamic range in time and
    frequency instead.
    """
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(1234)
    floor = rng.normal(0, NOISE_FLOOR, n)
    harmonics = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in (1, 2, 3, 4, 5))
    syllables = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    return {
        "silence": np.zeros(n, dtype=np.int16),
        "white noise": rng.normal(0, 4000, n).astype(np.int16),
        "sweep": (9000 * np.sin(2 * np.pi * (200 + 1200 * t) * t) + floor).astype(
            np.int16
        ),
        "voiced": (6000 * harmonics * syllables + floor).astype(np.int16),
    }


class TestMatchesTheLibrary:
    """The safety net for driving WakeWordModel's private attributes.

    Scoring incrementally means reimplementing what predict() does -- mel
    geometry, the 76-frame/stride-8 embedding grid, the newest-16 cache --
    against attributes the library does not promise to keep. That is only
    acceptable while a test makes a divergence loud, so this one runs both
    paths over identical audio and compares the scores.
    """

    def test_the_library_still_exposes_the_private_api(self):
        """A rename shows up here, on a real loaded model, rather than on
        the Pi at 3am."""
        detector = LiveKitWakeWordDetector(MODEL)
        detector.load()
        model = detector._model
        for attr in REQUIRED_MODEL_ATTRS:
            assert hasattr(model, attr), (
                f"livekit's WakeWordModel no longer has {attr}; the "
                "incremental path drives it directly"
            )
        session, input_name = next(iter(model._classifiers.values()))
        assert callable(session.run)
        assert isinstance(input_name, str)

    def test_incremental_scores_match_predict_on_varied_audio(self):
        """Feed identical audio to both paths and compare.

        The tolerance is measured, not chosen for comfort. Three of these
        four signals come out bit-identical and the fourth differs by one
        float32 ulp (3e-8); the only thing that can separate the paths is
        reassociation inside the ONNX mel, which the incremental one feeds
        shorter inputs. 1e-6 is ~30x that floor, which is room for a
        different CPU's kernels in CI, and ~60x below the smallest
        divergence a wrong-but-plausible implementation produced here (the
        mutant that dropped the mel left context, 5.8e-5).

        That separation is narrower than it looks: the classifier squashes
        everything that is not the wake word down to ~0.004, so even gross
        corruption only moves these scores a little. If this ever fails,
        the answer is to find out which stage moved -- not to widen the
        number until it passes.
        """
        reference = LiveKitWakeWordDetector(MODEL)
        reference.load()
        scores = {}
        for name, samples in agreement_signals(AGREEMENT_SAMPLES).items():
            detector = LiveKitWakeWordDetector(MODEL, model=reference._model)
            score = 0.0
            for frame in frames_of(samples):
                score = detector.process(frame)
            window = samples[AGREEMENT_SAMPLES - WINDOW_SAMPLES :]
            expected = max(reference._model.predict(window).values())
            assert score == pytest.approx(expected, abs=1e-6), (
                f"{name}: incremental {score!r} vs predict() {expected!r}"
            )
            scores[name] = score

        # ...and the comparison has to have had something to compare. If
        # every signal scored the same, agreement would prove nothing.
        assert len(set(scores.values())) == len(scores), scores
        assert max(scores.values()) - min(scores.values()) > 1e-3, scores

    def test_the_mel_frontend_clamps_against_its_own_peak(self):
        """Why the signals above carry a noise floor, pinned so the next
        person does not rediscover it the hard way.

        The frontend's last step is a power_to_db with top_db=80: every bin
        below (peak - 80dB) is clamped to that floor, and the peak is taken
        over whatever was passed in. It is therefore *not* a pure function
        of each frame, and audio with more than 80dB of range inside one
        window -- a synthetic tone with no noise floor, say -- mels
        differently in 80ms chunks than in one 2s call. Real microphone
        audio has a noise floor and stays inside the clamp, which is why
        the agreement above is exact to float noise.
        """
        detector = LiveKitWakeWordDetector(MODEL)
        detector.load()
        frontend = detector._model._mel_frontend
        t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
        tone = (9000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)

        mel = np.asarray(frontend(tone.astype(np.float32) / INT16_SCALE))
        # 80dB, in the frontend's post-processed units (x / 10 + 2).
        assert np.ptp(mel) == pytest.approx(8.0, abs=1e-5)
        assert (mel == mel.min()).mean() > 0.1, "expected a clamped floor"
