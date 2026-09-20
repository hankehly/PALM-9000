"""Local wake-word detection.

Audio in, score out. Deliberately free of pipecat imports so it can be tested
as a pure function, and behind a Protocol so the engine can be swapped without
touching the gate.
"""

from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np

# 2.0s at 16kHz. The classifier wants 16 speech embeddings, which span
# 196 mel frames; see MIN_WINDOW_SAMPLES for the exact sample count.
WINDOW_SAMPLES = 32000

# 80ms, the model's own embedding stride (8 mel frames). One new embedding
# per scored window; scoring every 20ms frame would be 50 classifier passes
# a second, which a Zero 2W cannot afford.
HOP_SAMPLES = 1280

# The wake-word stack needs EMBEDDING_WINDOW (76) mel frames for its first
# embedding, plus EMBEDDING_STRIDE (8) more per additional embedding, and the
# classifier needs MIN_EMBEDDINGS (16) of them before it will run at all:
# 76 + (16 - 1) * 8 = 196 mel frames minimum (see
# livekit.wakeword.inference.model). Below that, the library's predict()
# takes an early return and reports a flat 0.0 for every input, forever.
#
# 196 mel frames is NOT 196 * 160 samples (10ms hops at 16kHz = 31360): that
# under-counts, because the analysis window itself has to fill before the
# first hop even starts, so reaching the Nth mel frame costs more than N
# hops' worth of samples. The real minimum was found by bisecting
# WakeWordModel.predict() -- bundled mel frontend plus the committed
# hey_livekit.onnx -- directly on sample count: 31711 samples still yields
# only 195 mel frames (flat 0.0); 31712 yields 196 (the classifier runs,
# ~0.0047 on silence). Use the measured boundary, not the under-counted one.
#
# It is exactly MEL_WINDOW_SAMPLES + 195 * MEL_HOP_SAMPLES, which is what the
# mel geometry below predicts -- the bisected number and the measured frontend
# parameters agree.
MIN_WINDOW_SAMPLES = 31712

# --- mel frontend geometry --------------------------------------------------
#
# Measured against the bundled melspectrogram.onnx, not taken from a paper:
# frame counts were mapped for every input length from 600 to 4000 samples,
# giving transitions exactly every 160 samples with the first at 672, i.e.
#
#     frames(n) = (n - 512) // 160 + 1      (n >= 512; n < 512 raises)
#
# so the analysis window is 512 samples and the hop is 160. (A 640-sample
# window also reproduces frames(32000) == 197 by coincidence, but predicts
# 1 frame for n=672 where the real frontend yields 2.)
#
# There is no padding: melling any 160-aligned slice reproduces the frames of
# the longer signal it came from to within float32 noise (~1e-6 measured),
# which is what makes incremental melling exact rather than approximate.
#
# One caveat, worth knowing before chasing a phantom bug. The frontend ends in
# a power_to_db with top_db=80, and the 80dB floor is measured against the peak
# of *whatever was passed in*, so a chunk whose own peak is lower than the
# window's clamps lower too. Values that low are floor either way once the
# window-wide clamp is applied, so real audio is unaffected -- but a synthetic
# tone with no noise floor has more than 80dB of range and does mel differently
# in 80ms chunks than in one 2s call. openWakeWord, which this model family
# comes from, streams its mel exactly this way. TestMatchesTheLibrary pins both
# the agreement and the clamp.
MEL_WINDOW_SAMPLES = 512
MEL_HOP_SAMPLES = 160
MEL_BINS = 32

# Mel frames per speech embedding, and the stride between embeddings, both in
# mel frames; and how many embeddings the classifier consumes. These mirror
# EMBEDDING_WINDOW / EMBEDDING_STRIDE / MIN_EMBEDDINGS in
# livekit.wakeword.inference.model.
EMBEDDING_WINDOW = 76
EMBEDDING_STRIDE = 8
MIN_EMBEDDINGS = 16

# What livekit's predict() divides int16 samples by to get float32 audio.
# Reproduced here so the incremental path is bit-comparable with it.
INT16_SCALE = 32768.0

# The private attributes of WakeWordModel this class drives directly. They are
# private, so a library upgrade may rename them; load() checks for them and
# fails loudly rather than letting the failure surface on the Pi.
REQUIRED_MODEL_ATTRS = ("_mel_frontend", "_speech_embedding", "_classifiers")


class WakeWordDetector(Protocol):
    """Scores audio for the presence of the wake word."""

    def process(self, audio: bytes) -> float:
        """Return the highest wake-word score for this frame, 0.0-1.0."""
        ...  # pragma: no cover

    def reset(self) -> None:
        """Discard buffered audio, e.g. after a detection."""
        ...  # pragma: no cover


class LiveKitWakeWordDetector:
    """WakeWordDetector backed by livekit-wakeword, scored incrementally.

    livekit-wakeword's WakeWordModel.predict() is stateless: hand it ~2s of
    audio and it mels the lot, cuts 16 overlapping 76-frame windows out of
    the mel, runs the speech-embedding CNN on every one of them, and runs
    the classifier over the 16 resulting vectors. Calling it once per hop
    redoes all of that for audio it already saw: on a Pi Zero 2W a single
    call costs 614ms wall / 1572ms CPU -- 496% of one core at an 80ms hop,
    so the pipeline's event loop stalls and detection never keeps up.

    Each hop only advances the window by EMBEDDING_STRIDE mel frames, which
    is exactly *one* new embedding. This class therefore keeps the state
    predict() lacks and does only the new work:

      * `_pending` -- raw int16 audio that has not been melled yet. Its
        first sample is always the start of the next mel frame, so the
        350-odd samples left over after a mel call are the left-context the
        next call needs; drop them and the boundary frames come out subtly
        wrong (a smoke test still passes, detection quietly degrades).
      * `_mel` -- mel frames not yet consumed by an embedding, trimmed to
        EMBEDDING_WINDOW - EMBEDDING_STRIDE frames of overlap.
      * `_embeddings` -- the last MIN_EMBEDDINGS speech embeddings.

    Per hop that is one mel call over the new audio, one embedding, and one
    classifier pass: 22.8ms of the 614ms, measured stage by stage on the Pi.

    It drives WakeWordModel's private attributes to do that, which is a real
    coupling to livekit-wakeword 0.2.x. load() fails loudly if they are gone,
    and TestMatchesTheLibrary asserts the incremental score agrees with
    predict() on the same audio, so a divergence is noisy rather than silent.

    `window_samples` is how much audio must accumulate before scoring
    starts, unchanged. What gets scored is the trailing MIN_WINDOW_SAMPLES
    (the 196 mel frames the 16 embeddings span), ending at the newest
    complete mel frame -- so a larger window only delays the first score, it
    does not widen the context the classifier sees. predict() behaves the
    same way for the same reason: it keeps the *last* 16 embeddings and
    ignores whatever else the window held.

    Pass `model` to inject a stub in tests. Call `load()` at startup so a
    missing or unreadable model fails there rather than on the first frame.
    """

    def __init__(
        self,
        model_path: str | Path,
        model: object | None = None,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
    ) -> None:
        if window_samples < MIN_WINDOW_SAMPLES:
            raise ValueError(
                f"window_samples={window_samples} is below the model's real "
                f"minimum of {MIN_WINDOW_SAMPLES} (~2s of 16kHz audio, the "
                "span of the 16 embeddings the classifier wants). Anything "
                "smaller never scores at all -- permanent deafness, not an "
                "error."
            )
        if hop_samples < 1:
            raise ValueError(
                f"hop_samples={hop_samples} must be >= 1. 0 or negative "
                "scores on every process() call instead of throttling, "
                "defeating the point of the hop."
            )
        self.model_path = Path(model_path)
        self._model = model
        self._checked = False
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._samples_seen = 0
        self._samples_since_score = 0
        self._pending = np.zeros(0, dtype=np.int16)
        self._mel = np.zeros((0, MEL_BINS), dtype=np.float32)
        self._embeddings: deque[np.ndarray] = deque(maxlen=MIN_EMBEDDINGS)

    def load(self) -> None:
        """Open the model now.

        Raises if it is missing, unreadable, or no longer exposes the
        attributes the incremental path drives.
        """
        if self._checked:
            return
        if self._model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Wake-word model not found: {self.model_path}. "
                    "Wake-word gating is enabled, so this is fatal: running "
                    "without it would stream audio continuously."
                )
            from livekit.wakeword import WakeWordModel

            self._model = WakeWordModel(models=[str(self.model_path)])

        missing = [a for a in REQUIRED_MODEL_ATTRS if not hasattr(self._model, a)]
        if missing:
            raise RuntimeError(
                f"{type(self._model).__name__} no longer exposes "
                f"{', '.join(missing)}. Wake-word gating scores incrementally "
                "by driving those attributes directly, because livekit's own "
                "predict() re-mels and re-embeds the whole 2s window every "
                "hop -- 27x the work, far more than a Pi Zero 2W can do in "
                "realtime. It cannot run against this version of "
                "livekit-wakeword. Re-check the attribute names in "
                "livekit.wakeword.inference.model and re-run the agreement "
                "test before changing this."
            )
        self._checked = True

    def process(self, audio: bytes) -> float:
        if not audio:
            return 0.0
        self.load()
        frame = np.frombuffer(audio, dtype=np.int16)
        self._pending = np.concatenate([self._pending, frame])
        self._samples_seen = min(self._samples_seen + len(frame), self.window_samples)
        self._samples_since_score += len(frame)

        window_full = self._samples_seen >= self.window_samples
        hop_elapsed = self._samples_since_score >= self.hop_samples
        if not (window_full and hop_elapsed):
            return 0.0

        # Modulo, not zero and not repeated subtraction.
        #
        # Zeroing would discard whatever backlog was already past
        # hop_samples, rounding the effective hop up to the next frame-size
        # multiple whenever the frame size doesn't evenly divide
        # hop_samples (e.g. a hop tuned on the Pi).
        #
        # Subtracting a single hop keeps that remainder, but the counter
        # also accumulates through the whole window fill -- so on the frame
        # where the buffer first fills it holds ~a full window of backlog
        # and drains it one hop per frame, scoring on 33 consecutive frames.
        # That is seconds of solid, event-loop-blocking inference on a slow
        # Pi, at startup and again after every single wake, since reset()
        # empties the buffer.
        #
        # Modulo keeps the remainder AND discards the backlog in one step.
        self._samples_since_score %= self.hop_samples
        self._advance()
        return self._classify()

    def _advance(self) -> None:
        """Turn the newly arrived audio into mel frames and embeddings.

        Nothing here re-touches audio that has already been melled or mel
        frames that have already been embedded: this is the whole point of
        the class.
        """
        if len(self._pending) >= MEL_WINDOW_SAMPLES:
            # The frontend wants float32; scale exactly as predict() does so
            # the two paths stay comparable sample for sample.
            audio = self._pending.astype(np.float32) / INT16_SCALE
            # (1, frames, 32) for 1-D input; reshape rather than branch on
            # ndim, since the bin count is fixed by the ONNX graph.
            frames = np.asarray(self._model._mel_frontend(audio)).reshape(-1, MEL_BINS)
            # Keep the samples of the first frame that is *not* complete yet:
            # its start is where the next mel call has to begin.
            self._pending = self._pending[len(frames) * MEL_HOP_SAMPLES :]
            self._mel = np.concatenate([self._mel, frames])

        # _mel always begins at the start of the next embedding window, so
        # an embedding is due as soon as EMBEDDING_WINDOW frames are there.
        while len(self._mel) >= EMBEDDING_WINDOW:
            window = self._mel[:EMBEDDING_WINDOW]
            embedding = self._model._speech_embedding(window[np.newaxis, :, :])
            self._embeddings.append(embedding[0])
            self._mel = self._mel[EMBEDDING_STRIDE:]

    def _classify(self) -> float:
        """Run the classifiers over the cached embeddings, newest last."""
        emb_input = np.stack(self._embeddings, axis=0)[np.newaxis, :, :].astype(
            np.float32
        )
        scores = []
        for session, input_name in self._model._classifiers.values():
            outputs = session.run(None, {input_name: emb_input})
            scores.append(float(outputs[0][0, 0]))
        if not scores:
            return 0.0
        return max(scores)

    def reset(self) -> None:
        """Discard buffered audio, e.g. after a detection.

        Clears the embedding cache too, not just the audio: a surviving
        embedding would let a re-armed gate score on audio from before the
        wake it just handled.

        Keeps the model -- rebuilding the ONNX sessions costs seconds on a
        Pi, and they hold no per-utterance state.
        """
        self._samples_seen = 0
        self._samples_since_score = 0
        self._pending = np.zeros(0, dtype=np.int16)
        self._mel = np.zeros((0, MEL_BINS), dtype=np.float32)
        self._embeddings.clear()
