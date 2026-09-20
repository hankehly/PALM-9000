"""Local wake-word detection.

Audio in, score out. Deliberately free of pipecat imports so it can be tested
as a pure function, and behind a Protocol so the engine can be swapped without
touching the gate.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

# 2.0s at 16kHz. predict() is stateless: it needs 76 + 15*8 = 196 mel
# frames in a single call to produce the 16 embeddings its classifier
# wants, and returns a flat 0.0 for anything shorter.
WINDOW_SAMPLES = 32000

# 80ms, the model's own embedding stride (8 mel frames). One new embedding
# per scored window; scoring every 20ms frame would be 50 full-window ONNX
# passes a second, which a Zero 2W cannot afford.
HOP_SAMPLES = 1280


class WakeWordDetector(Protocol):
    """Scores audio for the presence of the wake word."""

    def process(self, audio: bytes) -> float:
        """Return the highest wake-word score for this frame, 0.0-1.0."""
        ...  # pragma: no cover

    def reset(self) -> None:
        """Discard buffered audio, e.g. after a detection."""
        ...  # pragma: no cover


class LiveKitWakeWordDetector:
    """WakeWordDetector backed by livekit-wakeword.

    livekit-wakeword's WakeWordModel.predict() is stateless: it scores
    whatever chunk it is handed and remembers nothing between calls. This
    class supplies the state predict() lacks: it keeps a rolling int16
    buffer of the last `window_samples`, and only calls predict() -- on the
    full buffer -- once that buffer is full and at least `hop_samples` of
    new audio have arrived since the last call. That is what lets pipeline
    frames far smaller than a window (e.g. 20ms/320 samples) accumulate
    into something predict() can actually score.

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
        self.model_path = Path(model_path)
        self._model = model
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._buffer = np.zeros(0, dtype=np.int16)
        self._samples_since_score = 0

    def load(self) -> None:
        """Open the model now. Raises if it is missing or unreadable."""
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Wake-word model not found: {self.model_path}. "
                "Wake-word gating is enabled, so this is fatal: running "
                "without it would stream audio continuously."
            )
        from livekit.wakeword import WakeWordModel

        self._model = WakeWordModel(models=[str(self.model_path)])

    def process(self, audio: bytes) -> float:
        if not audio:
            return 0.0
        self.load()
        frame = np.frombuffer(audio, dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, frame])[-self.window_samples :]
        self._samples_since_score += len(frame)

        window_full = len(self._buffer) >= self.window_samples
        hop_elapsed = self._samples_since_score >= self.hop_samples
        if not (window_full and hop_elapsed):
            return 0.0

        self._samples_since_score = 0
        scores = self._model.predict(self._buffer)
        if not scores:
            return 0.0
        return float(max(scores.values()))

    def reset(self) -> None:
        """Discard buffered audio, e.g. after a detection.

        Keeps the model: predict() is stateless, so there is nothing else
        to clear, and rebuilding the ONNX session costs seconds on a Pi.
        """
        self._buffer = np.zeros(0, dtype=np.int16)
        self._samples_since_score = 0
