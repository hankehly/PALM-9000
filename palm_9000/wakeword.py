"""Local wake-word detection.

Audio in, score out. Deliberately free of pipecat imports so it can be tested
as a pure function, and behind a Protocol so the engine can be swapped without
touching the gate.
"""

from pathlib import Path
from typing import Protocol

import numpy as np


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

    Pass `model` to inject a stub in tests. Call `load()` at startup so a
    missing or unreadable model fails there rather than on the first frame.
    """

    def __init__(self, model_path: str | Path, model: object | None = None) -> None:
        self.model_path = Path(model_path)
        self._model = model

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
        scores = self._model.predict(frame)
        if not scores:
            return 0.0
        return float(max(scores.values()))

    def reset(self) -> None:
        # livekit-wakeword keeps its own rolling buffers; dropping the model
        # is the only reliable way to clear them. Reloading costs far less
        # than the conversation that follows a detection.
        self._model = None
