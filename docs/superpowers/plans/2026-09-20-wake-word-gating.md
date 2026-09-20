# Wake-word gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop PALM-9000 uploading microphone audio to Gemini until a wake word is detected locally, and re-arm after the conversation goes quiet.

**Architecture:** `GeminiLiveLLMService` is built with `start_audio_paused=True` so it boots muted. A `WakeWordGate` processor sits between the transport and the context aggregator, scoring input audio with a local ONNX model and calling `set_audio_input_paused(False)` on detection. A silence timer, driven by turn frames from a locally-run Silero VAD, re-pauses it. The whole feature is off by default.

**Tech Stack:** Python 3.12, pipecat-ai 1.11, livekit-wakeword 0.2.1 (numpy + onnxruntime only), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-20-wake-word-gating-design.md`

## Global Constraints

- Python `>=3.12`; the Pi runs 3.12.11 on aarch64.
- Dependency additions must resolve to **exactly one** new production package (`livekit-wakeword>=0.2.1`). Verify with `uv pip install --dry-run` against a `--no-dev` environment.
- `pipecat-ai` is pinned `>=1.11.0,<2`. Do not use `model=`, `voice_id=` or `params=` on `GeminiLiveLLMService`; they are removed in 2.0.
- Coverage gate is **100%** including branches (`[tool.coverage.report] fail_under`). CI fails below it.
- Lint and format with ruff, line length 88, rules `E`/`F`/`I`/`UP`/`B`.
- Branch names follow Conventional Branch; commits follow Conventional Commits (`<type>[scope]: <description>`).
- Run tests as CI does: `uv run --no-sync pytest`, never `python -m pytest`.
- `wake_word_enabled` defaults to `False`. Merging this must not change runtime behaviour.
- **Never fall back to ungated operation on error.** Failing closed is required; a fallback would silently restore continuous upload, which is the behaviour this feature exists to remove.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `palm_9000/wakeword.py` (new) | `WakeWordDetector` protocol + `LiveKitWakeWordDetector`. Audio bytes in, score out. No pipecat import. |
| `palm_9000/processors.py` (modify) | Add `WakeWordGate`. Owns the asleep/awake state machine. |
| `palm_9000/settings.py` (modify) | Four new settings. |
| `main.py` (modify) | `build_wake_gate()`, `build_context_aggregator()`, wiring. |
| `models/wakeword/hey_livekit.onnx` (new) | The committed classifier, 953 KB. |
| `.gitignore` (modify) | Un-ignore the model directory. |
| `pyproject.toml` (modify) | Add `livekit-wakeword>=0.2.1`. |
| `tests/test_wakeword.py` (new) | Detector tests, stub and real model. |
| `tests/test_processors.py` (modify) | `WakeWordGate` tests. |
| `tests/test_main.py` (modify) | Wiring tests. |
| `tests/test_packaging.py` (modify) | Dependency and asset tests. |

---

### Task 1: Add the dependency and commit the model

**Files:**
- Modify: `pyproject.toml`, `.gitignore`
- Create: `models/wakeword/hey_livekit.onnx`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `livekit.wakeword.WakeWordModel` importable under `--no-dev`; model readable at `models/wakeword/hey_livekit.onnx`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_packaging.py`:

```python
class TestWakeWordAssets:
    def test_livekit_wakeword_is_a_main_dependency(self):
        main_deps = pyproject()["project"]["dependencies"]
        assert any("livekit-wakeword" in dep for dep in main_deps), (
            "livekit-wakeword must be a production dependency; the gate runs "
            "on the Pi under --no-dev."
        )

    def test_model_file_is_committed_and_readable(self):
        model = PROJECT_ROOT / "models" / "wakeword" / "hey_livekit.onnx"
        assert model.exists(), f"{model} is missing"
        assert model.stat().st_size > 500_000, "model looks truncated"

    def test_model_is_tracked_by_git(self):
        """A .gitignore rule silently excluding it would break deploys."""
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch",
             "models/wakeword/hey_livekit.onnx"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, (
            "model is not tracked by git; check the .gitignore negations"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_packaging.py::TestWakeWordAssets -v`
Expected: FAIL — dependency and model both missing.

- [ ] **Step 3: Add the dependency**

In `pyproject.toml`, add to `[project.dependencies]` after the pipecat line:

```toml
    "livekit-wakeword>=0.2.1",
```

Then `uv lock`.

- [ ] **Step 4: Un-ignore the model directory**

In `.gitignore`, replace the `models/*` block with:

```gitignore
models/*
!models/.gitkeep
!models/wakeword/
!models/wakeword/**
```

The directory negation must precede the contents negation: git will not
re-include a file whose parent directory is excluded.

- [ ] **Step 5: Download and commit the model**

```bash
mkdir -p models/wakeword
curl -L -o models/wakeword/hey_livekit.onnx \
  https://raw.githubusercontent.com/livekit-examples/hello-wakeword/main/client/models/hey_livekit.onnx
ls -l models/wakeword/hey_livekit.onnx    # expect ~953,357 bytes
git add --dry-run models/wakeword/hey_livekit.onnx   # must report "add '...'"
```

Use `git add --dry-run`, **not** `git check-ignore -v`: the latter prints the
matching rule even when it is a negation and still exits 0, so it reads as
"ignored" when the file is includable.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_packaging.py::TestWakeWordAssets -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Verify the one-package claim**

```bash
UV_PROJECT_ENVIRONMENT=/tmp/prodcheck uv sync --no-dev
uv pip install --python /tmp/prodcheck/bin/python --dry-run "livekit-wakeword>=0.2.1"
```

Expected: `Would install 1 package` or `Audited`. Anything more means a
transitive dependency crept in; stop and investigate.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore models/wakeword/hey_livekit.onnx tests/test_packaging.py
git commit -m "feat(wakeword): add livekit-wakeword and the hey_livekit model"
```

---

### Task 2: The detector

**Files:**
- Create: `palm_9000/wakeword.py`
- Test: `tests/test_wakeword.py`

**Interfaces:**
- Consumes: `livekit.wakeword.WakeWordModel`; the model from Task 1.
- Produces:
  - `WakeWordDetector` protocol: `process(audio: bytes) -> float`, `reset() -> None`
  - `LiveKitWakeWordDetector(model_path: str | Path, model: object | None = None)` with `load() -> None`

`load()` exists so the gate can fail at startup on a missing model rather
than on the first audio frame, which the spec requires.

- [ ] **Step 1: Write the failing test**

Create `tests/test_wakeword.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from palm_9000.wakeword import LiveKitWakeWordDetector

MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "wakeword"
    / "hey_livekit.onnx"
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wakeword.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'palm_9000.wakeword'`

- [ ] **Step 3: Write the implementation**

Create `palm_9000/wakeword.py`:

```python
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
        ...

    def reset(self) -> None:
        """Discard buffered audio, e.g. after a detection."""
        ...


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_wakeword.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Add the real-model tests**

Append to `tests/test_wakeword.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_wakeword.py -v`
Expected: PASS (10 tests). The real-model tests take a second or two to load ONNX.

- [ ] **Step 7: Commit**

```bash
git add palm_9000/wakeword.py tests/test_wakeword.py
git commit -m "feat(wakeword): add local wake-word detector"
```

---

### Task 3: The gate processor

**Files:**
- Modify: `palm_9000/processors.py`
- Test: `tests/test_processors.py`

**Interfaces:**
- Consumes: `WakeWordDetector` (Task 2); `GeminiLiveLLMService.set_audio_input_paused(paused: bool)`, which is synchronous.
- Produces: `WakeWordGate(detector, llm, threshold=0.5, silence_timeout_secs=30.0, now=time.monotonic)`

- [ ] **Step 1: Extend the frame imports in the test file**

At the top of `tests/test_processors.py`, replace the pipecat frame import with:

```python
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InputAudioRawFrame,
    TextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_processors.py`:

```python
from palm_9000.processors import WakeWordGate


class FakeDetector:
    def __init__(self, scores=None):
        self.scores = list(scores or [])
        self.resets = 0
        self.seen = 0

    def process(self, audio):
        self.seen += 1
        return self.scores.pop(0) if self.scores else 0.0

    def reset(self):
        self.resets += 1


class FakeLLM:
    def __init__(self):
        self.paused_calls = []

    def set_audio_input_paused(self, paused):
        self.paused_calls.append(paused)


def audio_frame(n=8):
    return InputAudioRawFrame(
        audio=b"\x00\x00" * n, sample_rate=16000, num_channels=1
    )


class Clock:
    """Controllable monotonic clock, so tests never sleep."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def clock():
    return Clock()


def make_gate(detector, llm, clock, **kw):
    gate = WakeWordGate(
        detector=detector, llm=llm, silence_timeout_secs=30.0, now=clock, **kw
    )
    gate.pushed = []
    gate.push_frame = AsyncMock(side_effect=lambda f, d: gate.pushed.append((f, d)))
    return gate


class TestWakeWordGate:
    async def test_stays_asleep_below_threshold(self, clock):
        detector, llm = FakeDetector([0.1, 0.2]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == []

    async def test_score_above_threshold_unpauses(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False]
        assert detector.resets == 1

    async def test_does_not_detect_while_awake(self, clock):
        detector, llm = FakeDetector([0.9, 0.9, 0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)
        seen_after_wake = detector.seen
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert detector.seen == seen_after_wake, "detector ran while awake"
        assert llm.paused_calls == [False]

    async def test_silence_timeout_re_arms(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(31.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False, True]

    @pytest.mark.parametrize(
        "frame",
        [
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            BotStoppedSpeakingFrame(),
        ],
        ids=["user_start", "user_stop", "bot_start", "bot_stop"],
    )
    async def test_activity_frames_reset_the_deadline(self, clock, frame):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        clock.advance(20.0)
        await gate.process_frame(frame, FrameDirection.DOWNSTREAM)
        clock.advance(20.0)
        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [False], "deadline was not reset by activity"

    async def test_every_frame_is_forwarded(self, clock):
        detector, llm = FakeDetector([0.9]), FakeLLM()
        gate = make_gate(detector, llm, clock)
        frames = [audio_frame(), TextFrame(text="hi"), BotStoppedSpeakingFrame()]

        for f in frames:
            await gate.process_frame(f, FrameDirection.DOWNSTREAM)

        assert [f for f, _ in gate.pushed] == frames

    async def test_detector_failure_keeps_it_asleep(self, clock):
        class Exploding(FakeDetector):
            def process(self, audio):
                raise RuntimeError("onnx blew up")

        llm = FakeLLM()
        gate = make_gate(Exploding(), llm, clock)

        await gate.process_frame(audio_frame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == [], "must fail closed"
        assert len(gate.pushed) == 1, "frame must still be forwarded"

    async def test_activity_frames_are_ignored_while_asleep(self, clock):
        """A bot frame must not extend a deadline that is not running."""
        detector, llm = FakeDetector([0.0]), FakeLLM()
        gate = make_gate(detector, llm, clock)

        await gate.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        assert llm.paused_calls == []
        assert len(gate.pushed) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_processors.py -k WakeWordGate -v`
Expected: FAIL with `ImportError: cannot import name 'WakeWordGate'`

- [ ] **Step 4: Extend the imports in `palm_9000/processors.py`**

```python
import time

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
```

- [ ] **Step 5: Write the implementation**

Append to `palm_9000/processors.py`:

```python
class WakeWordGate(FrameProcessor):
    """Keeps user audio off the wire until a wake word is heard.

    The service is constructed with start_audio_paused=True, so the gate only
    ever has to open it. Every frame is forwarded unchanged; pausing and
    unpausing the service is the sole side effect.
    """

    _ACTIVITY_FRAMES = (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
    )

    def __init__(
        self,
        detector,
        llm,
        threshold: float = 0.5,
        silence_timeout_secs: float = 30.0,
        now=time.monotonic,
    ) -> None:
        super().__init__()
        self._detector = detector
        self._llm = llm
        self._threshold = threshold
        self._silence_timeout_secs = silence_timeout_secs
        self._now = now
        self._awake = False
        self._deadline = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._awake and isinstance(frame, self._ACTIVITY_FRAMES):
            self._deadline = self._now() + self._silence_timeout_secs
        elif isinstance(frame, InputAudioRawFrame):
            if self._awake:
                if self._now() >= self._deadline:
                    self._sleep()
            else:
                self._maybe_wake(frame.audio)

        await self.push_frame(frame, direction)

    def _maybe_wake(self, audio: bytes) -> None:
        try:
            score = self._detector.process(audio)
        except Exception:
            # Fail closed: a broken detector must never open the microphone.
            logger.exception("Wake-word detection failed; staying asleep")
            return
        if score > self._threshold:
            logger.info(f"Wake word detected (score {score:.2f})")
            self._detector.reset()
            self._awake = True
            self._deadline = self._now() + self._silence_timeout_secs
            self._llm.set_audio_input_paused(False)

    def _sleep(self) -> None:
        logger.info(
            f"No speech for {self._silence_timeout_secs}s - going back to sleep"
        )
        self._awake = False
        self._llm.set_audio_input_paused(True)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_processors.py -v`
Expected: PASS — 11 new gate tests plus the existing ones.

- [ ] **Step 7: Commit**

```bash
git add palm_9000/processors.py tests/test_processors.py
git commit -m "feat(wakeword): add WakeWordGate processor"
```

---

### Task 4: Settings

**Files:**
- Modify: `palm_9000/settings.py`, `tests/conftest.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings.wake_word_enabled: bool`, `.wake_word_model_path: str`, `.wake_word_threshold: float`, `.wake_silence_timeout_secs: float`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_settings.py`:

```python
class TestWakeWordSettings:
    def test_disabled_by_default(self, env):
        """Merging the feature must not change runtime behaviour."""
        settings = Settings(_env_file=None, google_api_key="k")
        assert settings.wake_word_enabled is False

    def test_defaults(self, env):
        settings = Settings(_env_file=None, google_api_key="k")
        assert settings.wake_word_model_path == "models/wakeword/hey_livekit.onnx"
        assert settings.wake_word_threshold == 0.5
        assert settings.wake_silence_timeout_secs == 30.0

    def test_env_overrides(self, env):
        env.setenv("GOOGLE_API_KEY", "k")
        env.setenv("WAKE_WORD_ENABLED", "true")
        env.setenv("WAKE_WORD_THRESHOLD", "0.8")
        env.setenv("WAKE_SILENCE_TIMEOUT_SECS", "12.5")

        settings = Settings(_env_file=None)

        assert settings.wake_word_enabled is True
        assert settings.wake_word_threshold == 0.8
        assert settings.wake_silence_timeout_secs == 12.5
```

- [ ] **Step 2: Add the new names to the `env` fixture**

In `tests/conftest.py`, add to the tuple of cleared variables:

```python
        "WAKE_WORD_ENABLED",
        "WAKE_WORD_MODEL_PATH",
        "WAKE_WORD_THRESHOLD",
        "WAKE_SILENCE_TIMEOUT_SECS",
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_settings.py -k WakeWord -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'wake_word_enabled'`

- [ ] **Step 4: Write the implementation**

In `palm_9000/settings.py`, add after `gemini_live_model`:

```python
    # Wake-word gating. Disabled by default: with it off the app behaves
    # exactly as before and streams audio continuously while running.
    wake_word_enabled: bool = False
    wake_word_model_path: str = "models/wakeword/hey_livekit.onnx"
    wake_word_threshold: float = 0.5
    wake_silence_timeout_secs: float = 30.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_settings.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add palm_9000/settings.py tests/test_settings.py tests/conftest.py
git commit -m "feat(wakeword): add wake-word settings, disabled by default"
```

---

### Task 5: Wire it into the pipeline

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `LiveKitWakeWordDetector` (Task 2), `WakeWordGate` (Task 3), settings (Task 4).
- Produces: `build_wake_gate(llm) -> WakeWordGate | None`, `build_context_aggregator() -> LLMContextAggregatorPair`, `build_pipeline(..., wake_gate=None)`.

- [ ] **Step 1: Add the settings helper to the test file**

Add near the top of `tests/test_main.py`:

```python
def _settings(**overrides):
    """A real Settings object with wake-word fields overridden."""
    from palm_9000.settings import Settings

    base = {
        "google_api_key": "k",
        "wake_word_enabled": False,
        "wake_word_model_path": "models/wakeword/hey_livekit.onnx",
        "wake_word_threshold": 0.5,
        "wake_silence_timeout_secs": 30.0,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_main.py`:

```python
class TestWakeWordWiring:
    def test_no_gate_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=False)
        )
        assert main_module.build_wake_gate(llm=MagicMock()) is None

    def test_gate_built_when_enabled(self, monkeypatch):
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=True)
        )
        captured = {}
        monkeypatch.setattr(
            main_module, "WakeWordGate", lambda **kw: captured.update(kw) or "GATE"
        )

        class FakeDetector:
            def __init__(self, path):
                self.path = path
                self.loaded = False

            def load(self):
                self.loaded = True

        monkeypatch.setattr(main_module, "LiveKitWakeWordDetector", FakeDetector)

        assert main_module.build_wake_gate(llm="LLM") == "GATE"
        assert captured["threshold"] == 0.5
        assert captured["silence_timeout_secs"] == 30.0
        assert captured["llm"] == "LLM"

    def test_model_is_loaded_at_build_time(self, monkeypatch):
        """A missing model must fail at startup, not on the first frame."""
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=True)
        )
        monkeypatch.setattr(main_module, "WakeWordGate", lambda **kw: "GATE")

        loaded = []

        class FakeDetector:
            def __init__(self, path):
                self.path = path

            def load(self):
                loaded.append(self.path)

        monkeypatch.setattr(main_module, "LiveKitWakeWordDetector", FakeDetector)
        main_module.build_wake_gate(llm="LLM")

        assert loaded == ["models/wakeword/hey_livekit.onnx"]

    def test_missing_model_raises_at_build_time(self, monkeypatch):
        monkeypatch.setattr(
            main_module,
            "get_settings",
            lambda: _settings(wake_word_enabled=True, wake_word_model_path="nope.onnx"),
        )
        with pytest.raises(FileNotFoundError):
            main_module.build_wake_gate(llm="LLM")

    def test_pipeline_places_the_gate_before_the_aggregator(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "Pipeline",
            lambda processors: captured.setdefault("p", processors),
        )
        transport = MagicMock()
        transport.input.return_value = "IN"
        transport.output.return_value = "OUT"
        aggregators = MagicMock()
        aggregators.user.return_value = "AGG_USER"
        aggregators.assistant.return_value = "AGG_ASSISTANT"

        main_module.build_pipeline(
            transport=transport,
            llm="LLM",
            context_aggregator=aggregators,
            recording_control="CTL",
            audio_buffer="BUF",
            wake_gate="GATE",
        )

        order = captured["p"]
        assert order.index("IN") < order.index("GATE") < order.index("AGG_USER")

    def test_pipeline_omits_the_gate_when_none(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "Pipeline",
            lambda processors: captured.setdefault("p", processors),
        )
        transport = MagicMock()
        transport.input.return_value = "IN"
        transport.output.return_value = "OUT"
        aggregators = MagicMock()
        aggregators.user.return_value = "AGG_USER"
        aggregators.assistant.return_value = "AGG_ASSISTANT"

        main_module.build_pipeline(
            transport=transport,
            llm="LLM",
            context_aggregator=aggregators,
            recording_control="CTL",
            audio_buffer="BUF",
            wake_gate=None,
        )

        assert None not in captured["p"]
        assert captured["p"][0] == "IN"
        assert captured["p"][1] == "AGG_USER"

    def test_service_starts_paused_only_when_gating(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "GeminiLiveLLMService",
            lambda **kw: captured.update(kw) or "LLM",
        )

        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=True)
        )
        main_module.build_llm()
        assert captured["start_audio_paused"] is True

        captured.clear()
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=False)
        )
        main_module.build_llm()
        assert captured["start_audio_paused"] is False

    def test_aggregator_gets_a_vad_analyzer(self, monkeypatch):
        """Without it the silence timer has no activity signal."""
        captured = {}
        monkeypatch.setattr(
            main_module,
            "LLMContextAggregatorPair",
            lambda context, **kw: captured.update(kw) or MagicMock(),
        )
        main_module.build_context_aggregator()
        assert captured["user_params"].vad_analyzer is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_main.py -k WakeWord -v`
Expected: FAIL with `AttributeError: module 'main' has no attribute 'build_wake_gate'`

- [ ] **Step 4: Update the imports in `main.py`**

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from palm_9000.processors import AudioRecordingControlProcessor, WakeWordGate
from palm_9000.wakeword import LiveKitWakeWordDetector
```

- [ ] **Step 5: Add the two new builders**

Add to `main.py` after `build_llm`:

```python
def build_wake_gate(llm: GeminiLiveLLMService) -> WakeWordGate | None:
    """The gate, or None when wake-word gating is disabled.

    Returning None rather than a pass-through keeps the disabled path
    byte-identical to the pre-feature pipeline.

    The model is loaded here, not lazily, so a missing or unreadable model
    fails at startup. Failing later would leave the gate permanently asleep
    and the plant permanently mute, which is harder to diagnose.
    """
    settings = get_settings()
    if not settings.wake_word_enabled:
        return None

    detector = LiveKitWakeWordDetector(settings.wake_word_model_path)
    detector.load()
    return WakeWordGate(
        detector=detector,
        llm=llm,
        threshold=settings.wake_word_threshold,
        silence_timeout_secs=settings.wake_silence_timeout_secs,
    )


def build_context_aggregator() -> LLMContextAggregatorPair:
    """Aggregators with local VAD.

    GeminiLiveLLMService does not emit user turn frames, so the gate's silence
    timer needs a local source. vad_analyzer lives on LLMUserAggregatorParams
    in pipecat 1.x, not on TransportParams.
    """
    return LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
```

- [ ] **Step 6: Start the service paused when gating**

In `build_llm`, add to the `GeminiLiveLLMService(...)` call:

```python
        start_audio_paused=settings.wake_word_enabled,
```

- [ ] **Step 7: Let `build_pipeline` take the gate**

Add `wake_gate: WakeWordGate | None = None` to the signature and replace the
list literal with:

```python
    processors = [transport.input()]
    if wake_gate is not None:
        processors.append(wake_gate)
    processors += [
        context_aggregator.user(),
        llm,
        transport.output(),
        recording_control,
        audio_buffer,
        context_aggregator.assistant(),
    ]
    return Pipeline(processors)
```

- [ ] **Step 8: Use them in `run_pipeline`**

Replace the aggregator construction and the `build_pipeline` call:

```python
    context_aggregator = build_context_aggregator()
    llm = build_llm()

    pipeline = build_pipeline(
        transport=build_transport(),
        llm=llm,
        context_aggregator=context_aggregator,
        recording_control=AudioRecordingControlProcessor(audio_buffer),
        audio_buffer=audio_buffer,
        wake_gate=build_wake_gate(llm),
    )
```

- [ ] **Step 9: Remove the stale VAD comment**

Delete these two lines from `build_transport` — `vad_analyzer` is not a
`TransportParams` field in pipecat 1.x, so the comment misdirects:

```python
            # Needs: from pipecat.audio.vad.silero import SileroVADAnalyzer
            # vad_analyzer=SileroVADAnalyzer(),
```

- [ ] **Step 10: Run the full suite**

Run: `uv run --no-sync pytest --cov`
Expected: PASS, coverage 100%.

- [ ] **Step 11: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat(wakeword): gate the pipeline behind a local wake word"
```

---

### Task 6: Verify on hardware, then document

**Files:**
- Modify: `CLAUDE.md`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Run the mutation sweep**

Each mutation must turn the suite red. Assert the target string exists before
replacing it — a mutation that silently fails to apply reports as "survived",
which is indistinguishable from an unprotected behaviour.

| Mutation | Must fail |
| --- | --- |
| `wake_word_enabled` default `False` → `True` | settings test |
| Drop `start_audio_paused` from `build_llm` | wiring test |
| Remove the gate from `build_pipeline` | ordering test |
| Delete the `_sleep()` call on timeout | re-arm test |
| Make `_maybe_wake` wake on exception | fail-closed test |
| Remove `vad_analyzer` from the aggregator | VAD test |
| Remove `detector.load()` from `build_wake_gate` | startup-failure test |

- [ ] **Step 2: Measure CPU on the Pi**

The spec flags two unmeasured risks: Silero VAD and the wake-word chain both
run ONNX inference per frame. Measure before claiming this works on device.

```bash
rsync -av --exclude '.git/' --exclude '.venv/' --exclude '.env' \
      --exclude '__pycache__/' --exclude 'notebooks/' --exclude '.vscode/' \
      ./ raspberrypi-zero2w.local:/home/pi/Projects/PALM-9000/
ssh raspberrypi-zero2w.local
cd ~/Projects/PALM-9000 && uv sync --no-dev
WAKE_WORD_ENABLED=true PULSE_LATENCY_MSEC=60 uv run --no-dev main.py
```

From a second session, sample CPU while it idles asleep:

```bash
top -b -n 12 -d 5 -p "$(pgrep -f '[m]ain.py' | head -1)" | grep -E '^ *[0-9]+ pi'
```

A Zero 2W has four cores, so sustained >100% in `top` means one core is
saturated and audio will begin to stutter. Record the figure in the PR. If it
is too high, the spec's fallback is to drop Silero and use Gemini's
`TranscriptionFrame` plus bot-speaking frames as the activity signal.

- [ ] **Step 3: Verify it wakes and sleeps on device**

Say "hey livekit", then speak Japanese. Confirm in the log:

```
Wake word detected (score 0.NN)
[Transcription:user] ...
```

Then stay quiet for 30s and confirm:

```
No speech for 30.0s - going back to sleep
```

Also confirm the inverse — that speaking *without* the wake word produces no
transcription at all, which is the whole point of the feature.

- [ ] **Step 4: Document it**

Add to the gotchas section of `CLAUDE.md`:

```markdown
**Wake-word gating is off by default.** With `wake_word_enabled=False` the app
streams microphone audio to Gemini continuously while running — a privacy
posture and roughly $0.30/hour. Set `WAKE_WORD_ENABLED=true` before leaving
PALM-9000 running unattended.

The gate fails **closed**: a detector error or a missing model keeps the
microphone shut rather than falling back to ungated streaming. `build_wake_gate`
loads the model eagerly so a missing file fails at startup. Do not add a
fallback — it would silently restore continuous upload.
```

Add a short "Wake word" subsection to `README.md` under Run, giving the
setting, the phrase, and the fact that it must be enabled explicitly.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs(wakeword): document gating, defaults and the fail-closed rule"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
| --- | --- |
| Pause rather than disconnect | 3, 5 |
| `start_audio_paused=True` when gating | 5 |
| livekit-wakeword, exactly one package | 1 |
| `hey_livekit.onnx` committed | 1 |
| `.gitignore` negations, verified with `git add --dry-run` | 1 |
| Detector Protocol + implementation | 2 |
| Gate state machine | 3 |
| Silence timeout, four activity frames | 3 |
| Silero VAD via `LLMUserAggregatorParams` | 5 |
| Stale `vad_analyzer` comment removed | 5 |
| Four settings, disabled by default | 4 |
| Fail closed on detector error | 3 |
| Fail at startup on missing model | 2 (`load()`), 5 (`build_wake_gate` calls it) |
| Real-model tests | 2 |
| Mutation checks | 6 |
| CPU risk measurement | 6 |

Every spec requirement maps to a task. The "fail at startup" requirement,
which an earlier draft of this plan deviated from by loading lazily, is now
met by `load()` being called in `build_wake_gate`, with a test asserting a
missing model raises at build time.

**Placeholder scan:** none. Every step carries runnable content.

**Type consistency:** `WakeWordGate(detector=, llm=, threshold=,
silence_timeout_secs=, now=)` is identical in Tasks 3 and 5.
`LiveKitWakeWordDetector(model_path, model=None)` with `load()`, `process()`,
`reset()` matches between Tasks 2, 3 and 5. `build_pipeline(...,
wake_gate=None)` matches between its implementation and its tests.
