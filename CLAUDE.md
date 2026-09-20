# CLAUDE.md

Guidance for working in this repository.

## What this is

PALM-9000 is a talking houseplant: a Raspberry Pi Zero 2W runs a pipecat
pipeline that streams microphone audio to the Gemini Live API and plays the
reply through a speaker, while an 8x8 MAX7219 LED matrix pulses like a
heartbeat in time with the bot's voice. It speaks Japanese only.

`main.py` is the whole application. `palm_9000/` holds the supporting pieces.

## Commands

```sh
uv sync --group test                 # install test deps (lean; excludes notebook libs)
uv run --no-sync pytest              # run the suite (coverage gate: 100%)
uv run --no-sync pytest --cov --cov-report=term-missing
uv run --no-sync ruff check --fix main.py palm_9000/ tests/
uv run --no-sync ruff format main.py palm_9000/ tests/
PULSE_LATENCY_MSEC=60 uv run --no-dev main.py   # run for real (on the Pi)
```

`uv sync` (no flags) pulls the full `dev` group: torch, whisper, langchain,
llama-index. That is for the notebooks. Use `--group test` for test work and
`--no-dev` for running on the Pi.

## Pipeline order matters

```
transport.input() -> context_aggregator.user() -> llm -> transport.output()
  -> AudioRecordingControlProcessor -> AudioBufferProcessor
  -> context_aggregator.assistant()
```

The recording-control and buffer processors sit *after* `transport.output()`
because they react to bot-speaking frames, which originate downstream. The
buffer's `on_audio_data` is what drives the LED heart, so it only sees bot
audio, never user audio.

## Gotchas that have already cost a day

**The context aggregators are load-bearing, even though nothing here needs
conversation history.** `GeminiLiveLLMService` gates outgoing audio behind
`_ready_for_realtime_input`, which only flips to `True` once an
`LLMContextFrame` reaches the service. Remove the aggregators or the
`LLMRunFrame` kickoff and every frame of user audio is silently discarded:
the websocket connects, the log looks healthy, and the bot simply never
answers. There is no error. This gate did not exist in pipecat 0.0.84.

**`WorkerRunner.add_workers` is a coroutine.** `inspect.signature` renders it
identically to a sync function (`-> None`); only
`inspect.iscoroutinefunction` distinguishes them. Forgetting `await`
registers no worker, and the app connects to nothing while looking fine.
Several pipecat 1.x methods are async this way — check before calling.

**pipecat deprecations become removals.** `model=`, `voice_id=` and `params=`
on `GeminiLiveLLMService` are deprecated and disappear in 2.0; use
`settings=GeminiLiveLLMSettings(...)`. `PipelineTask` and `PipelineRunner`
are likewise superseded by `PipelineWorker` and `WorkerRunner`. The dependency
is pinned `>=1.11.0,<2` to keep those removals from landing silently. When
upgrading, verify APIs against an actual install rather than the docs.

**Gemini Live model IDs churn.** `gemini_live_model` is a setting
(`models/gemini-3.8-live` by default), not a hardcoded string. Preview model
IDs get retired; check Google's current lineup before assuming a failure is
in this code.

**The `.env` uses `export KEY=value` shell syntax.** python-dotenv handles the
prefix, so pydantic-settings reads it fine. The Pi has its own `.env` that is
separate from the development machine's — updating one does not update the
other, and `INPUT_DEVICE` legitimately differs between them, so do not copy
the file wholesale.

## Tests

`tests/conftest.py` injects fakes for `RPi.GPIO`, `pyaudio` and `sounddevice`
into `sys.modules` before the modules under test import them, so the suite runs
on any machine. It also sets a placeholder `GOOGLE_API_KEY`, because
`palm_9000/settings.py` constructs `Settings()` at import time and would
otherwise depend on a local `.env`.

`luma` is a real dependency and imports everywhere, so `test_gpio.py` patches
`palm_9000.gpio.spi` / `.max7219` / `.canvas` instead of faking the module.

Coverage is gated at 100% (branch coverage included) via
`[tool.coverage.report] fail_under`. The `main.py` tests are deliberately
regression tests for the two silent-failure bugs above; if you change the
pipeline wiring, expect `TestRealtimeGateRegression` and
`TestWorkerRegistrationRegression` to be what catches you.

## Hardware

Wiring for the mic, speaker, LED matrix and echo cancellation is documented in
`README.md`. Worth knowing:

- The LED matrix is on **SPI0 / CE0**, single module (`cascaded=1`). The `pi`
  user must be in the `spi` group.
- `Max7219AmplitudeHeart.__init__` opens SPI eagerly, so it cannot be imported
  on a machine without the bus. Tests patch around this; be careful adding
  imports of `palm_9000.gpio` to code that runs off-device.
- Acoustic echo cancellation runs in PulseAudio (`module-echo-cancel`,
  webrtc). The app must capture from `echosource`, not the raw ALSA input, or
  the bot hears itself and talks to itself. Verify with
  `pactl list short sources`.
- To check the audio path independently of the app, record from `echosource`
  with `parecord` while the app runs — both can read the source at once, which
  makes it easy to tell an input problem from a pipeline problem.

## Deploying to the Pi

The Pi has no SSH key for GitHub, so it cannot `git pull`. Deploy with rsync
from a development machine, excluding `.env` and `.venv`:

```sh
rsync -av --exclude '.git/' --exclude '.venv/' --exclude '.env' \
      --exclude '__pycache__/' --exclude 'notebooks/' --exclude '.vscode/' \
      ./ raspberrypi-zero2w.local:/home/pi/Projects/PALM-9000/
```

Then `uv sync --no-dev` on the Pi. A Zero 2W has 416 MB of RAM, so installs
are slow; expect several minutes.

When killing a run over SSH, note that `pkill -f main.py` will match the SSH
command's own line and kill your session. Use a bracket pattern
(`pkill -f "[m]ain.py"`) and keep the launch command in a separate invocation.

## Conventions

- Lint and format with ruff (line length 88, rules E/F/I/UP/B); CI enforces
  both `ruff check` and `ruff format --check`.
- `palm_9000/utils.py` imports `pyaudio`/`sounddevice`/`scipy` at module scope.
  Those are not production dependencies, so the module is unimportable under
  `--no-dev`. It exists for the notebooks. The same is true of
  `adc0834.py`, which needs `RPi.GPIO` — this blocks the moisture-sensor
  roadmap item until the dependency groups are reorganized.
