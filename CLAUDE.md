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
uv sync --no-default-groups --group test   # test deps only (no notebook stack)
uv run --no-sync pytest              # run the suite (coverage gate: 100%)
uv run --no-sync pytest --cov --cov-report=term-missing
uv run --no-sync ruff check --fix main.py palm_9000/ tests/
uv run --no-sync ruff format main.py palm_9000/ tests/
PULSE_LATENCY_MSEC=60 uv run --no-dev main.py   # run for real (on the Pi)
```

`uv sync` (no flags) pulls the full `dev` group: torch, whisper, langchain,
llama-index. That is for the notebooks.

`--group` *adds* to the default groups rather than replacing them, and `dev`
is a default, so `uv sync --group test` still installs the whole notebook
stack (~374 extra packages). Use `--no-default-groups --group test` for test
work, and `--no-dev` for running on the Pi.

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

Deploy by pulling from GitHub. The repository is public and the Pi's remote
uses HTTPS, so this needs no credentials on the device — no SSH key, no token,
nothing to rotate or leak:

```sh
ssh raspberrypi-zero2w.local
cd ~/Projects/PALM-9000
git pull
uv sync --no-dev
```

A Zero 2W has 416 MB of RAM, so installs are slow; expect several minutes.

`.env` is gitignored and lives only on the Pi, so a pull never touches it.
The Pi keeps its own copy with its own values — `INPUT_DEVICE` in particular
differs from a development machine, so never copy that file between them.

Because the remote is HTTPS and anonymous, the Pi can pull but cannot push.
That is intentional for a deploy target. If the repository is ever made
private this stops working; the fix then is a read-only deploy key for the
Pi's existing `~/.ssh/id_ed25519.pub`, not an account-wide SSH key.

rsync is still useful for deploying uncommitted work while debugging:

```sh
rsync -av --exclude '.git/' --exclude '.venv/' --exclude '.env' \
      --exclude '__pycache__/' --exclude 'notebooks/' --exclude '.vscode/' \
      ./ raspberrypi-zero2w.local:/home/pi/Projects/PALM-9000/
```

Note that rsyncing leaves the Pi's working tree dirty relative to its commit,
which blocks a later `git pull`. Clear it with `git reset --hard origin/main`
once the change is committed upstream (`.env` is safe: it is ignored).

When killing a run over SSH, note that `pkill -f main.py` will match the SSH
command's own line and kill your session. Use a bracket pattern
(`pkill -f "[m]ain.py"`) and keep the launch command in a separate invocation.

## Conventions

- Lint and format with ruff (line length 88, rules E/F/I/UP/B); CI enforces
  both `ruff check` and `ruff format --check`.
- `palm_9000/utils.py` is notebook tooling; nothing in the runtime pipeline
  imports it. Its `scipy`/`sounddevice`/`pyaudio` imports are deferred into
  the functions that use them, so the module itself loads under `--no-dev`
  and only a call that needs a missing package raises. Keep them deferred —
  `tests/test_packaging.py` asserts it via the AST.
- Linux-only hardware wheels (`spidev` for the MAX7219, `rpi-gpio` for the
  ADC0834) are **main** dependencies carrying a `sys_platform == 'linux'`
  marker, so they install on the Pi and are simply absent on a development
  Mac. Tests fake them; see `tests/conftest.py`.
- Branch names follow the Conventional Branch spec
  (<https://conventionalbranch.org/>): `<type>/<description>`, where type is
  one of `feature`, `bugfix`, `hotfix`, `release` or `chore`, and the
  description is lowercase alphanumerics and hyphens only — e.g.
  `feature/led-heartbeat`, `bugfix/echo-cancel-source`.
- Commit messages follow Conventional Commits
  (<https://www.conventionalcommits.org/>):
  `<type>[optional scope]: <description>`, with `feat` and `fix` carrying
  semantic meaning, plus `docs`, `chore`,
  `refactor`, `test` and friends. Breaking changes take a `!` before the colon
  or a `BREAKING CHANGE:` footer — e.g. `fix(gpio): defer opening the SPI bus`.
