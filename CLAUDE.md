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
WAKE_WORD_ENABLED=true PULSE_LATENCY_MSEC=60 uv run --no-dev main.py   # gated
```

`uv sync` (no flags) pulls the full `dev` group: torch, whisper, langchain,
llama-index. That is for the notebooks.

`--group` *adds* to the default groups rather than replacing them, and `dev`
is a default, so `uv sync --group test` still installs the whole notebook
stack (~374 extra packages). Use `--no-default-groups --group test` for test
work, and `--no-dev` for running on the Pi.

## Pipeline order matters

```
transport.input() -> [WakeWordGate] -> context_aggregator.user() -> llm
  -> transport.output() -> AudioRecordingControlProcessor
  -> AudioBufferProcessor -> context_aggregator.assistant()
```

`WakeWordGate` is present only when wake-word gating is enabled.
`build_pipeline` takes `wake_gate=None` by default and, when it is
`None`, leaves the step out entirely rather than inserting a
pass-through, so the disabled pipeline is the sequence above minus the
gate, unchanged from before the feature existed. Its position — upstream
of everything else — is load-bearing; see the gotcha below.

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

**Wake-word gating is off by default.** With `wake_word_enabled=False`
the app streams microphone audio to Gemini continuously while running —
a privacy posture and roughly $0.30/hour. Rates change; check Google's
current Gemini Live pricing before treating that figure as exact. Set
`WAKE_WORD_ENABLED=true` before leaving PALM-9000 running unattended.

**The gate fails closed.** A detector error or a missing model keeps the
microphone shut rather than falling back to ungated streaming.
`build_wake_gate` loads the model eagerly so a missing file fails at
startup. Do not add a fallback — it would silently restore continuous
upload.

**The wake gate's pipeline position is load-bearing.** `WakeWordGate`
sits between `transport.input()` and `context_aggregator.user()`,
upstream of every processor that emits the six frames its silence timer
resets on: `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`,
`UserSpeakingFrame`, `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`,
`BotSpeakingFrame`. All six come from
processors downstream of the gate — the user aggregator and the output
transport — and both broadcast every frame in *both* directions: the
aggregator via `FrameProcessor.broadcast_frame`
(`pipecat/processors/frame_processor.py:1053-1054`), called from
`llm_response_universal.py:1324,1410`; the output transport the same
way by hand at `pipecat/transports/base_output.py:716-726,787-798`. The
gate, upstream of both, only ever sees the upstream copy. Move it
downstream of the aggregator and it stops seeing all six: the deadline
never resets, and the plant goes deaf 30 seconds into a conversation,
with no error.

**The *continuing* speech frames are load-bearing, not padding.**
`_ACTIVITY_FRAMES` must include `BotSpeakingFrame` and
`UserSpeakingFrame`, not only the start/stop pairs. Nothing arrives
between a start and a stop, so with start/stop alone a 45-second bot
reply trips the 30-second silence timeout *while the bot is audibly
talking* — the log reads "No speech for 30.0s" over the sound of it
speaking — and the `BotStoppedSpeakingFrame` that follows is ignored,
because the deadline only refreshes while awake. The user's follow-up
is then discarded and they have to say the wake word again. Both
continuing frames broadcast about every 0.2s (`base_output.py:805` for
the bot; `VADController.on_speech_activity` via the user aggregator).

**The wake-word model is stateless; the detector holds the buffer.**
`WakeWordModel.predict()` mels exactly the chunk you hand it and needs
76 + 15×8 = 196 mel frames — about 2 seconds — in a *single* call
(`livekit/wakeword/inference/model.py:96-145`). It accumulates nothing
between calls. `LocalAudioTransport` pushes 20 ms frames (320 samples
at 16 kHz, `pipecat/transports/local/audio.py:76`), so handing a frame
straight to `predict()` raises `InvalidArgument: Invalid input shape:
{320}` about fifty times a second and never scores anything. Shorter-
but-valid chunks are worse: below ~2 s `predict()` returns **exactly
0.0 for every possible input**, including a perfect wake word — which
is why a test that only asserts "silence scores below the threshold"
proves nothing at all. `LiveKitWakeWordDetector` therefore keeps its
own 2-second rolling buffer and scores it on an 80 ms hop. `reset()`
clears that buffer and keeps the model; dropping the model forces a
full ONNX session rebuild on the next frame, which on a Pi is seconds
of stall after every wake.

**With gating on, local VAD can interrupt the bot.** The turn-start
strategy defaults to `enable_interruptions=True`
(`base_user_turn_start_strategy.py:56`), so echo leakage or a second
person talking will cut a reply off mid-sentence without any wake word.
This is accepted rather than fixed: there is no knob for it on
`LLMUserAggregatorParams`, and suppressing it means pinning pipecat's
whole default strategy list — the kind of coupling to internals that
caused the silent-audio bug above. Barge-in is also often wanted. Watch
for it when testing on device, at higher speaker volume and on longer
replies.

**With gating on, the 10-minute idle timeout effectively stops firing.**
Local VAD emits `UserStartedSpeakingFrame` and `UserSpeakingFrame`,
both in `PipelineWorker`'s default idle set (`worker.py:301-307`), so
any room noise — a television, a conversation nearby — keeps resetting
`IDLE_TIMEOUT_SECS`. Harmless, since nothing uploads while the gate is
asleep, but do not rely on that timeout as a backstop when gating is on.

**`vad_analyzer` is attached only when gating is on.** This is not a
tidiness choice. A `vad_analyzer` is what constructs pipecat's
`VADController` at all (`if self._params.vad_analyzer:`,
`llm_response_universal.py:756`) and brings the default
`VADUserTurnStartStrategy` to life; that strategy broadcasts an
interruption on every detected turn start
(`llm_response_universal.py:1327-1329`), which `GeminiLiveLLMService`
turns into a `TTSStoppedFrame` that cuts the bot off mid-sentence
(`gemini_live/llm.py:1036-1037`, `:960-966`). Attaching Silero
unconditionally woke all of that on every run, including gating off,
where nothing needs it: imperfect echo cancellation — the risk the
Hardware section already names as "the bot hears itself and talks to
itself" — lets a sliver of the bot's own TTS reach the mic, local VAD
reads it as speech, and the bot interrupts itself, a failure that did
not exist before this branch. `GeminiLiveLLMService` also logs
pipecat's "not emitting turn frames" warning once at startup either
way — `service_metadata_frame()` never looks at the aggregator, so
seeing it does not mean local VAD is missing. Its own suggested fix is
"set a vad_analyzer in LLMUserAggregatorParams"; that is not a reason
to attach one unconditionally. Doing so to quiet the warning is the
same mistake that caused this bug.

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

**Run tests the way CI does — `uv run pytest`, not `python -m pytest`.** The
two differ: `python -m pytest` injects the working directory into
`sys.path`, `uv run` does not, and there is no `[build-system]` so the
project is never installed into the venv. A suite that passes locally under
`python -m pytest` can fail in CI with `ModuleNotFoundError: No module named
'palm_9000'`. `pythonpath = ["."]` in `[tool.pytest.ini_options]` is what
makes both work; do not remove it.

A passing test proves little on its own. Before trusting a regression test,
reintroduce the bug it covers and confirm the suite goes red. Every
regression suite here was checked that way, and the mutation counts are
recorded in the PR that added each one.

## Hardware

Wiring for the mic, speaker, LED matrix and echo cancellation is documented in
`README.md`. Worth knowing:

- The LED matrix is on **SPI0 / CE0**, single module (`cascaded=1`). The `pi`
  user must be in the `spi` group.
- `Max7219AmplitudeHeart` opens the bus in `_open_device()`, called from
  `start()` — **not** in `__init__`. Constructing one therefore works on a
  machine with no SPI, which is what makes the class testable off-device.
  Keep it that way; `spidev` is Linux-only, so moving the open back into the
  constructor would make the class unusable anywhere but the Pi.
- The render loop must **not** redraw the pattern every frame. The heart is
  static and only its brightness changes, so `_run()` draws once and writes
  the intensity register only when the computed brightness actually differs.
  A steady audio level produces zero SPI traffic; the old per-frame loop did
  ~90 full-frame flushes and ~90 intensity writes per second. `refresh_secs`
  (default 5s) redraws occasionally as insurance against the display losing
  state, and `0` disables that. `TestRedrawIsNotRepeated` enforces all of it.
- To measure real SPI traffic, wrap the luma device in a proxy that counts
  `contrast()` and `display()` calls and forwards to the original. That is
  how the numbers above were taken on actual hardware rather than simulated.
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

When killing a run over SSH, `pkill -f main.py` matches the SSH command's own
line and kills your session. A bracket pattern (`pkill -f "[m]ain.py"`) fixes
the self-match **only if the literal string is absent from the rest of the
command** — killing and relaunching in one invocation re-introduces it via the
launch arguments and kills the session anyway. Put the kill and the launch in
separate SSH calls.

`pgrep -cf "<pattern>"` has the same trap in reverse: it counts its own
command line, so a finished process can look like it is still running. Check
`ps` output or the log instead of trusting the count.

## CI

`.github/workflows/ci.yml` runs on every PR and push to `main`:

- **test** — installs PortAudio (needed by `pipecat-ai[local]` → pyaudio),
  then `ruff check`, `ruff format --check`, and pytest with the 100% gate.
- **lockfile** — `uv lock --check`. The lockfile is what actually pins
  pipecat for deploys to the Pi, so drift between it and `pyproject.toml`
  means the Pi installs something the tests never saw. Run `uv lock` after
  touching dependencies.

CI needs a `GOOGLE_API_KEY` to exist because `palm_9000/settings.py`
constructs `Settings()` at import time; the workflow sets a placeholder. No
request is ever made with it.

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
