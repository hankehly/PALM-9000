# Wake-word gating: on-device verification

Everything below needs the real Pi. Nothing here has been done yet — the
branch ships with `wake_word_enabled=False`, so none of it blocks merging,
but all of it blocks *trusting* the feature.

Work through it in order; step 2 can disqualify the default configuration,
and there is no point testing behaviour that cannot keep up with realtime.

## 0. Deploy

```sh
rsync -avn --exclude '.git' --exclude '.venv/' --exclude '.env' \
      --exclude '__pycache__/' --exclude 'notebooks/' --exclude '.vscode/' \
      --exclude '.superpowers/' --exclude '.pytest_cache/' \
      --exclude '.ruff_cache/' --exclude '.coverage' --exclude 'images/' \
      ./ raspberrypi-zero2w.local:/home/pi/Projects/PALM-9000/
```

**Dry-run it first (`-n`), and read the first lines, not just the last.**

**`--exclude '.git'` has no trailing slash, and that matters.** A trailing
slash matches directories only. When deploying *from a git worktree* —
which is where this feature was built — `.git` is a 79-byte **file**
pointing at the real git directory, so `.git/` does not exclude it and
rsync will try to replace the Pi's `.git` **directory** with that file. It
announces itself:

```
could not make way for new regular file: .git
cannot delete non-empty directory: .git
```

Let that through and the Pi's repository is broken — no more `git pull` to
deploy. Drop the slash and it is excluded properly.

Once the dry run looks right, drop the `-n` and send it.

Then, on the Pi:

```sh
ssh raspberrypi-zero2w.local
export PATH="$HOME/.local/bin:$PATH"   # uv is not on a non-interactive PATH
cd ~/Projects/PALM-9000 && uv sync --no-dev
```

**`uv` is not on the PATH over non-interactive SSH** — it lives at
`~/.local/bin/uv` and only the login profile adds it. `ssh host 'uv ...'`
fails with `uv: command not found`. Worse, if you pipe it (`| tail`) the
shell reports the *pipe's* exit code, so the failure looks like success.
Export the PATH explicitly, and use `set -o pipefail` when piping.

**Ad-hoc scripts need `PYTHONPATH=.`** For the same reason the repo pins
`pythonpath = ["."]` for pytest: there is no `[build-system]`, so the
project is never installed into the venv, and `uv run` does not put the
working directory on `sys.path`. A one-off script fails with
`ModuleNotFoundError: No module named 'palm_9000'` even when run from the
project root. `PYTHONPATH=. uv run --no-sync --no-dev python script.py`
works. `main.py` itself is fine — it is run as a top-level script, so its
own directory is `sys.path[0]`.

Installs take several minutes on a Zero 2W.

rsync leaves the Pi's tree dirty relative to its commit, which blocks a
later `git pull`. Clear it with `git reset --hard origin/main` once the work
is merged — `.env` is gitignored and survives.

## 1. Confirm the model is there

```sh
ls -l models/wakeword/hey_livekit.onnx   # expect 953,357 bytes
```

If it is missing, startup will fail with a clear `FileNotFoundError` rather
than falling back to ungated streaming. That is deliberate.

## 2. Measure CPU — this is the step that can fail

**Measured on the actual Pi Zero 2W, with the whole app running:**

| Configuration | CPU | Load avg | Memory |
| --- | --- | --- | --- |
| gating disabled (baseline) | **42.7%** | 3.5 | 35.1%, flat |
| `WAKE_WORD_HOP_SAMPLES=1280` (default) | **345%** | 8.55 | 42%, climbing |
| `WAKE_WORD_HOP_SAMPLES=8000` | **284%** | 1.99 | 37.4%, stable |

The detector costs roughly **eight times the entire rest of the
application**. On four cores, 284% is ~71% of the machine.

**Do not trust single-threaded micro-benchmarks here.** onnxruntime spreads
this model over ~3.5 cores (measured: `wall 574 ms / cpu 2004 ms`), so any
figure quoted as "% of one core" from a wall-clock timing understates true
machine load by that factor. Measure the running app with `top` and read
the CPU column, which is what the table above does.

**The hop is a weak lever.** Raising it 6.25× bought only a 20% CPU
reduction, because embeddings are computed whenever enough mel frames
accumulate — at the model's fixed cadence — while the hop throttles only
the classifier. Load average does improve markedly (8.55 → 1.99), so the
machine stops queueing work even though total CPU barely moves.

Run it asleep — which is the normal state, and already the worst case,
because the gate forwards every frame to the detector whether awake or not:

```sh
WAKE_WORD_ENABLED=true PULSE_LATENCY_MSEC=60 uv run --no-dev main.py
```

From a second SSH session, sample for a minute while saying nothing:

```sh
top -b -n 12 -d 5 -p "$(pgrep -f '[m]ain.py' | head -1)" | grep -E '^ *[0-9]+ pi'
```

`pgrep -cf` counts its own command line, so do not trust a count — read the
`ps`/`top` output or the log.

**Decision:**

| Sustained CPU | Action |
| --- | --- |
| under ~60% | keep the default hop, continue to step 3 |
| 60–100% | raise `WAKE_WORD_HOP_SAMPLES`, re-measure |
| over 100% | raise the hop substantially before anything else |

Cost scales linearly as `16000 / hop`:

| `WAKE_WORD_HOP_SAMPLES` | scores/sec | embeddings/sec | app CPU | added wake latency |
| --- | --- | --- | --- | --- |
| 1280 (default, 80 ms) | 12.5 | 12.5 | 345% | up to 80 ms |
| 2560 (160 ms) | 6.2 | 12.5 | — | up to 160 ms |
| 4000 (250 ms) | 4.0 | 12.5 | — | up to 250 ms |
| 8000 (500 ms) | 2.0 | 12.5 | 284% | up to 500 ms |

Note the **embeddings column does not move**. An earlier version of this
table projected cost as `1 / hop`, which was wrong — it assumed the whole
chain scaled with the hop. Only the classifier does.

A larger hop does not cost detections. The wake word stays in the 2-second
window, so at a 250 ms hop it is still examined about eight times. It costs
only latency.

## 3. Verify it wakes

Say **"hey livekit"** — English, then speak Japanese. The wake word is
livekit's pretrained model; a custom 「へい やっし」 model is a separate task.

Expect in the log:

```
Wake word detected (score 0.NN)
[Transcription:user] ...
```

If it never fires, try `WAKE_WORD_THRESHOLD=0.3` before concluding the model
is wrong — 0.5 is a guess, not a measured value.

## 4. Verify it sleeps

Stay quiet for 10 seconds after the exchange ends. Expect:

```
No speech for 10.0s - going back to sleep
```

Then confirm the ~2 s re-arm delay is not a problem in practice: say the
wake word immediately after that line and check it still wakes. It should,
roughly 2 seconds late, because the buffer refills before it can score.

## 5. Verify it stays shut — the point of the whole feature

Speak, at length, **without** the wake word. Expect **no**
`[Transcription:user]` lines at all. If any appear, audio is reaching Google
ungated and the feature is not doing its job.

## 6. Watch for the bot interrupting itself

With gating on, local VAD is attached and pipecat's turn-start strategy
broadcasts an interruption on every detected turn start, which the Gemini
service turns into a `TTSStoppedFrame`. If echo cancellation leaks, the bot
can cut off its own reply mid-sentence with no wake word involved.

Provoke it: ask something that produces a long answer, and turn the speaker
up. Watch for replies truncating mid-sentence.

**This has been observed, not merely anticipated.** On the first real
exchange the log showed `broadcasting interruption` 0.7 s after the user's
speech was transcribed, and three more times during the reply; the bot
audibly cut itself off mid-sentence.

The remedy is narrower than the design originally assumed — Silero does not
have to go. `enable_interruptions` is an independent constructor parameter
on the turn-start strategy, so it can be switched off while the gate keeps
the speaking frames its silence timer depends on. See the CLAUDE.md gotcha
for the exact mechanism and citations.

Check `pactl list short sources` regardless — the app must be capturing
from `echosource`, not the raw ALSA input.

## 7. Record the numbers

Put the sustained CPU figure and the hop you settled on in the PR. The
design document lists CPU as an unmeasured risk; this is what closes it.
