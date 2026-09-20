# Wake-word gating: on-device verification

Everything below needs the real Pi. Nothing here has been done yet — the
branch ships with `wake_word_enabled=False`, so none of it blocks merging,
but all of it blocks *trusting* the feature.

Work through it in order; step 2 can disqualify the default configuration,
and there is no point testing behaviour that cannot keep up with realtime.

## 0. Deploy

```sh
rsync -av --exclude '.git/' --exclude '.venv/' --exclude '.env' \
      --exclude '__pycache__/' --exclude 'notebooks/' --exclude '.vscode/' \
      ./ raspberrypi-zero2w.local:/home/pi/Projects/PALM-9000/
ssh raspberrypi-zero2w.local
cd ~/Projects/PALM-9000 && uv sync --no-dev
```

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

The wake-word chain is the expensive part, not Silero. Measured on Apple
silicon, per second of audio:

| Component | Cost | Share of one core |
| --- | --- | --- |
| Wake-word inference | 202 ms | 20.2% |
| Silero VAD | 7.2 ms | 0.72% |
| Buffer bookkeeping | 0.45 ms | 0.05% |

It is single-threaded in one process. A Zero 2W is far slower at ONNX, so
the same workload could land anywhere from ~60% to over 200% of a core.
**Above 100% it cannot keep up with realtime**, and audio and the LED heart
will stutter.

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

| `WAKE_WORD_HOP_SAMPLES` | scores/sec | relative cost | added wake latency |
| --- | --- | --- | --- |
| 1280 (default, 80 ms) | 12.5 | 1.0× | up to 80 ms |
| 2560 (160 ms) | 6.2 | 0.50× | up to 160 ms |
| 4000 (250 ms) | 4.0 | 0.32× | up to 250 ms |
| 8000 (500 ms) | 2.0 | 0.16× | up to 500 ms |

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

Stay quiet for 30 seconds after the exchange ends. Expect:

```
No speech for 30.0s - going back to sleep
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

If it happens, the remedy in the design is to drop Silero and use Gemini's
`TranscriptionFrame` plus the bot-speaking frames as the activity signal
instead. Check `pactl list short sources` first — the app must be capturing
from `echosource`, not the raw ALSA input.

## 7. Record the numbers

Put the sustained CPU figure and the hop you settled on in the PR. The
design document lists CPU as an unmeasured risk; this is what closes it.
