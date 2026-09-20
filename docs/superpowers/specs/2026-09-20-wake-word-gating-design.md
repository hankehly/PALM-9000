# Wake-word gating

**Status:** design approved, not yet implemented
**Date:** 2026-09-20

## Problem

PALM-9000 uses Gemini's server-side VAD. Pipecat's own docstring is explicit
about what that entails:

> With server-side VAD enabled we stream continuously, since Gemini's VAD
> needs the uninterrupted stream to detect turns.

So whenever the app is running, it is uploading microphone audio to Google
continuously — not only when someone speaks to it. Today a 10-minute idle
timeout caps that, but only until the process is restarted.

That is the blocker for running PALM-9000 as an always-on appliance. A live
microphone streaming to a cloud API around the clock is a cost and a privacy
posture that should be chosen deliberately, not inherited from a systemd
restart policy.

**Goal:** no microphone audio leaves the Pi until a wake word is detected
locally, and audio stops flowing again shortly after the conversation ends.

## Decisions

Each was settled against measurements or library constraints, not preference.

### Gate by pausing, not disconnecting

`GeminiLiveLLMService` is constructed with `start_audio_paused=True` and woken
with `set_audio_input_paused(False)`. `_send_user_audio` returns early while
paused, so no audio is transmitted.

The alternative — disconnecting from Gemini entirely while asleep — was
measured on the Pi:

| Design | Wake-to-ready |
| --- | --- |
| Pause/unpause (chosen) | 0.011 ms (synchronous flag) |
| Rebuild pipeline per wake | 1.81s / 2.13s steady state, 6.42s first run |

Two seconds of deafness after the wake word would lose the first utterance,
since people speak immediately after saying it. Recovering that would require
buffering and replaying audio across the connect.

`_connect` and `_disconnect` are also private in pipecat, so a
disconnect-based design would either depend on internals — the same coupling
that caused the silent-audio-drop bug — or rebuild the pipeline per wake.

**Accepted trade-off:** a websocket to Google stays open while asleep. No
audio crosses it. Gemini Live sessions have server-side duration limits, so
pipecat will reconnect periodically on its own.

### openWakeWord models, run directly on onnxruntime

The `openwakeword` package cannot be installed on the Pi: it declares
`tflite-runtime` as a hard Linux dependency, and `tflite-runtime` has no
`cp312` aarch64 wheel (latest 2.14.0 tops out at `cp311`). The Pi runs Python
3.12. It would also pull `scikit-learn` and `scipy` into production, neither
of which is there today, on a device with roughly 260 MB of free RAM.

Its models are plain ONNX, and `onnxruntime` is **already installed and
loading on the Pi** via `pipecat[silero]`. So we run the models directly and
add no dependencies at all.

The inference chain is audio → melspectrogram → embedding → classifier, each
shipped as a separate `.onnx` at the v0.5.1 release. All six are confirmed
downloadable; a complete chain is ~3.5 MB.

### English wake word, for now

All openWakeWord pretrained models are English phrases. `hey_jarvis` is the
choice; `alexa` was rejected to avoid triggering on real devices.

A Japanese wake word was investigated and is **not achievable on this
engine**: openWakeWord's training extra pins `tensorflow-cpu==2.8.1` (no
Python 3.12 wheels) and uses `deep-phonemizer`, which is English-oriented.

Porcupine *can* do it — it ships `porcupine_params_ja.pv`, and Picovoice
Console generates a custom `.ppn` from a typed Japanese phrase with no
training. The repo's empty `PORCUPINE_KEYWORD_PATH` and
`PORCUPINE_MODEL_PATH` settings suggest the original author intended exactly
that. It was rejected here only because it requires a Picovoice account, and
keylessness was preferred.

**Consequence to accept:** you say an English phrase, then speak Japanese.
If that grates, switching to Porcupine is the remedy, and the detector
interface below keeps that a new class rather than a rewrite.

### Silence timeout for re-arming

After a wake, the gate re-arms once there has been no user speech and no bot
speech for `wake_silence_timeout_secs` (default 30). Conversation continues as
long as someone is talking; a single wake cannot leave the microphone open
indefinitely.

### Activity signal: Silero VAD via the aggregator

The obvious signal — `UserStartedSpeakingFrame` — is not emitted by
`GeminiLiveLLMService`. That is the warning the app has printed since the
pipecat 1.x migration:

> GeminiLiveLLMService#0 is not emitting turn frames … You can enable local
> VAD/turn detection by setting a vad_analyzer in LLMUserAggregatorParams.

`vad_analyzer` is a field on `LLMUserAggregatorParams` (it is **not** on
`TransportParams` in 1.x, so the commented-out `vad_analyzer=` line in
`main.py` is stale and should be removed). Setting it produces local user turn
frames independent of whether Gemini reports turns:

```python
LLMContextAggregatorPair(
    LLMContext(),
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
```

Silero ships with `pipecat[silero]`, so this adds no dependency, and it also
silences the long-standing startup warning.

An earlier draft proposed computing RMS locally instead. That was rejected on
review: it would have reinvented turn detection badly, resetting the timer on
any background noise.

## Architecture

### `palm_9000/wakeword.py`

```python
class WakeWordDetector(Protocol):
    def process(self, audio: bytes) -> float: ...   # 0.0-1.0 score
    def reset(self) -> None: ...
```

`OpenWakeWordDetector` implements it. Holds three `onnxruntime` sessions and
the rolling mel-frame and embedding buffers. No pipecat import, so it is
testable as a pure function of audio in, score out.

The `Protocol` exists so a `PorcupineDetector` can be added later without
touching the gate.

### `palm_9000/processors.py` — `WakeWordGate`

A `FrameProcessor` holding a detector, the LLM service, and the timeout. It
forwards every frame unchanged; its only side effect is pausing and unpausing
the service.

```
transport.input() -> WakeWordGate -> context_aggregator.user() -> llm -> ...
```

State machine:

```
ASLEEP  --(score > threshold)-------------------> AWAKE
AWAKE   --(no speech for silence_timeout_secs)--> ASLEEP
```

- **ASLEEP:** feed `InputAudioRawFrame` audio to the detector. On a score above
  threshold, call `set_audio_input_paused(False)`, reset the detector, log at
  INFO.
- **AWAKE:** do not run detection. Reset the silence deadline on
  `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`,
  `BotStartedSpeakingFrame` and `BotStoppedSpeakingFrame`. When the deadline
  passes, call `set_audio_input_paused(True)`.

Starting asleep is structural: the service is built with
`start_audio_paused=True`, so audio cannot reach Google before a wake even if
the gate fails to run.

### Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `wake_word_enabled` | `False` | Master switch; off changes nothing |
| `wake_word_model` | `hey_jarvis_v0.1` | Classifier model name |
| `wake_word_threshold` | `0.5` | Score above which to wake |
| `wake_silence_timeout_secs` | `30.0` | Re-arm after this much quiet |
| `wake_word_model_dir` | `models/wakeword` | Where the `.onnx` files live |

Defaulting `wake_word_enabled` to `False` means merging this changes no
runtime behaviour until it is switched on.

### Model files

The three `.onnx` files are committed under `models/wakeword/`, ~3.5 MB total.
Committing keeps deploys self-contained — the Pi pulls and runs, with no fetch
step — and lets CI exercise the detector against real models.

Two `.gitignore` rules currently exclude them, and `models/*` is the one that
actually matches first:

```
$ git check-ignore -v models/wakeword/melspectrogram.onnx
.gitignore:29:models/*   models/wakeword/melspectrogram.onnx
```

Because git will not re-include a file whose parent directory is excluded, the
directory must be un-ignored before its contents:

```gitignore
models/*
!models/.gitkeep
!models/wakeword/
!models/wakeword/**
```

`*.onnx` (line 25) also matches, so the `!models/wakeword/**` negation has to
come after it. Verified with `git add --dry-run`, which reports the file as
addable.

Check it that way, **not** with `git check-ignore -v`: that command prints the
matching rule even when the match is a negation, and still exits 0, so its
output reads as "ignored" when the file is in fact includable. It misled the
author of this spec once already.

## Error handling

- **Missing or unreadable models:** if `wake_word_enabled` is true and the
  models cannot be loaded, fail at startup with a clear message. Do *not* fall
  back to running ungated — that would silently restore continuous upload,
  which is the behaviour this feature exists to prevent.
- **Detector raises mid-stream:** log via `logger.exception` and stay asleep.
  Failing closed keeps audio off the wire.
- **Gate disabled:** `main.py` omits the processor entirely and constructs the
  service without `start_audio_paused`, so the current behaviour is bit-for-bit
  unchanged.

## Testing

- `OpenWakeWordDetector` against a fake `onnxruntime` session, so no model
  files are needed: buffer management, threshold behaviour, reset.
- Detector against the **real committed models**: silence scores low; a
  synthetic burst does not false-trigger. Keeps the ONNX wiring honest.
- `WakeWordGate` like `AudioRecordingControlProcessor`: frames in, assertions
  on `set_audio_input_paused` calls, on the silence deadline resetting for
  each of the four activity frames, and on every frame being forwarded.
- `main.py` wiring: the gate is present when enabled and absent when not, and
  the service is constructed with `start_audio_paused=True` only when enabled.
- **Mutation checks** on each claim: removing the gate, defaulting
  `wake_word_enabled` to `True`, dropping `start_audio_paused`, and removing
  the silence re-arm must each turn the suite red.

## Risks

- **Silero VAD CPU cost on a Pi Zero 2W is unmeasured.** It runs onnxruntime
  inference per frame alongside the wake-word chain. Measure before wiring it
  in; if it is too expensive, the fallback is Gemini's `TranscriptionFrame`
  plus bot-speaking frames as the activity signal.
- **Wake-word CPU cost is likewise unmeasured** on that hardware.
- **False accepts upload audio.** A false wake opens the microphone for up to
  the silence timeout. Threshold tuning is empirical and wants real-device
  testing.
- **A resumed stream may confuse Gemini's server VAD**, since it sees a
  discontinuity when audio starts mid-session.

## Out of scope

- The systemd unit and always-on operation. This is its prerequisite.
- A custom Japanese wake word (see above).
- Any change to the existing conversation behaviour once awake.
