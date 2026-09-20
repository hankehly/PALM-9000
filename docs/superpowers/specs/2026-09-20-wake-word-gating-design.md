# Wake-word gating

**Status:** design approved, not yet implemented
**Date:** 2026-09-20

## Problem

PALM-9000 uses Gemini's server-side VAD. Pipecat's own docstring is explicit
about what that entails:

> With server-side VAD enabled we stream continuously, since Gemini's VAD
> needs the uninterrupted stream to detect turns.

So whenever the app is running, it uploads microphone audio to Google
continuously — not only when someone speaks to it. Today a 10-minute idle
timeout caps that, but only until the process restarts.

That is the blocker for running PALM-9000 as an always-on appliance.

The cost is not only privacy. Live API audio input is billed per token, quoted
at roughly $0.005/min, so streaming continuously runs about $0.30/hour — on
the order of **$216/month to listen to an empty room**. (Rates change; check
current pricing before relying on the figure.)

**Goal:** no microphone audio leaves the Pi until a wake word is detected
locally, and audio stops flowing again shortly after the conversation ends.

## Decisions

Each was settled against a measurement or a library constraint.

### Gate by pausing, not disconnecting

`GeminiLiveLLMService` is constructed with `start_audio_paused=True` and woken
with `set_audio_input_paused(False)`. `_send_user_audio` returns early while
paused, so nothing is transmitted.

Measured on the Pi:

| Design | Wake-to-ready |
| --- | --- |
| Pause/unpause (chosen) | 0.011 ms (synchronous flag) |
| Rebuild pipeline per wake | 1.81s / 2.13s steady, 6.42s first run |

Two seconds of deafness would lose the first utterance, since people speak
immediately after the wake word. `_connect`/`_disconnect` are also private in
pipecat, so a disconnect design would depend on internals — the same coupling
that caused the silent-audio-drop bug.

**Holding the session open costs nothing.** The service has no keepalive, ping
or heartbeat path; its only senders are `_send_user_audio` (returns early while
paused), `_send_user_text` and `_send_user_video`, and the latter two are never
called here. The Live API bills per token with no charge for connection
duration. The open socket is a privacy consideration, not a financial one.

### Engine: livekit-wakeword

Chosen after ruling out the alternatives:

| Engine | Verdict |
| --- | --- |
| **livekit-wakeword** | **Chosen.** Keyless, ONNX, trains custom multilingual models. |
| Porcupine | Out — now requires a company email; the project's account is gone. |
| openWakeWord (package) | Out — declares `tflite-runtime`, which has no `cp312` aarch64 wheel. |
| Rhasspy Raven | Out — archived Nov 2023, pins `scipy==1.6.0`. |
| Vosk | Out — small models want ~300 MB runtime; the Pi has ~260 MB free. |

Verified empirically rather than assumed:

- **It adds exactly one package to production.** `uv pip install --dry-run`
  against a real `--no-dev` environment reports `Would install 1 package:
  livekit-wakeword`. Its only declared deps are `numpy` and `onnxruntime`,
  both already present via `pipecat[silero]`.
- **It loads openWakeWord's pretrained models.** `WakeWordModel(models=
  ["hey_jarvis_v0.1.onnx"])` loads and `predict()` returns
  `{'hey_jarvis_v0.1': 0.0}` on silence.
- **The wheel bundles the feature extractors** (`melspectrogram.onnx`,
  `embedding_model.onnx`), so there is no download at runtime and no network
  dependency on the Pi or in CI.

Input is 16 kHz int16 — exactly what the transport already produces.

### English wake word now, Japanese later

Ship with openWakeWord's pretrained `hey_jarvis_v0.1.onnx`. `alexa` was
rejected to avoid triggering real devices.

The intended wake word is 「へい やっし」 — *hey yashi*, ヤシ being Japanese for
palm tree. livekit-wakeword can train it: set `tts_backend: voxcpm` and
`target_phrases`, then train off-device and drop the resulting `.onnx` in.
Because that is the same file slot and the same runtime call, it is a
configuration change, not a rewrite.

Deferred rather than done now because the project documents that
*"multilingual models currently achieve lower accuracy than English models"* —
the frozen speech embedding is English-dominant and VoxCPM produces less
diverse synthetic speech than Piper. Training time and hardware are
unspecified upstream. Gating is worth having before that is worked out.

**Consequence to accept meanwhile:** you say an English phrase, then speak
Japanese.

### Silence timeout for re-arming

After a wake, the gate re-arms once there has been no user speech and no bot
speech for `wake_silence_timeout_secs` (default 30). A single wake cannot
leave the microphone open indefinitely.

### Activity signal: Silero VAD via the aggregator

`GeminiLiveLLMService` does not emit `UserStartedSpeakingFrame` — that is the
warning the app has printed since the pipecat 1.x migration:

> GeminiLiveLLMService#0 is not emitting turn frames … You can enable local
> VAD/turn detection by setting a vad_analyzer in LLMUserAggregatorParams.

`vad_analyzer` is a field on `LLMUserAggregatorParams`. It is **not** on
`TransportParams` in 1.x, so the commented-out `vad_analyzer=` line in
`main.py` is stale and should be deleted.

```python
LLMContextAggregatorPair(
    LLMContext(),
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)
```

Silero ships with `pipecat[silero]`, so this adds nothing, and it silences the
long-standing startup warning.

An earlier draft proposed computing RMS locally. Rejected on review: it would
have reinvented turn detection badly, resetting the timer on any background
noise.

## Architecture

### `palm_9000/wakeword.py`

```python
class WakeWordDetector(Protocol):
    def process(self, audio: bytes) -> float: ...   # 0.0-1.0
    def reset(self) -> None: ...
```

`LiveKitWakeWordDetector` wraps `livekit.wakeword.WakeWordModel`, converting
`bytes` to the `int16` array it expects and reducing the returned score dict to
a single float. No pipecat import, so it is testable as audio in, score out.

The `Protocol` keeps the gate independent of the engine.

### `palm_9000/processors.py` — `WakeWordGate`

A `FrameProcessor` holding a detector, the LLM service and the timeout. It
forwards every frame unchanged; its only side effect is pausing and unpausing.

```
transport.input() -> WakeWordGate -> context_aggregator.user() -> llm -> ...
```

```
ASLEEP  --(score > threshold)-------------------> AWAKE
AWAKE   --(no speech for silence_timeout_secs)--> ASLEEP
```

- **ASLEEP:** feed `InputAudioRawFrame` audio to the detector. Above
  threshold, call `set_audio_input_paused(False)`, reset the detector, log at
  INFO.
- **AWAKE:** skip detection. Reset the silence deadline on
  `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`,
  `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`. On expiry, call
  `set_audio_input_paused(True)`.

Starting asleep is structural: the service is built with
`start_audio_paused=True`, so audio cannot reach Google before a wake even if
the gate fails to run.

### Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `wake_word_enabled` | `False` | Master switch; off changes nothing |
| `wake_word_model_path` | `models/wakeword/hey_jarvis_v0.1.onnx` | Classifier |
| `wake_word_threshold` | `0.5` | Score above which to wake |
| `wake_silence_timeout_secs` | `30.0` | Re-arm after this much quiet |

### Model file

One file, `models/wakeword/hey_jarvis_v0.1.onnx` (1.27 MB), committed. The
feature extractors come from the wheel, so nothing else ships and nothing is
fetched at runtime.

Two `.gitignore` rules currently exclude it, and `models/*` matches first.
Because git will not re-include a file whose parent directory is excluded, the
directory must be un-ignored before its contents:

```gitignore
models/*
!models/.gitkeep
!models/wakeword/
!models/wakeword/**
```

Verify with `git add --dry-run`, **not** `git check-ignore -v`: the latter
prints the matching rule even when the match is a negation and still exits 0,
so its output reads as "ignored" when the file is includable. It misled the
author of this spec once already.

## Error handling

- **Missing or unreadable model:** if enabled and the model cannot load, fail
  at startup with a clear message. Do **not** fall back to ungated operation —
  that would silently restore continuous upload, the exact behaviour this
  feature removes.
- **Detector raises mid-stream:** log via `logger.exception` and stay asleep.
  Failing closed keeps audio off the wire.
- **Disabled:** `main.py` omits the processor and constructs the service
  without `start_audio_paused`, leaving today's behaviour unchanged.

## Testing

- `LiveKitWakeWordDetector` against a fake `WakeWordModel`: byte-to-array
  conversion, score reduction, reset.
- Detector against the **real committed model**: silence scores low, and a
  synthetic burst does not false-trigger. Keeps the wiring honest.
- `WakeWordGate` like `AudioRecordingControlProcessor`: assertions on
  `set_audio_input_paused` calls, on the deadline resetting for each of the
  four activity frames, and on every frame being forwarded.
- `main.py` wiring: gate present only when enabled; `start_audio_paused=True`
  only when enabled.
- **Mutation checks** on each claim: removing the gate, defaulting
  `wake_word_enabled` to `True`, dropping `start_audio_paused`, and removing
  the silence re-arm must each turn the suite red.

## Risks

- **Silero VAD CPU cost on a Pi Zero 2W is unmeasured.** It runs ONNX
  inference per frame alongside the wake-word chain. Measure before wiring; if
  too expensive, fall back to Gemini's `TranscriptionFrame` plus bot-speaking
  frames as the activity signal.
- **Wake-word CPU cost is likewise unmeasured** on that hardware.
- **livekit-wakeword is young** (v0.2.1). Mitigated by its models being plain
  ONNX and openWakeWord-compatible, so they outlive the library.
- **False accepts upload audio** for up to the silence timeout. Threshold
  tuning is empirical and wants real-device testing.
- **A resumed stream may confuse Gemini's server VAD**, which sees a
  discontinuity when audio restarts mid-session.

## Out of scope

- The systemd unit and always-on operation. This is its prerequisite.
- Training the Japanese 「へい やっし」 model (follow-up; same file slot).
- Any change to conversation behaviour once awake.
