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

### Pretrained model now, custom Japanese model later

Ship with livekit-wakeword's own pretrained **`hey_livekit.onnx`** (953 KB).
Verified: it loads and scores `0.0` on both silence and white noise, so it
rejects non-speech cleanly.

It is fetched from the `livekit-examples/hello-wakeword` repository rather
than bundled in the wheel, so it is committed here like any other asset.

openWakeWord's `hey_jarvis_v0.1.onnx` is an equally valid drop-in — verified
loadable by the same runtime — and has a closer cadence to the eventual
Japanese phrase, which may ease the eventual switch. Changing between them is
one setting. `alexa` was rejected to avoid triggering real devices.

**Consequence to accept meanwhile:** you say an English phrase, then speak
Japanese.

The intended wake word is 「へい やっし」 — *hey yashi*, ヤシ being Japanese for
palm tree. Training it is **a separate task**, not part of this work: set
`tts_backend: voxcpm` and `target_phrases`, train off-device, drop the
resulting `.onnx` into the same slot. Same file path, same runtime call, so it
is a configuration change rather than a rewrite.

Deferred because the project documents that *"multilingual models currently
achieve lower accuracy than English models"* — the frozen speech embedding is
English-dominant and VoxCPM produces less diverse synthetic speech than Piper.
Training time and hardware are unspecified upstream. Gating is worth having
before that is worked out.

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

Attach it **only when gating is enabled**. Silero ships with
`pipecat[silero]`, so it costs no dependency, but it is not otherwise free.

> **Corrected during implementation.** This section originally claimed the
> analyzer "adds nothing, and it silences the long-standing startup warning."
> Both halves were wrong, and both made attaching it unconditionally look
> harmless:
>
> - **It adds an interruption path.** A `vad_analyzer` is what constructs
>   pipecat's `VADController` at all (`llm_response_universal.py:756`); with
>   `None`, the default `VADUserTurnStartStrategy` is inert. Once live, every
>   detected turn start broadcasts an interruption (`:1327-1329`), which
>   `GeminiLiveLLMService` turns into a `TTSStoppedFrame`
>   (`gemini_live/llm.py:1036-1037`, `:960-966`). With gating off — where
>   nothing needs the signal — imperfect echo cancellation would then let the
>   bot interrupt itself, a failure that did not exist before this feature.
> - **It does not silence the warning.** `service_metadata_frame()` hardcodes
>   `emits_turn_frames=False` (`gemini_live/llm.py:441`) and never inspects
>   the aggregator, so the warning fires either way. Its text merely
>   *suggests* a `vad_analyzer` as a remedy.
>
> So the only true reason to attach it is the one that survives: the gate's
> silence timer needs an activity signal. That reason applies only when there
> is a gate, which is why it is now conditional on `wake_word_enabled`.

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

> **Corrected during implementation — this section's omission shipped a bug
> that made the whole feature inert.** The detector must hold a rolling
> **2-second** buffer and score that window; it cannot pass a pipeline frame
> straight through.
>
> `WakeWordModel.predict()` is **stateless**. It computes a mel spectrogram
> over exactly the chunk it is given, slices 76-frame windows at stride 8,
> and returns all-zeros below 16 embeddings — so it needs 76 + 15×8 = 196
> mel frames, about 1.96 s, **in a single call**
> (`livekit/wakeword/inference/model.py:96-145`).
>
> `LocalAudioTransport` pushes 20 ms frames — 320 samples at 16 kHz
> (`pipecat/transports/local/audio.py:76`). Measured against the committed
> model: 320 samples raises `InvalidArgument: Invalid input shape: {320}`;
> 1.0 s and 1.5 s both return exactly `0.0`; only at ~2 s does a real score
> (`0.0047` for silence) appear. Feeding 25 consecutive 1280-sample chunks
> to one model instance still returns `0.0` — it does not accumulate.
>
> An earlier draft of this spec assumed the library kept its own rolling
> buffers. It does not, and that assumption reached the code as a comment
> asserting it. The result passed 236 tests at 100% branch coverage while
> being incapable of ever firing.
>
> Two constants follow, both constructor arguments so the on-device CPU
> measurement can tune them: a **32000-sample window** (2.0 s) and a
> **1280-sample hop** (80 ms — the model's own embedding stride, so one new
> embedding per scored window). Scoring every 20 ms frame would mean 50
> full-window ONNX passes a second, which a Zero 2W cannot afford.
>
> `reset()` clears the buffer and **keeps** the model. Dropping the model
> forces a full ONNX session rebuild on the next frame — seconds of stall on
> a Pi, after every wake.

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
  `UserSpeakingFrame`, `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`,
  `BotSpeakingFrame`. On expiry, call `set_audio_input_paused(True)`.

  The two *continuing* frames are load-bearing, not padding. Nothing
  arrives between a start and a stop, so with only the four start/stop
  frames a 45-second bot reply trips the 30-second timeout while the bot is
  still audibly talking — and the `BotStoppedSpeakingFrame` that follows is
  then ignored, because the deadline only refreshes while awake. Both
  continuing frames are broadcast about every 0.2 s
  (`base_output.py:805` for the bot; `VADController.on_speech_activity` via
  the user aggregator for the user).

Starting asleep is structural: the service is built with
`start_audio_paused=True`, so audio cannot reach Google before a wake even if
the gate fails to run.

### Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `wake_word_enabled` | `False` | Master switch; off changes nothing |
| `wake_word_model_path` | `models/wakeword/hey_livekit.onnx` | Classifier |
| `wake_word_threshold` | `0.5` | Score above which to wake |
| `wake_silence_timeout_secs` | `30.0` | Re-arm after this much quiet |

### Model file

One file, `models/wakeword/hey_livekit.onnx` (953 KB), committed. The feature
extractors come from the wheel, so nothing else ships and nothing is fetched
at runtime.

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
- **Disabled:** `main.py` omits the processor, constructs the service
  without `start_audio_paused`, and builds the aggregators with no
  `user_params` — leaving today's behaviour unchanged. "Unchanged" is
  literal: the disabled path must match the pre-feature construction
  exactly, which is what the VAD correction above enforces.

## Testing

- `LiveKitWakeWordDetector` against a fake `WakeWordModel`: byte-to-array
  conversion, score reduction, reset.
- Detector against the **real committed model**: silence and white noise both
  score 0.0 (verified during design), so neither false-triggers. Keeps the
  ONNX wiring honest rather than only exercising a stub.

  > **Corrected during implementation: as written, this test is a fiction.**
  > `predict()` returns exactly `0.0` for *every* input shorter than ~2 s,
  > so an assertion that silence and noise score below a threshold passes
  > against a model that would also score `0.0` on a perfect wake word. It
  > proves nothing.
  >
  > A real-model test must (a) drive the detector at the transport's actual
  > **320-sample / 640-byte** frame size, and (b) assert the score is
  > **non-zero** once a full window has accumulated — `0.0047` for silence.
  > Non-zero is the load-bearing part: it is the only evidence the
  > embedding and classifier path ran at all, rather than hitting the
  > not-enough-data early return. That one assertion would have caught the
  > Critical described in the `wakeword.py` section above.
- `WakeWordGate` like `AudioRecordingControlProcessor`: assertions on
  `set_audio_input_paused` calls, on the deadline resetting for each of the
  six activity frames, and on every frame being forwarded. Include a test
  that a long bot turn does not trip the timeout mid-reply.
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
- Training the Japanese 「へい やっし」 model. Tracked as a separate task; it
  drops into the same file slot with no code change.
- Any change to conversation behaviour once awake.
