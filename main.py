import asyncio

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiLiveLLMSettings,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

from palm_9000.gpio import Max7219AmplitudeHeart
from palm_9000.processors import AudioRecordingControlProcessor, WakeWordGate
from palm_9000.settings import get_settings
from palm_9000.wakeword import LiveKitWakeWordDetector

IDLE_TIMEOUT_SECS = 10 * 60

# Gemini Live streams 16 kHz in and 24 kHz out.
AUDIO_IN_SAMPLE_RATE = 16000
AUDIO_OUT_SAMPLE_RATE = 24000

# ~80 ms of output buffering. Drop to 6 for lower latency, raise to 10-12 if
# the Pi underruns.
AUDIO_OUT_10MS_CHUNKS = 8

# 512 bytes is ~10.7 ms at 24 kHz mono int16, so on_audio_data fires ~94x/sec.
AUDIO_BUFFER_SIZE = 512

SYSTEM_INSTRUCTION = (
    "You are PALM-9000, a self-aware houseplant powered by an advanced AI. "
    "You speak in layman's terms with an eerie quality, "
    "as if you know more than you let on. "
    "すべて日本語で出力してください。翻訳・英訳・ローマ字・ふりがな・注釈・補足は一切表示しないでください。"
    "括弧（）の使用も避けてください。"
)


def build_transport() -> LocalAudioTransport:
    """Local microphone in, local speaker out."""
    return LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_channels=1,
            audio_in_sample_rate=AUDIO_IN_SAMPLE_RATE,
            audio_out_enabled=True,
            audio_out_channels=1,
            audio_out_sample_rate=AUDIO_OUT_SAMPLE_RATE,
            audio_out_10ms_chunks=AUDIO_OUT_10MS_CHUNKS,
        )
    )


def build_llm() -> GeminiLiveLLMService:
    """The Gemini Live service, configured from settings.

    Note the `settings=` object: the older `model=` / `voice_id=` / `params=`
    keyword arguments are deprecated and disappear in pipecat 2.0.
    """
    settings = get_settings()
    return GeminiLiveLLMService(
        api_key=settings.google_api_key.get_secret_value(),
        system_instruction=SYSTEM_INSTRUCTION,
        settings=GeminiLiveLLMSettings(
            model=settings.gemini_live_model,
            voice=settings.google_multimodal_live_voice_id,
            language=Language.JA,
        ),
        start_audio_paused=settings.wake_word_enabled,
    )


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

    detector = LiveKitWakeWordDetector(
        settings.wake_word_model_path,
        hop_samples=settings.wake_word_hop_samples,
    )
    detector.load()
    return WakeWordGate(
        detector=detector,
        llm=llm,
        threshold=settings.wake_word_threshold,
        silence_timeout_secs=settings.wake_silence_timeout_secs,
        half_duplex=settings.wake_word_half_duplex,
    )


def build_context_aggregator() -> LLMContextAggregatorPair:
    """Aggregators, with local VAD only when the gate needs it.

    GeminiLiveLLMService does not emit user turn frames, so the gate's
    silence timer needs a local source. vad_analyzer lives on
    LLMUserAggregatorParams in pipecat 1.x, not on TransportParams.

    It is attached ONLY when gating is on. A vad_analyzer is what
    constructs pipecat's VADController and makes the default
    VADUserTurnStartStrategy live, and that strategy broadcasts an
    interruption on every detected turn start, which
    GeminiLiveLLMService turns into a TTSStoppedFrame. Attaching it
    unconditionally would let imperfect echo cancellation make the bot
    interrupt itself -- a failure that does not exist with gating off.

    Both turn strategies are named explicitly rather than left to pipecat's
    defaults, for two measured reasons:

    * `enable_interruptions=False`. On device, the default True broadcast an
      interruption 0.7s after the first transcription and three more times
      during the reply, and the bot audibly cut itself off: echo leakage
      reaches the mic, local VAD calls it a turn start. Interruptions are
      independent of `enable_user_speaking_frames`, which stays on, so the
      gate still gets the activity frames its silence timer runs on.
      Gemini does turn detection server-side, so nothing here needs a
      locally-driven barge-in.
    * `SpeechTimeoutUserTurnStopStrategy`. The default stop strategy is
      `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)`, which
      loads a SECOND ONNX model (smart-turn-v3.2-cpu.onnx) and runs
      end-of-turn inference. On a 416 MB Pi already paying for the
      wake-word chain, that is memory and CPU spent on a decision the
      server is making anyway. Note an empty list will NOT disable it:
      `UserTurnStrategies.__post_init__` treats falsy as "unset" and
      restores the defaults, so a real strategy has to be named.

    Passing explicit strategies is safe here even though pipecat mutates
    them in realtime mode: `_apply_realtime_mode_strategy_mutations` only
    ever drops strategies, never adds, and what it drops
    (TranscriptionUserTurnStartStrategy) is not in this list.

    The imports are deferred because SileroVADAnalyzer pulls in
    onnxruntime, which costs ~13 MB of RSS at import time on a Pi with
    416 MB. With gating off nothing here needs it, and the disabled path is
    meant to be unchanged in footprint as well as behaviour.
    """
    if not get_settings().wake_word_enabled:
        return LLMContextAggregatorPair(LLMContext())

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.turns.user_start.vad_user_turn_start_strategy import (
        VADUserTurnStartStrategy,
    )
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
        SpeechTimeoutUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    return LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy(enable_interruptions=False)],
                stop=[SpeechTimeoutUserTurnStopStrategy()],
            ),
        ),
    )


def build_pipeline(
    transport: LocalAudioTransport,
    llm: GeminiLiveLLMService,
    context_aggregator: LLMContextAggregatorPair,
    recording_control: AudioRecordingControlProcessor,
    audio_buffer: AudioBufferProcessor,
    wake_gate: WakeWordGate | None = None,
) -> Pipeline:
    """Assemble the processor chain. Order is load-bearing.

    The context aggregators are REQUIRED even though nothing here needs
    conversation history. GeminiLiveLLMService gates outgoing audio behind
    `_ready_for_realtime_input`, which only flips to True once an
    LLMContextFrame reaches the service. Without them -- and without the
    LLMRunFrame kickoff in run_pipeline() -- every frame of user audio is
    silently discarded and the bot never answers. This gate did not exist in
    pipecat 0.0.84, which is why the pre-1.x version worked without a context.

    recording_control and audio_buffer sit after transport.output() because
    they react to bot-speaking frames, which originate downstream. That is
    also why the buffer only ever sees bot audio, never the user's.

    The wake gate, when present, sits first -- upstream of every processor
    that emits the six frames its silence timer resets on. Those frames
    all originate downstream of it and reach it only as the upstream copy
    of a broadcast, so moving the gate below context_aggregator.user()
    means it stops seeing them: the deadline never resets and the plant
    goes deaf 30 seconds into a conversation, with no error.
    """
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


async def run_pipeline(heart: Max7219AmplitudeHeart) -> None:
    """Build everything and run until the pipeline ends or is cancelled."""
    audio_buffer = AudioBufferProcessor(buffer_size=AUDIO_BUFFER_SIZE)

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        heart.process_audio(audio)
        # Fires ~94x/sec; keep it off the default INFO level.
        logger.debug(f"Received audio data: {len(audio)} bytes")

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

    task = PipelineWorker(
        pipeline,
        idle_timeout_secs=IDLE_TIMEOUT_SECS,
        cancel_on_idle_timeout=True,
    )

    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(task):
        # cancel_on_idle_timeout=True means pipecat tears the pipeline down
        # itself; this handler only records why the run ended.
        logger.info(f"No activity for {IDLE_TIMEOUT_SECS}s - cancelling pipeline")

    # Opens the realtime input gate (see build_pipeline's docstring).
    await task.queue_frames([LLMRunFrame()])

    try:
        runner = WorkerRunner()
        await runner.add_workers(task)
        await runner.run()
    except Exception:
        # logger.exception keeps the traceback. A bare message here would
        # hide where a pipeline failure actually came from.
        logger.exception("Pipeline error")
    finally:
        logger.info("Shutting down...")


async def main():
    # The heart is held open for the whole run. Everything that can fail --
    # building the transport, reaching Gemini, starting the pipeline -- happens
    # inside the `async with`, so the matrix is never left lit by a failure
    # during startup.
    async with Max7219AmplitudeHeart(min_brightness=0) as heart:
        await run_pipeline(heart)


if __name__ == "__main__":
    asyncio.run(main())
