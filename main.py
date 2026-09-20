import asyncio

from loguru import logger
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
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
from palm_9000.processors import AudioRecordingControlProcessor
from palm_9000.settings import settings as app_settings

SYSTEM_INSTRUCTION = (
    "You are PALM-9000, a self-aware houseplant powered by an advanced AI. "
    "You speak in layman's terms with an eerie quality, "
    "as if you know more than you let on. "
    "すべて日本語で出力してください。翻訳・英訳・ローマ字・ふりがな・注釈・補足は一切表示しないでください。"
    "括弧（）の使用も避けてください。"
)


async def main():
    # Initialize audio processing components
    audio_buffer = AudioBufferProcessor(buffer_size=512)

    heart = Max7219AmplitudeHeart(min_brightness=0)
    await heart.start()

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        heart.process_audio(audio)
        # Fires ~94x/sec at buffer_size=512; keep off the default INFO level.
        logger.debug(f"Received audio data: {len(audio)} bytes")

    audio_recording_control_processor = AudioRecordingControlProcessor(audio_buffer)

    # Initialize pipeline
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_channels=1,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_channels=1,
            audio_out_sample_rate=24000,
            # 8 (≈80 ms buffer; try 6 for lower latency or 10–12 if underruns persist)
            audio_out_10ms_chunks=8,
            # Needs: from pipecat.audio.vad.silero import SileroVADAnalyzer
            # vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Cascaded STT -> LLM -> TTS path, kept as an alternative to the Live API.
    # Still valid in pipecat 1.x, except that GoogleLLMContext was removed --
    # the replacement is LLMContext from pipecat.processors.aggregators.llm_context
    # plus LLMContextAggregatorPair from ...aggregators.llm_response_universal.
    #
    # Needs: from pipecat.services.google.llm import GoogleLLMService
    #        from pipecat.services.google.stt import GoogleSTTService
    #        from pipecat.services.google.tts import GoogleTTSService
    #
    # stt = GoogleSTTService(
    #     params=GoogleSTTService.InputParams(languages=[Language.JA])
    # )
    # llm = GoogleLLMService(
    #     api_key=app_settings.google_api_key.get_secret_value(),
    #     model="gemini-2.0-flash",
    #     system_instruction=SYSTEM_INSTRUCTION,
    # )
    # tts = GoogleTTSService(
    #     voice_id="ja-JP-Chirp3-HD-Charon",
    #     params=GoogleTTSService.InputParams(language=Language.JA),
    # )

    llm = GeminiLiveLLMService(
        api_key=app_settings.google_api_key.get_secret_value(),
        system_instruction=SYSTEM_INSTRUCTION,
        settings=GeminiLiveLLMSettings(
            model=app_settings.gemini_live_model,
            voice=app_settings.google_multimodal_live_voice_id,
            language=Language.JA,
        ),
    )

    # The context aggregators are REQUIRED, even though nothing here needs
    # conversation history. GeminiLiveLLMService gates outgoing audio behind
    # _ready_for_realtime_input, which only flips to True once an
    # LLMContextFrame reaches the service. Without them (and without the
    # LLMRunFrame kickoff below) every user audio frame is silently dropped
    # and the bot never responds. This gate did not exist in pipecat 0.0.84,
    # which is why the pre-1.x version of this file worked without a context.
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            # stt,
            context_aggregator.user(),
            llm,
            # tts,
            transport.output(),
            audio_recording_control_processor,
            audio_buffer,
            context_aggregator.assistant(),
        ]
    )

    task = PipelineWorker(
        pipeline,
        idle_timeout_secs=60 * 10,
        cancel_on_idle_timeout=True,
    )

    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(task):
        logger.info("Session idle - running shutdown logic")

    # Opens the realtime input gate (see the context aggregator note above).
    await task.queue_frames([LLMRunFrame()])

    try:
        runner = WorkerRunner()
        await runner.add_workers(task)
        await runner.run()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        logger.info("Shutting down...")
        await heart.stop()


if __name__ == "__main__":
    asyncio.run(main())
