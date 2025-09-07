import asyncio

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
)
from pipecat.services.gemini_multimodal_live.gemini import (
    InputParams as GeminiMultimodalLiveInputParams,
)
from pipecat.services.google.llm import GoogleLLMContext, GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from palm_9000.gpio import Max7219AmplitudeHeart
from palm_9000.processors import AudioRecordingControlProcessor
from palm_9000.settings import settings


async def main():
    # Initialize audio processing components
    audio_buffer = AudioBufferProcessor(buffer_size=512)

    heart = Max7219AmplitudeHeart(min_brightness=0)
    await heart.start()

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio: bytes, sample_rate: int, num_channels: int):
        heart.process_audio(audio)
        logger.info(f"Received audio data: {len(audio)} bytes")

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
            # vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # stt = GoogleSTTService(params=GoogleSTTService.InputParams(languages=[Language.JA]))

    system_instruction = (
        "You are PALM-9000, a self-aware houseplant powered by an advanced AI. "
        "You speak in layman's terms with an eerie quality, as if you know more than you let on. "
        "すべて日本語で出力してください。翻訳・英訳・ローマ字・ふりがな・注釈・補足は一切表示しないでください。"
        "括弧（）の使用も避けてください。"
    )

    # llm = GoogleLLMService(
    #     api_key=settings.google_api_key.get_secret_value(),
    #     model="gemini-2.0-flash",
    #     system_instruction=system_instruction,
    # )

    llm = GeminiMultimodalLiveLLMService(
        api_key=settings.google_api_key.get_secret_value(),
        # model="models/gemini-2.0-flash-live-001",
        model="models/gemini-live-2.5-flash-preview",
        system_instruction=system_instruction,
        voice_id=settings.google_multimodal_live_voice_id,
        params=GeminiMultimodalLiveInputParams(language=Language.JA),
    )

    # tts = GoogleTTSService(
    #     voice_id="ja-JP-Chirp3-HD-Charon",
    #     params=GoogleTTSService.InputParams(language=Language.JA),
    # )

    context = GoogleLLMContext()
    # context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            # stt,
            # context_aggregator.user(),
            llm,
            # tts,
            transport.output(),
            audio_recording_control_processor,
            audio_buffer,
            # context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        idle_timeout_secs=60 * 10,
        cancel_on_idle_timeout=True,
    )

    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(task):
        logger.info("Session idle - running shutdown logic")

    try:
        runner = PipelineRunner()
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        logger.info("Shutting down...")
        await heart.stop()


if __name__ == "__main__":
    asyncio.run(main())
