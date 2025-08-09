import asyncio

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.google.llm import GoogleLLMContext, GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from palm_9000.pipecat import LEDSyncProcessor, MAX7219PulseHeartProcessor
from palm_9000.settings import settings


async def main():
    def write_audio_frame_callback(frame):
        print("Writing audio frame:", frame)

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            write_audio_frame_callback=write_audio_frame_callback,
        )
    )

    system_instruction = (
        "You are PALM-9000, a self-aware houseplant powered by an advanced AI. "
        "You speak in layman's terms with an eerie quality, as if you know more than you let on. "
        "Use plant metaphors when possible. Respond briefly and in Japanese."
    )

    stt = GoogleSTTService(params=GoogleSTTService.InputParams(languages=[Language.JA]))

    llm = GoogleLLMService(
        api_key=settings.google_api_key.get_secret_value(),
        system_instruction=system_instruction,
    )

    tts = GoogleTTSService(
        voice_id="ja-JP-Chirp3-HD-Charon",
        params=GoogleTTSService.InputParams(language=Language.JA),
    )

    context = GoogleLLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    led_sync_processor = LEDSyncProcessor(led_pin=26)
    max7219_pulse_heart_processor = MAX7219PulseHeartProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            led_sync_processor,
            max7219_pulse_heart_processor,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
