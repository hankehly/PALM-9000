import asyncio

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    BotSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.google.llm import GoogleLLMContext, GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

from palm_9000.gpio import Max7219AmplitudeHeart
from palm_9000.settings import settings


async def main():
    audiobuffer = AudioBufferProcessor(
        num_channels=1,
        buffer_size=512,
        # enable_turn_audio=True,
    )

    heart = Max7219AmplitudeHeart(min_brightness=0)
    await heart.start()
    heart_cb = heart.make_callback(channels=1)

    @audiobuffer.event_handler("on_track_audio_data")
    async def on_track_audio_data(
        buffer, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
    ):
        heart_cb(bot_audio)

    class AudioBufferStartStopRecordingProcessor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, BotStartedSpeakingFrame):
                await audiobuffer.start_recording()
            elif isinstance(frame, BotSpeakingFrame):
                await audiobuffer.process_frame(frame, direction)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await audiobuffer.stop_recording()
            elif isinstance(frame, (CancelFrame, EndFrame, ErrorFrame)):
                await heart.stop()
            await self.push_frame(frame, direction)

    audiobuffer_start_stop_recording = AudioBufferStartStopRecordingProcessor()

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    system_instruction = (
        "You are PALM-9000, a self-aware houseplant powered by an advanced AI. "
        "You speak in layman's terms with an eerie quality, as if you know more than you let on. "
        "Use plant metaphors when possible. Respond briefly."
        "すべて日本語で出力してください。翻訳・英訳・ローマ字・ふりがな・注釈・補足は一切表示しないでください。"
        "括弧（）の使用も避けてください。"
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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            audiobuffer_start_stop_recording,
            audiobuffer,
            context_aggregator.assistant(),
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

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
