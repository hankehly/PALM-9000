import numpy as np
import pvleopard

from palm_9000.settings import settings

# import tempfile
# import scipy.io.wavfile
# import whisper
# whisper_model = whisper.load_model(settings.whisper_model)
# def speech_to_text_whisper(audio_bytes: bytes, language: str) -> str:
#     """
#     Transcribe audio bytes using Whisper.
#     """
#     pcm = np.frombuffer(audio_bytes, dtype=np.int16)
#     with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
#         scipy.io.wavfile.write(tmpfile.name, settings.sample_rate, pcm)
#         # disable fp16 for CPU compatibility
#         decode_options = {"fp16": False, "language": language}
#         result = whisper_model.transcribe(tmpfile.name, **decode_options)
#         return result["text"].strip()


leopard = pvleopard.create(
    access_key=settings.picovoice_access_key.get_secret_value(),
    model_path=settings.pvleopard_model_path,
)

STT_SAMPLE_RATE = leopard.sample_rate


def speech_to_text(audio_bytes: bytes):
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)
    # filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    # scipy.io.wavfile.write(filename, settings.sample_rate, pcm)
    transcript, _ = leopard.process(pcm)
    return transcript.strip()
