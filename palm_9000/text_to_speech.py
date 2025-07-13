import io
import wave

import numpy as np
import pyaudio
from google import genai
from google.genai import types

from palm_9000.settings import settings


def play_audio(audio: bytes, sample_rate=16000, volume=1.0):
    """
    volume is a multiplier for the audio volume, so 1.0 is normal volume, 2.0 is double the volume, etc.
    Don't set it too high (>=3) or it will clip and distort the audio.
    """
    # Convert raw bytes to NumPy array of int16 samples
    pcm = np.frombuffer(audio, dtype=np.int16)

    # Apply volume gain (with clipping to int16 range)
    amplified = np.clip(pcm * volume, -32768, 32767).astype(np.int16)

    # Convert back to bytes
    amplified_bytes = amplified.tobytes()

    # Wrap PCM data in WAV headers in-memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(amplified_bytes)

    buffer.seek(0)

    # Play audio with PyAudio
    wf = wave.open(buffer, "rb")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    pa.terminate()


def generate_google_gemini_audio(text: str) -> bytes:
    """
    Generates speech from text using Google Gemini TTS.
    Returns the audio data as bytes.

    Voice options can be found here:
    https://ai.google.dev/gemini-api/docs/speech-generation?_gl=1*16uz4h8*_up*MQ..*_ga*MTk1NjU5MzM4Ny4xNzUxNzc2MTE0*_ga_P1DBVKWT6V*czE3NTE3NzYxMTMkbzEkZzAkdDE3NTE3NzYxMTMkajYwJGwwJGg3MjIzMjgwMDY.#voices

    Test it here:
    https://aistudio.google.com/generate-speech
    """
    client = genai.Client(api_key=settings.google_api_key.get_secret_value())
    prompt = f"Say quickly with an eerie calm: {text}"
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=settings.google_tts_voice_name,
                )
            )
        ),
    )
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-preview-tts", contents=prompt, config=config
    )
    result = response.candidates[0].content.parts[0].inline_data.data
    return result
