import tempfile
import subprocess

from google import genai
from google.genai import types
from pydantic import BaseModel
import langdetect
import scipy.io.wavfile

from palm_9000.settings import settings


class TextToSpeechResult(BaseModel):
    audio_data: bytes
    sample_rate: int


def text_to_speech_gemini_api(text: str) -> TextToSpeechResult:
    """
    Generates speech from text using Google Gemini TTS.
    Returns the audio data as bytes.

    Voice options can be found here:
    https://ai.google.dev/gemini-api/docs/speech-generation?_gl=1*16uz4h8*_up*MQ..*_ga*MTk1NjU5MzM4Ny4xNzUxNzc2MTE0*_ga_P1DBVKWT6V*czE3NTE3NzYxMTMkbzEkZzAkdDE3NTE3NzYxMTMkajYwJGwwJGg3MjIzMjgwMDY.#voices

    Test it here:
    https://aistudio.google.com/generate-speech

    The output uses a sample rate of 24kHz.

    Note: This should be replaced with the Live API
        > The TTS capability differs from speech generation provided through the Live API, which is designed for interactive,
        > unstructured audio, and multimodal inputs and outputs. While the Live API excels in dynamic conversational contexts,
        > TTS through the Gemini API is tailored for scenarios that require exact text recitation with fine-grained control over
        > style and sound, such as podcast or audiobook generation.

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
    return TextToSpeechResult(audio_data=result, sample_rate=24000)


def text_to_speech_offline(text: str) -> TextToSpeechResult:
    lang = langdetect.detect(text)

    if lang not in ["ja", "en"]:
        raise NotImplementedError(f"Language '{lang}' not supported.")

    with (
        tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".txt") as txt_file,
        tempfile.NamedTemporaryFile(suffix=".wav") as wav_file,
    ):
        txt_file.write(text)
        txt_file.flush()

        if lang == "ja":
            # Requires downloading the HTS voice package:
            # sudo apt install open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001
            subprocess.run(
                [
                    "open_jtalk",
                    "-x",
                    "/var/lib/mecab/dic/open-jtalk/naist-jdic",
                    "-m",
                    "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice",
                    "-ow",
                    wav_file.name,
                    txt_file.name,
                ],
                check=True,
            )
        else:
            # Requires installing espeak:
            # sudo apt install espeak
            subprocess.run(["espeak", "-w", wav_file.name, text], check=True)

        sample_rate, audio_data = scipy.io.wavfile.read(wav_file.name)
        return TextToSpeechResult(
            audio_data=audio_data.tobytes(), sample_rate=sample_rate
        )
