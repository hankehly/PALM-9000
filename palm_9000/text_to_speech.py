from google import genai
from google.genai import types

from palm_9000.settings import settings


def text_to_speech(text: str) -> bytes:
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
