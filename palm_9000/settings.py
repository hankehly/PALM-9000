from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    picovoice_access_key: SecretStr
    porcupine_keyword: str
    porcupine_keyword_path: str
    porcupine_model_path: str
    pvleopard_model_path: str

    whisper_model: str = "base"

    input_device: int = 1
    sample_rate: int = 44100
    silence_timeout: float = 1.0  # seconds of silence to trigger stop
    vad_mode: int = 3  # 0-3: 0 is least aggressive about filtering out non-speech

    google_api_key: SecretStr
    google_tts_voice_name: str = "Enceladus"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
