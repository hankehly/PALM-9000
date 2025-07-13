from pydantic import BaseSettings, SecretStr
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    picovoice_access_key: SecretStr
    porcupine_keywords: list[str] = ["computer"]
    porcupine_keyword_path: str
    porcupine_model_path: str
    pvleopard_model_path: str

    whisper_model: str = "base"

    pvrecorder_input_device: int = 0
    sounddevice_input_device: int | None = 1  # 1
    sample_rate: int = 44100
    frame_duration_ms: int = 30
    silence_timeout: float = 1.0  # seconds of silence to trigger stop
    vad_mode: int = 3  # 0-3: 0 is least aggressive about filtering out non-speech

    google_api_key: SecretStr
    google_tts_voice_name: str = "Enceladus"

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )


settings = Settings()
