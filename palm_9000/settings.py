from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: SecretStr
    google_multimodal_live_voice_id: str = "Puck"
    google_cloud_project: str

    # Legacy settings
    picovoice_access_key: SecretStr = None
    porcupine_keyword: str = None
    porcupine_keyword_path: str = None
    porcupine_model_path: str = None
    pvleopard_model_path: str = None
    whisper_model: str = "base"
    input_device: int = 1
    sample_rate: int = 44100
    silence_timeout: float = 1.0  # seconds of silence to trigger stop
    vad_mode: int = 3  # 0-3: 0 is least aggressive about filtering out non-speech

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
