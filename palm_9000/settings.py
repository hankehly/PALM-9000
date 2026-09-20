from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: SecretStr
    google_multimodal_live_voice_id: str = "Puck"
    gemini_live_model: str = "models/gemini-3.8-live"

    # Only needed by the cascaded GoogleSTTService / GoogleTTSService path,
    # not by the Gemini Live pipeline in main.py.
    google_cloud_project: str | None = None

    # Legacy settings
    picovoice_access_key: SecretStr | None = None
    porcupine_keyword: str | None = None
    porcupine_keyword_path: str | None = None
    porcupine_model_path: str | None = None
    pvleopard_model_path: str | None = None
    whisper_model: str = "base"
    input_device: int = 1
    sample_rate: int = 44100
    silence_timeout: float = 1.0  # seconds of silence to trigger stop
    vad_mode: int = 3  # 0-3: 0 is least aggressive about filtering out non-speech

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
