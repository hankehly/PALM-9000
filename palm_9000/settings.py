from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_api_key: SecretStr
    google_multimodal_live_voice_id: str = "Puck"
    gemini_live_model: str = "models/gemini-3.8-live"

    # Wake-word gating. Disabled by default: with it off the app behaves
    # exactly as before and streams audio continuously while running.
    wake_word_enabled: bool = False
    wake_word_model_path: str = "models/wakeword/hey_livekit.onnx"
    wake_word_threshold: float = 0.5
    wake_silence_timeout_secs: float = 30.0
    wake_word_hop_samples: int = 1280

    # Only needed by the cascaded GoogleSTTService / GoogleTTSService path,
    # not by the Gemini Live pipeline in main.py.
    google_cloud_project: str | None = None

    # Legacy settings, still referenced by the notebooks.
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once, on first use.

    Constructing Settings() at import time made importing anything in this
    package fail without a GOOGLE_API_KEY, which is why the test suite and CI
    both have to set a placeholder. Deferring it means import and
    configuration are independent concerns.
    """
    return Settings()


def __getattr__(name: str):
    """Keep `from palm_9000.settings import settings` working, but lazily.

    Several notebooks import the module-level name. PEP 562 lets us serve it
    on first access instead of at import time, so the old spelling keeps
    working without reintroducing the import-time dependency on config.
    """
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
