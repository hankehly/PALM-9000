import pytest
from pydantic import SecretStr, ValidationError

from palm_9000.settings import Settings


def test_google_api_key_is_required(env):
    """google_api_key has no default, so construction must fail without it."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_api_key_is_wrapped_in_secretstr(env):
    settings = Settings(_env_file=None, google_api_key="abc123")
    assert isinstance(settings.google_api_key, SecretStr)
    assert settings.google_api_key.get_secret_value() == "abc123"
    # SecretStr must not leak the value when printed.
    assert "abc123" not in repr(settings.google_api_key)


def test_defaults(env):
    settings = Settings(_env_file=None, google_api_key="k")
    assert settings.google_multimodal_live_voice_id == "Puck"
    assert settings.gemini_live_model == "models/gemini-3.8-live"
    assert settings.whisper_model == "base"
    assert settings.input_device == 1
    assert settings.sample_rate == 44100
    assert settings.silence_timeout == 1.0
    assert settings.vad_mode == 3


def test_google_cloud_project_is_optional(env):
    """Only the cascaded STT/TTS path needs it, so it must not block startup."""
    settings = Settings(_env_file=None, google_api_key="k")
    assert settings.google_cloud_project is None


@pytest.mark.parametrize(
    "field",
    [
        "picovoice_access_key",
        "porcupine_keyword",
        "porcupine_keyword_path",
        "porcupine_model_path",
        "pvleopard_model_path",
    ],
)
def test_legacy_fields_default_to_none(env, field):
    settings = Settings(_env_file=None, google_api_key="k")
    assert getattr(settings, field) is None


def test_env_vars_override_defaults(env):
    env.setenv("GOOGLE_API_KEY", "from-env")
    env.setenv("GEMINI_LIVE_MODEL", "models/other-model")
    env.setenv("GOOGLE_MULTIMODAL_LIVE_VOICE_ID", "Charon")
    env.setenv("SAMPLE_RATE", "16000")
    env.setenv("SILENCE_TIMEOUT", "2.5")

    settings = Settings(_env_file=None)

    assert settings.google_api_key.get_secret_value() == "from-env"
    assert settings.gemini_live_model == "models/other-model"
    assert settings.google_multimodal_live_voice_id == "Charon"
    assert settings.sample_rate == 16000
    assert settings.silence_timeout == 2.5


def test_unknown_env_vars_are_ignored(env):
    """model_config sets extra='ignore'; a stray var must not raise."""
    env.setenv("GOOGLE_API_KEY", "k")
    env.setenv("SOME_UNRELATED_VARIABLE", "whatever")
    assert Settings(_env_file=None).google_api_key.get_secret_value() == "k"


def test_reads_from_env_file(env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_API_KEY=from-file\nGEMINI_LIVE_MODEL=models/from-file\n"
    )

    settings = Settings(_env_file=str(env_file))

    assert settings.google_api_key.get_secret_value() == "from-file"
    assert settings.gemini_live_model == "models/from-file"


def test_env_file_supports_export_prefix(env, tmp_path):
    """The real .env on the Pi uses `export KEY=value` shell syntax."""
    env_file = tmp_path / ".env"
    env_file.write_text("export GOOGLE_API_KEY=exported-value\n")

    settings = Settings(_env_file=str(env_file))

    assert settings.google_api_key.get_secret_value() == "exported-value"


def test_module_level_singleton_is_importable(env, monkeypatch):
    """`from palm_9000.settings import settings` must work at import time."""
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    import importlib

    import palm_9000.settings as settings_module

    reloaded = importlib.reload(settings_module)
    assert reloaded.settings.google_api_key.get_secret_value() == "k"


class TestWakeWordSettings:
    def test_disabled_by_default(self, env):
        """Merging the feature must not change runtime behaviour."""
        settings = Settings(_env_file=None, google_api_key="k")
        assert settings.wake_word_enabled is False

    def test_defaults(self, env):
        settings = Settings(_env_file=None, google_api_key="k")
        assert settings.wake_word_model_path == "models/wakeword/hey_livekit.onnx"
        assert settings.wake_word_threshold == 0.5
        assert settings.wake_silence_timeout_secs == 30.0

    def test_env_overrides(self, env):
        env.setenv("GOOGLE_API_KEY", "k")
        env.setenv("WAKE_WORD_ENABLED", "true")
        env.setenv("WAKE_WORD_MODEL_PATH", "models/wakeword/other.onnx")
        env.setenv("WAKE_WORD_THRESHOLD", "0.8")
        env.setenv("WAKE_SILENCE_TIMEOUT_SECS", "12.5")

        settings = Settings(_env_file=None)

        assert settings.wake_word_enabled is True
        assert settings.wake_word_model_path == "models/wakeword/other.onnx"
        assert settings.wake_word_threshold == 0.8
        assert settings.wake_silence_timeout_secs == 12.5
