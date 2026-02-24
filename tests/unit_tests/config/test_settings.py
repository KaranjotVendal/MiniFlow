from src.config.settings import AppSettings


def test_settings_defaults():
    settings = AppSettings.from_env()
    assert settings.miniflow_env in {"dev", "staging", "prod"}
    assert settings.miniflow_request_timeout_seconds > 0
    assert settings.miniflow_max_audio_upload_bytes > 0


def test_resolve_config_path_relative():
    settings = AppSettings(miniflow_config="configs/baseline.yml")
    resolved = settings.resolve_config_path()
    assert resolved.is_absolute()
    assert resolved.name == "baseline.yml"


def test_profile_config_used_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("MINIFLOW_CONFIG", raising=False)
    monkeypatch.delenv("RELEASE_ID", raising=False)
    monkeypatch.setenv("MINIFLOW_ENV", "prod")

    settings = AppSettings.from_env()

    assert settings.miniflow_config == "configs/3_TTS-to-vibevoice.yml"
    assert settings.release_id == "prod"


def test_env_var_overrides_profile(monkeypatch):
    monkeypatch.setenv("MINIFLOW_ENV", "prod")
    monkeypatch.setenv("MINIFLOW_CONFIG", "configs/baseline.yml")
    monkeypatch.setenv("RELEASE_ID", "override-release")

    settings = AppSettings.from_env()

    assert settings.miniflow_config == "configs/baseline.yml"
    assert settings.release_id == "override-release"
