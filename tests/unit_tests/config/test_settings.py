import pytest

from src.config.settings import AppSettings


def test_settings_requires_miniflow_config(monkeypatch):
    monkeypatch.delenv("MINIFLOW_CONFIG", raising=False)
    monkeypatch.delenv("MINIFLOW_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MINIFLOW_MAX_AUDIO_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("RELEASE_ID", raising=False)

    with pytest.raises(ValueError):
        AppSettings.from_env()


def test_settings_requires_release_id(monkeypatch):
    monkeypatch.setenv("MINIFLOW_CONFIG", "configs/baseline.yml")
    monkeypatch.delenv("RELEASE_ID", raising=False)

    with pytest.raises(ValueError):
        AppSettings.from_env()


# TODO: this test can have another case we can set timeout and max audio upload
def test_settings_defaults_with_required_config(monkeypatch):
    monkeypatch.setenv("MINIFLOW_CONFIG", "configs/baseline.yml")
    monkeypatch.setenv("RELEASE_ID", "dev-pseudo")
    settings = AppSettings.from_env()
    assert settings.miniflow_config == "configs/baseline.yml"
    assert settings.release_id == "dev-pseudo"
    assert settings.miniflow_request_timeout_seconds > 0
    assert settings.miniflow_max_audio_upload_bytes > 0


def test_resolve_config_path_relative():
    settings = AppSettings(
        miniflow_config="configs/baseline.yml",
        release_id="test-release",
    )
    resolved = settings.resolve_config_path()
    assert resolved.is_absolute()
    assert resolved.name == "baseline.yml"


def test_resolve_config_path_missing_file_raises():
    settings = AppSettings(
        miniflow_config="configs/does_not_exist.yml",
        release_id="test-release",
    )

    with pytest.raises(FileNotFoundError):
        settings.resolve_config_path()


def test_release_id_read_from_env(monkeypatch):
    monkeypatch.setenv("MINIFLOW_CONFIG", "configs/baseline.yml")
    monkeypatch.setenv("RELEASE_ID", "release-test")

    settings = AppSettings.from_env()

    assert settings.miniflow_config == "configs/baseline.yml"
    assert settings.release_id == "release-test"


def test_env_var_overrides_defaults(monkeypatch):
    monkeypatch.setenv("MINIFLOW_CONFIG", "configs/baseline.yml")
    monkeypatch.setenv("RELEASE_ID", "override-release")

    settings = AppSettings.from_env()

    assert settings.miniflow_config == "configs/baseline.yml"
    assert settings.release_id == "override-release"
