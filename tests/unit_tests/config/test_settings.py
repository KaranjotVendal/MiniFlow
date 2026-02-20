from pathlib import Path

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


def test_resolve_metrics_path_with_base_dir():
    settings = AppSettings(miniflow_metrics_config="metrics.yml")
    resolved = settings.resolve_metrics_path(base_dir=Path("configs"))
    assert resolved is not None
    assert resolved.is_absolute()
    assert resolved.name == "metrics.yml"
