from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class AppSettings(BaseModel):
    miniflow_env: str = Field(default="dev")
    miniflow_config: str = Field(default="configs/baseline.yml")
    miniflow_metrics_config: str | None = Field(default=None)
    miniflow_request_timeout_seconds: float = Field(default=120.0)
    miniflow_max_audio_upload_bytes: int = Field(default=10 * 1024 * 1024)
    release_id: str = Field(default="dev")

    @classmethod
    def from_env(cls) -> "AppSettings":
        data = {
            "miniflow_env": _env("MINIFLOW_ENV", "dev"),
            "miniflow_config": _env("MINIFLOW_CONFIG", "configs/baseline.yml"),
            "miniflow_metrics_config": _env("MINIFLOW_METRICS_CONFIG", None),
            "miniflow_request_timeout_seconds": _env(
                "MINIFLOW_REQUEST_TIMEOUT_SECONDS", "120"
            ),
            "miniflow_max_audio_upload_bytes": _env(
                "MINIFLOW_MAX_AUDIO_UPLOAD_BYTES", str(10 * 1024 * 1024)
            ),
            "release_id": _env("RELEASE_ID", "dev"),
        }
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid application settings: {exc}") from exc

    def resolve_config_path(self) -> Path:
        path = Path(self.miniflow_config)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def resolve_metrics_path(self, base_dir: Path | None = None) -> Path | None:
        if self.miniflow_metrics_config is None:
            return None
        path = Path(self.miniflow_metrics_config)
        if path.is_absolute():
            return path
        if base_dir is None:
            return (Path.cwd() / path).resolve()
        return (base_dir / path).resolve()


def _env(key: str, default: str | None) -> str | None:
    import os

    return os.getenv(key, default)
