from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.config.load_config import load_yaml_config


DEFAULT_CONFIG_PATH = "configs/3_TTS-to-vibevoice.yml"
DEFAULT_ENV_PROFILE_DIR = "configs/env"
DEFAULT_RELEASE_ID = "dev"


def _env() -> dict:
    import os

    data = {
        "miniflow_env": os.getenv("MINIFLOW_ENV", "dev"),
        "miniflow_config": os.getenv("MINIFLOW_CONFIG"),
        "miniflow_metrics_config": os.getenv("MINIFLOW_METRICS_CONFIG", None),
        "miniflow_request_timeout_seconds": os.getenv(
            "MINIFLOW_REQUEST_TIMEOUT_SECONDS", "120"
        ),
        "miniflow_max_audio_upload_bytes": os.getenv(
            "MINIFLOW_MAX_AUDIO_UPLOAD_BYTES", str(10 * 1024 * 1024)
        ),
        "release_id": os.getenv("RELEASE_ID"),
    }
    return data


class AppSettings(BaseModel):
    miniflow_env: str = Field(default="dev")
    miniflow_config: str = Field(default=DEFAULT_CONFIG_PATH)
    miniflow_metrics_config: str | None = Field(default=None)
    miniflow_request_timeout_seconds: float = Field(default=120.0)
    miniflow_max_audio_upload_bytes: int = Field(default=10 * 1024 * 1024)
    release_id: str = Field(default=DEFAULT_RELEASE_ID)

    @staticmethod
    def _resolve_path(path_value: str, base_dir: Path | None = None) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        root = Path.cwd() if base_dir is None else base_dir
        return (root / path).resolve()

    @classmethod
    def _load_env_profile(cls, miniflow_env: str) -> dict:
        profile_path = cls._resolve_path(
            f"{DEFAULT_ENV_PROFILE_DIR}/{miniflow_env}.yml"
        )
        if not profile_path.exists():
            return {}
        profile_data = load_yaml_config(profile_path)
        return profile_data if isinstance(profile_data, dict) else {}

    @classmethod
    def from_env(cls) -> "AppSettings":
        raw_data = _env()
        validation_data = {
            **raw_data,
            "miniflow_config": raw_data.get("miniflow_config") or DEFAULT_CONFIG_PATH,
            "release_id": raw_data.get("release_id") or DEFAULT_RELEASE_ID,
        }
        try:
            validated_raw = cls.model_validate(validation_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid application settings: {exc}") from exc

        profile_data = cls._load_env_profile(validated_raw.miniflow_env)

        # Precedence: env vars > env profile > defaults.
        merged_data = {
            "miniflow_env": validated_raw.miniflow_env,
            "miniflow_config": (
                raw_data.get("miniflow_config")
                or profile_data.get("config")
                or DEFAULT_CONFIG_PATH
            ),
            "miniflow_metrics_config": (
                raw_data.get("miniflow_metrics_config")
                or profile_data.get("metrics")
            ),
            "miniflow_request_timeout_seconds": validated_raw.miniflow_request_timeout_seconds,
            "miniflow_max_audio_upload_bytes": validated_raw.miniflow_max_audio_upload_bytes,
            "release_id": (
                raw_data.get("release_id")
                or profile_data.get("release_id")
                or DEFAULT_RELEASE_ID
            ),
        }

        try:
            return cls.model_validate(merged_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid merged application settings: {exc}") from exc

    def resolve_config_path(self) -> Path:
        return self._resolve_path(self.miniflow_config)

    def resolve_metrics_path(self, base_dir: Path | None = None) -> Path | None:
        if self.miniflow_metrics_config is None:
            return None
        return self._resolve_path(self.miniflow_metrics_config, base_dir=base_dir)
