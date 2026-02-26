from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


def _env() -> dict:
    import os

    data = {
        "miniflow_config": os.getenv("MINIFLOW_CONFIG"),
        "miniflow_request_timeout_seconds": os.getenv("MINIFLOW_REQUEST_TIMEOUT_SECONDS"),
        "miniflow_max_audio_upload_bytes": os.getenv("MINIFLOW_MAX_AUDIO_UPLOAD_BYTES"),
        "release_id": os.getenv("RELEASE_ID"),
    }
    return data


class AppSettings(BaseModel):
    miniflow_config: str
    miniflow_request_timeout_seconds: float = Field(default=120.0)
    miniflow_max_audio_upload_bytes: int = Field(default=10 * 1024 * 1024)
    release_id: str

    @staticmethod
    def _resolve_path(path_value: str, base_dir: Path | None = None) -> Path:
        """Resolve a path string into an absolute filesystem path.

        If `path_value` is relative, it is resolved against `base_dir` when provided,
        otherwise against the current working directory.

        Example:
            path_value="configs/baseline.yml", cwd="/repo"
            -> "/repo/configs/baseline.yml"
        """
        path = Path(path_value)
        if path.is_absolute():
            return path
        root = Path.cwd() if base_dir is None else base_dir
        return (root / path).resolve()

    @classmethod
    def from_env(cls) -> "AppSettings":
        raw_data = _env()
        sanitized_raw = {key: value for key, value in raw_data.items() if value is not None}
        try:
            return cls.model_validate(sanitized_raw)
        except ValidationError as exc:
            raise ValueError(f"Invalid application settings: {exc}") from exc

    def resolve_config_path(self) -> Path:
        """Return absolute path for `miniflow_config`.

        Example:
            MINIFLOW_CONFIG="configs/3_TTS-to-vibevoice.yml"
            cwd="/repo"
            -> "/repo/configs/3_TTS-to-vibevoice.yml"
        """
        resolved_path = self._resolve_path(self.miniflow_config)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_path}")
        return resolved_path
