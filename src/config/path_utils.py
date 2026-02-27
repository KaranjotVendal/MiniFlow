from pathlib import Path


def resolve_path(path_value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve path string/path object into an absolute path.

    Relative paths are resolved against `base_dir` when provided, otherwise
    against current working directory.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path.cwd() if base_dir is None else base_dir
    return (root / path).resolve()


def resolve_path_relative_to_file(path_value: str | Path, file_path: Path) -> Path:
    """Resolve a path relative to the parent directory of `file_path`."""
    return resolve_path(path_value, base_dir=file_path.parent)

