import yaml
from pathlib import Path


def load_yaml_config(path: str | Path) -> dict:
    # TODO: make a dataclass for config.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(config, dict):
        raise ValueError(
            f"Config must be a dictionary, got {type(config).__name__}: {path}"
        )

    return config
