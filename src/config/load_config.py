import yaml
from pathlib import Path


def load_yaml_config(path: str | Path) -> dict:
    # save config values to a dataclasses for easier usage.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
