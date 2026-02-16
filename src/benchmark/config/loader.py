import yaml
from pathlib import Path


def load_benchmark_config(config_path: str | Path) -> dict:
    """Load a benchmark configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)
