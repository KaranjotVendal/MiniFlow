from pathlib import Path

import pytest


@pytest.fixture
def temp_benchmark_dir(tmp_path: Path) -> Path:
    """Temporary benchmark output root used by integration runner tests."""
    output_dir = tmp_path / "benchmark_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

