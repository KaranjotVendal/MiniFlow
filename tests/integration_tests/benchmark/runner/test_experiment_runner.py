"""
Integration tests for ExperimentRunner.

These tests verify the full experiment flow including:
- Configuration loading and validation
- Storage operations
"""

from pathlib import Path

import pytest
import yaml

from src.benchmark.runner.experiment_runner import ExperimentRunner
from src.benchmark.storage.jsonl_storage import JSONLStorage


class TestExperimentRunnerIntegration:
    """Integration tests for ExperimentRunner initialization and storage."""

    @pytest.fixture
    def temp_benchmark_dir(self, tmp_path: Path) -> Path:
        """Temporary benchmark output root for this test module."""
        output_dir = tmp_path / "benchmark_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def metrics_config(self, tmp_path: Path) -> Path:
        """Create a metrics config file."""
        config = {
            "enabled": ["timing", "hardware_basic", "quality", "tokens", "model_lifecycle"],
            "configurations": {
                "timing": {"stages": ["asr", "llm", "tts", "pipeline"]},
                "hardware_basic": {"device": 0, "track_power": False},
                "quality": {"evaluators": ["wer"]},
                "tokens": {"track_ttft": False},
                "model_lifecycle": None,
            },
        }
        metrics_path = tmp_path / "metrics.yml"
        with open(metrics_path, "w") as f:
            yaml.dump(config, f)
        return metrics_path

    @pytest.fixture
    def experiment_config(self, tmp_path: Path, metrics_config: Path) -> dict:
        """Create a full experiment config."""
        return {
            "experiment": {"name": "integration_test", "description": "Integration test"},
            "dataset": {"name": "globe", "split": "test", "num_samples": 2, "warmup_samples": 1},
            "metrics": str(metrics_config),
            "asr": {"model_id": "openai/whisper-small", "model_name": "whisper-small"},
            "llm": {
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "model_name": "Qwen2.5-3B",
                "quantization": {
                    "enabled": True,
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "use_double_quant": True,
                    "compute_dtype": "bf16",
                },
                "kv_cache": False,
                "max_new_tokens": 50,
            },
            "tts": {
                "model_name": "xtts",
                "model_id": "tts_models/multilingual/multi-dataset/xtts_v2",
                "speaker": "Gracie Wise",
                "language": "en",
            },
            "benchmark": {"enable_streaming_audio": False, "save_processed_samples": False},
        }

    def test_runner_initialization(self, experiment_config, temp_benchmark_dir):
        """Test that runner initializes correctly with full config."""
        runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

        assert runner.experiment == "integration_test"
        assert runner.output_dir.exists()
        assert "timing" in runner.metrics
        assert "hardware_basic" in runner.metrics
        assert "quality" in runner.metrics

    def test_runner_creates_output_directory(self, experiment_config, temp_benchmark_dir):
        """Test that runner creates the output directory structure."""
        runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

        # Check output dir exists - the runner creates temp_benchmark_dir/integration_test/<run_id>
        assert temp_benchmark_dir.exists()
        assert "integration_test" in str(runner.output_dir)

    def test_storage_initialization(self, experiment_config, temp_benchmark_dir):
        """Test that storage is properly initialized."""
        runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

        assert runner.storage is not None
        assert isinstance(runner.storage, JSONLStorage)

        # Check config was saved
        saved_config = runner.output_dir / "config.json"
        assert saved_config.exists()

    def test_metrics_loaded_from_config(self, experiment_config, temp_benchmark_dir):
        """Test that metrics are correctly loaded from config."""
        runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

        # Verify all expected metrics are loaded
        expected_metrics = {"timing", "hardware_basic", "quality", "tokens", "model_lifecycle"}
        assert set(runner.metrics.keys()) == expected_metrics

    def test_config_persisted(self, experiment_config, temp_benchmark_dir):
        """Test that experiment config is saved to disk."""
        runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

        # Load saved config
        saved_config = runner.storage.load_config()
        assert saved_config is not None
        assert saved_config["experiment"]["name"] == "integration_test"
