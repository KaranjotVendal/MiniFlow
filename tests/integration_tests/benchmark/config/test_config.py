import yaml
import pytest

from src.benchmark.config.loader import load_benchmark_config
from src.benchmark.config.validation import (
    MetricsConfig,
    BenchmarkConfig,
)

class TestConfigIntegration:
    """Integration tests for config loading and validation."""

    def test_load_and_validate_roundtrip(self, tmp_path):
        """Load config, create BenchmarkConfig, convert back to dict."""
        yml_file = tmp_path / "temp_config.yml"
        with open(yml_file, "w") as f:
            yaml.dump(
                {
                    "experiment": {"name": "roundtrip_test"},
                    "dataset": {"num_samples": 15, "warmup_samples": 2},
                    "benchmark": {
                        "metrics": {
                            "enabled": ["timing"],
                            "configurations": {"timing": {"stages": ["asr"]}},
                        }
                    },
                },
                f,
            )

        loaded = load_benchmark_config(yml_file)
        benchmark_config = BenchmarkConfig(
            experiment_name=loaded["experiment"]["name"],
            num_samples=loaded["dataset"]["num_samples"],
            warmup_samples=loaded["dataset"]["warmup_samples"],
            metrics=MetricsConfig(
                enabled=loaded["benchmark"]["metrics"]["enabled"],
                configurations=loaded["benchmark"]["metrics"]["configurations"],
            ),
        )

        result = benchmark_config.to_dict()
        assert result["experiment_name"] == "roundtrip_test"
        assert result["num_samples"] == 15
        assert result["metrics"]["enabled"] == ["timing"]

    def test_missing_optional_fields_use_defaults(self, tmp_path):
        yml_file = tmp_path / "temp_config.yml"
        with open(yml_file, "w") as f:
            yaml.dump(
                {
                    "experiment": {"name": "defaults_test"},
                    "dataset": {"num_samples": 10},
                },
                f,
            )
        loaded = load_benchmark_config(yml_file)
        config = BenchmarkConfig(
            experiment_name=loaded["experiment"]["name"],
            num_samples=loaded["dataset"]["num_samples"],
        )
        assert config.warmup_samples == 3  # default
        assert config.output_dir == "./Benchmark"  # default
