import yaml
import pytest

from src.benchmark.config.loader import load_benchmark_config


class TestLoadBenchmarkConfig:
    def test_load_valid_config(self, tmp_path):
        yml_file = tmp_path / "temp_config.yml"
        with open(yml_file, "w") as f:
            yaml.dump(
                {
                    "experiment": {"name": "test_exp"},
                    "dataset": {"num_samples": 10},
                },
                f,
            )
        config = load_benchmark_config(yml_file)
        assert config["experiment"]["name"] == "test_exp"
        assert config["dataset"]["num_samples"] == 10

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_benchmark_config("/nonexistent/path/config.yml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        yml_file = tmp_path / "temp_config.yml"
        with open(yml_file, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_benchmark_config(yml_file)

    def test_load_existing_baseline_config(self):
        config = load_benchmark_config("configs/baseline.yml")
        assert config["experiment"]["name"] == "baseline"
        assert "dataset" in config
        assert "asr" in config
        assert "llm" in config
        assert "tts" in config
