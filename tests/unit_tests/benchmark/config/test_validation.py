import pytest

from src.benchmark.config.validation import (
    MetricConfig,
    MetricsConfig,
    BenchmarkConfig,
)


class TestMetricConfig:
    def test_default_values(self):
        config = MetricConfig()
        assert config.enabled is True
        assert config.parameters == {}

    def test_custom_values(self):
        config = MetricConfig(enabled=False, parameters={"key": "value"})
        assert config.enabled is False
        assert config.parameters == {"key": "value"}

    def test_parameters_default_factory(self):
        config1 = MetricConfig()
        config2 = MetricConfig()
        config1.parameters["test"] = True
        assert "test" not in config2.parameters


class TestMetricsConfig:
    def test_default_values(self):
        config = MetricsConfig()
        assert config.enabled == []
        assert config.configurations == {}

    def test_get_config_existing_metric(self):
        config = MetricsConfig(
            enabled=["timing"],
            configurations={"timing": {"stages": ["asr", "llm"]}},
        )
        result = config.get_config("timing")
        assert result == {"stages": ["asr", "llm"]}

    def test_get_config_nonexistent_metric(self):
        config = MetricsConfig(enabled=["timing"], configurations={})
        result = config.get_config("nonexistent")
        assert result == None

    def test_with_multiple_metrics(self):
        config = MetricsConfig(
            enabled=["timing", "hardware"],
            configurations={
                "timing": {"stages": ["asr"]},
                "hardware": {"device": 0},
            },
        )
        assert len(config.enabled) == 2
        assert config.get_config("timing")["stages"] == ["asr"]
        assert config.get_config("hardware")["device"] == 0


class TestBenchmarkConfig:
    def test_required_fields(self):
        with pytest.raises(TypeError):
            BenchmarkConfig()

    def test_with_required_fields_only(self):
        config = BenchmarkConfig(experiment_name="test")
        assert config.experiment_name == "test"
        assert config.num_samples == 20  # default
        assert config.warmup_samples == 3  # default

    def test_all_fields(self):
        config = BenchmarkConfig(
            experiment_name="full_test",
            num_samples=50,
            warmup_samples=5,
            output_dir="./test_output",
            metrics=MetricsConfig(enabled=["timing"]),
            pipeline_config={"key": "value"},
        )
        assert config.experiment_name == "full_test"
        assert config.num_samples == 50
        assert config.warmup_samples == 5
        assert config.output_dir == "./test_output"
        assert config.metrics.enabled == ["timing"]
        assert config.pipeline_config == {"key": "value"}

    def test_to_dict_basic(self):
        """to_dict returns dictionary representation."""
        config = BenchmarkConfig(experiment_name="test_exp")
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["experiment_name"] == "test_exp"
        assert result["num_samples"] == 20
        assert result["warmup_samples"] == 3
        assert result["output_dir"] == "./Benchmark"

    def test_to_dict_with_metrics(self):
        metrics_config = MetricsConfig(
            enabled=["timing", "hardware"],
            configurations={"timing": {"stages": ["asr"]}},
        )
        config = BenchmarkConfig(
            experiment_name="test",
            metrics=metrics_config,
        )
        result = config.to_dict()
        assert "metrics" in result
        assert result["metrics"]["enabled"] == ["timing", "hardware"]
        assert result["metrics"]["configurations"]["timing"] == {"stages": ["asr"]}

    def test_to_dict_with_pipeline_config(self):
        config = BenchmarkConfig(
            experiment_name="test",
            pipeline_config={"asr_model": "whisper"},
        )
        result = config.to_dict()
        assert result["pipeline_config"] == {"asr_model": "whisper"}
