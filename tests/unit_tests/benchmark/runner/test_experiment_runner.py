from pathlib import Path
from unittest.mock import patch

import yaml

from src.benchmark.runner.experiment_runner import ExperimentRunner, ExperimentSummary


def _write_metrics_config(tmp_path: Path, enabled: list[str], configurations: dict) -> Path:
    metrics_path = tmp_path / "metrics.yml"
    metrics_path.write_text(
        yaml.safe_dump({"enabled": enabled, "configurations": configurations})
    )
    return metrics_path


def _make_config(experiment: str, metrics_path: Path) -> dict:
    return {
        "experiment": {"name": experiment},
        "dataset": {"num_samples": 2, "warmup_samples": 1, "split": "test"},
        "benchmark": {"enable_streaming_audio": False},
        "metrics": str(metrics_path),
    }


class TestExperimentRunnerConfig:
    def test_from_config_creates_runner(self, tmp_path: Path):
        metrics_path = _write_metrics_config(tmp_path, enabled=[], configurations={})
        config = _make_config("test_experiment", metrics_path)

        runner = ExperimentRunner.from_config(config, tmp_path)

        assert runner.experiment == "test_experiment"
        assert runner.run_id.endswith("_test_experiment")
        assert runner.output_dir.exists()
        assert runner.metrics == {}
        assert runner.config["experiment"]["name"] == "test_experiment"

    def test_from_config_with_timing_metric(self, tmp_path: Path):
        metrics_path = _write_metrics_config(
            tmp_path,
            enabled=["timing"],
            configurations={"timing": {"stages": ["asr", "llm", "tts"]}},
        )
        config = _make_config("with_metrics", metrics_path)

        runner = ExperimentRunner.from_config(config, tmp_path)

        assert "timing" in runner.metrics
        assert runner.metrics["timing"].is_enabled()

    def test_get_metric_names(self, tmp_path: Path):
        metrics_path = _write_metrics_config(
            tmp_path,
            enabled=["timing"],
            configurations={"timing": {"stages": ["asr"]}},
        )
        config = _make_config("metric_names", metrics_path)
        runner = ExperimentRunner.from_config(config, tmp_path)

        assert runner.get_metric_names() == ["timing"]
        assert runner.get_trial_count() == 0


class TestExperimentSummary:
    def test_create_summary(self):
        summary = ExperimentSummary(
            experiment="test_exp",
            run_id="123_test_exp",
            num_trials=10,
            summary=None,
        )

        assert summary.experiment == "test_exp"
        assert summary.run_id == "123_test_exp"
        assert summary.num_trials == 10
        assert summary.summary is None

# Queestion: should this be in integration test folder? is this a reeal integration test?
class TestRunnerIntegration:
    @patch("src.benchmark.runner.experiment_runner.process_sample")
    @patch("src.benchmark.runner.experiment_runner.stream_dataset_samples")
    @patch("src.benchmark.runner.experiment_runner.BenchmarkCollector")
    def test_run_with_mocks(
        self,
        mock_collector_cls,
        mock_stream_samples,
        mock_process_sample,
        tmp_path: Path,
    ):
        metrics_path = _write_metrics_config(tmp_path, enabled=[], configurations={})
        config = _make_config("mock_run", metrics_path)

        warmup_samples = [object()]
        main_samples = [object(), object()]
        mock_stream_samples.side_effect = [iter(warmup_samples), iter(main_samples)]

        collector = mock_collector_cls.return_value
        collector.start_trial.return_value = None
        collector.end_trial.side_effect = [
            {
                "trial_id": "warmup_1",
                "sample_id": "warmup_1",
                "is_warmup": True,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 0.01,
                "latencies": {"asr": 0.001},
            },
            {
                "trial_id": "trial_1",
                "sample_id": "trial_1",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 0.02,
                "latencies": {"asr": 0.002},
            },
            {
                "trial_id": "trial_2",
                "sample_id": "trial_2",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 0.03,
                "latencies": {"asr": 0.003},
            },
        ]
        mock_process_sample.return_value = None

        runner = ExperimentRunner.from_config(config, tmp_path)
        summary = runner.run()

        assert isinstance(summary, ExperimentSummary)
        assert summary.num_trials == 2
        assert runner.get_trial_count() == 2
        assert mock_process_sample.call_count == 3
        assert mock_stream_samples.call_count == 2
