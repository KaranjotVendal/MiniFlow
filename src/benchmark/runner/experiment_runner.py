from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from src.benchmark.storage import BaseStorage
from src.benchmark.core.base import BaseMetric
from src.benchmark.collectors import BenchmarkCollector
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.storage.jsonl_storage import JSONLStorage
from src.config.load_config import load_yaml_config
from src.sts_pipeline import process_sample
from src.utils import clear_gpu_cache
from src.prepare_data import stream_dataset_samples, AudioSample
from src.utils import get_device
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)

DEVICE = get_device()


@dataclass
class MetricStats:
    mean: float
    median: float
    min: float
    max: float
    std: float
    p95: float
    p99: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "p95": self.p95,
            "p99": self.p99,
            "count": self.count,
        }


@dataclass
class ExperimentSummary:
    """Aggregated summary of benchmark experiment.

    Attributes:
        experiment: Experiment name.
        run_id: Unique run identifier.
        num_trials: Number of main trials executed.
        metric_summaries: Dictionary of per-metric summary statistics.
    """
    experiment: str
    run_id: str
    num_trials: int
    metric_summaries: dict[str, "MetricStats"] = field(default_factory=dict)


class ExperimentRunner:
    """Main orchestrator for benchmark experiments.

    The ExperimentRunner coordinates all aspects of a benchmark run:
    - Configuration parsing and metric selection
    - Trial execution with metric collection
    - Result storage and summary generation

    Example:
        >>> config = load_benchmark_config("configs/baseline.yml")
        >>> runner = ExperimentRunner.from_config(config)
        >>> summary = runner.run()

    Attributes:
        experiment: Experiment name.
        run_id: Unique identifier for this run.
        output_dir: Directory for results.
        storage: Storage backend for persisting results.
        metrics: Dictionary of metric name to metric instance.
        config: Full configuration dictionary.
    """

    def __init__(
        self,
        experiment: str,
        run_id: str,
        storage: BaseStorage,
        metrics: dict[str, BaseMetric],
        config: dict,
        device: torch.device | str,
        output_dir: Path = Path("/home/childofprophecy/Desktop/Personal_projects/Machine_Learning/MiniFlow/Benchmark"),
    ):
        """Initialize the experiment runner.

        Args:
            experiment: Experiment name.
            run_id: Unique identifier for this run.
            storage: Storage backend instance.
            metrics: Dictionary of metric name to metric instance.
            config: Full configuration dictionary.
            output_dir: Directory for storing results.
        """
        self.experiment = experiment
        self.run_id = run_id
        self.output_dir = output_dir
        self.storage = storage
        self.metrics = metrics
        self.config = config
        self.device = device
        self._trial_count = 0

    def _stream_samples(self, num_samples: int, split: str):
        """Stream samples from dataset.

        This is a placeholder that should be replaced with actual dataset
        integration in the full implementation.

        Args:
            num_samples: Number of samples to yield.
            split: Dataset split to use.

        Yields:
            Sample dictionaries with audio and transcript data.
        """

        return stream_dataset_samples(num_samples=num_samples, split=split)

    @staticmethod
    def _configure_metric_classes(config_path: Path) -> dict:
        config = load_yaml_config(config_path)

        # Create metrics from registry
        enabled_metrics = config["enabled"]
        metric_configurations: dict[str, dict] = config["configurations"]

        # Pass runtime device through config; hardware metric resolves device/index.
        if "hardware_basic" in metric_configurations:
            metric_configurations["hardware_basic"]["device"] = DEVICE
        if "hardware_detailed" in metric_configurations:
            metric_configurations["hardware_detailed"]["device"] = DEVICE

        # NOTE: all the metrics that are enabled.
        metrics: dict[str, BaseMetric] = {}
        for metric_name in enabled_metrics:
            metric_class = MetricRegistry.get(metric_name)
            metric_config = metric_configurations[metric_name]
            metrics[metric_name] = metric_class(metric_config)

        # TODO: Can the benchmarkCollector be just initialised once here??
        # collector = BenchmarkCollector(metrics=self.metrics, config=self.config)
        return metrics

    def _run_trial(
        self, sample: AudioSample, trial_id: str, is_warmup: bool = False
    ) -> None:
        """Execute a single trial with metric collection.

        This method:
        1. Creates a BenchmarkCollector for the trial
        2. Calls collector.start_trial()
        3. Executes the pipeline (ASR -> LLM -> TTS) via process_sample()
        4. Calls collector.end_trial() and collects results
        5. Saves trial results to storage

        Args:
            sample: Sample data dictionary.
            trial_id: Unique trial identifier.
            is_warmup: Whether this is a warmup trial (excluded from summary).

        """
        collector = BenchmarkCollector(metrics=self.metrics, config=self.config)

        collector.start_trial(
            trial_id=trial_id,
            sample_id=trial_id,
            is_warmup=is_warmup,
        )

        try:
            process_sample(
                sample=sample,
                config=self.config,
                collector=collector,
                run_id=self.output_dir.name,
                folder=str(self.output_dir),
                device=self.device
            )
            trial_metrics = collector.end_trial(status="success")
        except Exception as e:
            logger.exception(f"Trial {trial_id} failed: {e}")
            trial_metrics = collector.end_trial(status="error", error=type(e).__name__)
            raise
        finally:
            # Ensure GPU cleanup after each trial
            clear_gpu_cache()

        self.storage.save_trial(trial_id, trial_metrics)

        logger.debug(f"Trial {trial_id} completed in {trial_metrics["trial_wall_time_seconds"]}" + (" (warmup)" if is_warmup else ""))
        self._trial_count += 1

    def _run_warmup_trials(self, warmup_samples: int, split: str) -> None:
        if warmup_samples == 0:
            logger.info("running experiment with 0 warmup samples")
            return

        logger.info(f"Running {warmup_samples} warmup trials...")
        warmup_iter = self._stream_samples(num_samples=warmup_samples, split=split)

        for i, sample in enumerate(warmup_iter):
            trial_id = f"warmup_{i + 1}"
            self._run_trial(sample=sample, trial_id=trial_id, is_warmup=True)
        self._trial_count = 0
        logger.info("Warmup complete.")

    def _generate_summary(self) -> ExperimentSummary:
        """Generate aggregated summary statistics from trial results.

        Loads trial data from storage (raw_logs.jsonl) and computes summary
        statistics including mean, median, min, max, std, p95, p99, and count
        for each numeric metric.

        Returns:
            ExperimentSummary with aggregated statistics.
        """
        # TODO: we need to make sure we are summarises all the data collected and
        # it is done in a way that help us with the project.
        # For example we need latencies, gpu metrics, quality metrics,

        # Load trials from storage - load_trials() returns list of dicts
        trials: list[dict] = self.storage.load_trials()

        # Filter out warmup trials
        main_trials = [t for t in trials if not t.get("is_warmup", False)]
        num_trials = len(main_trials)

        metric_summaries: dict[str, MetricStats] = {}
        all_metrics: dict[str, list[float]] = {}

        # Keys to exclude from metric aggregation
        exclude_keys = {
            "trial_id",
            "is_warmup",
            "sample_id",
            "exp_name",
            "timestamp_start",
            "status",
            "error",
        }

        def _normalize_metric_key(key: str) -> str:
            normalized = []
            for ch in key.lower():
                if ch.isalnum() or ch == "_":
                    normalized.append(ch)
                else:
                    normalized.append("_")
            value = "".join(normalized)
            while "__" in value:
                value = value.replace("__", "_")
            return value.strip("_")

        def _flatten_numeric_metrics(data: Any, prefix: str = "") -> dict[str, float]:
            flattened: dict[str, float] = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    next_prefix = f"{prefix}_{key}" if prefix else str(key)
                    flattened.update(_flatten_numeric_metrics(value, next_prefix))
                return flattened

            # bool is a subclass of int; exclude it from numeric aggregation.
            if isinstance(data, bool):
                return flattened
            if isinstance(data, (int, float)):
                metric_key = _normalize_metric_key(prefix)
                if metric_key:
                    flattened[metric_key] = float(data)
            return flattened

        for trial in main_trials:
            filtered_trial = {
                key: value for key, value in trial.items() if key not in exclude_keys
            }
            flattened_metrics = _flatten_numeric_metrics(filtered_trial)
            for key, value in flattened_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        for metric_name, values in all_metrics.items():
            sorted_values = sorted(values)
            n = len(sorted_values)

            if n == 0:
                continue

            mean_val = mean(values)
            metric_summaries[metric_name] = MetricStats(
                mean=round(mean_val, 6),
                median=round(median(values), 6),
                min=round(min(values), 6),
                max=round(max(values), 6),
                std=round((sum((x - mean_val) ** 2 for x in values) / n) ** 0.5, 6)
                if n > 1
                else 0.0,
                p95=round(sorted_values[int(0.95 * n)], 6)
                if n >= 20
                else round(sorted_values[-1], 6),
                p99=round(sorted_values[int(0.99 * n)], 6)
                if n >= 100
                else round(sorted_values[-1], 6),
                count=n,
            )

        return ExperimentSummary(
            experiment=self.experiment,
            run_id=self.run_id,
            num_trials=num_trials,
            metric_summaries=metric_summaries,
        )

    def _run_benchmark_samples(self, num_samples: int, split: str) -> None:
        trial_iter = self._stream_samples(num_samples=num_samples, split=split)
        for i, sample in enumerate(trial_iter):
            trial_id = f"trial_{i + 1}"
            self._run_trial(sample, trial_id, is_warmup=False)

    def get_metric_names(self) -> list[str]:
        """Get list of enabled metric names.

        Returns:
            List of metric names currently enabled.
        """
        return list(self.metrics.keys())

    def get_trial_count(self) -> int:
        """Get the number of main trials run.

        Returns:
            Number of main trials executed.
        """
        return self._trial_count

    def run(self) -> ExperimentSummary:
        """This method runs warmup trials first, then main trials, collects all
        metrics, generates summary statistics, and persists results.

        Returns:
            ExperimentSummary containing all results and statistics.
        """
        dataset_config = self.config["dataset"]
        num_samples = dataset_config["num_samples"]
        warmup_samples = dataset_config["warmup_samples"]
        split = dataset_config.get("split", "test")

        logger.info("Starting experiment...")
        logger.info(f"Run ID: {self.output_dir.name}")
        # TODO: we can probably have one dataset iter objec instead of two different ones.
        self._run_warmup_trials(warmup_samples=warmup_samples, split=split)

        logger.info(f"Running {num_samples} main trials...")
        self._run_benchmark_samples(num_samples=num_samples, split=split)

        # Generate summary
        logger.info("Generating summary statistics...")
        summary = self._generate_summary()

        # Save summary with run metadata plus aggregated metric summaries.
        summary_payload = {
            "experiment": summary.experiment,
            "run_id": summary.run_id,
            "timestamp": datetime.now().isoformat(),
            "num_trials": summary.num_trials,
            "metric_summaries": {
                metric_name: metric_stats.to_dict()
                for metric_name, metric_stats in summary.metric_summaries.items()
            },
        }
        self.storage.save_summary(summary_payload)

        logger.info("Experiment complete!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Summary: {summary.num_trials} trials completed")

        return summary

    @classmethod
    def from_config(cls, config: dict) -> "ExperimentRunner":
        """Create an ExperimentRunner from a configuration dictionary.

        This method parses the configuration, creates the appropriate metrics
        from the registry, and initializes the storage backend.

        Args:
            config: Benchmark configuration dictionary. Expected structure:
                {
                    "experiment": {"name": "experiment_name"},
                    "dataset": {"num_samples": 20, "warmup_samples": 3},
                    "metrics": "/path/to/metrics.yml",
                    ...
                }

        Returns:
            Initialized ExperimentRunner instance.

        Raises:
            KeyError: If required config keys are missing.
            ValueError: If metric configuration is invalid.
        """
        # Load metrics configuration from separate file
        metrics_config_path = Path(config["metrics"])
        if not metrics_config_path.exists():
            logger.warning(f"Metrics config doesn't exist: {metrics_config_path}, stopping..")
            raise FileNotFoundError(f"Metrics config not found: {metrics_config_path}")

        experiment_name = config["experiment"]["name"]

        # Create unique run ID and output directory
        run_id = f"{int(datetime.timestamp(datetime.now()))}_{experiment_name}"
        output_dir = Path("Benchmark") / experiment_name / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        storage = JSONLStorage(output_dir)
        metrics = cls._configure_metric_classes(config_path=metrics_config_path)
        storage.save_config(config)

        logger.info(f"Initialized experiment '{experiment_name}'")
        logger.info(f"Run ID: {storage.output_dir.name}")
        logger.info(f"Enabled metrics: {list(metrics.keys())}")

        return cls(
            experiment=experiment_name,
            run_id=run_id,
            output_dir=output_dir,
            storage=storage,
            metrics=metrics,
            config=config,
            device=DEVICE
        )
