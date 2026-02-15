from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from src.benchmark.storage import BaseStorage
from src.benchmark.core.base import BaseMetric
from src.benchmark.collectors import BenchmarkCollector
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.runner.summary_models import MetricStats, SummaryRecord
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
class ExperimentSummary:
    """Structured benchmark summary metadata and payload."""
    experiment: str
    run_id: str
    num_trials: int
    summary: SummaryRecord | None = None


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

    @staticmethod
    def _compute_stats(values: list[float]) -> MetricStats:
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n == 0:
            return MetricStats(
                mean=0.0, median=0.0, min=0.0, max=0.0, std=0.0, p95=0.0, p99=0.0, count=0
            )

        mean_val = mean(values)
        return MetricStats(
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

    @staticmethod
    def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
        current = target
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _generate_summary(self) -> SummaryRecord:
        trials: list[dict] = self.storage.load_trials()
        main_trials = [t for t in trials if not t.get("is_warmup", False)]
        warmup_trials = [t for t in trials if t.get("is_warmup", False)]

        status_counts = {"success": 0, "error": 0}
        for trial in main_trials:
            status = trial.get("status")
            if status == "success":
                status_counts["success"] += 1
            else:
                status_counts["error"] += 1

        summary_dict: dict[str, Any] = {
            "meta": {
                "experiment": self.experiment,
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "num_trials": len(main_trials),
                "num_warmup_trials": len(warmup_trials),
                "status_counts": status_counts,
            },
            "pipeline": {
                "latency": {},
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
            "asr": {
                "latency": {},
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
            "llm": {
                "latency": {},
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
            "tts": {
                "latency": {},
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
        }

        numeric_series: dict[tuple[str, ...], list[float]] = {}
        ttft_mode_counts: dict[str, int] = {}

        def add_numeric(path: list[str], value: Any) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return
            key = tuple(path)
            numeric_series.setdefault(key, []).append(float(value))

        hardware_phase_map = {
            "asr_model_load": ("asr", "hardware", "model_load"),
            "asr_inference": ("asr", "hardware", "inference"),
            "llm_model_load": ("llm", "hardware", "model_load"),
            "llm_tokenizer": ("llm", "hardware", "tokenizer_load"),
            "llm_pipeline": ("llm", "hardware", "pipeline_load"),
            "llm_inference": ("llm", "hardware", "inference"),
            "tts_model_load": ("tts", "hardware", "model_load"),
            "tts_processor_load": ("tts", "hardware", "processor_load"),
            "tts_inference": ("tts", "hardware", "inference"),
        }

        for trial in main_trials:
            add_numeric(["pipeline", "latency", "trial_wall_time_seconds"], trial.get("trial_wall_time_seconds"))
            add_numeric(["pipeline", "latency", "total_latency_seconds"], trial.get("total_latency_seconds"))
            add_numeric(["pipeline", "load_times", "total_model_load_time_seconds"], trial.get("total_model_load_time"))
            add_numeric(["pipeline", "hardware", "trial", "gpu_memory_allocated_mb"], trial.get("gpu_memory_allocated_mb"))
            add_numeric(["pipeline", "hardware", "trial", "gpu_memory_reserved_mb"], trial.get("gpu_memory_reserved_mb"))
            add_numeric(["pipeline", "hardware", "trial", "gpu_memory_peak_mb"], trial.get("gpu_memory_peak_mb"))
            add_numeric(["pipeline", "hardware", "trial", "gpu_memory_efficiency"], trial.get("gpu_memory_efficiency"))

            stage_latencies = trial.get("latencies", {})
            add_numeric(["asr", "latency", "inference_seconds"], stage_latencies.get("asr"))
            add_numeric(["llm", "latency", "inference_seconds"], stage_latencies.get("llm"))
            add_numeric(["tts", "latency", "inference_seconds"], stage_latencies.get("tts"))

            add_numeric(["asr", "load_times", "model_load_seconds"], (trial.get("asr_model_load") or {}).get("total_time"))
            add_numeric(["llm", "load_times", "model_load_seconds"], (trial.get("llm_model_load") or {}).get("total_time"))
            add_numeric(["llm", "load_times", "tokenizer_load_seconds"], (trial.get("llm_tokenizer_load") or {}).get("total_time"))
            add_numeric(["llm", "load_times", "pipeline_load_seconds"], (trial.get("llm_pipeline_load") or {}).get("total_time"))
            add_numeric(["tts", "load_times", "model_load_seconds"], (trial.get("tts_model_load") or {}).get("total_time"))
            add_numeric(["tts", "load_times", "processor_load_seconds"], (trial.get("tts_processor_load") or {}).get("total_time"))

            add_numeric(["llm", "inference", "tokens_generated"], trial.get("tokens_generated"))
            add_numeric(["llm", "inference", "tokens_per_sec"], trial.get("tokens_per_sec"))
            add_numeric(["llm", "inference", "time_per_token_seconds"], trial.get("time_per_token"))
            add_numeric(["llm", "inference", "total_generation_time_seconds"], trial.get("total_generation_time"))
            add_numeric(["llm", "inference", "ttft_seconds"], trial.get("ttft"))
            add_numeric(["llm", "inference", "cache_hits"], trial.get("cache_hits"))
            add_numeric(["llm", "inference", "cache_misses"], trial.get("cache_misses"))
            add_numeric(["llm", "inference", "models_loaded"], trial.get("models_loaded"))

            add_numeric(["asr", "quality", "wer"], trial.get("wer"))
            add_numeric(["tts", "quality", "utmos"], trial.get("utmos"))

            ttft_mode = trial.get("ttft_mode")
            if isinstance(ttft_mode, str) and ttft_mode:
                ttft_mode_counts[ttft_mode] = ttft_mode_counts.get(ttft_mode, 0) + 1

            hardware_phase_metrics = trial.get("hardware_phase_metrics", {})
            if isinstance(hardware_phase_metrics, dict):
                for phase_name, base_path in hardware_phase_map.items():
                    phase_data = hardware_phase_metrics.get(phase_name)
                    if not isinstance(phase_data, dict):
                        continue
                    for metric_name in (
                        "gpu_memory_allocated_mb",
                        "gpu_memory_reserved_mb",
                        "gpu_memory_peak_mb",
                        "gpu_memory_efficiency",
                    ):
                        add_numeric(
                            [*base_path, metric_name],
                            phase_data.get(metric_name),
                        )

        for path, values in numeric_series.items():
            self._set_nested(summary_dict, list(path), self._compute_stats(values).to_dict())

        self._set_nested(
            summary_dict, ["llm", "inference", "ttft_mode_counts"], ttft_mode_counts
        )
        return SummaryRecord.from_dict(summary_dict)

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
        """Run warmups/main trials and persist structured summary."""
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

        logger.info("Generating summary statistics...")
        summary_record = self._generate_summary()
        summary_payload = summary_record.to_dict()
        self.storage.save_summary(summary_payload)
        meta = summary_record.meta

        # TODO: It seems dataclasses for ExperimentSummary can be simplified.
        summary = ExperimentSummary(
            experiment=meta.experiment,
            run_id=meta.run_id,
            num_trials=meta.num_trials,
            summary=summary_record,
        )

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
