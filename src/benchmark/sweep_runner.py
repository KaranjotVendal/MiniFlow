import json
import time
from pathlib import Path

import yaml

from src.config.inspect_config import inspect_config
from src.benchmark.runner.experiment_runner import ExperimentRunner
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def load_sweep_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sweep config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_sweep(sweep_config_path: str | Path):
    config = load_sweep_config(sweep_config_path)

    sweep_name = config["sweep_name"]
    sweep_list = config["sweep"]
    output_dir = Path(config["output_dir"])

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    sweep_folder = output_dir / f"{sweep_name}_{timestamp}"
    sweep_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running sweep: {sweep_name}")
    logger.info(f"saving results to: {sweep_folder}")

    sweep_results = []
    for config_path in sweep_list:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"config does not exist: {config_path}, skipping")
            continue

        # kv_cache.yaml -> kv_cache
        exp_name = config_path.stem
        logger.info(f"\n==== Running experiment: {exp_name} ===")

        _config = inspect_config(config_path)
        runner = ExperimentRunner.from_config(_config)
        summary = runner.run()

        sweep_results.append(
            {
                "experiment": exp_name,
                "config_path": str(config_path),
                "summary": {
                    "experiment": summary.experiment,
                    "run_id": summary.run_id,
                    "num_trials": summary.num_trials,
                    "metric_summaries": summary.metric_summaries,
                },
            }
        )

        exp_dir = sweep_folder / exp_name
        exp_dir.mkdir(exist_ok=True)
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    logger.info("Building sweep summary...")

    def numeric(x) -> float | int:
        """helper to extract numeric mean from nested dict"""
        if isinstance(x, dict) and "mean" in x:
            return x["mean"]
        if isinstance(x, dict):
            return 0.0
        return x

    sweep_summary = {
        "sweep_name": sweep_name,
        "timestamp": timestamp,
        "experiments": [],
    }

    for r in sweep_results:
        exp_summary = r["summary"]
        metric_summaries = exp_summary.get("metric_summaries", {})
        sweep_summary["experiments"].append(
            {
                "experiment": r["experiment"],
                "num_trials": exp_summary.get("num_trials", 0),
                "asr_latency_mean": numeric(metric_summaries.get("asr_latency_seconds", {})),
                "llm_latency_mean": numeric(metric_summaries.get("llm_latency_seconds", {})),
                "tts_latency_mean": numeric(metric_summaries.get("tts_latency_seconds", {})),
                "total_latency_mean": numeric(metric_summaries.get("trial_wall_time_seconds", {})),
            }
        )

    with open(sweep_folder / "sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)

    logger.info(f"sweep complete. Results saved in {sweep_folder}")

    return sweep_summary
