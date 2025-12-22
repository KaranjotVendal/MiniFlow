import json
import time
import uuid
from pathlib import Path

import yaml

from src.config.inspect_config import inspect_config
from src.benchmark.runner import run_benchmark
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
        summary = run_benchmark(_config)

        sweep_results.append(
            {
                "experiment": exp_name,
                "config_path": str(config_path),
                "summary": summary,
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
        return x

    sweep_summary = {
        "sweep_name": sweep_name,
        "timestamp": timestamp,
        "experiments": [],
    }

    for r in sweep_results:
        exp_summary = r["summary"]
        sweep_summary["experiments"].append(
            {
                "experiment": r["experiment"],
                "asr_latency_mean": numeric(exp_summary["asr_latency"]),
                "llm_latency_mean": numeric(exp_summary["llm_latency"]),
                "tts_latency_mean": numeric(exp_summary["tts_latency"]),
                "total_latency_mean": numeric(exp_summary["total_latency"]),
                "asr_wer_mean": exp_summary["asr_wer_mean"],
                "tts_utmos_mean": exp_summary["tts_utmos_mean"],
            }
        )

    with open(sweep_folder / "sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)

    logger.info(f"sweep complete. Results saved in {sweep_folder}")

    return sweep_summary
