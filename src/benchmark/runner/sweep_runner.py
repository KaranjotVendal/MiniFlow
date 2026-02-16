import json
from datetime import datetime
from pathlib import Path
from statistics import mean
import yaml

from src.benchmark.runner.experiment_runner import ExperimentRunner
from src.config.load_config import load_yaml_config
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def _extract_metric_value(summary_dict: dict, *keys, default: float = 0.0) -> float:
    """Helper to extract numeric mean from nested dict with fallback."""
    current = summary_dict
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, {})
        else:
            return default

    if isinstance(current, dict) and "mean" in current:
        return float(current["mean"])
    return default


def _sweep_summary(sweep_name: str, sweep_folder: Path, sweep_results: list[dict]) -> None:
    # Build aggregated sweep summary
    sweep_summary = {
        "sweep_name": sweep_name,
        "timestamp": datetime.now().isoformat(),
        "num_experiments": len(sweep_results),
        "experiments": [],
    }

    for result in sweep_results:
        exp_name = result["experiment"]
        summary_dict = result["summary"]

        # Extract metrics using the helper with correct paths
        # The summary dict has structure: {pipeline: {latency: {...}}, asr: {...}, llm: {...}, tts: {...}}
        pipeline = summary_dict.get("pipeline", {})
        asr = summary_dict.get("asr", {})
        llm = summary_dict.get("llm", {})
        tts = summary_dict.get("tts", {})

        exp_summary = {
            "experiment": exp_name,
            "run_id": result["run_id"],
            "num_trials": result["num_trials"],
            "asr_latency_mean": _extract_metric_value(asr, "latency", "inference_seconds"),
            "llm_latency_mean": _extract_metric_value(llm, "latency", "inference_seconds"),
            "tts_latency_mean": _extract_metric_value(tts, "latency", "inference_seconds"),
            "total_latency_mean": _extract_metric_value(pipeline, "latency", "trial_wall_time_seconds"),
            "total_model_load_time_mean": _extract_metric_value(pipeline, "load_times", "total_model_load_time_seconds"),
            "asr_wer_mean": _extract_metric_value(asr, "quality", "wer"),
            "tts_utmos_mean": _extract_metric_value(tts, "quality", "utmos"),
            "llm_tokens_per_sec_mean": _extract_metric_value(llm, "inference", "tokens_per_sec"),
            "llm_ttft_mean": _extract_metric_value(llm, "inference", "ttft_seconds"),
        }

        sweep_summary["experiments"].append(exp_summary)

    # Calculate aggregate statistics across experiments
    if sweep_summary["experiments"]:
        experiments = sweep_summary["experiments"]

        # Add aggregate comparison table
        sweep_summary["aggregates"] = {
            "best_asr_latency": min(exp["asr_latency_mean"] for exp in experiments),
            "best_llm_latency": min(exp["llm_latency_mean"] for exp in experiments),
            "best_tts_latency": min(exp["tts_latency_mean"] for exp in experiments),
            "best_total_latency": min(exp["total_latency_mean"] for exp in experiments),
            "best_asr_wer": min(exp["asr_wer_mean"] for exp in experiments),
            "best_tts_utmos": max(exp["tts_utmos_mean"] for exp in experiments),
            "avg_total_latency": mean(exp["total_latency_mean"] for exp in experiments),
        }

    # Save sweep summary
    with open(sweep_folder / "sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)

    logger.info(f"Sweep complete. Results saved in {sweep_folder}")
    logger.info(f"Summary: {len(sweep_results)} experiments completed")


def run_sweep(sweep_config_path: str | Path) -> None:
    """Run a parameter sweep across multiple experiment configurations.

    Args:
        sweep_config_path: Path to sweep YAML configuration file.

    Returns:
        Sweep summary dictionary with aggregated results.
    """
    config = load_yaml_config(sweep_config_path)

    sweep_name = config["sweep_name"]
    sweep_list = config["sweep"]
    output_dir = Path(config.get("output_dir", "sweeps"))

    sweep_folder = output_dir / f"{int(datetime.timestamp(datetime.now()))}_{sweep_name}"
    sweep_folder.mkdir(parents=True, exist_ok=True)

    # Save sweep config to output folder
    with open(sweep_folder / "sweep_config.yml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Running sweep: {sweep_name}")
    logger.info(f"Saving results to: {sweep_folder}")

    sweep_results: list[dict] = []
    for config_path in sweep_list:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config does not exist: {config_path}, skipping")
            continue

        # config filename stem (e.g., "baseline", "2_TTS-to-vibevoice")
        exp_name = config_path.stem
        logger.info(f"\n==== Running experiment: {exp_name} ===")

        # Load full experiment configuration
        _config = load_yaml_config(config_path)

        # Create and run experiment
        runner = ExperimentRunner.from_config(_config, sweep_folder)
        summary = runner.run()

        # Store result with experiment metadata
        sweep_results.append(
            {
                "experiment": exp_name,
                "config_path": str(config_path),
                "run_id": summary.run_id,
                "num_trials": summary.num_trials,
                # Convert SummaryRecord to dict for serialization
                "summary": summary.summary.to_dict(),
            }
        )

    logger.info("Building sweep summary...")
    _sweep_summary(sweep_name=sweep_name, sweep_folder=sweep_folder, sweep_results=sweep_results)


def main():
    """Entry point for running sweeps from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a parameter sweep.")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Starting sweep with config: {config_path}")

    if args.dry_run:
        config = load_sweep_config(config_path)
        logger.info("Dry run enabled. Config loaded successfully:")
        logger.info(f"  Sweep name: {config['sweep_name']}")
        logger.info(f"  Experiments: {len(config['sweep'])}")
        for cfg in config["sweep"]:
            logger.info(f"    - {cfg}")
        return

    run_sweep(config_path)
    logger.info("Sweep completed successfully!")


if __name__ == "__main__":
    main()
