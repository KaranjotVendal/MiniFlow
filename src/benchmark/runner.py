import json
import time
import uuid
import yaml
from pathlib import Path
from statistics import mean, median

from src.sts_pipeline import process_sample
from src.prepare_data import stream_dataset_samples
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def run_benchmark(config: dict) -> dict:
    exp_name = config["experiment"]["name"]
    num_samples = config["dataset"]["num_samples"]
    warmup_samples = config["dataset"]["warmup_samples"]
    dataset_split = config["dataset"]["split"]

    run_id = f"{exp_name}_{uuid.uuid4().hex[:6]}"
    exp_dir = Path(f"Benchmark/{exp_name}/{run_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # saving config inside experiment folder
    with open(exp_dir / "config.yml", "w") as f:
        yaml.dump(config, f)

    raw_logs_path = exp_dir / "raw_logs.jsonl"

    metadata = {
        "experiment": exp_name,
        "run_id": run_id,
        "timestamp": time.time(),
        "num_samples": num_samples,
        "warmup": warmup_samples,
        "dataset_split": dataset_split,
    }

    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"created experiment directory: {exp_dir}")

    # warmup runs
    logger.info(f"Running {warmup_samples} warmup samples...")
    warmup_iter = stream_dataset_samples(
        num_samples=warmup_samples, split=dataset_split
    )
    for i, sample in enumerate(warmup_iter):
        process_sample(sample, config=config, run_id=f"warmup{i + 1}", folder=exp_dir)

    logger.info("warmup complete.\n")

    # benchmark runs
    results = []
    for i, sample in enumerate(
        stream_dataset_samples(num_samples=num_samples, split=dataset_split)
    ):
        trial_id = f"trail_{i + 1}"
        logger.info(f"Running {trial_id}")

        processed, metrics = process_sample(
            sample,
            config=config,
            run_id=trial_id,
            folder=exp_dir,
            jsonl_file=raw_logs_path,
        )

        md = metrics.to_dict()
        md["trial"] = i + 1
        results.append(md)

    logger.info("Benchmark run complete.\n")

    # aggregration
    def aggregate(field: str) -> dict:
        arr = [r[field] for r in results]
        # NOTE: need to sort for median and for percentile.
        s = sorted(arr)
        return {
            "mean": mean(arr),
            "median": median(arr),
            "p95": s[int(0.95 * len(s))],
            "p99": s[int(0.99 * len(s))],
        }

    summary = {
        "experiment": exp_name,
        "run_id": run_id,
        "num_samples": len(results),
        "asr_latency": aggregate("asr_latency"),
        "llm_latency": aggregate("llm_latency"),
        "tts_latency": aggregate("tts_latency"),
        "total_latency": aggregate("total_latency"),
        "asr_wer_mean": mean([r["asr_wer"] for r in results]),
        "tts_utmos_mean": mean([r["tts_utmos"] for r in results]),
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary written to: {exp_dir / 'summary.json'}")
    logger.info(f"Benchmark '{exp_name}' complete!")
    return summary
