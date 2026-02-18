# run_benchmark_one_sample.py
from pathlib import Path

from src.config.inspect_config import inspect_config
from src.benchmark.runner.experiment_runner import ExperimentRunner


def main():
    print("Loading config...")
    cfg = inspect_config("configs/3_TTS-to-vibevoice.yml")

    # Override for quick test
    cfg["dataset"]["num_samples"] = 1
    cfg["dataset"]["warmup_samples"] = 0

    print("Running full benchmark (1 sample, no warmup)...")
    try:
        runner = ExperimentRunner.from_config(cfg)
        summary = runner.run()
    except Exception as e:
        print(f"Benchmark failed:\n{e}")
        raise

    print("\nSUCCESS")
    print(f"Run ID: {summary.run_id}")
    print(f"Total samples processed: {summary.num_trials}")

    # Validate output files
    output_dir = Path("Benchmark") / summary.experiment / summary.run_id
    print(f"output dir: {output_dir}")
    raw_logs = output_dir / "raw_logs.jsonl"
    summary_json = output_dir / "summary.json"

    if raw_logs.exists():
        lines = raw_logs.read_text().strip().split("\n")
        print(f"raw_logs.jsonl: {len(lines)} line(s)")
    else:
        print("raw_logs.jsonl not found")

    if summary_json.exists():
        print("summary.json: exists")
    else:
        print("summary.json not found")


if __name__ == "__main__":
    main()
