import json
from datetime import datetime
from pathlib import Path

from src.benchmark.storage.base import BaseStorage


class JSONLStorage(BaseStorage):
    """JSONL-based storage for benchmark trial results.

    This storage backend persists benchmark results in a format compatible
    with the existing MiniFlow benchmark system. It writes trial metrics
    as JSONL (one JSON object per line), summary statistics as JSON,
    and configuration as JSON.

    Attributes:
        output_dir: Directory where benchmark results are stored.
        raw_logs_path: Path to the raw logs JSONL file.
        summary_path: Path to the summary JSON file.
        config_path: Path to the config JSON file.

    Example:
        ```python
        storage = JSONLStorage(Path("Benchmark/experiment_001"))
        storage.save_trial("trial_001", {"latency": 1.5, "wer": 0.1})
        storage.save_summary({"mean_latency": 1.5})
        storage.save_config({"num_samples": 10})
        trials = storage.load_trials()
        ```
    """

    RAW_LOGS_FILENAME = "raw_logs.jsonl"
    SUMMARY_FILENAME = "summary.json"
    CONFIG_FILENAME = "config.json"

    def __init__(self, output_dir: Path):
        super().__init__(output_dir)

        self.raw_logs_path = self.output_dir / self.RAW_LOGS_FILENAME
        self.summary_path = self.output_dir / self.SUMMARY_FILENAME
        self.config_path = self.output_dir / self.CONFIG_FILENAME

    def save_trial(self, trial_id: str, metrics: dict) -> None:
        """Save a single trial's metrics to the JSONL file.

        Args:
            trial_id: Unique identifier for the trial (e.g., "trial_001").
            metrics: Dictionary of collected metric values for this trial.

        Raises:
            IOError: If writing to the file fails.
            TypeError: If metrics contains non-serializable values.
        """
        trial_data = {
            "sample_id": trial_id,
            "exp_name": self.output_dir.name,
            "timestamp_start": datetime.now().timestamp(),
            **metrics,
        }

        with open(self.raw_logs_path, "a") as f:
            f.write(json.dumps(trial_data) + "\n")

    def save_summary(self, summary: dict) -> None:
        """Save experiment summary statistics to JSON file.

        Writes aggregated summary statistics for all trials. The summary
        includes experiment metadata and per-metric statistics.

        Args:
            summary: Dictionary of aggregated summary statistics.

        Raises:
            IOError: If writing to the file fails.
            TypeError: If summary contains non-serializable values.
        """
        # Add metadata to summary
        summary_with_metadata = {
            "experiment": self.output_dir.name,
            "run_id": self.output_dir.name,
            "timestamp": datetime.now().isoformat(),
            **summary,
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary_with_metadata, f, indent=2)

    def save_config(self, config: dict) -> None:
        """Save experiment configuration to JSON file.

        Persists the benchmark configuration used for reproducibility.

        Args:
            config: Benchmark configuration dictionary.

        Raises:
            IOError: If writing to the file fails.
            TypeError: If config contains non-serializable values.
        """
        # Add metadata to config
        config_with_metadata = {
            "saved_at": datetime.now().isoformat(),
            "config": config,
        }

        with open(self.config_path, "w") as f:
            json.dump(config_with_metadata, f, indent=2)

    def load_trials(self) -> list[dict]:
        """Load all trial results from the JSONL file.

        Reads and parses all lines from raw_logs.jsonl, returning a list
        of trial dictionaries. Each dictionary contains the trial_id and
        all collected metric values.

        Returns:
            List of trial dictionaries in order of execution.
            Returns empty list if raw_logs.jsonl does not exist.

        Raises:
            IOError: If reading from the file fails.
            JSONDecodeError: If the file contains invalid JSON.
        """
        trials = []

        if not self.raw_logs_path.exists():
            return trials

        with open(self.raw_logs_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    trial_data = json.loads(line)
                    trials.append(trial_data)

        return trials

    def load_summary(self) -> dict | None:
        """Load experiment summary from JSON file.

        Returns:
            Summary dictionary, or None if summary.json does not exist.
        """
        if not self.summary_path.exists():
            return None

        with open(self.summary_path, "r") as f:
            return json.load(f)

    def load_config(self) -> dict | None:
        """Load experiment configuration from JSON file.

        Returns:
            Configuration dictionary wrapped in metadata dict,
            or None if config.json does not exist.
        """
        if not self.config_path.exists():
            return None

        with open(self.config_path, "r") as f:
            return json.load(f)

    def get_trial_count(self) -> int:
        """Get the number of trials stored.

        Returns:
            Number of trials in raw_logs.jsonl.
        """
        return len(self.load_trials())

    def clear_trials(self) -> None:
        """Clear all stored trial data.

        Removes the raw_logs.jsonl file if it exists.
        This is useful for resetting storage between experiments.
        """
        if self.raw_logs_path.exists():
            self.raw_logs_path.unlink()
