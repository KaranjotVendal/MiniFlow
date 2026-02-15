import json
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator

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
    SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"
    TRIAL_SCHEMA_FILENAME = "trial.schema.json"
    SUMMARY_SCHEMA_FILENAME = "summary.schema.json"

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
        trial_data = dict(metrics)
        self._validate_trial_payload(trial_data)

        with open(self.raw_logs_path, "a") as f:
            # Enforce strict JSONL: one compact JSON object per line.
            f.write(json.dumps(trial_data, separators=(",", ":")) + "\n")

    def save_summary(self, summary: dict) -> None:
        """Save structured experiment summary statistics to JSON file."""
        summary_data = dict(summary)
        self._validate_summary_payload(summary_data)

        with open(self.summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

    def save_config(self, config: dict) -> None:
        """Save experiment configuration to JSON file.

        Persists the benchmark configuration used for reproducibility.

        Args:
            config: Benchmark configuration dictionary.

        Raises:
            IOError: If writing to the file fails.
            TypeError: If config contains non-serializable values.
        """
        config["saved_at"] = datetime.now().isoformat()

        # TODO: we should save config as a yml file.
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

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

        raw_text = self.raw_logs_path.read_text().strip()
        if not raw_text:
            return trials

        # Preferred format: strict JSONL (one JSON object per line).
        if "\n{" in raw_text or not raw_text.startswith("{"):
            with open(self.raw_logs_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trials.append(json.loads(line))
            return trials

        # Backward-compatibility fallback: legacy pretty-printed single JSON object.
        legacy_trial = json.loads(raw_text)
        if isinstance(legacy_trial, dict):
            trials.append(legacy_trial)
        elif isinstance(legacy_trial, list):
            trials.extend(legacy_trial)

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

    @classmethod
    def _load_schema(cls, schema_filename: str) -> dict:
        schema_path = cls.SCHEMAS_DIR / schema_filename
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        with open(schema_path, "r") as f:
            return json.load(f)

    @classmethod
    def _validate_trial_payload(cls, payload: dict) -> None:
        schema = cls._load_schema(cls.TRIAL_SCHEMA_FILENAME)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        if errors:
            first_error = errors[0]
            path = ".".join(str(p) for p in first_error.path) or "<root>"
            raise ValueError(
                f"Invalid trial payload at {path}: {first_error.message}"
            )

    @classmethod
    def _validate_summary_payload(cls, payload: dict) -> None:
        schema = cls._load_schema(cls.SUMMARY_SCHEMA_FILENAME)
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
        if errors:
            first_error = errors[0]
            path = ".".join(str(p) for p in first_error.path) or "<root>"
            raise ValueError(
                f"Invalid summary payload at {path}: {first_error.message}"
            )
