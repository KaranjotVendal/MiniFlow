from abc import ABC, abstractmethod
from pathlib import Path


class BaseStorage(ABC):
    """Abstract base class for benchmark result storage.

    This class defines the contract for persisting trial metrics, experiment
    summaries, and configurations. Subclasses implement this interface for
    specific storage backends such as JSONL files, databases, or cloud storage.

    Attributes:
        output_dir: Directory path where results are stored.

    Example:
        ```python
        class JSONLStorage(BaseStorage):
            def save_trial(self, trial_id: str, metrics: dict) -> None:
                with open(self.output_dir / "trials.jsonl", "a") as f:
                    f.write(json.dumps({"trial_id": trial_id, **metrics}) + "\\n")
        ```
    """

    def __init__(self, output_dir: Path):
        """Initialize storage with output directory.
        Creates the output directory if it does not exist.

        Args:
            output_dir: Directory path for storing benchmark results.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save_trial(self, trial_id: str, metrics: dict) -> None:
        """Save a single trial's metrics to storage.

        Args:
            trial_id: Unique identifier for the trial (e.g., "trial_001").
            metrics: Dictionary of collected metric values for this trial.
        """
        pass

    @abstractmethod
    def save_summary(self, summary: dict) -> None:
        """Save experiment summary statistics.
        This method is called after all trials complete to persist the
        aggregated summary statistics.

        Args:
            summary: Dictionary of aggregated summary statistics for each metric.

        Raises:
            StorageError: If saving fails due to I/O or other errors.
        """
        pass

    @abstractmethod
    def save_config(self, config: dict) -> None:
        """Save experiment configuration to persist
        the configuration used, enabling reproducibility.

        Args:
            config: Benchmark configuration dictionary including experiment
                name, metrics configuration, and pipeline settings.

        Raises:
            StorageError: If saving fails due to I/O or other errors.
        """
        pass

    @abstractmethod
    def load_trials(self) -> list[dict]:
        """Load all trial results from storage.
        This method retrieves all previously saved trial results for analysis
        or comparison purposes.

        Returns:
            List of trial dictionaries, each containing the trial_id and
            all collected metric values for that trial.

        Raises:
            StorageError: If loading fails due to I/O or other errors.
        """
        pass
