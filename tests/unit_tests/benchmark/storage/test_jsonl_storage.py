import json
from pathlib import Path

import pytest

from src.benchmark.storage.jsonl_storage import JSONLStorage


class TestJSONLStorage:
    """Tests for JSONLStorage class."""

    def test_init_creates_directory(self, tmp_path: Path):
        """Test that initialization creates the output directory."""
        output_dir = tmp_path / "benchmark" / "experiment_001"
        _ = JSONLStorage(output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_init_existing_directory(self, tmp_path: Path):
        """Test that initialization works with existing directory."""
        output_dir = tmp_path / "benchmark"
        output_dir.mkdir()
        _ = JSONLStorage(output_dir)

        assert output_dir.exists()

    def test_save_trial_creates_raw_logs(self, tmp_path: Path):
        """Test that saving a trial creates raw_logs.jsonl."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        trial_id = "trial_001"
        metrics = {"latency": 1.5, "wer": 0.1}
        storage.save_trial(trial_id, metrics)

        assert storage.raw_logs_path.exists()

    def test_save_single_trial(self, tmp_path: Path):
        """Test saving a single trial."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        trial_id = "trial_001"
        metrics = {"latency": 1.5, "wer": 0.1}
        storage.save_trial(trial_id, metrics)

        trials = storage.load_trials()
        assert len(trials) == 1
        assert trials[0]["sample_id"] == trial_id
        assert trials[0]["latency"] == 1.5
        assert trials[0]["wer"] == 0.1

    def test_save_multiple_trials(self, tmp_path: Path):
        """Test saving multiple trials."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        for i in range(5):
            trial_id = f"trial_{i:03d}"
            metrics = {"latency": 1.0 + i, "wer": 0.1 * i}
            storage.save_trial(trial_id, metrics)

        trials = storage.load_trials()
        assert len(trials) == 5

        # Verify each trial
        for i, trial in enumerate(trials):
            assert trial["sample_id"] == f"trial_{i:03d}"
            assert trial["latency"] == 1.0 + i
            assert trial["wer"] == 0.1 * i

    def test_save_trial_preserves_exp_name(self, tmp_path: Path):
        """Test that trial data includes experiment name."""
        output_dir = tmp_path / "my_experiment"
        storage = JSONLStorage(output_dir)

        storage.save_trial("trial_001", {"latency": 1.5})

        trials = storage.load_trials()
        assert trials[0]["exp_name"] == "my_experiment"

    def test_save_trial_includes_timestamp(self, tmp_path: Path):
        """Test that trial data includes timestamp."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        import time

        before = time.time()
        storage.save_trial("trial_001", {"latency": 1.5})
        after = time.time()

        trials = storage.load_trials()
        assert "timestamp_start" in trials[0]
        assert before <= trials[0]["timestamp_start"] <= after

    def test_save_summary(self, tmp_path: Path):
        """Test saving summary statistics."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        summary = {
            "mean_latency": 1.5,
            "median_latency": 1.4,
            "p95_latency": 2.0,
        }
        storage.save_summary(summary)

        assert storage.summary_path.exists()

        loaded = storage.load_summary()
        assert loaded["mean_latency"] == 1.5
        assert loaded["median_latency"] == 1.4
        assert loaded["p95_latency"] == 2.0
        assert loaded["experiment"] == "experiment_001"
        assert loaded["run_id"] == "experiment_001"
        assert "timestamp" in loaded

    def test_save_config(self, tmp_path: Path):
        """Test saving configuration."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        config = {
            "num_samples": 20,
            "metrics": ["latency", "wer"],
        }
        storage.save_config(config)

        assert storage.config_path.exists()

        loaded = storage.load_config()
        assert loaded["config"]["num_samples"] == 20
        assert loaded["config"]["metrics"] == ["latency", "wer"]
        assert "saved_at" in loaded

    def test_load_trials_empty_file(self, tmp_path: Path):
        """Test loading from non-existent file returns empty list."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        trials = storage.load_trials()
        assert trials == []

    def test_load_summary_nonexistent(self, tmp_path: Path):
        """Test loading non-existent summary returns None."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        summary = storage.load_summary()
        assert summary is None

    def test_load_config_nonexistent(self, tmp_path: Path):
        """Test loading non-existent config returns None."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        config = storage.load_config()
        assert config is None

    def test_jsonl_format_valid(self, tmp_path: Path):
        """Test that raw_logs.jsonl contains valid JSONL format."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        storage.save_trial("trial_001", {"latency": 1.5, "wer": 0.1})
        storage.save_trial("trial_002", {"latency": 2.0, "wer": 0.2})

        # Read raw file and verify each line is valid JSON
        with open(storage.raw_logs_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2

        for line in lines:
            data = json.loads(line)
            assert "sample_id" in data
            assert "exp_name" in data
            assert "timestamp_start" in data

    def test_get_trial_count(self, tmp_path: Path):
        """Test getting trial count."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        assert storage.get_trial_count() == 0

        storage.save_trial("trial_001", {"latency": 1.5})
        assert storage.get_trial_count() == 1

        storage.save_trial("trial_002", {"latency": 1.5})
        assert storage.get_trial_count() == 2

    def test_clear_trials(self, tmp_path: Path):
        """Test clearing trial data."""
        output_dir = tmp_path / "experiment_001"
        storage = JSONLStorage(output_dir)

        storage.save_trial("trial_001", {"latency": 1.5})
        assert storage.get_trial_count() == 1

        storage.clear_trials()
        assert storage.get_trial_count() == 0
        assert not storage.raw_logs_path.exists()
