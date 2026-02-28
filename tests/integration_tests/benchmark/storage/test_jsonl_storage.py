"""
Integration tests for storage components.

These tests verify storage operations with real file I/O.
"""

import json
from pathlib import Path

import pytest

from src.benchmark.storage.jsonl_storage import JSONLStorage


class TestJSONLStorageIntegration:
    """Integration tests for JSONLStorage with real file operations."""

    @pytest.fixture
    def storage_dir(self, tmp_path: Path) -> Path:
        """Create a temporary storage directory."""
        return tmp_path / "storage"

    @pytest.fixture
    def storage(self, storage_dir: Path) -> JSONLStorage:
        """Create a JSONLStorage instance."""
        return JSONLStorage(storage_dir)

    @pytest.fixture
    def valid_trial(self) -> dict:
        """Create a valid trial payload matching the schema."""
        return {
            "trial_id": "test_trial_1",
            "sample_id": "sample_1",
            "is_warmup": False,
            "status": "success",
            "error": None,
            "trial_wall_time_seconds": 1.5,
            "latencies": {
                "asr": 1.0,
                "llm": 2.0,
                "tts": 3.0,
            },
        }

    def test_storage_creates_directories(self, storage_dir):
        """Test that storage creates required directories."""
        JSONLStorage(storage_dir)
        assert storage_dir.exists()
        assert storage_dir.is_dir()

    def test_save_and_retrieve_trial(self, storage: JSONLStorage, valid_trial: dict):
        """Test saving and retrieving a trial."""
        storage.save_trial("test_trial_1", valid_trial)

        # Load trials back
        trials = storage.load_trials()

        assert len(trials) == 1
        assert trials[0]["trial_id"] == "test_trial_1"
        assert trials[0]["latencies"]["asr"] == 1.0

    def test_multiple_trials_persistence(self, storage: JSONLStorage):
        """Test saving multiple trials."""
        for i in range(5):
            trial = {
                "trial_id": f"trial_{i}",
                "sample_id": f"sample_{i}",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": i * 1.5,
                "latencies": {"asr": float(i)},
            }
            storage.save_trial(f"trial_{i}", trial)

        trials = storage.load_trials()

        assert len(trials) == 5
        # Verify order is preserved
        assert trials[0]["trial_id"] == "trial_0"
        assert trials[4]["trial_id"] == "trial_4"

    def test_save_and_load_summary(self, storage: JSONLStorage):
        """Test saving and loading experiment summary."""
        summary = {
            "meta": {
                "experiment": "test_exp",
                "run_id": "123_test",
                "timestamp": "2024-01-01T00:00:00",
                "num_trials": 10,
                "num_warmup_trials": 2,
                "status_counts": {"success": 10, "error": 0},
            },
            "pipeline": {
                "latency": {
                    "trial_wall_time_seconds": {
                        "mean": 1.5,
                        "median": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "std": 0.5,
                        "p95": 1.9,
                        "p99": 2.0,
                        "count": 10,
                    }
                },
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
            "asr": {
                "latency": {
                    "inference_seconds": {
                        "mean": 1.0,
                        "median": 1.0,
                        "min": 1.0,
                        "max": 1.0,
                        "std": 0.0,
                        "p95": 1.0,
                        "p99": 1.0,
                        "count": 10,
                    }
                },
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
            "llm": {
                "latency": {
                    "inference_seconds": {
                        "mean": 2.0,
                        "median": 2.0,
                        "min": 2.0,
                        "max": 2.0,
                        "std": 0.0,
                        "p95": 2.0,
                        "p99": 2.0,
                        "count": 10,
                    }
                },
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
            "tts": {
                "latency": {
                    "inference_seconds": {
                        "mean": 3.0,
                        "median": 3.0,
                        "min": 3.0,
                        "max": 3.0,
                        "std": 0.0,
                        "p95": 3.0,
                        "p99": 3.0,
                        "count": 10,
                    }
                },
                "hardware": {},
                "load_times": {},
                "inference": {},
                "quality": {},
            },
        }

        storage.save_summary(summary)

        # Check file exists
        summary_file = storage.output_dir / "summary.json"
        assert summary_file.exists()

        # Load and verify
        loaded = storage.load_summary()
        assert loaded["meta"]["experiment"] == "test_exp"
        assert loaded["meta"]["num_trials"] == 10

    def test_save_config(self, storage: JSONLStorage):
        """Test saving configuration."""
        config = {
            "experiment": {"name": "test"},
            "dataset": {"num_samples": 10},
            "metrics": {"enabled": ["timing"]},
        }

        storage.save_config(config)

        config_file = storage.output_dir / "config.json"
        assert config_file.exists()

    def test_trial_count(self, storage: JSONLStorage):
        """Test trial count functionality."""
        # Empty initially
        assert storage.get_trial_count() == 0

        # Add trials
        for i in range(3):
            trial = {
                "trial_id": f"trial_{i}",
                "sample_id": f"sample_{i}",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 1.0,
                "latencies": {"asr": 1.0},
            }
            storage.save_trial(f"trial_{i}", trial)

        assert storage.get_trial_count() == 3

    def test_jsonl_format_valid(self, storage: JSONLStorage):
        """Test that saved data is valid JSONL format."""
        trials = [
            {
                "trial_id": "t1",
                "sample_id": "s1",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 1.0,
                "latencies": {"asr": 1.0},
            },
            {
                "trial_id": "t2",
                "sample_id": "s2",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 2.0,
                "latencies": {"asr": 2.0},
            },
        ]

        for trial in trials:
            storage.save_trial(trial["trial_id"], trial)

        # Read raw file
        raw_logs = storage.output_dir / "raw_logs.jsonl"
        with open(raw_logs) as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Each line should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "trial_id" in parsed

    def test_clear_trials(self, storage: JSONLStorage):
        """Test clearing trial data."""
        # Add trials
        for i in range(3):
            trial = {
                "trial_id": f"trial_{i}",
                "sample_id": f"sample_{i}",
                "is_warmup": False,
                "status": "success",
                "error": None,
                "trial_wall_time_seconds": 1.0,
                "latencies": {"asr": 1.0},
            }
            storage.save_trial(f"trial_{i}", trial)

        assert storage.get_trial_count() == 3

        # Clear
        storage.clear_trials()

        assert storage.get_trial_count() == 0

        # Verify file is empty/removed
        trials = storage.load_trials()
        assert len(trials) == 0


class TestStorageErrorHandling:
    """Integration tests for storage error handling."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> JSONLStorage:
        return JSONLStorage(tmp_path / "storage")

    def test_invalid_trial_rejected(self, storage: JSONLStorage):
        """Test that invalid trial data raises validation error."""
        # Trial missing required fields should raise ValueError
        invalid_trial = {"status": "success"}  # missing required fields

        with pytest.raises(ValueError):
            storage.save_trial("unnamed", invalid_trial)

    def test_nonexistent_summary_returns_none(self, tmp_path: Path):
        """Test loading non-existent summary returns None."""
        storage = JSONLStorage(tmp_path / "storage")

        result = storage.load_summary()
        assert result is None


class TestStorageFileOperations:
    """Integration tests for storage file operations."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> JSONLStorage:
        return JSONLStorage(tmp_path / "storage")

    def test_config_file_saved(self, storage: JSONLStorage):
        """Test that config file is saved correctly."""
        config = {"test": "config"}

        storage.save_config(config)

        config_file = storage.output_dir / "config.json"
        assert config_file.exists()

    def test_output_dir_structure(self, storage: JSONLStorage):
        """Test that output directory has correct structure."""
        # Storage should create the directory
        assert storage.output_dir.exists()
        assert storage.output_dir.is_dir()
