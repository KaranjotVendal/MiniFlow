import json
from pathlib import Path

import pytest

from src.benchmark.storage.jsonl_storage import JSONLStorage


def _valid_trial(trial_id: str, is_warmup: bool = False) -> dict:
    return {
        "trial_id": trial_id,
        "sample_id": trial_id,
        "is_warmup": is_warmup,
        "status": "success",
        "error": None,
        "trial_wall_time_seconds": 0.01,
        "latencies": {"asr": 0.001},
    }


def _stats(value: float, count: int = 1) -> dict:
    return {
        "mean": value,
        "median": value,
        "min": value,
        "max": value,
        "std": 0.0,
        "p95": value,
        "p99": value,
        "count": count,
    }


def _valid_summary() -> dict:
    return {
        "meta": {
            "experiment": "experiment_001",
            "run_id": "123_experiment_001",
            "timestamp": "2026-02-15T00:00:00",
            "num_trials": 1,
            "num_warmup_trials": 0,
            "status_counts": {"success": 1, "error": 0},
        },
        "pipeline": {
            "latency": {"total_latency_seconds": _stats(1.0)},
            "hardware": {},
            "load_times": {},
            "inference": {},
            "quality": {},
        },
        "asr": {"latency": {}, "hardware": {}, "load_times": {}, "inference": {}, "quality": {}},
        "llm": {"latency": {}, "hardware": {}, "load_times": {}, "inference": {}, "quality": {}},
        "tts": {"latency": {}, "hardware": {}, "load_times": {}, "inference": {}, "quality": {}},
    }


class TestJSONLStorage:
    def test_init_creates_directory(self, tmp_path: Path):
        output_dir = tmp_path / "benchmark" / "experiment_001"
        _ = JSONLStorage(output_dir)
        assert output_dir.exists()

    def test_save_and_load_trials(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        storage.save_trial("trial_001", _valid_trial("trial_001"))
        storage.save_trial("trial_002", _valid_trial("trial_002"))

        trials = storage.load_trials()
        assert len(trials) == 2
        assert trials[0]["trial_id"] == "trial_001"
        assert trials[1]["trial_id"] == "trial_002"

    def test_jsonl_format_valid(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        storage.save_trial("trial_001", _valid_trial("trial_001"))
        storage.save_trial("trial_002", _valid_trial("trial_002"))

        lines = storage.raw_logs_path.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "trial_id" in data
            assert "latencies" in data

    def test_save_and_load_summary(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        payload = _valid_summary()
        storage.save_summary(payload)

        loaded = storage.load_summary()
        assert loaded is not None
        assert loaded["meta"]["experiment"] == "experiment_001"
        assert loaded["pipeline"]["latency"]["total_latency_seconds"]["mean"] == 1.0

    def test_save_config(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        cfg = {"dataset": {"num_samples": 20}}
        storage.save_config(cfg)
        loaded = storage.load_config()
        assert loaded["dataset"]["num_samples"] == 20
        assert "saved_at" in loaded

    def test_get_trial_count_and_clear(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        assert storage.get_trial_count() == 0
        storage.save_trial("trial_001", _valid_trial("trial_001"))
        assert storage.get_trial_count() == 1
        storage.clear_trials()
        assert storage.get_trial_count() == 0
        assert not storage.raw_logs_path.exists()

    def test_invalid_trial_rejected(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        with pytest.raises(ValueError, match="Invalid trial payload"):
            storage.save_trial("trial_001", {"latencies": {"asr": 0.1}})

    def test_invalid_summary_rejected(self, tmp_path: Path):
        storage = JSONLStorage(tmp_path / "experiment_001")
        with pytest.raises(ValueError, match="Invalid summary payload"):
            storage.save_summary({"run_id": 123})
