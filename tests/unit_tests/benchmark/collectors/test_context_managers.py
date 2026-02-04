import time
from typing import Any

import pytest
import torch

from src.benchmark.collectors.context_managers import track_latency, track_memory


class TestTrackLatency:
    """Tests for track_latency context manager."""

    def test_basic_latency_tracking(self):
        """Test that latency is recorded correctly in metrics dict."""
        metrics: dict[str, Any] = {}
        with track_latency("inference", metrics):
            time.sleep(0.01)

        assert "inference_latency_seconds" in metrics
        assert metrics["inference_latency_seconds"] >= 0.01
        assert metrics["inference_latency_seconds"] < 1.0

    def test_latency_with_no_operations(self, monkeypatch):
        """Test latency tracking with minimal work."""
        metrics: dict[str, Any] = {}
        with track_latency("empty", metrics):
            pass

        assert "empty_latency_seconds" in metrics
        assert metrics["empty_latency_seconds"] < 0.1

    def test_multiple_context_managers(self):
        """Test multiple latency context managers in sequence."""
        metrics1: dict[str, Any] = {}
        metrics2: dict[str, Any] = {}

        with track_latency("first", metrics1):
            time.sleep(0.005)
        with track_latency("second", metrics2):
            time.sleep(0.01)

        assert "first_latency_seconds" in metrics1
        assert "second_latency_seconds" in metrics2
        assert metrics1["first_latency_seconds"] < metrics2["second_latency_seconds"]

    def test_exception_handling(self):
        """Test that latency is recorded even when exception occurs."""
        metrics: dict[str, Any] = {}
        with pytest.raises(ValueError):
            with track_latency("failing", metrics):
                time.sleep(0.01)
                raise ValueError("Test error")

        assert "failing_latency_seconds" in metrics
        assert metrics["failing_latency_seconds"] >= 0.01

    def test_latency_stored_in_shared_dict(self):
        """Test that multiple metrics accumulate in same dict."""
        metrics: dict[str, Any] = {}
        with track_latency("op1", metrics):
            time.sleep(0.005)
        with track_latency("op2", metrics):
            time.sleep(0.01)

        assert "op1_latency_seconds" in metrics
        assert "op2_latency_seconds" in metrics
        assert metrics["op2_latency_seconds"] > metrics["op1_latency_seconds"]

    def test_latency_precision(self):
        """Test that latency is rounded to 6 decimal places."""
        metrics: dict[str, Any] = {}
        with track_latency("precise", metrics):
            pass

        value = metrics["precise_latency_seconds"]
        assert isinstance(value, float)
        assert len(str(value).split(".")[1]) <= 6 if "." in str(value) else True


class TestTrackMemory:
    """Tests for track_memory context manager."""

    def test_memory_tracking_with_cuda(self):
        """Test memory delta tracking when CUDA is available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metrics: dict[str, Any] = {}
        with track_memory("test_alloc", metrics):
            _tensor = torch.zeros(100, 100, device="cuda")

        assert "test_alloc_memory_delta_mb" in metrics
        assert isinstance(metrics["test_alloc_memory_delta_mb"], int)

    def test_memory_tracking_no_cuda(self, monkeypatch):
        """Test memory delta tracking when CUDA is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        metrics: dict[str, Any] = {}
        with track_memory("test_no_cuda", metrics):
            pass

        assert "test_no_cuda_memory_delta_mb" in metrics
        assert metrics["test_no_cuda_memory_delta_mb"] == 0

    def test_memory_exception_handling(self):
        """Test that memory delta is recorded even when exception occurs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metrics: dict[str, Any] = {}
        with pytest.raises(RuntimeError):
            with track_memory("failing_alloc", metrics):
                tensor = torch.zeros(100, 100, device="cuda")
                raise RuntimeError("Test error")

        assert "failing_alloc_memory_delta_mb" in metrics

    def test_memory_tracking_with_deallocation(self):
        """Test memory tracking with allocation and deallocation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metrics: dict[str, Any] = {}
        with track_memory("alloc_dealloc", metrics):
            tensor = torch.zeros(100, 100, device="cuda")
            del tensor

        assert "alloc_dealloc_memory_delta_mb" in metrics

    def test_multiple_memory_operations(self):
        """Test multiple memory tracking operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metrics: dict[str, Any] = {}
        with track_memory("first_alloc", metrics):
            t1 = torch.zeros(50, 50, device="cuda")
        with track_memory("second_alloc", metrics):
            t2 = torch.zeros(100, 100, device="cuda")

        assert "first_alloc_memory_delta_mb" in metrics
        assert "second_alloc_memory_delta_mb" in metrics
        assert (
            metrics["second_alloc_memory_delta_mb"]
            >= metrics["first_alloc_memory_delta_mb"]
        )
