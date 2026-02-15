from unittest.mock import MagicMock, patch

import pytest
import torch

from src.benchmark.core.base import MetricContext, Stage
from src.benchmark.metrics.hardware import HardwareMetrics


def _base_config(**overrides) -> dict:
    cfg = {
        "device": 0,
        "track_power": False,
        "track_fragmentation": False,
        "waste_threshold": 0.3,
    }
    cfg.update(overrides)
    return cfg


def _context() -> MetricContext:
    return MetricContext(stage=Stage.PIPELINE, trial_id="trial_001", config={}, timestamp=0.0)


@pytest.mark.parametrize("class_name", ["hardware_basic", "hardware_detailed"])
def test_registry_registration(class_name):
    from src.benchmark.core.registry import MetricRegistry

    assert MetricRegistry.get(class_name) is HardwareMetrics


def test_init_requires_device_key():
    with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
        mock_device.return_value = MagicMock()
        with pytest.raises(KeyError):
            HardwareMetrics({"track_power": False})


def test_config_assignment():
    with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
        mock_device.return_value = MagicMock()
        metric = HardwareMetrics(_base_config(device=1, track_power=True, track_fragmentation=True, waste_threshold=0.5))

    assert metric.device_index == 1
    assert metric.track_power is True
    assert metric.track_fragmentation is True
    assert metric.waste_threshold == 0.5


def test_returns_empty_dict_without_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
            mock_device.return_value = MagicMock()
            metric = HardwareMetrics(_base_config())
            metric.start(_context())
            assert metric.end(_context()) == {}


def test_basic_mode_collects_memory_metrics():
    mb = HardwareMetrics.GPU_MEM_BLOCK
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.memory_allocated", return_value=mb * 10):
            with patch("torch.cuda.memory_reserved", return_value=mb * 20):
                with patch("torch.cuda.max_memory_allocated", return_value=mb * 15):
                    with patch("torch.cuda.reset_peak_memory_stats"):
                        with patch("torch.cuda.synchronize"):
                            with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
                                mock_device.return_value = MagicMock()
                                metric = HardwareMetrics(_base_config())
                                metric.start(_context())
                                results = metric.end(_context())

    assert results["gpu_memory_allocated_mb"] == 10
    assert results["gpu_memory_reserved_mb"] == 20
    assert results["gpu_memory_peak_mb"] == 15
    assert results["gpu_memory_efficiency"] == 0.5


def test_detailed_mode_collects_fragmentation_metrics():
    mb = HardwareMetrics.GPU_MEM_BLOCK
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.memory_allocated", return_value=mb * 10):
            with patch("torch.cuda.memory_reserved", return_value=mb * 20):
                with patch("torch.cuda.max_memory_allocated", return_value=mb * 15):
                    with patch(
                        "torch.cuda.memory_stats",
                        return_value={
                            "segment.count": 5,
                            "inactive_split.all.alloc_count": 3,
                            "pool_fraction": 0.4,
                        },
                    ):
                        with patch("torch.cuda.reset_peak_memory_stats"):
                            with patch("torch.cuda.synchronize"):
                                with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
                                    mock_device.return_value = MagicMock()
                                    metric = HardwareMetrics(
                                        _base_config(track_fragmentation=True)
                                    )
                                    metric.start(_context())
                                    results = metric.end(_context())

    assert results["fragmentation_waste_ratio"] == 0.5
    assert results["inactive_blocks"] == 3
    assert results["segment_count"] == 5
    assert results["pool_fraction"] == 0.4
    assert results["is_fragmented"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "cfg",
    [
        _base_config(),
        _base_config(track_fragmentation=True),
    ],
    ids=["hardware_basic", "hardware_detailed"],
)
def test_real_cuda_metrics_smoke(cfg: dict):
    """Smoke test on real CUDA for both hardware_basic and hardware_detailed modes."""
    metric = HardwareMetrics({**cfg, "device": torch.cuda.current_device()})
    ctx = _context()
    metric.start(ctx)

    tensor = torch.randn(1024, 1024, device="cuda")
    _ = tensor * 2
    torch.cuda.synchronize()

    results = metric.end(ctx)
    del tensor

    assert results["gpu_memory_allocated_mb"] >= 0
    assert results["gpu_memory_reserved_mb"] >= 0
    assert results["gpu_memory_peak_mb"] >= 0
    assert results["gpu_memory_efficiency"] >= 0
    if cfg["track_fragmentation"]:
        assert "fragmentation_waste_ratio" in results
        assert "inactive_blocks" in results
        assert "segment_count" in results
        assert "pool_fraction" in results
        assert "is_fragmented" in results
