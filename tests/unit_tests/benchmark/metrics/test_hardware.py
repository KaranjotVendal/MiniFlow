import pytest
from unittest.mock import MagicMock, patch

from src.benchmark.metrics.hardware import HardwareMetrics
from src.benchmark.core.base import MetricContext, Stage


@pytest.mark.parametrize("class_name", [("hardware_basic"), ("hardware_detailed")])
def test_registers_under_hardware_basic(class_name):
    from src.benchmark.core.registry import MetricRegistry

    metric_class = MetricRegistry.get(class_name)
    assert metric_class is HardwareMetrics


@pytest.mark.parametrize(
    "config,expected",
    [
        (
            None,
            {
                "device": 0,
                "track_power": False,
                "track_fragmentation": False,
                "waste_threshold": 0.3,
            },
        ),
        (
            {
                "device": 1,
                "track_power": True,
                "track_fragmentation": True,
                "waste_threshold": 0.5,
            },
            {
                "device": 1,
                "track_power": True,
                "track_fragmentation": True,
                "waste_threshold": 0.5,
            },
        ),
        (
            {"device": 0},
            {
                "device": 0,
                "track_power": False,
                "track_fragmentation": False,
                "waste_threshold": 0.3,
            },
        ),
    ],
)
def test_config_options(config, expected):
    """Test configuration options with parameterization."""
    with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
        mock_device.return_value = MagicMock()
        metric = HardwareMetrics(config)
        assert metric.device == expected["device"]
        assert metric.track_power == expected["track_power"]
        assert metric.track_fragmentation == expected["track_fragmentation"]
        assert metric.waste_threshold == expected["waste_threshold"]


def test_gpu_mem_block_constant():
    """Verify the GPU_MEM_BLOCK constant is correctly defined."""
    assert HardwareMetrics.GPU_MEM_BLOCK == 1024 * 1024


class TestHardwareMetricsBasicMode:
    """Test basic mode (hardware_basic) without CUDA."""

    def test_returns_empty_dict_without_cuda(self):
        """Should return empty dict when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
                mock_device.return_value = MagicMock()
                metric = HardwareMetrics()
                context = MetricContext(
                    stage=Stage.PIPELINE,
                    trial_id="trial_001",
                    config={},
                    timestamp=0.0,
                )
                metric.start(context)
                results = metric.end(context)
                assert results == {}

    def test_basic_mode_collects_memory_metrics(self):
        """Should collect 4 basic memory metrics."""
        MB = HardwareMetrics.GPU_MEM_BLOCK
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=MB * 10):
                with patch("torch.cuda.memory_reserved", return_value=MB * 20):
                    with patch("torch.cuda.max_memory_allocated", return_value=MB * 15):
                        with patch("torch.cuda.reset_peak_memory_stats"):
                            with patch("torch.cuda.synchronize"):
                                with patch(
                                    "src.benchmark.metrics.hardware.nvitop.Device"
                                ) as mock_device:
                                    mock_device.return_value = MagicMock()
                                    metric = HardwareMetrics(
                                        {
                                            "track_power": False,
                                            "track_fragmentation": False,
                                        }
                                    )
                                    context = MetricContext(
                                        stage=Stage.PIPELINE,
                                        trial_id="trial_001",
                                        config={},
                                        timestamp=0.0,
                                    )
                                    metric.start(context)
                                    results = metric.end(context)

                                    assert "gpu_memory_allocated_mb" in results
                                    assert "gpu_memory_reserved_mb" in results
                                    assert "gpu_memory_peak_mb" in results
                                    assert "gpu_memory_efficiency" in results

                                    assert results["gpu_memory_allocated_mb"] == 10
                                    assert results["gpu_memory_reserved_mb"] == 20
                                    assert results["gpu_memory_peak_mb"] == 15
                                    assert results["gpu_memory_efficiency"] == 0.5


class TestHardwareMetricsDetailedMode:
    def test_detailed_mode_collects_fragmentation_metrics(self):
        """Should collect fragmentation metrics when enabled."""
        MB = HardwareMetrics.GPU_MEM_BLOCK
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=MB * 10):
                with patch("torch.cuda.memory_reserved", return_value=MB * 20):
                    with patch("torch.cuda.max_memory_allocated", return_value=MB * 15):
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
                                    with patch(
                                        "src.benchmark.metrics.hardware.nvitop.Device"
                                    ) as mock_device:
                                        mock_device.return_value = MagicMock()
                                        metric = HardwareMetrics(
                                            {
                                                "track_power": False,
                                                "track_fragmentation": True,
                                                "waste_threshold": 0.3,
                                            }
                                        )
                                        context = MetricContext(
                                            stage=Stage.PIPELINE,
                                            trial_id="trial_001",
                                            config={},
                                            timestamp=0.0,
                                        )
                                        metric.start(context)
                                        results = metric.end(context)

                                        assert "fragmentation_waste_ratio" in results
                                        assert "inactive_blocks" in results
                                        assert "segment_count" in results
                                        assert "pool_fraction" in results
                                        assert "is_fragmented" in results

                                        assert (
                                            results["fragmentation_waste_ratio"] == 0.5
                                        )
                                        assert results["inactive_blocks"] == 3
                                        assert results["segment_count"] == 5
                                        assert results["pool_fraction"] == 0.4
                                        assert results["is_fragmented"] is True

    def test_is_fragmented_based_on_threshold(self):
        """is_fragmented should be True when waste exceeds threshold."""
        MB = HardwareMetrics.GPU_MEM_BLOCK
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=MB * 18):
                with patch("torch.cuda.memory_reserved", return_value=MB * 20):
                    with patch("torch.cuda.max_memory_allocated", return_value=MB * 18):
                        with patch(
                            "torch.cuda.memory_stats",
                            return_value={
                                "segment.count": 1,
                                "inactive_split.all.alloc_count": 0,
                                "pool_fraction": 0.9,
                            },
                        ):
                            with patch("torch.cuda.reset_peak_memory_stats"):
                                with patch("torch.cuda.synchronize"):
                                    with patch(
                                        "src.benchmark.metrics.hardware.nvitop.Device"
                                    ) as mock_device:
                                        mock_device.return_value = MagicMock()
                                        metric = HardwareMetrics(
                                            {
                                                "track_power": False,
                                                "track_fragmentation": True,
                                                "waste_threshold": 0.3,
                                            }
                                        )
                                        context = MetricContext(
                                            stage=Stage.PIPELINE,
                                            trial_id="trial_001",
                                            config={},
                                            timestamp=0.0,
                                        )
                                        metric.start(context)
                                        results = metric.end(context)

                                        assert results["is_fragmented"] is False
                                        assert (
                                            results["fragmentation_waste_ratio"] == 0.1
                                        )


class TestHardwareMetricsPowerTracking:
    def test_power_tracking_with_nvitop(self):
        """Should collect power, temperature, and utilization when enabled."""
        MB = HardwareMetrics.GPU_MEM_BLOCK
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=MB * 10):
                with patch("torch.cuda.memory_reserved", return_value=MB * 20):
                    with patch("torch.cuda.max_memory_allocated", return_value=MB * 15):
                        with patch("torch.cuda.reset_peak_memory_stats"):
                            with patch("torch.cuda.synchronize"):
                                with patch(
                                    "src.benchmark.metrics.hardware.nvitop.Device"
                                ) as mock_device_class:
                                    mock_device = MagicMock()
                                    mock_device.power_draw.return_value = (
                                        18000  # 18W in milliwatts
                                    )
                                    mock_device.temperature.return_value = 65
                                    mock_device.utilization.return_value = 45
                                    mock_device_class.return_value = mock_device

                                    metric = HardwareMetrics(
                                        {
                                            "track_power": True,
                                            "track_fragmentation": False,
                                        }
                                    )
                                    context = MetricContext(
                                        stage=Stage.PIPELINE,
                                        trial_id="trial_001",
                                        config={},
                                        timestamp=0.0,
                                    )
                                    metric.start(context)
                                    results = metric.end(context)

                                    assert "gpu_power_draw_watts" in results
                                    assert "gpu_temperature_celsius" in results
                                    assert "gpu_utilization_percent" in results

                                    assert results["gpu_power_draw_watts"] == 18.0
                                    assert results["gpu_temperature_celsius"] == 65.0
                                    assert results["gpu_utilization_percent"] == 45


class TestHardwareMetricsHelperMethods:
    def test_helper_methods_return_correct_values(self):
        """Helper methods should return values in MB."""
        MB = HardwareMetrics.GPU_MEM_BLOCK
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=MB * 8):
                with patch("torch.cuda.memory_reserved", return_value=MB * 16):
                    with patch("torch.cuda.max_memory_allocated", return_value=MB * 12):
                        with patch(
                            "torch.cuda.max_memory_reserved",
                            return_value=MB * 20,
                        ):
                            with patch("torch.cuda.memory_stats", return_value={}):
                                with patch("torch.cuda.reset_peak_memory_stats"):
                                    with patch(
                                        "src.benchmark.metrics.hardware.nvitop.Device"
                                    ) as mock_device:
                                        mock_device.return_value = MagicMock()
                                        metric = HardwareMetrics()

                                        assert metric._get_gpu_memory_allocated() == 8
                                        assert metric._get_gpu_memory_reserved() == 16
                                        assert (
                                            metric._get_gpu_memory_peak_allocated()
                                            == 12
                                        )
                                        assert (
                                            metric._get_gpu_memory_peak_reserved() == 20
                                        )


class TestHardwareMetricsEdgeCases:
    """Test edge cases and error handling."""

    def test_efficiency_calculation_with_zero_reserved(self):
        """Should handle zero reserved memory gracefully."""
        MB = HardwareMetrics.GPU_MEM_BLOCK
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.memory_allocated", return_value=MB * 10):
                with patch("torch.cuda.memory_reserved", return_value=0):
                    with patch("torch.cuda.max_memory_allocated", return_value=MB * 10):
                        with patch("torch.cuda.reset_peak_memory_stats"):
                            with patch("torch.cuda.synchronize"):
                                with patch(
                                    "src.benchmark.metrics.hardware.nvitop.Device"
                                ) as mock_device:
                                    mock_device.return_value = MagicMock()
                                    metric = HardwareMetrics()
                                    context = MetricContext(
                                        stage=Stage.PIPELINE,
                                        trial_id="trial_001",
                                        config={},
                                        timestamp=0.0,
                                    )
                                    metric.start(context)
                                    results = metric.end(context)

                                    assert results["gpu_memory_efficiency"] == 0.0
