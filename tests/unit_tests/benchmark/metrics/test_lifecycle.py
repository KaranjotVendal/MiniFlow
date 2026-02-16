import pytest
import time

from src.benchmark.core.base import MetricContext, Stage
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.metrics.lifecycle import ModelLifecycleMetrics


def test_registers_model_lifecycle():
    """Verify registration with registry."""
    metric_class = MetricRegistry.get("model_lifecycle")
    assert metric_class is ModelLifecycleMetrics


# TODO: this tests can be made into one tests by using pytest.mark.parameterize
class TestModelLifecycleMetricsConfig:
    """Test configuration options."""

    def test_default_config(self):
        """Test with None/empty config."""
        metric = ModelLifecycleMetrics(None)
        assert metric.track_gpu_transfer is True

        metric = ModelLifecycleMetrics({})
        assert metric.track_gpu_transfer is True

    def test_custom_config(self):
        """Test with custom config."""
        config = {"track_gpu_transfer": False}
        metric = ModelLifecycleMetrics(config)
        assert metric.track_gpu_transfer is False


class TestModelLifecycleMetricsBasic:

    def test_record_load_start_end(self):
        """Test basic load start and end recording."""
        metric = ModelLifecycleMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        metric.record_load_start("whisper", "disk")
        time.sleep(0.01)
        event = metric.record_load_end(cached=False)

        assert event is not None
        assert event["model_name"] == "whisper"
        assert event["source"] == "disk"
        assert event["cached"] is False
        assert event["total_time"] >= 0.01
        assert event["disk_load_time"] >= 0.01
        assert event["gpu_transfer_time"] == 0.0  # Not tracked

    def test_cache_hit_tracking(self):
        """Test cache hit detection."""
        metric = ModelLifecycleMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        metric.record_load_start("llama", "cache")
        event = metric.record_load_end(cached=True)

        assert event["cached"] is True
        assert event["source"] == "cache"

    def test_multiple_loads(self):
        """Test multiple model loads in one trial."""
        metric = ModelLifecycleMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        # Load first model
        metric.record_load_start("whisper", "disk")
        time.sleep(0.01)
        metric.record_load_end(cached=False)

        # Load second model
        metric.record_load_start("llama", "remote")
        time.sleep(0.02)
        metric.record_load_end(cached=False)

        # Load cached model
        metric.record_load_start("tts", "cache")
        metric.record_load_end(cached=True)

        results = metric.end(context)

        assert results["models_loaded"] == 3
        assert results["cache_hits"] == 1
        assert results["cache_misses"] == 2
        assert len(results["load_events"]) == 3


class TestModelLifecycleMetricsGPUTransfer:
    """Test GPU transfer tracking."""

    def test_gpu_transfer_tracking(self):
        """Test tracking of disk load vs GPU transfer time."""
        metric = ModelLifecycleMetrics({"track_gpu_transfer": True})
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        # TODO: this also makes me question the design of the ModelLifeCycleMetrics class
        # standard practice is load_model.to(device="cuda")
        # how are we supposed measure times accurately for a chained operation?
        metric.record_load_start("whisper", "disk")
        time.sleep(0.01)  # Disk load
        metric.record_gpu_transfer_start()
        time.sleep(0.02)  # GPU transfer
        event = metric.record_load_end(cached=False)

        assert event is not None
        assert event["disk_load_time"] >= 0.01
        assert event["gpu_transfer_time"] >= 0.02
        assert event["total_time"] >= 0.03

    def test_gpu_transfer_disabled(self):
        """Test with GPU transfer tracking disabled."""
        metric = ModelLifecycleMetrics({"track_gpu_transfer": False})
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        metric.record_load_start("whisper", "disk")
        # GPU transfer start should be ignored
        metric.record_gpu_transfer_start()
        event = metric.record_load_end(cached=False)

        assert event["gpu_transfer_time"] == 0.0


def test_total_load_time_calculation():
    """Test total load time is sum of all events."""
    metric = ModelLifecycleMetrics()
    context = MetricContext(
        stage=Stage.PIPELINE,
        trial_id="trial_001",
        config={},
        timestamp=0.0,
    )
    metric.start(context)

    metric.record_load_start("model1", "disk")
    time.sleep(0.01)
    metric.record_load_end(cached=False)

    metric.record_load_start("model2", "disk")
    time.sleep(0.02)
    metric.record_load_end(cached=False)

    results = metric.end(context)

    assert results["total_model_load_time"] >= 0.03
    assert results["models_loaded"] == 2


def test_multiple_trials_independent():
    """Verify trials have independent tracking."""
    metric = ModelLifecycleMetrics()
    context1 = MetricContext(
        stage=Stage.PIPELINE,
        trial_id="trial_001",
        config={},
        timestamp=0.0,
    )
    context2 = MetricContext(
        stage=Stage.PIPELINE,
        trial_id="trial_002",
        config={},
        timestamp=0.0,
    )

    # First trial
    metric.start(context1)
    metric.record_load_start("model1", "disk")
    metric.record_load_end(cached=False)
    results1 = metric.end(context1)

    # Second trial
    metric.start(context2)
    metric.record_load_start("model2", "cache")
    metric.record_load_end(cached=True)
    results2 = metric.end(context2)

    assert results1["models_loaded"] == 1
    assert results2["models_loaded"] == 1
    assert results1["cache_hits"] == 0
    assert results2["cache_hits"] == 1
