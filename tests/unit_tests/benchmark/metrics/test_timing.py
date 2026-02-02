import pytest
import time

from src.benchmark.core.base import MetricContext, Stage
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.metrics.timing import TimingMetrics


def test_registers_timing():
    """Verify registration with registry."""
    metric_class = MetricRegistry.get("timing")
    assert metric_class is TimingMetrics


class TestTimingMetricsConfig:
    """Test configuration options."""

    def test_default_config(self):
        """Test with None/empty config."""
        metric = TimingMetrics(None)
        assert metric.stages == []

        metric = TimingMetrics({})
        assert metric.stages == []

    def test_custom_stages_config(self):
        """Test with custom stages list."""
        config = {"stages": ["asr", "llm", "tts"]}
        metric = TimingMetrics(config)
        assert metric.stages == ["asr", "llm", "tts"]


class TestTimingMetricsTotalLatency:
    """Test total latency calculation."""

    def test_total_latency(self):
        """Verify total latency calculation."""
        metric = TimingMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)
        # Simulate some work
        time.sleep(0.01)
        results = metric.end(context)

        assert "total_latency_seconds" in results
        assert results["total_latency_seconds"] >= 0.01
        assert results["total_latency_seconds"] < 1.0  # Should be fast

    def test_total_latency_zero_when_no_start(self):
        """Handle case where start was not called."""
        metric = TimingMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        results = metric.end(context)

        assert results["total_latency_seconds"] == 0.0
        assert results["stage_latencies"] == {}


class TestTimingMetricsStageTiming:
    """Test individual stage timing."""

    def test_single_stage_timing(self):
        """Test start/end for one stage."""
        metric = TimingMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)
        metric.record_stage_start("asr")
        time.sleep(0.01)
        metric.record_stage_end("asr")
        results = metric.end(context)

        assert "asr" in results["stage_latencies"]
        assert results["stage_latencies"]["asr"] >= 0.01

    def test_multiple_stage_timing(self):
        """Test multiple stages tracked."""
        metric = TimingMetrics({"stages": ["asr", "llm", "tts"]})
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        # ASR
        metric.record_stage_start("asr")
        time.sleep(0.01)
        metric.record_stage_end("asr")

        # LLM
        metric.record_stage_start("llm")
        time.sleep(0.02)
        metric.record_stage_end("llm")

        # TTS
        metric.record_stage_start("tts")
        time.sleep(0.01)
        metric.record_stage_end("tts")

        results = metric.end(context)

        assert "asr" in results["stage_latencies"]
        assert "llm" in results["stage_latencies"]
        assert "tts" in results["stage_latencies"]

        assert results["stage_latencies"]["asr"] >= 0.01
        assert results["stage_latencies"]["llm"] >= 0.02
        assert results["stage_latencies"]["tts"] >= 0.01

        assert len(results["stage_latencies"]) == 3

    def test_stage_order_recorded(self):
        """Verify stages are tracked in execution order."""
        metric = TimingMetrics()
        context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)
        metric.record_stage_start("first")
        metric.record_stage_end("first")
        metric.record_stage_start("second")
        metric.record_stage_end("second")
        metric.record_stage_start("third")
        metric.record_stage_end("third")
        results = metric.end(context)

        assert list(results["stage_latencies"].keys()) == ["first", "second", "third"]


def test_double_stage_end():
    """Handle multiple end calls for same stage - uses last duration."""
    metric = TimingMetrics()
    context = MetricContext(
        stage=Stage.PIPELINE,
        trial_id="trial_001",
        config={},
        timestamp=0.0,
    )
    metric.start(context)
    metric.record_stage_start("double_end")
    time.sleep(0.01)
    metric.record_stage_end("double_end")
    first_duration = metric._stage_latencies["double_end"]
    # Call end again - should update to new duration
    metric.record_stage_end("double_end")
    results = metric.end(context)

    # Second end call updates to new measurement
    assert results["stage_latencies"]["double_end"] >= first_duration
