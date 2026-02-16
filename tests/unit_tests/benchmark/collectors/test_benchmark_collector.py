import time
from unittest.mock import MagicMock, patch

import pytest

from src.benchmark.collectors import BenchmarkCollector
from src.benchmark.metrics.hardware import HardwareMetrics
from src.benchmark.metrics.lifecycle import ModelLifecycleMetrics
from src.benchmark.metrics.quality import QualityMetrics
from src.benchmark.metrics.timing import TimingMetrics
from src.benchmark.metrics.tokens import TokenMetrics


class TestBenchmarkCollector:
    def test_start_end_trial_collects_metrics(self):
        timing = TimingMetrics({})
        tokens = TokenMetrics({"track_ttft": True})
        lifecycle = ModelLifecycleMetrics({})
        with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
            mock_device.return_value = MagicMock()
            hardware = HardwareMetrics(
                {
                    "device": 0,
                    "track_power": False,
                    "track_fragmentation": False,
                    "waste_threshold": 0.3,
                }
            )
        quality = QualityMetrics({"evaluators": ["wer"]})
        collector = BenchmarkCollector(
            metrics={
                "timing": timing,
                "tokens": tokens,
                "model_lifecycle": lifecycle,
                "hardware_basic": hardware,
                "quality": quality,
            },
            config={"benchmark": {"enable_streaming_audio": False}},
        )

        collector.start_trial(trial_id="trial_1", sample_id="sample_1", is_warmup=False)

        collector.timing_metrics.record_stage_start("llm")
        time.sleep(0.01)
        collector.start_token_metrics()
        collector.token_metrics.add_tokens(8)
        collector.finalize_token_metrics()
        collector.timing_metrics.record_stage_end("llm")

        collector.lifecycle_metrics.record_load_start(
            model_name="Qwen/Qwen2.5-3B-Instruct", source="disk"
        )
        time.sleep(0.005)
        event = collector.lifecycle_metrics.record_load_end(cached=False)
        if event is not None:
            event["stage"] = "llm"
            event["success"] = True

        result = collector.end_trial(status="success")

        assert result["trial_id"] == "trial_1"
        assert result["sample_id"] == "sample_1"
        assert result["status"] == "success"
        assert result["error"] is None
        assert result["tokens_generated"] == 8
        assert result["ttft_mode"] == "proxy_non_streaming"
        assert result["ttft"] == result["total_generation_time"]
        assert result["tokens_per_sec_total"] > 0
        assert "total_latency_seconds" in result
        assert "latencies" in result
        assert "llm" in result["latencies"]
        assert "llm_model_load" in result
        assert event is not None
        assert event["stage"] == "llm"
        assert event["success"] is True

    def test_context_requires_active_trial(self):
        with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
            mock_device.return_value = MagicMock()
            collector = BenchmarkCollector(
                metrics={
                    "timing": TimingMetrics({}),
                    "tokens": TokenMetrics({"track_ttft": True}),
                    "model_lifecycle": ModelLifecycleMetrics({}),
                    "hardware_basic": HardwareMetrics(
                        {
                            "device": 0,
                            "track_power": False,
                            "track_fragmentation": False,
                            "waste_threshold": 0.3,
                        }
                    ),
                    "quality": QualityMetrics({"evaluators": ["wer"]}),
                },
                config={"benchmark": {"enable_streaming_audio": False}},
            )
        with pytest.raises(RuntimeError):
            _ = collector.context

    def test_record_phase_metrics_saved(self):
        with patch("src.benchmark.metrics.hardware.nvitop.Device") as mock_device:
            mock_device.return_value = MagicMock()
            collector = BenchmarkCollector(
                metrics={
                    "timing": TimingMetrics({}),
                    "tokens": TokenMetrics({"track_ttft": True}),
                    "model_lifecycle": ModelLifecycleMetrics({}),
                    "hardware_basic": HardwareMetrics(
                        {
                            "device": 0,
                            "track_power": False,
                            "track_fragmentation": False,
                            "waste_threshold": 0.3,
                        }
                    ),
                    "quality": QualityMetrics({"evaluators": ["wer"]}),
                },
                config={"benchmark": {"enable_streaming_audio": False}},
            )
        collector.start_trial("trial_1")
        collector.record_phase_metrics(
            "asr_inference_gpu_metrics", {"gpu_memory_peak_mb": 123}
        )
        result = collector.end_trial(status="success")
        assert "hardware_phase_metrics" in result
        assert (
            result["hardware_phase_metrics"]["asr_inference"]["gpu_memory_peak_mb"]
            == 123
        )
