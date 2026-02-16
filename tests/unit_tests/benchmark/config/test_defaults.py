import pytest

from src.benchmark.config.defaults import (
    get_default_metrics_config,
    get_default_hardware_config,
    get_default_timing_config,
    get_default_tokens_config,
    get_default_quality_config,
)


class TestDefaultConfigs:
    def test_get_default_metrics_config(self):
        result = get_default_metrics_config()
        assert result == {"enabled": [], "configurations": {}}

    def test_get_default_hardware_config(self):
        result = get_default_hardware_config()
        assert result["device"] == 0
        assert result["track_power"] is False
        assert result["track_fragmentation"] is False
        assert result["waste_threshold"] == 0.3

    def test_get_default_timing_config(self):
        result = get_default_timing_config()
        assert "asr" in result["stages"]
        assert "llm" in result["stages"]
        assert "tts" in result["stages"]
        assert "pipeline" in result["stages"]

    def test_get_default_tokens_config(self):
        result = get_default_tokens_config()
        assert result["track_ttft"] is True

    def test_get_default_quality_config(self):
        result = get_default_quality_config()
        assert "wer" in result["evaluators"]
        assert "utmos" in result["evaluators"]
