from unittest.mock import MagicMock

import pytest
import torch

from src.stt.stt_pipeline import run_asr


class _FakeASRPipeline:
    def __call__(self, audio):
        assert "array" in audio
        assert "sampling_rate" in audio
        return {"text": "hello world"}


def _collector_mock() -> MagicMock:
    collector = MagicMock()
    collector.timing_metrics = MagicMock()
    collector.lifecycle_metrics = MagicMock()
    collector.hardware_metrics = MagicMock()
    collector.quality_metrics = MagicMock()
    collector.context = MagicMock()
    collector.current_trial = MagicMock()
    collector.current_trial.quality = MagicMock()
    collector.record_phase_metrics = MagicMock()
    return collector


def test_run_asr_uses_collector_hooks(monkeypatch):
    monkeypatch.setattr(
        "src.stt.stt_pipeline.pipeline", lambda *args, **kwargs: _FakeASRPipeline()
    )
    monkeypatch.setattr("src.stt.stt_pipeline.clear_gpu_cache", lambda: None)

    collector = _collector_mock()
    collector.quality_metrics.evaluate.return_value = {"wer": 0.0}
    transcription = run_asr(
        config={"model_name": "whisper_small", "model_id": "openai/whisper-small"},
        audio_tensor=torch.zeros(1, 16000),
        sampling_rate=16000,
        groundtruth="hello world",
        collector=collector,
        device="cpu",
    )

    assert transcription == "hello world"
    collector.timing_metrics.record_stage_start.assert_called_once_with(
        "asr_inference_latency"
    )
    collector.lifecycle_metrics.record_load_start.assert_called_once()
    collector.lifecycle_metrics.record_load_end.assert_called_once_with(cached=False)
    assert collector.hardware_metrics.start.call_count == 2
    assert collector.hardware_metrics.end.call_count == 2
    assert collector.record_phase_metrics.call_count == 2
    collector.timing_metrics.record_stage_end.assert_called_once_with(
        "asr_inference_latency"
    )
    collector.quality_metrics.evaluate.assert_called_once()
    assert collector.current_trial.quality.wer == 0.0
