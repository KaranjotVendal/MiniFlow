import io
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import src.app as app_mod


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    return buffer.getvalue()


def _ready_config() -> dict:
    return {
        "asr": {"model_id": "openai/whisper-small"},
        "llm": {"model_id": "Qwen/Qwen2.5-3B-Instruct"},
        "tts": {
            "model_name": "vibevoice",
            "model_id": "microsoft/VibeVoice-Realtime-0.5B",
        },
    }


def test_metrics_endpoint_exports_prometheus_text(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", _ready_config())

    def fake_process_sample(**kwargs):
        return SimpleNamespace(
            asr_transcript="hello",
            llm_response="world",
            tts_waveform=np.zeros(16000, dtype=np.float32),
            tts_waveform_output_sr=16000,
        )

    monkeypatch.setattr(app_mod, "process_sample", fake_process_sample)
    response = client.post(
        "/s2s",
        files={"audio_file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "miniflow_requests_total" in body
    assert "miniflow_request_latency_seconds" in body
    assert "miniflow_stage_latency_seconds" in body


def test_metrics_error_counter_increments(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", _ready_config())

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_mod, "process_sample", boom)
    response = client.post(
        "/s2s",
        files={"audio_file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 400

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'miniflow_requests_total{status="error"}' in metrics.text
