import io
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import src.app as app_mod

# NOTE: TestClient is synchronous, but our FastAPI app is asynchronous.
# TestClient handles this automatically by running the async code in an event loop internally.
def _wav_bytes(sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    data = np.zeros(int(sample_rate * seconds), dtype=np.float32)
    sf.write(buffer, data, sample_rate, format="WAV")
    return buffer.getvalue()


def _ready_config() -> dict:
    return {
        "asr": {"model_id": "openai/whisper-small"},
        "llm": {"model_id": "Qwen/Qwen2.5-3B-Instruct"},
        "tts": {"model_name": "vibevoice", "model_id": "microsoft/VibeVoice-Realtime-0.5B"},
    }


def test_s2s_success_contract(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", _ready_config())

    def fake_process_sample(**kwargs):
        return SimpleNamespace(
            asr_transcript="hello world",
            llm_response="hi there",
            tts_waveform=np.zeros(24000, dtype=np.float32),
            tts_waveform_output_sr=24000,
        )

    monkeypatch.setattr(app_mod, "process_sample", fake_process_sample)
    response = client.post(
        "/s2s",
        files={"audio_file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    for key in ("transcript", "response", "audio", "sample_rate", "request_id", "latency_ms", "release_id"):
        assert key in payload


def test_s2s_rejects_non_audio_content_type(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", _ready_config())
    response = client.post(
        "/s2s",
        files={"audio_file": ("sample.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["detail"] == "unsupported_media_type"


def test_s2s_rejects_empty_audio(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", _ready_config())
    response = client.post(
        "/s2s",
        files={"audio_file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "empty_audio_file"


def test_s2s_rejects_when_service_not_ready(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", {"asr": {}, "llm": {}, "tts": {}})
    response = client.post(
        "/s2s",
        files={"audio_file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 503
    payload = response.json()["detail"]
    assert payload["code"] == "service_not_ready"
    assert "missing_fields" in payload


def test_s2s_timeout_path(monkeypatch):
    client = TestClient(app_mod.app)
    monkeypatch.setattr(app_mod, "APP_CONFIG", _ready_config())
    monkeypatch.setattr(app_mod, "REQUEST_TIMEOUT_SECONDS", 0.01)

    def slow_process_sample(**kwargs):
        time.sleep(0.1)
        return SimpleNamespace(
            asr_transcript="x",
            llm_response="y",
            tts_waveform=np.zeros(10, dtype=np.float32),
            tts_waveform_output_sr=16000,
        )

    monkeypatch.setattr(app_mod, "process_sample", slow_process_sample)
    response = client.post(
        "/s2s",
        files={"audio_file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 504
    assert response.json()["detail"] == "request_timeout"
