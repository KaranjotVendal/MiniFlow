"""
MiniFlow-S2S FastAPI Application
--------------------------------
Entry point for both:
  • REST-style /s2s endpoint (upload full audio, get full response)
  • WebSocket /ws endpoint for low-latency streaming audio interaction
"""

import base64
import asyncio
import io
import time
import uuid

import soundfile as sf
import torch
import torchaudio
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    WebSocket,
)
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from src.instrumentation.adapters import RuntimeTelemetryAdapter
from src.sts_pipeline import process_sample
from src.config import AppSettings
from src.config.load_config import load_yaml_config
from src.logger.logging import initialise_logger
from src.observability.metrics import render_metrics
from src.prepare_data import AudioSample

logger = initialise_logger(__name__)

app = FastAPI(
    title="MiniFlow",
    version="0.0.1",
    description="Low-latency speech-to-speech agent",
)

SETTINGS = AppSettings.from_env()
MAX_AUDIO_UPLOAD_BYTES = SETTINGS.miniflow_max_audio_upload_bytes
REQUEST_TIMEOUT_SECONDS = SETTINGS.miniflow_request_timeout_seconds

def load_app_config() -> dict:
    config_path = SETTINGS.resolve_config_path()
    config = load_yaml_config(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary: {config_path}")
    return config

APP_CONFIG = load_app_config()


def _readiness_missing_fields(config: dict) -> list[str]:
    required_paths = [
        ("asr", "model_id"),
        ("llm", "model_id"),
        ("tts", "model_name"),
        ("tts", "model_id"),
    ]
    missing: list[str] = []
    for section, key in required_paths:
        if not isinstance(config.get(section), dict) or not config[section].get(key):
            missing.append(f"{section}.{key}")
    return missing


@app.get("/")
def hello_world():
    return {
        "message": "Welcome to MiniFlow API",
        "routes": ["/health", "/ready", "/s2s", "/ws"],
    }


@app.get("/health")
def health_check():
    """Basic liveness probe."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    """Readiness probe - checks if service is ready to handle requests."""
    if not APP_CONFIG:
        return JSONResponse(
            {"status": "not_ready", "reason": "configuration not loaded"},
            status_code=503
        )

    missing_fields = _readiness_missing_fields(APP_CONFIG)
    if missing_fields:
        return JSONResponse(
            {
                "status": "not_ready",
                "reason": "configuration incomplete",
                "missing_fields": missing_fields,
            },
            status_code=503,
        )

    cuda_available = torch.cuda.is_available()

    return {
        "status": "ready",
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
    }


@app.get("/metrics")
def metrics_stub():
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


class S2SResponse(BaseModel):
    """Response model for /s2s endpoint."""
    transcript: str
    response: str
    audio: str  # base64 encoded WAV
    sample_rate: int
    request_id: str
    latency_ms: float
    release_id: str


# REST Endpoint: simple non-streaming baseline
@app.post("/s2s", response_model=S2SResponse)
async def speech_to_speech(audio_file: UploadFile):
    """
    Synchronous S2S inference:
      1. Read entire uploaded audio file
      2. Run baseline process_sample()
      3. Return transcript, response text, and base64 WAV
    """
    request_id = str(uuid.uuid4())
    instrumentation = RuntimeTelemetryAdapter(release_id=SETTINGS.release_id)
    instrumentation.on_request_start(request_id, metadata={"route": "/s2s"})
    start_time = time.time()

    try:
        if audio_file.content_type and not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail="unsupported_media_type")

        contents = await audio_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="empty_audio_file")
        if len(contents) > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="audio_file_too_large")

        missing_fields = _readiness_missing_fields(APP_CONFIG)
        if missing_fields:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "service_not_ready",
                    "missing_fields": missing_fields,
                },
            )

        waveform, sampling_rate = torchaudio.load(io.BytesIO(contents))

        sample = AudioSample(
            audio_tensor=waveform,
            transcript="",  # No groundtruth for inference
            accent=None,
            duration=None,
            sample_to_noise_ratio=None,
            utmos=None,
            sampling_rate=sampling_rate,
        )

        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Process sample (collector=None for production mode - lightweight telemetry)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                process_sample,
                config=APP_CONFIG,
                sample=sample,
                run_id=request_id,
                instrumentation=instrumentation,
                device=device,
                history=None,
                stream_audio=False,
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        # Encode output waveform to base64 WAV for JSON transport
        buffer = io.BytesIO()
        sf.write(
            buffer,
            result.tts_waveform,
            result.tts_waveform_output_sr,
            format="WAV",
        )
        wav_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        latency_ms = (time.time() - start_time) * 1000

        response_payload = {
            "transcript": result.asr_transcript,
            "response": result.llm_response,
            "audio": wav_base64,
            "sample_rate": result.tts_waveform_output_sr,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2),
            "release_id": SETTINGS.release_id,
        }

        logger.info(
            f"/s2s request completed",
            extra={
                "request_id": request_id,
                "latency_ms": latency_ms,
                "release_id": response_payload["release_id"],
            }
        )
        instrumentation.on_request_end(request_id, status="success")

        return JSONResponse(response_payload)

    except HTTPException:
        instrumentation.on_request_end(request_id, status="error")
        raise
    except asyncio.TimeoutError:
        logger.exception(f"/s2s request timed out: {request_id}")
        instrumentation.on_request_end(request_id, status="error", error="request_timeout")
        raise HTTPException(status_code=504, detail="request_timeout")
    except RuntimeError as e:
        # Covers invalid/unsupported audio decode errors surfaced by torchaudio stack.
        logger.exception(f"Invalid audio payload for request {request_id}")
        instrumentation.on_request_end(request_id, status="error", error="invalid_audio_payload")
        raise HTTPException(status_code=400, detail="invalid_audio_payload") from e
    except Exception:
        logger.exception(f"Error in /s2s request {request_id}")
        instrumentation.on_request_end(request_id, status="error", error="internal_error")
        raise HTTPException(status_code=500, detail="internal_error")


# WebSocket Endpoint: stub with clear message
@app.websocket("/ws")
async def websocket_s2s(ws: WebSocket):
    """
    Real-time speech-to-speech streaming interface.

    NOTE: This endpoint is deferred to v2. Currently returns clear error.
    """
    await ws.accept()
    await ws.send_json(
        {
            "status": "not_implemented",
            "message": "WebSocket streaming deferred to v2. Use /s2s for synchronous requests.",
        }
    )
    await ws.close(code=1008)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
