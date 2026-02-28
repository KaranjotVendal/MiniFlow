"""
MiniFlow-S2S FastAPI Application
--------------------------------
Entry point for both:
  • REST-style /s2s endpoint (upload full audio, get full response)
  • WebSocket /ws endpoint for low-latency streaming audio interaction
"""

import asyncio
import base64
import io
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    WebSocket,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import AppSettings
from src.config.load_config import load_yaml_config
from src.logger.logging import initialise_logger
from src.prepare_data import AudioSample
from src.sts_pipeline import process_sample

logger = initialise_logger(__name__)


def load_app_config(settings: AppSettings) -> dict:
    config_path = settings.resolve_config_path()
    config = load_yaml_config(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary: {config_path}")
    return config


def get_settings() -> AppSettings | None:
    """Return runtime application settings loaded at startup."""
    return getattr(app.state, "settings", None)


def get_app_config() -> dict | None:
    """Return runtime application configuration loaded at startup."""
    return getattr(app.state, "app_config", None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load configuration during application startup."""
    try:
        app.state.settings = AppSettings.from_env()
        app.state.app_config = load_app_config(app.state.settings)
        app.state.config_error = None
        logger.info("Application configuration loaded successfully.")
    except Exception as exc:
        app.state.settings = None
        app.state.app_config = None
        app.state.config_error = str(exc)
        logger.exception("Failed to load application configuration on startup.")
    yield


app = FastAPI(
    title="MiniFlow",
    version="0.0.1",
    description="Low-latency speech-to-speech agent",
    lifespan=lifespan,
)


def _encode_wav_base64(waveform, sample_rate: int) -> str:
    """Encode waveform and sample rate into base64 WAV payload."""
    if isinstance(waveform, torch.Tensor):
        audio = waveform.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        audio = np.asarray(waveform, dtype=np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
    settings = get_settings()
    if settings is None:
        return JSONResponse(
            {
                "status": "not_ready",
                "reason": "settings not loaded",
                "error": getattr(app.state, "config_error", None),
            },
            status_code=503,
        )

    app_config = get_app_config()
    if not app_config:
        return JSONResponse(
            {
                "status": "not_ready",
                "reason": "configuration not loaded",
                "error": getattr(app.state, "config_error", None),
            },
            status_code=503,
        )

    missing_fields = _readiness_missing_fields(app_config)
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
    """Placeholder for Prometheus metrics integration."""
    return {"latency_avg_ms": "N/A", "sessions_active": 0}


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
    start_time = time.perf_counter()

    try:
        settings = get_settings()
        if settings is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "service_not_ready",
                    "reason": "settings not loaded",
                    "error": getattr(app.state, "config_error", None),
                },
            )

        if audio_file.content_type and not audio_file.content_type.startswith("audio/"):
            raise HTTPException(status_code=415, detail="unsupported_media_type")

        contents = await audio_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="empty_audio_file")
        if len(contents) > settings.miniflow_max_audio_upload_bytes:
            raise HTTPException(status_code=413, detail="audio_file_too_large")

        app_config = get_app_config()
        if app_config is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "service_not_ready",
                    "reason": "configuration not loaded",
                    "error": getattr(app.state, "config_error", None),
                },
            )

        missing_fields = _readiness_missing_fields(app_config)
        if missing_fields:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "service_not_ready",
                    "missing_fields": missing_fields,
                },
            )

        try:
            waveform, sampling_rate = torchaudio.load(io.BytesIO(contents))
        except RuntimeError as exc:
            # Map only decode/format errors from torchaudio to client-side 400.
            logger.exception(f"Invalid audio payload for request {request_id}")
            raise HTTPException(status_code=400, detail="invalid_audio_payload") from exc

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
                config=app_config,
                sample=sample,
                run_id=request_id,
                # Production mode - no benchmark collection
                collector=None,
                device=device,
                history=None,
                stream_audio=False,
            ),
            timeout=settings.miniflow_request_timeout_seconds,
        )

        wav_base64 = _encode_wav_base64(result.tts_waveform, result.tts_waveform_output_sr)

        latency_ms = (time.perf_counter() - start_time) * 1000

        response_payload = {
            "transcript": result.asr_transcript,
            "response": result.llm_response,
            "audio": wav_base64,
            "sample_rate": result.tts_waveform_output_sr,
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2),
            "release_id": settings.release_id,
        }

        logger.info(
            "/s2s request completed",
            extra={
                "request_id": request_id,
                "latency_ms": latency_ms,
                "release_id": response_payload["release_id"],
            },
        )

        return JSONResponse(response_payload)

    except HTTPException:
        raise
    except TimeoutError as exc:
        logger.exception(f"/s2s request timed out: {request_id}")
        raise HTTPException(status_code=504, detail="request_timeout") from exc
    except Exception as exc:
        logger.exception(f"Error in /s2s request {request_id}")
        raise HTTPException(status_code=500, detail="internal_error") from exc


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
