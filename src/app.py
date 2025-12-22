"""
MiniFlow-S2S FastAPI Application
--------------------------------
Entry point for both:
  • REST-style /s2s endpoint (upload full audio, get full response)
  • WebSocket /ws endpoint for low-latency streaming audio interaction
"""

import base64
import io

import soundfile as sf
import torch
import torchaudio
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from src.sts_pipeline import process_sample
from src.logger.logging import initialise_logger
from src.prepare_data import AudioSample

logger = initialise_logger(__name__)

app = FastAPI(
    title="MiniFlow",
    version="0.1",
    description="Low-latency speech-to-speech agent",
)


@app.get("/")
def hello_world():
    return {
        "message": "Welcome to MiniFlow API",
        "routes": ["/health", "/s2s", "/ws"],
    }


@app.get("/health")
def health_check():
    """Basic liveness probe."""
    return {"status": "healthy"}


@app.get("/metrics")
def metrics_stub():
    """Placeholder for Prometheus metrics integration."""
    return {"latency_avg_ms": "N/A", "sessions_active": 0}


# REST Endpoint: simple non-streaming baseline
@app.post("/s2s")
async def speech_to_speech(audio_file: UploadFile):
    """
    Synchronous S2S inference:
      1. Read entire uploaded audio file
      2. Run baseline process_sample()
      3. Return transcript, response text, latencies, and base64 WAV
    """
    try:
        contents = await audio_file.read()
        waveform, sampling_rate = torchaudio.load(io.BytesIO(contents))

        sample = AudioSample(
            audio_tensor=waveform,
            transcript="",
            accent=None,
            duration=None,
            sample_to_noise_ratio=None,
            utmos=None,
            sampling_rate=sampling_rate,
        )
        result, metrics = process_sample(sample, folder="./metrics")

        # Encode output waveform to base64 WAV for JSON transport
        buffer = io.BytesIO()
        sf.write(
            buffer,
            result.tts_waveform,
            result.tts_waveform_output_sr,
            format="WAV",
        )
        wav_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        response_payload = {
            "transcript": result.asr_transcript,
            "response": result.llm_response,
            "tts_wavform": wav_base64,
            "wer": metrics.asr_wer,
            "mos": metrics.tts_utmos,
            "latencies": {
                "asr": metrics.asr_latency,
                "llm": metrics.llm_latency,
                "tts": metrics.tts_latency,
                "total": metrics.total_latency,
            },
            "gpu_util": {
                "asr": metrics.asr_gpu_peak_mem,
                "llm": metrics.llm_gpu_peak_mem,
                "tts": metrics.tts_gpu_peak_mem,
            },
        }
        return JSONResponse(response_payload)

    except Exception as e:
        logger.exception("Error in /s2s")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoint: real-time streaming (stub until full pipeline ready)
@app.websocket("/ws")
async def websocket_s2s(ws: WebSocket):
    """
    Real-time speech-to-speech streaming interface.
    Clients send:
        { "type": "audio", "seq": int, "pcm16_base64": str }
    Server responds with streaming events:
        { "type": "asr_partial"/"llm_partial"/"tts_audio", ... }
    """
    await ws.accept()
    logger.info("New WebSocket connection established")

    try:
        while True:
            msg = await ws.receive_json()

            # Basic handshake (future: handle "start"/"stop"/"config" messages)
            if msg["type"] == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if msg["type"] == "audio":
                # Decode audio bytes (float32 in [-1, 1])
                pcm = base64.b64decode(msg["pcm16_base64"])
                waveform = (
                    torch.frombuffer(pcm, dtype=torch.int16)
                    .float()
                    .div(32768.0)
                    .unsqueeze(0)
                )

                # Placeholder: run baseline pipeline on each chunk (for now)
                class Sample:
                    def __init__(self, audio_tensor):
                        self.audio_tensor = audio_tensor
                        self.transcript = ""

                result = process_sample(Sample(waveform))
                text = result.get("response", "")
                await ws.send_json({"type": "llm_partial", "text": text})
                # In future: stream partial ASR/LLM/TTS frames progressively

            elif msg["type"] == "stop":
                await ws.close(code=1000)
                logger.info("WebSocket closed cleanly by client")
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        await ws.close(code=1011, reason=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
