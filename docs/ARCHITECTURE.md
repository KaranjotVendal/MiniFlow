# MiniFlow Architecture Guide

A comprehensive guide to understanding MiniFlow's architecture, data flow, and component interactions.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Component Breakdown](#component-breakdown)
5. [Configuration System](#configuration-system)
6. [Deployment Architecture](#deployment-architecture)
7. [Benchmark Framework](#benchmark-framework)
8. [Key Architectural Patterns](#key-architectural-patterns)

---

## Overview

MiniFlow is a real-time speech-to-speech conversational AI system that processes audio input through three main stages:

```
Audio Input → ASR → LLM → TTS → Audio Output
    │         │      │      │         │
    │      (Listen) (Think) (Speak)    │
    │                                   │
    └───────────────────────────────────┘
         End-to-End Latency: 50-120s
```

**Core Capabilities:**
- Real-time speech recognition (Whisper)
- Conversational AI response generation (Qwen)
- High-quality speech synthesis (XTTS/VibeVoice)
- Comprehensive benchmarking and metrics collection
- GPU-optimized inference with 6GB VRAM target

---

## System Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MINIFLOW S2S SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ENTRY POINTS                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │   FastAPI    │  │   Pipeline   │  │   Benchmark Runner       │  │   │
│  │  │   (app.py)   │  │   (sts_)     │  │   (experiment_runner.py) │  │   │
│  │  │              │  │              │  │                          │  │   │
│  │  │  /s2s (REST) │  │  Direct API  │  │  Batch evaluation        │  │   │
│  │  │  /ws (WebSock)│  │  for scripts │  │  with metrics            │  │   │
│  │  │  /health     │  │              │  │                          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONFIGURATION SYSTEM                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │   AppSettings│  │   YAML Config│  │   Centralized Path       │  │   │
│  │  │   (Pydantic) │  │   Loader     │  │   Resolution             │  │   │
│  │  │              │  │              │  │                          │  │   │
│  │  │  Env vars    │  │  Pipeline    │  │  REQUIRED:              │  │   │
│  │  │  Validation  │  │  configs     │  │  MINIFLOW_CONFIG,       │  │   │
│  │  │              │  │              │  │  RELEASE_ID             │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CORE PIPELINE (sts_pipeline.py)                  │   │
│  │                                                                     │   │
│  │   Input Audio ──► ASR ──► Text ──► LLM ──► Response ──► TTS ──► Audio │   │
│  │                                                                     │   │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐                        │   │
│  │   │  ASR    │───►│   LLM   │───►│  TTS    │                        │   │
│  │   │ Module  │    │ Module  │    │ Module  │                        │   │
│  │   │         │    │         │    │         │                        │   │
│  │   │Whisper  │    │Qwen2.5  │    │XTTS/    │                        │   │
│  │   │         │    │         │    │VibeVoice│                        │   │
│  │   └─────────┘    └─────────┘    └─────────┘                        │   │
│  │        │              │               │                            │   │
│  │        ▼              ▼               ▼                            │   │
│  │   ┌─────────────────────────────────────────┐                      │   │
│  │   │      BENCHMARK COLLECTOR (Optional)     │                      │   │
│  │   │  - Timing metrics                       │                      │   │
│  │   │  - Hardware metrics (GPU memory, power) │                      │   │
│  │   │  - Quality metrics (WER, UTMOS)         │                      │   │
│  │   │  - Token metrics (TTFT, tokens/sec)     │                      │   │
│  │   │  - Model lifecycle tracking             │                      │   │
│  │   └─────────────────────────────────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA & STORAGE LAYER                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ AudioSample  │  │  JSONL       │  │   GLOBE Dataset          │  │   │
│  │  │  (dataclass) │  │  Storage     │  │   (streaming)            │  │   │
│  │  │              │  │              │  │                          │  │   │
│  │  │  waveform    │  │  raw_logs    │  │  HuggingFace             │  │   │
│  │  │  transcript  │  │  summary     │  │  MushanW/GLOBE_V3        │  │   │
│  │  │  metadata    │  │  config      │  │                          │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCKER DEPLOYMENT                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04                         │   │
│  │  ├── uv (package manager)                                           │   │
│  │  ├── Python 3 + PyTorch                                             │   │
│  │  ├── VibeVoice submodule (TTS model)                                │   │
│  │  └── FastAPI server on port 8000                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Entry Points

| Entry Point | File | Purpose | Usage |
|-------------|------|---------|-------|
| **FastAPI App** | `src/app.py` | Production HTTP API server | `uv run uvicorn src.app:app` |
| **Pipeline Script** | `src/sts_pipeline.py` | Direct programmatic API | Import and call `process_sample()` |
| **Benchmark Runner** | `src/scripts/run_experiment.py` | Batch evaluation with metrics | `uv run python -m src.scripts.run_experiment` |
| **Debug Scripts** | `src/debug_scripts/` | Component testing | `uv run python -m src.debug_scripts.debug_tts` |

---

## Data Flow

### Speech-to-Speech Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    S2S PIPELINE DATA FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  USER SPEECH
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────────────────────────────┐
│  Audio Input │────►│  process_sample() in src/sts_pipeline.py                │
│  (wav bytes  │     │                                                         │
│   or tensor) │     │  1. Create NoOpCollector (production)                   │
└──────────────┘     │     or BenchmarkCollector (benchmarking)                │
                     │                                                         │
                     │  2. ASR STAGE                                           │
                     │     ┌─────────────────────────────────────────────┐    │
                     │     │ run_asr()                                   │    │
                     │     │ ├── Load Whisper pipeline                   │    │
                     │     │ ├── Record GPU metrics during load          │    │
                     │     │ ├── Transcribe audio                        │    │
                     │     │ ├── Record inference latency                │    │
                     │     │ └── Evaluate WER vs groundtruth             │    │
                     │     └─────────────────────────────────────────────┘    │
                     │                          │                              │
                     │                          ▼                              │
                     │                   "transcription"                       │
                     │                          │                              │
                     │  3. LLM STAGE                                           │
                     │     ┌─────────────────────────────────────────────┐    │
                     │     │ run_llm()                                   │    │
                     │     │ ├── Load quantized Qwen model               │    │
                     │     │ ├── Load tokenizer                          │    │
                     │     │ ├── Build conversation history              │    │
                     │     │ ├── Generate response                       │    │
                     │     │ ├── Record token metrics (TTFT, etc.)       │    │
                     │     │ └── Return response + updated history       │    │
                     │     └─────────────────────────────────────────────┘    │
                     │                          │                              │
                     │                          ▼                              │
                     │                    "response"                         │
                     │                          │                              │
                     │  4. TTS STAGE                                           │
                     │     ┌─────────────────────────────────────────────┐    │
                     │     │ run_tts()                                   │    │
                     │     │ ├── Route to XTTS/VibeVoice/CosyVoice       │    │
                     │     │ ├── Load TTS model + processor              │    │
                     │     │ ├── Synthesize speech                       │    │
                     │     │ ├── Evaluate UTMOS quality                  │    │
                     │     │ └── Return waveform + sample_rate           │    │
                     │     └─────────────────────────────────────────────┘    │
                     │                          │                              │
                     │                          ▼                              │
                     │  5. RETURN ProcessedSample                              │
                     │     ├── asr_transcript                                  │
                     │     ├── llm_response                                    │
                     │     ├── tts_waveform (numpy array)                      │
                     │     └── new_history                                     │
                     │                                                         │
                     └─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │   RESPONSE AUDIO    │
                              │   (WAV format)      │
                              └─────────────────────┘
```

### Request/Response Lifecycle (FastAPI)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FASTAPI REQUEST/RESPONSE LIFECYCLE                              │
└─────────────────────────────────────────────────────────────────────────────┘

  CLIENT REQUEST
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────────────────────────────┐
│   POST /s2s  │────►│  speech_to_speech() endpoint                            │
│  (audio file)│     │  (src/app.py)                                           │
└──────────────┘     │                                                         │
                     │  1. VALIDATION                                          │
                     │     ├── Check content-type is audio/*                   │
                     │     ├── Check file not empty                            │
                     │     └── Check file size < max (default 10MB)            │
                     │                                                         │
                     │  2. CONFIG CHECK                                        │
                     │     ├── get_app_config() - must be loaded               │
                     │     └── _readiness_missing_fields() validation          │
                     │                                                         │
                     │  3. AUDIO PREPROCESSING                                 │
                     │     ├── Read file contents (bytes)                      │
                     │     └── torchaudio.load() -> (waveform, sample_rate)    │
                     │                                                         │
                     │  4. CREATE AudioSample                                  │
                     │     ├── audio_tensor=waveform                           │
                     │     ├── transcript="" (no groundtruth for inference)    │
                     │     └── sampling_rate=sample_rate                       │
                     │                                                         │
                     │  5. PROCESS (async wrapper)                             │
                     │     └── asyncio.to_thread(                              │
                     │           process_sample(),                             │
                     │           config=app_config,                            │
                     │           sample=sample,                                │
                     │           collector=None,  # No benchmark in prod       │
                     │           device="cuda" if available else "cpu"         │
                     │         )                                               │
                     │                                                         │
                     │  6. AUDIO ENCODING                                      │
                     │     ├── soundfile.write() to BytesIO buffer             │
                     │     └── base64.b64encode() for JSON transport           │
                     │                                                         │
                     │  7. BUILD RESPONSE                                      │
                     │     └── S2SResponse:                                    │
                     │         ├── transcript (ASR output)                     │
                     │         ├── response (LLM output)                       │
                     │         ├── audio (base64 WAV)                          │
                     │         ├── sample_rate                                 │
                     │         ├── request_id (UUID)                           │
                     │         ├── latency_ms                                  │
                     │         └── release_id                                  │
                     │                                                         │
                     └─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌─────────────────────┐
                              │   JSONResponse      │
                              │   (S2SResponse)     │
                              └─────────────────────┘
                                          │
                                          ▼
                                    CLIENT RECEIVES
                                    JSON with base64
                                    audio payload
```

---

## Component Breakdown

### ASR Module (`src/stt/stt_pipeline.py`)

```python
def run_asr(
    config: dict,                    # ASR config section
    audio_tensor: torch.Tensor,      # Input audio waveform
    sampling_rate: int,              # Audio sample rate (e.g., 16000)
    groundtruth: str,                # Reference transcript (for WER)
    collector: BenchmarkCollector,   # Metrics collector
    device: torch.device | str,      # cuda or cpu
) -> str:                           # Returns: transcription
```

**Implementation Details:**
- Uses HuggingFace `transformers.pipeline` for automatic speech recognition
- Default model: `openai/whisper-small`
- Supports English transcription with forced language setting
- Records load time, inference latency, GPU metrics, and WER quality score

### LLM Module (`src/llm/llm_pipeline.py`)

```python
def run_llm(
    config: dict | None,             # LLM config section
    transcription: str,              # ASR output text
    device: torch.device | str,      # cuda or cpu
    collector: BenchmarkCollector,   # Metrics collector
    history: list[dict] | None = None,  # Conversation history
) -> tuple[str, list[dict]]:        # Returns: (response, updated_history)
```

**Implementation Details:**
- Default model: `Qwen/Qwen2.5-3B-Instruct`
- Supports 4-bit quantization via BitsAndBytesConfig
- Builds conversation history with system prompt
- Records token metrics (TTFT, tokens/sec), inference latency, GPU metrics

**Quantization Config:**
```yaml
quantization:
  enabled: true
  load_in_4bit: true
  quant_type: "nf4"
  use_double_quant: true
  compute_dtype: "bf16"
```

### TTS Module (`src/tts/tts_pipelines.py`)

```python
def run_tts(
    config: dict,                    # TTS config section
    llm_response: str,               # Text to synthesize
    device: torch.device | str,      # cuda or cpu
    collector: BenchmarkCollector,   # Metrics collector
) -> tuple[torch.Tensor, int]:      # Returns: (waveform, sample_rate)
```

**Supported TTS Backends:**

| Backend | File | Model | Features |
|---------|------|-------|----------|
| **XTTS** | `src/tts/xtts.py` | `tts_models/multilingual/multi-dataset/xtts_v2` | Multi-speaker, voice cloning |
| **VibeVoice** | `src/tts/vibevoice.py` | `microsoft/VibeVoice-Realtime-0.5B` | Real-time streaming, git submodule |
| **CosyVoice** | `src/tts/cosyvoice.py` | (planned) | - |

**VibeVoice Integration Notes:**
- Git submodule at `vibevoice/`
- Local compatibility fixes for transformers 4.57.3
- Voice files stored in `vibevoice/demo/voices/streaming_model/`
- Uses flash_attention_2 on CUDA, SDPA on MPS/CPU

---

## Configuration System

### Configuration Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION HIERARCHY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: Environment Variables (API runtime)                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  REQUIRED:                                                       │
│  ├── MINIFLOW_CONFIG=<path>           # Pipeline YAML path     │
│  └── RELEASE_ID=<identifier>          # Release version        │
│                                                                  │
│  OPTIONAL (with defaults):                                       │
│  ├── MINIFLOW_REQUEST_TIMEOUT_SECONDS=120                      │
│  └── MINIFLOW_MAX_AUDIO_UPLOAD_BYTES=10485760 (10MB)          │
│                                                                  │
│  Level 2: Pipeline Configuration (configs/*.yml)                │
│  ├── experiment: name, description                              │
│  ├── dataset: num_samples, warmup_samples, split                │
│  ├── asr: model_id, model_name                                  │
│  ├── llm: model_id, quantization, kv_cache, max_new_tokens       │
│  ├── tts: model_name, model_id, speaker/voice, cfg_scale       │
│  └── benchmark: enable_streaming_audio, save_processed_samples  │
│                                                                  │
│  Level 3: Metrics Configuration (configs/metrics.yml)           │
│  ├── enabled: [timing, hardware_basic, model_lifecycle, ...]   │
│  └── configurations: per-metric settings                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MINIFLOW_CONFIG` | ✅ Yes | - | Path to pipeline YAML config |
| `RELEASE_ID` | ✅ Yes | - | Release identifier (e.g., `prod-pseudo`, `v1.0.0`) |
| `MINIFLOW_REQUEST_TIMEOUT_SECONDS` | ❌ No | 120 | Request timeout in seconds |
| `MINIFLOW_MAX_AUDIO_UPLOAD_BYTES` | ❌ No | 10485760 (10MB) | Max upload size in bytes |

### Configuration Loading Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 CONFIGURATION LOADING FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  APPLICATION STARTUP (FastAPI)
         │
         ▼
┌─────────────────────┐
│  AppSettings.from_env()  │
│  (src/config/settings.py) │
└─────────────────────┘
         │
         ├──► Load environment variables
         │    ├── MINIFLOW_CONFIG (REQUIRED)
         │    ├── RELEASE_ID (REQUIRED)
         │    ├── MINIFLOW_REQUEST_TIMEOUT_SECONDS (default: 120)
         │    └── MINIFLOW_MAX_AUDIO_UPLOAD_BYTES (default: 10MB)
         │
         ├──► Pydantic validation
         │    ├── MINIFLOW_CONFIG required? → Error if missing
         │    └── RELEASE_ID required? → Error if missing
         │
         └──► Returns validated AppSettings instance
                      │
                      ▼
         ┌────────────────────────┐
         │  resolve_config_path() │
         │  (src/config/settings.py)
         └────────────────────────┘
                      │
                      ├──► Resolve path (relative → absolute vs CWD)
                      │
                      ├──► Validate file exists → FileNotFoundError if missing
                      │
                      ▼
         ┌────────────────────────┐
         │  load_yaml_config()   │
         │  (src/config/load_config.py)
         └────────────────────────┘
                      │
                      ├──► Parse YAML
                      │
                      ├──► Validate not empty → ValueError if empty
                      │
                      ├──► Validate is dict → ValueError if not
                      │
                      ▼
         ┌────────────────────────┐
         │  FastAPI Lifespan     │
         │  (src/app.py)         │
         └────────────────────────┘
                      │
                      ├──► Store in app.state.app_config
                      │
                      ▼
         ┌────────────────────────┐
         │  /ready endpoint       │
         │  validates required    │
         │  fields: asr.model_id, │
         │  llm.model_id,         │
         │  tts.model_name,       │
         │  tts.model_id          │
         └────────────────────────┘
```

### Configuration Modules

```
src/config/
├── __init__.py              → Exports: AppSettings
├── load_config.py           → load_yaml_config() with validation
├── path_utils.py            → resolve_path(), resolve_path_relative_to_file()
├── settings.py              → AppSettings (Pydantic model)
└── inspect_config.py        → inspect_config() for benchmarks
```

### Path Resolution

Path resolution is centralized in `src/config/path_utils.py`:

```python
from src.config.path_utils import resolve_path, resolve_path_relative_to_file

# Resolve relative to CWD
resolve_path("configs/baseline.yml")
# → "/project/configs/baseline.yml"

# Resolve relative to a file's parent directory
resolve_path_relative_to_file("metrics.yml", Path("/project/configs/baseline.yml"))
# → "/project/configs/metrics.yml"
```

> **Note:** The Benchmark Framework is independent of AppSettings. Benchmark
> configurations are self-contained YAML files passed via `--config` CLI
> argument. Use `inspect_config()` to normalize paths before passing to
> `ExperimentRunner.from_config()`.


## Deployment Architecture

### Dockerfile Structure

```
┌─────────────────────────────────────────────────────────────────┐
│  nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04  (Base image)       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: System dependencies                                    │
│  ├── python3, python3-pip, python3-venv                         │
│  ├── ffmpeg (audio processing)                                  │
│  ├── git (for submodules)                                       │
│  └── libportaudio2 (sounddevice)                                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: UV package manager (from ghcr.io/astral-sh/uv)        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Python dependencies                                    │
│  ├── Copy pyproject.toml + uv.lock                              │
│  └── uv sync --frozen --no-install-project                      │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Application code                                       │
│  ├── Copy entire project (including vibevoice submodule)        │
│  └── uv sync --frozen (install project)                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: Runtime configuration                                  │
│  ├── Create non-root user (appuser)                             │
│  ├── Healthcheck on /health                                     │
│  └── CMD: uv run uvicorn src.app:app --host 0.0.0.0 --port 8000│
└─────────────────────────────────────────────────────────────────┘
```

### Docker Compose Configuration

**Production:**
```yaml
services:
  api:
    environment:
      - COQUI_TOS_AGREED=1        # Required for XTTS
      - MINIFLOW_CONFIG=configs/3_TTS-to-vibevoice.yml
      - RELEASE_ID=prod-pseudo
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    command: uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:8000/health || exit 1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Development:**
```yaml
services:
  api:
    environment:
      - LOG_LEVEL=debug
      - MINIFLOW_CONFIG=configs/baseline.yml
      - RELEASE_ID=dev-pseudo
    stdin_open: true
    tty: true
    environment:
      - HF_HOME=/app/.cache/huggingface
    volumes:
      - .:/app                      # Live code updates
      - app_venv:/app/.venv         # Preserve dependencies
      - hf_cache:/app/.cache/huggingface  # Model cache
      - ./Benchmark:/app/Benchmark  # Persist benchmark outputs
    command: uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### Deployment Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Development   │────►│  Build Docker   │────►│  Deploy to      │
│   (local uv)    │     │  Image          │     │  Production     │
│                 │     │                 │     │                 │
│ uv sync         │     │ docker build    │     │ docker-compose  │
│ uv run python   │     │ docker-compose  │     │ up -d           │
│                 │     │ build           │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Local testing          Image registry          GPU-enabled
   with hot reload        (optional)              cloud instance
```

---

## Benchmark Framework

### Benchmark Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BENCHMARK DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  run_experiment.py --config configs/baseline.yml
         │
         ▼
┌─────────────────────────┐
│  ExperimentRunner       │
│  .from_config(config)   │
└─────────────────────────┘
         │
         ├──► inspect_config() resolves metrics path from benchmark YAML
         │    └── Loads config["metrics"] relative to main config directory
         │
         ├──► Load metrics config (from resolved path)
         │    └── MetricRegistry creates metric instances
         │
         ├──► Create JSONLStorage for results
         │
         └──► Initialize runner with config + metrics + storage
                      │
                      ▼
         ┌────────────────────────┐
         │      runner.run()      │
         └────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌────────┐  ┌──────────┐  ┌──────────┐
    │ Warmup │  │  Main    │  │ Summary  │
    │ Trials │  │  Trials  │  │ Generation│
    └────────┘  └──────────┘  └──────────┘
         │            │            │
         │            │            │
         ▼            ▼            ▼
    For each sample:         ┌─────────────────────────┐
    ┌─────────────────┐      │  _generate_summary()    │
    │ 1. Create       │      │                         │
    │    Benchmark-   │      │  Aggregate all trials:  │
    │    Collector    │      │  - Compute mean, median │
    │                 │      │  - Compute p95, p99     │
    │ 2. collector.   │      │  - Status counts        │
    │    start_trial()│      │  - Per-stage metrics    │
    │                 │      │                         │
    │ 3. process_     │      │  Save to:               │
    │    sample()     │─────►│  - raw_logs.jsonl       │
    │    (with        │      │  - summary.json         │
    │    collector)   │      │  - config.json          │
    │                 │      └─────────────────────────┘
    │ 4. collector.   │
    │    end_trial()  │
    │                 │
    │ 5. storage.     │
    │    save_trial() │
    │                 │
    │ 6. clear_gpu_   │
    │    cache()      │
    └─────────────────┘
```

### Available Metrics

| Metric | Registered Name | Purpose |
|--------|-----------------|---------|
| `TimingMetrics` | `timing` | Stage and total latency tracking |
| `HardwareMetrics` | `hardware_basic` / `hardware_detailed` | GPU memory, power, temperature |
| `ModelLifecycleMetrics` | `model_lifecycle` | Model load times, cache hits/misses |
| `QualityMetrics` | `quality` | WER (ASR) and UTMOS (TTS) evaluation |
| `TokenMetrics` | `tokens` | TTFT, tokens/sec, generation time |

---

## Key Architectural Patterns

### 1. Collector Pattern for Metrics

The benchmark framework uses a collector pattern where `BenchmarkCollector` wraps metric instances and provides a consistent interface for recording data across all pipeline stages.

### 2. NoOp Pattern for Production

When running in production mode (API), a `NoOpCollector` is used that implements the same interface but performs no actual metric collection, minimizing overhead.

### 3. Registry Pattern for Extensibility

Metrics are registered via decorators and retrieved by name, allowing new metrics to be added without modifying core code.

### 4. Configuration Layering

Settings follow a clear precedence: Environment Variables > Environment Profile > Pipeline Config > Defaults

### 5. GPU Memory Management

Every pipeline component follows the pattern:
1. Load model
2. Run inference
3. Delete model references
4. Call `clear_gpu_cache()` (torch.cuda.empty_cache())

---

## File Structure

```
MiniFlow/
├── src/
│   ├── app.py                    # FastAPI entry point
│   ├── sts_pipeline.py           # Core S2S pipeline
│   ├── prepare_data.py           # AudioSample + dataset streaming
│   ├── utils.py                  # get_device(), clear_gpu_cache()
│   ├── config/
│   │   ├── settings.py           # AppSettings (Pydantic)
│   │   └── load_config.py        # YAML loader
│   ├── stt/stt_pipeline.py       # ASR (Whisper)
│   ├── llm/llm_pipeline.py       # LLM (Qwen)
│   ├── tts/
│   │   ├── tts_pipelines.py      # TTS router
│   │   ├── xtts.py               # XTTS implementation
│   │   └── vibevoice.py          # VibeVoice implementation
│   ├── benchmark/
│   │   ├── runner/
│   │   │   └── experiment_runner.py  # Benchmark orchestration
│   │   ├── collectors/
│   │   │   ├── benchmark_collector.py # Metrics collection
│   │   │   └── trial_models.py       # Data models
│   │   ├── core/
│   │   │   ├── base.py           # BaseMetric, MetricContext
│   │   │   └── registry.py       # MetricRegistry
│   │   ├── metrics/              # Metric implementations
│   │   └── storage/
│   │       └── jsonl_storage.py  # Results persistence
│   └── scripts/
│       └── run_experiment.py     # CLI entry point
├── configs/
│   ├── baseline.yml              # Default pipeline config
│   ├── 3_TTS-to-vibevoice.yml    # VibeVoice config
│   └── metrics.yml               # Metrics configuration
├── vibevoice/                    # Git submodule (TTS model)
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Reference

### Mental Model

Think of MiniFlow as:

> **A conversation pipeline factory**
>
> You configure it with YAML files (what models to use)
> You deploy it with Docker (where to run)
> You interact with it via HTTP API (how to use)
> You measure it with benchmarks (how well it works)

### Key Design Principles

1. **Modularity**: Each stage (ASR/LLM/TTS) can be swapped independently
2. **Observability**: Comprehensive metrics at every stage
3. **Flexibility**: Environment-driven configuration
4. **Efficiency**: GPU memory management for 6GB VRAM target
5. **Reproducibility**: Version-controlled configs and Docker images
