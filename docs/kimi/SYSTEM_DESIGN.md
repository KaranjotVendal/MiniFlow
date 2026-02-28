# MiniFlow System Design Document

## 1. Executive Summary

MiniFlow is a Speech-to-Speech (S2S) ML pipeline that processes audio input through:
- **ASR** (Automatic Speech Recognition): Audio → Text
- **LLM** (Large Language Model): Text → Response Text
- **TTS** (Text-to-Speech): Response Text → Audio

This document defines the system architecture for development and production environments.

---

## 2. Current State Analysis

### 2.1 Existing Components

#### Core Pipeline (`src/`)
```
sts_pipeline.py          # Main entry point
├── prepare_data.py      # AudioSample dataclass
├── stt/stt_pipeline.py  # ASR stage
├── llm/llm_pipeline.py  # LLM stage
├── tts/                 # TTS implementations
│   ├── tts_pipelines.py
│   ├── vibevoice.py
│   ├── xtts.py
│   └── cosyvoice.py
└── utils.py             # GPU utilities
```

#### Benchmark Framework (`src/benchmark/`)
```
core/                    # Base classes
├── base.py             # BaseMetric, BaseCollector
└── registry.py         # MetricRegistry

metrics/                 # Metric implementations
├── timing.py
├── hardware.py
├── lifecycle.py
├── quality.py
└── tokens.py

collectors/              # Data collection
├── benchmark_collector.py
├── trial_models.py
├── decorators.py
└── context_managers.py

storage/                 # Persistence
├── jsonl_storage.py
└── schemas/            # JSON schemas

runner/                  # Experiment execution
└── experiment_runner.py

analysis/                # Post-processing
```

#### Configuration (`configs/`)
- `baseline.yml` - XTTS baseline
- `2_TTS-to-vibevoice.yml` - VibeVoice config
- `metrics.yml` - Metric definitions

#### Tests (`tests/`)
- `unit_tests/` - 144+ unit tests
- `integration_tests/` - API/pipeline tests (4 marked as integration)

#### API (`src/app.py`)
- FastAPI service
- `/health`, `/s2s`, `/ws` endpoints
- **CURRENTLY STALE** - uses old process_sample signature

---

### 2.2 Component Health Matrix

| Component | Status | Notes |
|-----------|--------|-------|
| Benchmark Framework | ✅ Healthy | Well-tested, clean architecture |
| Experiment Runner | ✅ Healthy | experiment_runner.py is current |
| Pipeline Core | ✅ Healthy | process_sample works correctly |
| TTS Implementations | ✅ Healthy | XTTS, VibeVoice working |
| Storage Layer | ✅ Healthy | JSONL with schemas |
| Data Models | ✅ Healthy | AudioSample, ProcessedSample consistent |
| Unit Tests | ✅ Healthy | 165 passing |
| API (app.py) | ❌ BROKEN | Stale contract, needs fix |
| Docker | ⚠️ Dev-only | Needs prod hardening |
| CI/CD | ❌ Missing | No GitHub Actions |
| Config Management | ⚠️ Partial | Paths fixed but no env handling |
| Observability | ❌ Missing | No metrics/logs export |

---

## 3. Target Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT REQUESTS                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Load Balancer │  (AWS ALB / Nginx)
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ API Pod │   │ API Pod │   │ API Pod │  (ECS/EKS)
    └────┬────┘   └────┬────┘   └────┬────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
              ┌────────▼────────┐
              │   /s2s Handler  │  FastAPI Endpoint
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │   ASR   │──▶│   LLM   │──▶│   TTS   │  Pipeline Stages
    │ Whisper │   │  Qwen   │   │ XTTS/   │
    │  Small  │   │  3B/7B  │   │VibeVoice│
    └─────────┘   └─────────┘   └─────────┘
                       │
              ┌────────▼────────┐
              │  Response Audio │  (base64 WAV)
              └─────────────────┘
```

### 3.2 Data Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Client  │───▶│  /s2s    │───▶│ Pipeline │───▶│ Response │
│  Audio   │    │ Endpoint │    │ Execute  │    │  Audio   │
└──────────┘    └──────────┘    └────┬─────┘    └──────────┘
                                     │
                          ┌──────────┼──────────┐
                          │          │          │
                    ┌─────▼────┐ ┌───▼────┐ ┌──▼─────┐
                    │Collector │ │Metrics │ │Storage │
                    │  Track   │ │ Export │ │ JSONL  │
                    └──────────┘ └────────┘ └────────┘
```

---

## 4. Dev vs Prod Architecture

### 4.1 Development Environment

```
┌─────────────────────────────────────────┐
│           DEVELOPMENT SETUP             │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────┐  ┌──────────────┐    │
│  │   Local      │  │   Docker     │    │
│  │   Python     │  │  (optional)  │    │
│  │   (uv run)   │  │              │    │
│  └──────┬───────┘  └──────┬───────┘    │
│         │                 │             │
│         └────────┬────────┘             │
│                  │                      │
│           ┌──────▼──────┐               │
│           │  GPU Host   │               │
│           │ (optional)  │               │
│           └─────────────┘               │
│                                         │
│  Features:                              │
│  - Hot reload (--reload)                │
│  - Bind mounts for code                 │
│  - Debug logging                        │
│  - Local benchmark runs                 │
│                                         │
└─────────────────────────────────────────┘
```

**Dev Commands:**
```bash
# Local development
uv run python -m src.app

# With hot reload
uv run uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Run benchmarks
uv run python -m src.scripts.run_experiment --config configs/baseline.yml
```

### 4.2 Production Environment

```
┌─────────────────────────────────────────┐
│         PRODUCTION SETUP                │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         AWS / Cloud               │  │
│  │                                   │  │
│  │  ┌─────────────┐ ┌─────────────┐ │  │
│  │  │ ECS/Fargate │ │  EKS/K8s    │ │  │
│  │  │  (Docker)   │ │  (optional) │ │  │
│  │  └──────┬──────┘ └──────┬──────┘ │  │
│  │         │               │        │  │
│  │         └───────┬───────┘        │  │
│  │                 │                │  │
│  │  ┌──────────────▼──────────────┐ │  │
│  │  │         ALB / Ingress       │ │  │
│  │  └─────────────────────────────┘ │  │
│  │                                 │  │
│  │  ┌──────────────┐ ┌──────────┐ │  │
│  │  │ CloudWatch   │ │  ECR     │ │  │
│  │  │ Logs/Metrics │ │ Registry │ │  │
│  │  └──────────────┘ └──────────┘ │  │
│  │                                 │  │
│  └─────────────────────────────────┘  │
│                                         │
│  Features:                              │
│  - Immutable container images           │
│  - Auto-scaling                         │
│  - Health/readiness probes              │
│  - Structured logging                   │
│  - Metrics export (Prometheus)          │
│  - Secrets management                   │
│                                         │
└─────────────────────────────────────────┘
```

**Prod Characteristics:**
- No bind mounts
- Immutable image tags (git-sha)
- Environment-driven config
- Health/readiness endpoints
- Structured logs (JSON)
- Metrics endpoint for scraping

---

## 5. Component Specifications

### 5.1 API Service (`src/app.py`)

**Current:** Broken - uses stale process_sample signature

**Target:**
```python
# Target API Contract
@s2s_router.post("/s2s")
async def speech_to_speech(audio: UploadFile) -> S2SResponse:
    """
    Request: multipart/form-data with audio file
    Response: {
        "transcript": str,
        "response": str,
        "audio": base64,
        "metrics": {...}
    }
    """

@health_router.get("/health")
async def health_check() -> HealthStatus:
    """Liveness probe - always returns 200 if process up"""

@health_router.get("/ready")
async def readiness_check() -> ReadinessStatus:
    """Readiness probe - checks model loading, GPU availability"""
```

### 5.2 Pipeline (`src/sts_pipeline.py`)

**Current:** ✅ Working
- process_sample with BenchmarkCollector
- Supports history/conversation
- Handles ASR/LLM/TTS stages

**Target:** No changes needed (already correct)

### 5.3 Benchmark Framework

**Current:** ✅ Working
- ExperimentRunner with storage
- Comprehensive metrics
- JSONL persistence
- Summary generation

**Target:**
- Add API for benchmark triggering
- Export metrics for monitoring

### 5.4 Configuration System

**Current:** YAML files with relative paths

**Target:**
```python
# Pydantic Settings with env vars
class Settings(BaseSettings):
    env: Literal["dev", "staging", "prod"]
    model_paths: ModelPaths
    gpu_device: str = "cuda:0"
    max_audio_duration: float = 30.0
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
```

### 5.5 Observability

**Current:** Basic logging

**Target:**
```
Logs:
  - Structured JSON logs
  - Request IDs
  - Stage timings
  - Error traces

Metrics:
  - Request latency (p50, p95, p99)
  - Stage latency breakdown
  - GPU utilization
  - Queue depth
  - Error rates

Export:
  - /metrics endpoint (Prometheus format)
  - CloudWatch logs
  - Custom dashboard
```

---

## 6. Gap Analysis

### 6.1 Must Fix (PR1 Blockers)

| Issue | Impact | Effort |
|-------|--------|--------|
| API stale contract | Cannot deploy | 1-2 days |
| Missing CI/CD | No quality gates | 1 day |
| Dev-only Docker | Cannot deploy | 1 day |
| Missing health/ready | K8s cannot manage | 1 day |

### 6.2 Should Have (PR2-4)

| Issue | Impact | Effort |
|-------|--------|--------|
| Environment config | Portability | 1 day |
| API integration tests | Confidence | 1-2 days |
| Image versioning | Rollback capability | 1 day |
| Secrets management | Security | 1 day |

### 6.3 Nice to Have (PR5-8)

| Issue | Impact | Effort |
|-------|--------|--------|
| Full IaC | Production credibility | 2-3 days |
| Metrics dashboard | Observability | 2 days |
| Load testing | Performance validation | 2 days |
| Runbooks | Operations | 1 day |

---

## 7. Recommended Implementation Order

### Phase 1: Foundation (Week 1)
1. **PR1: Runtime Contract**
   - Fix API to use current pipeline
   - Add /ready endpoint
   - Update tests

2. **PR2: API Tests**
   - Add integration tests
   - Request validation
   - Error handling

### Phase 2: Packaging (Week 2)
3. **PR3: Container Hardening**
   - Production Dockerfile
   - .dockerignore
   - Dev/Prod split

4. **PR4: Config**
   - Pydantic settings
   - Environment profiles
   - Secrets handling

### Phase 3: Automation (Week 3)
5. **PR5: CI**
   - GitHub Actions
   - Lint/test/build

6. **PR6: CD**
   - Image publishing
   - Version tags

### Phase 4: Deployment (Week 4)
7. **PR7: IaC**
   - Terraform for staging
   - ECS/Fargate

8. **PR8: Observability**
   - Metrics endpoint
   - Dashboard
   - Runbooks

---

## 8. Open Questions

1. **GPU Requirements:**
   - Can we run CPU-only fallback for small models?
   - What's min GPU memory required?

2. **Model Caching:**
   - Should models be baked into image or downloaded at startup?
   - How to handle model updates?

3. **Scaling Strategy:**
   - Horizontal scaling (multiple pods) or vertical (bigger GPU)?
   - How to handle concurrent requests?

4. **Cost Optimization:**
   - Spot instances for batch processing?
   - Auto-scaling based on queue depth?

5. **Benchmark vs API:**
   - Should benchmark runs be separate deployment?
   - Or API endpoint to trigger benchmarks?

---

## 9. Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| FastAPI for API | Python-native, async support, good docs | 2024 |
| JSONL for storage | Simple, human-readable, no DB needed | 2024 |
| Docker containerization | Portable, reproducible, cloud-native | 2024 |
| AWS for deployment | Aligns with target JD | 2024 |

---

## 10. Appendix

### A. File Inventory

**Core (Must Keep):**
- `src/sts_pipeline.py` - Main pipeline
- `src/app.py` - API (needs fix)
- `src/benchmark/runner/experiment_runner.py` - Benchmark
- `src/benchmark/storage/` - Storage layer
- `configs/` - Configurations

**To Review:**
- `src/benchmark/runner.py` - Already deleted ✅
- `src/Untitled.ipynb` - Notebook, maybe archive
- `tests/integration_tests/` - Good coverage

**To Add:**
- `.github/workflows/ci.yml`
- `.github/workflows/cd.yml`
- `Dockerfile.prod`
- `docker-compose.prod.yml`
- `infra/terraform/`
- `docs/runbook.md`

### B. Technology Choices

| Layer | Current | Target |
|-------|---------|--------|
| Framework | FastAPI | FastAPI (keep) |
| Models | Whisper, Qwen, XTTS/VibeVoice | Same |
| Container | Docker (dev) | Docker (prod hardened) |
| Orchestration | None | AWS ECS/Fargate |
| CI/CD | None | GitHub Actions |
| IaC | None | Terraform |
| Monitoring | None | CloudWatch + Prometheus |
| Logs | Basic | Structured JSON |
