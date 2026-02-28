# Benchmark Framework + OpenTelemetry Integration Plan

## Core Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                     REQUEST PATH (API)                          │
├─────────────────────────────────────────────────────────────────┤
│  Path A: Lightweight Runtime Telemetry (Default)               │
│  ├── OpenTelemetry traces (optional, low overhead)             │
│  ├── Prometheus metrics (counts, histograms)                   │
│  └── Structured logs (correlation_id, latency, errors)         │
│                                                                 │
│  Path B: Deep Benchmark Collection (Opt-in, specific use)      │
│  ├── BenchmarkCollector (full trial-scoped metrics)            │
│  ├── WER, UTMOS, quality metrics                               │
│  └── Raw trial logs (audio, transcripts, artifacts)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OFFLINE PATH (Benchmark Framework)            │
├─────────────────────────────────────────────────────────────────┤
│  ExperimentRunner + BenchmarkCollector                          │
│  ├── Full evaluation dataset                                    │
│  ├── Deep metrics (WER, UTMOS, stage analysis)                 │
│  ├── Summary reports for release gating                        │
│  └── Comparison to baselines                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Decision: Three Modes of Operation

### Mode 1: Production Runtime (Default)
**Use Case:** Live API serving real requests

**Characteristics:**
- **Benchmark collection: DISABLED**
- **Telemetry: OpenTelemetry + Prometheus (lightweight)**
- **Overhead: Minimal**
- **Storage: Metrics only (no raw audio/transcripts)**

**Implementation:**
```python
# config/staging.yml
runtime:
  mode: "production"  # Default
  benchmark_on_request: false
  telemetry_enabled: true

telemetry:
  otel_enabled: true
  prometheus_enabled: true
  log_level: "INFO"
```

**Request Flow:**
```python
@app.post("/s2s")
async def speech_to_speech(request: S2SRequest):
    # 1. Lightweight telemetry only
    with tracer.start_as_current_span("s2s_request") as span:
        span.set_attribute("request.id", request_id)

        # 2. Process (NO benchmark collector)
        result = await process_sample(audio, config)

        # 3. Prometheus metrics
        REQUEST_LATENCY.observe(time_elapsed)

    return result
```

---

### Mode 2: Benchmark Evaluation Mode (Explicit Trigger)
**Use Case:** CI/CD pipeline, staging evaluation, A/B testing

**Characteristics:**
- **Benchmark collection: ENABLED via ExperimentRunner**
- **Telemetry: Still enabled for operational visibility**
- **Overhead: High (acceptable for evaluation, not production)**
- **Storage: Full trial logs + summary reports**

**Implementation:**
```python
# Triggered via API or CLI

# Option A: CLI (CI/CD)
$ python -m src.benchmark.runner \
    --config configs/eval-v1.yml \
    --dataset data/eval_set.jsonl \
    --output results/v1-benchmark/

# Option B: API endpoint (on-demand eval)
POST /admin/run-benchmark
{
  "config": "configs/eval-v1.yml",
  "dataset": "data/eval_set.jsonl",
  "release_id": "v1.2.3"
}
```

**Flow:**
```python
# src/benchmark/runner/eval_endpoint.py
@app.post("/admin/run-benchmark")
async def run_benchmark(request: BenchmarkRequest):
    # 1. Create ExperimentRunner
    runner = ExperimentRunner.from_config(request.config)

    # 2. Run full evaluation (offline, not on live traffic)
    results = runner.run(
        dataset=request.dataset,
        release_id=request.release_id
    )

    # 3. Generate summary
    summary = results.generate_summary()

    return {
        "release_id": request.release_id,
        "summary": summary,
        "report_url": f"/reports/{request.release_id}/summary.json"
    }
```

---

### Mode 3: Debug/Diagnostic Mode (Development Only)
**Use Case:** Local debugging, performance profiling

**Characteristics:**
- **Benchmark collection: ENABLED per-request**
- **Telemetry: Verbose**
- **Overhead: High (acceptable for debug)**
- **Storage: Local files only**

**Implementation:**
```python
# config/debug.yml
runtime:
  mode: "debug"
  benchmark_on_request: true  # Enable for debugging
  telemetry_enabled: true
  log_level: "DEBUG"

benchmark:
  output_dir: "./debug_benchmarks"
  collect_audio: true  # Store audio files for analysis
  collect_full_metrics: true
```

**Usage:**
```python
# Only for local debugging
with BenchmarkCollector(
    output_dir="./debug_run",
    collect_audio=True
) as collector:
    result = process_sample(audio, collector=collector)

# Analyze locally
# ./debug_run/raw_logs.jsonl
# ./debug_run/summary.json
# ./debug_run/audio_samples/
```

---

## Implementation: Configuration-Driven Approach

### 1. Central Configuration Model

```python
# src/config/settings.py
from pydantic import BaseSettings
from enum import Enum

class RuntimeMode(str, Enum):
    PRODUCTION = "production"      # API serving
    EVALUATION = "evaluation"      # Benchmark runner
    DEBUG = "debug"                # Local development

class BenchmarkConfig(BaseSettings):
    # When to enable benchmark collection
    enabled_in_production: bool = False  # Never in prod
    enabled_in_evaluation: bool = True   # Always in eval
    enabled_in_debug: bool = True        # Default in debug

    # What to collect
    collect_audio: bool = False
    collect_full_metrics: bool = True
    output_dir: str = "./benchmark_results"

    # Limits (safety)
    max_trials_in_production: int = 0  # 0 = disabled
    storage_backend: str = "jsonl"     # jsonl, s3, etc.

class TelemetryConfig(BaseSettings):
    # OpenTelemetry
    otel_enabled: bool = True
    otel_endpoint: str | None = None  # None = stdout/console
    otel_sampling_rate: float = 1.0   # 1.0 = all requests

    # Prometheus
    prometheus_enabled: bool = True
    metrics_port: int = 9090

    # Logging
    structured_logs: bool = True
    log_level: str = "INFO"

class Settings(BaseSettings):
    runtime_mode: RuntimeMode = RuntimeMode.PRODUCTION
    benchmark: BenchmarkConfig = BenchmarkConfig()
    telemetry: TelemetryConfig = TelemetryConfig()

    release_id: str = "dev"  # Git SHA or version tag
```

---

### 2. Request Path Implementation

```python
# src/app.py
from fastapi import FastAPI, Depends
from src.config.settings import Settings, RuntimeMode
from src.telemetry import get_tracer, REQUEST_LATENCY
from src.benchmark import BenchmarkCollector

app = FastAPI()
settings = Settings()

@app.post("/s2s")
async def speech_to_speech(
    request: S2SRequest,
    # Dependency injection for flexibility
    benchmark_enabled: bool = Depends(get_benchmark_flag)
):
    request_id = generate_request_id()
    start_time = time.time()

    # 1. Start OpenTelemetry trace (always in production)
    with get_tracer().start_as_current_span("s2s_request") as span:
        span.set_attribute("request.id", request_id)
        span.set_attribute("release.id", settings.release_id)

        # 2. Conditionally use BenchmarkCollector
        if benchmark_enabled:
            # Mode: Debug or Evaluation
            with BenchmarkCollector(
                trial_id=request_id,
                output_dir=settings.benchmark.output_dir,
                collect_audio=settings.benchmark.collect_audio
            ) as collector:

                result = await process_sample(
                    audio=request.audio,
                    collector=collector  # Full metrics
                )

                # Save benchmark data
                collector.finalize()
        else:
            # Mode: Production (default)
            # No collector, just telemetry
            result = await process_sample(
                audio=request.audio,
                collector=None  # Telemetry only
            )

        # 3. Prometheus metrics (always)
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        span.set_attribute("latency_ms", latency * 1000)

    # 4. Structured log (always)
    logger.info(
        "s2s_request_complete",
        extra={
            "request_id": request_id,
            "latency_ms": latency * 1000,
            "release_id": settings.release_id,
            "benchmark_enabled": benchmark_enabled
        }
    )

    return result

def get_benchmark_flag() -> bool:
    """Determine if benchmark collection should run."""
    match settings.runtime_mode:
        case RuntimeMode.PRODUCTION:
            # Never in production (safety)
            return settings.benchmark.enabled_in_production
        case RuntimeMode.EVALUATION:
            # Always in evaluation mode
            return settings.benchmark.enabled_in_evaluation
        case RuntimeMode.DEBUG:
            # Configurable in debug
            return settings.benchmark.enabled_in_debug
        case _:
            return False
```

---

### 3. Telemetry-Only `process_sample`

When `collector=None`, the pipeline still emits telemetry:

```python
# src/sts_pipeline.py
async def process_sample(
    audio: np.ndarray,
    config: PipelineConfig,
    collector: BenchmarkCollector | None = None
) -> ProcessedSample:
    """
    If collector is provided: full benchmark metrics.
    If collector is None: telemetry only (production).
    """
    tracer = get_tracer()

    # ASR Stage
    with tracer.start_as_current_span("asr_stage") as span:
        start = time.time()
        transcript = await run_asr(audio, config.stt)
        latency = time.time() - start

        span.set_attribute("asr.latency_ms", latency * 1000)
        span.set_attribute("asr.transcript_length", len(transcript))

        # Prometheus metric
        STAGE_LATENCY.labels(stage="asr").observe(latency)

        # Benchmark collector (if enabled)
        if collector:
            collector.record_asr_metrics(
                latency=latency,
                transcript=transcript
            )

    # LLM Stage
    with tracer.start_as_current_span("llm_stage") as span:
        start = time.time()
        response = await run_llm(transcript, config.llm)
        latency = time.time() - start

        span.set_attribute("llm.latency_ms", latency * 1000)
        span.set_attribute("llm.tokens", response.token_count)

        STAGE_LATENCY.labels(stage="llm").observe(latency)

        if collector:
            collector.record_llm_metrics(
                latency=latency,
                tokens=response.token_count
            )

    # TTS Stage
    with tracer.start_as_current_span("tts_stage") as span:
        start = time.time()
        audio_out = await run_tts(response.text, config.tts)
        latency = time.time() - start

        span.set_attribute("tts.latency_ms", latency * 1000)

        STAGE_LATENCY.labels(stage="tts").observe(latency)

        if collector:
            collector.record_tts_metrics(
                latency=latency,
                audio=audio_out  # Stored only if collect_audio=True
            )

    return ProcessedSample(
        transcript=transcript,
        response=response.text,
        audio=audio_out
    )
```

---

## Linking Telemetry and Benchmark Data

### The Problem
How do we correlate:
- Runtime telemetry (lightweight, always on)
- Benchmark results (deep, occasional)

### Solution: Release ID as Correlation Key

```python
# All telemetry includes release_id
release_id = "v1.2.3-abc123"  # Git SHA or version tag

# Runtime telemetry (always)
{
  "timestamp": "2026-02-17T10:30:00Z",
  "release_id": "v1.2.3-abc123",
  "request_id": "req-456",
  "latency_ms": 5234,
  "stage_latencies": {"asr": 1234, "llm": 2345, "tts": 1655}
}

# Benchmark report (evaluation mode)
{
  "release_id": "v1.2.3-abc123",
  "run_timestamp": "2026-02-17T10:00:00Z",
  "metrics": {
    "mean_latency_ms": 5200,
    "p95_latency_ms": 8500,
    "wer": 0.05,
    "utmos": 3.8
  }
}
```

**Usage in Grafana:**
```sql
-- Compare runtime metrics vs benchmark baseline
SELECT
  release_id,
  avg(latency_ms) as runtime_avg,
  benchmark_mean_latency as baseline
FROM telemetry t
JOIN benchmark_summary b ON t.release_id = b.release_id
WHERE release_id = 'v1.2.3-abc123'
GROUP BY release_id
```

---

## Operational Patterns

### Pattern 1: Release Gating (CI/CD)

```yaml
# .github/workflows/release-gate.yml
name: Benchmark Gate

on:
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - name: Run Benchmark
        run: |
          python -m src.benchmark.runner \
            --config configs/eval.yml \
            --release-id ${{ github.sha }}

      - name: Compare to Baseline
        run: |
          python -m src.benchmark.compare \
            --current results/${{ github.sha }}/summary.json \
            --baseline results/baseline/summary.json \
            --max-latency-regression 10% \
            --max-wer-regression 5%
```

### Pattern 2: Production Monitoring

```yaml
# Grafana dashboard query
# Show production latency vs benchmark baseline

groups:
  - name: s2s_alerts
    rules:
      - alert: HighLatencyVsBaseline
        expr: |
          (
            avg(s2s_request_latency_ms{env="production"})
            /
            benchmark_mean_latency_ms{release_id="$latest"}
          ) > 1.2
        for: 5m
        annotations:
          summary: "Production latency 20% above benchmark baseline"
```

### Pattern 3: Debug Collection (On-Demand)

```python
# Admin endpoint to capture debug benchmark
@app.post("/admin/capture-debug")
async def capture_debug_benchmark(request: DebugRequest):
    """Capture full benchmark for a specific request (debug only)."""

    with BenchmarkCollector(
        trial_id=request.request_id,
        output_dir=f"./debug/{request.request_id}",
        collect_audio=True
    ) as collector:

        result = await process_sample(
            audio=request.audio,
            collector=collector
        )

    return {
        "request_id": request.request_id,
        "debug_report": f"/debug/{request.request_id}/summary.json"
    }
```

---

## Configuration Examples

### Production Config
```yaml
# config/production.yml
runtime:
  mode: "production"

benchmark:
  enabled_in_production: false  # Safety: never in prod

telemetry:
  otel_enabled: true
  prometheus_enabled: true
  log_level: "INFO"

release:
  id_from: "git_sha"  # Auto-inject from env
```

### Evaluation Config
```yaml
# config/evaluation.yml
runtime:
  mode: "evaluation"

benchmark:
  enabled_in_evaluation: true
  output_dir: "./benchmark_results"
  collect_audio: true
  max_trials: 100

dataset:
  path: "data/eval_set.jsonl"

telemetry:
  otel_enabled: true  # Still want traces
  prometheus_enabled: false  # Not needed for eval
```

### Debug Config
```yaml
# config/debug.yml
runtime:
  mode: "debug"

benchmark:
  enabled_in_debug: true
  output_dir: "./debug_benchmarks"
  collect_audio: true
  collect_full_metrics: true

telemetry:
  otel_enabled: true
  otel_sampling_rate: 1.0  # All requests
  log_level: "DEBUG"
```

---

## Summary: Decision Matrix

| Scenario | Mode | Benchmark | Telemetry | Use Case |
|----------|------|-----------|-----------|----------|
| **Live API** | Production | ❌ Off | ✅ On | Serving real users |
| **CI/CD Gate** | Evaluation | ✅ On | ✅ On | Release approval |
| **Local Dev** | Debug | ✅ On (opt) | ✅ Verbose | Debugging |
| **Staging** | Production | ❌ Off | ✅ On | Pre-prod validation |
| **A/B Test** | Evaluation | ✅ On | ✅ On | Compare versions |

---

## Key Principles

1. **Benchmark collection is NEVER on in production** (safety)
2. **Telemetry is ALWAYS on** (operational visibility)
3. **ExperimentRunner is the only way to run full benchmarks** (controlled)
4. **Release ID links everything** (correlation)
5. **Configuration-driven** (no code changes to switch modes)

---

## Next Steps

1. **PR4 (Config):** Add `Settings` class with mode switching
2. **PR7 (Deploy):** Set `RUNTIME_MODE=production` in staging
3. **PR8 (Observability):** Implement OpenTelemetry + Prometheus
4. **Integration:** Add `/admin/run-benchmark` endpoint (PR8 or later)

This gives you a clean, safe, and flexible architecture.
