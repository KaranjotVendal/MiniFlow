# Pipeline Instrumentation Interface and Migration Plan

## Purpose
Define a clean split between:
1. Offline benchmark collection (`BenchmarkCollector` via `ExperimentRunner`)
2. Online runtime telemetry (OpenTelemetry + Prometheus in API request path)

This document prevents coupling benchmark internals to request-serving flow.

## Design Decision
`BenchmarkCollector` remains **offline-only**.
Request path uses a separate instrumentation interface, implemented by runtime telemetry adapters.

`NoOpCollector` is treated as temporary PR1 compatibility only and should be removed after migration.

---

## Target Interface

Create a new protocol under `src/instrumentation/interfaces.py`:

```python
from typing import Protocol, Any


class PipelineInstrumentation(Protocol):
    def on_request_start(self, request_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    def on_request_end(self, request_id: str, status: str, error: str | None = None) -> None: ...

    def on_stage_start(self, request_id: str, stage: str) -> None: ...
    def on_stage_end(
        self,
        request_id: str,
        stage: str,
        status: str = "success",
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None: ...

    def on_token_metrics(
        self,
        request_id: str,
        *,
        tokens_generated: int,
        ttft: float | None,
        tokens_per_sec: float | None,
    ) -> None: ...
```

Notes:
1. Keep methods explicit; no generic `record(dict)` payloads.
2. `extra` is optional and bounded to stage-level metadata.
3. Request and stage status must be explicit (`success`/`error`).

---

## Adapters

## 1) Benchmark Adapter (offline only)
File: `src/instrumentation/adapters/benchmark_adapter.py`

Responsibilities:
1. Translate `PipelineInstrumentation` calls to existing `BenchmarkCollector` methods.
2. No behavior change to current benchmark outputs.

Usage:
1. Constructed only in `ExperimentRunner`.
2. Never used directly by FastAPI request handlers.

## 2) Runtime Telemetry Adapter (online path)
File: `src/instrumentation/adapters/runtime_telemetry_adapter.py`

Responsibilities:
1. Emit request/stage timings and counters to Prometheus.
2. Emit trace spans/events to OpenTelemetry.
3. Emit structured logs with `request_id`, `release_id`, and status.

Usage:
1. Constructed in API layer (`src/app.py`) per request.
2. Used by `process_sample` in request path.

---

## Function Signatures After Migration

`src/sts_pipeline.py`:

```python
def process_sample(
    config: dict,
    sample: AudioSample,
    run_id: str,
    device: torch.device | str,
    instrumentation: PipelineInstrumentation | None = None,
    history: list[dict] | None = None,
    stream_audio: bool = False,
) -> ProcessedSample:
    ...
```

Stage modules (`run_asr`, `run_llm`, `run_tts`) should accept `instrumentation` instead of `collector` once migration is complete.

---

## Migration Steps

## Step 1: Introduce interface and adapters (no behavior change)
1. Add `PipelineInstrumentation` protocol.
2. Add `BenchmarkInstrumentationAdapter` that wraps existing `BenchmarkCollector`.
3. Add `NullInstrumentation` (true no-op) for temporary default.
4. Keep existing `collector` flow untouched for now.

Exit criteria:
1. No benchmark output schema changes.
2. Existing tests pass.

## Step 2: Update `process_sample` to use instrumentation
1. Replace `collector` arg with `instrumentation` arg.
2. Keep a temporary compatibility shim:
   - If `collector` is passed from `ExperimentRunner`, wrap with benchmark adapter.
3. Remove `NoOpCollector` from `src/sts_pipeline.py`.

Exit criteria:
1. API path works with `NullInstrumentation` or runtime adapter.
2. Benchmark path still produces current `raw_logs.jsonl` and `summary.json`.

## Step 3: Migrate stage functions
1. Update `run_asr`, `run_llm`, `run_tts` to call instrumentation hooks.
2. Benchmark adapter maps hooks to metric classes.
3. Runtime adapter maps hooks to OTel/Prometheus.

Exit criteria:
1. Stage timing and errors visible in runtime telemetry.
2. Offline benchmark metrics unchanged.

## Step 4: Remove compatibility shim
1. Remove `collector` parameters from pipeline and stage functions.
2. Use instrumentation interface only.
3. Delete temporary null/legacy compatibility paths not needed.

Exit criteria:
1. Single obvious instrumentation path.
2. No benchmark internals in request-serving code.

---

## PR Mapping
1. PR1: Temporary stability patch allowed (`NoOpCollector`) — already done.
2. PR2: Add integration tests asserting stable API behavior while instrumentation remains minimal.
3. PR8: Introduce runtime telemetry adapter (OTel + Prometheus) and remove no-op collector path.

If schedule allows, Step 1 and Step 2 can be brought earlier (between PR2 and PR3) to reduce technical debt.

---

## Risks and Controls
1. Risk: Metric drift between runtime and benchmark.
   - Control: Maintain explicit mapping doc (`runtime metric -> benchmark metric`).
2. Risk: Breaking benchmark artifacts during refactor.
   - Control: Add snapshot/regression tests for `summary.json` keys and trial schema.
3. Risk: Over-abstraction.
   - Control: Keep interface small (request + stage + token hooks only).

---

## Sign-off Criteria for This Plan
1. Team agrees benchmark stays offline-only.
2. Team agrees request path uses runtime telemetry adapter, not benchmark collector.
3. Team agrees `NoOpCollector` is temporary and scheduled for removal.
