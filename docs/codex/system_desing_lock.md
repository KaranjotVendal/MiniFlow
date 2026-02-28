# MiniFlow System Design Lock: Dev-to-Prod Demonstration Blueprint

## Summary
This planning package defines a decision-complete architecture and delivery plan for MiniFlow as a **production-style demo system** (not a final enterprise platform), optimized for:
1. Demonstrating CI/CD + deployment + operational ownership.
2. Preserving current baseline behavior as initial production release.
3. Enabling measurable latency-improvement releases afterward (model lifecycle, caching, streaming).

Chosen direction:
1. Deployment target: `AWS ECS/Fargate`.
2. Service mode v1: `Synchronous API first` (`/s2s`).
3. Planning depth: full spec pack (`PRD + architecture + interfaces + flows + DoD`).
4. Security scope: `basic service security` (API key + secrets + IAM).
5. Benchmark recommendation: `Hybrid`.
- Live path: lightweight request telemetry only.
- Offline path: full benchmark framework for CI/staging evaluation.
6. Model loading recommendation: `Phase-gated`.
- Release baseline (R1): current sequential per-request/per-trial loading preserved.
- Release improvement (R2+): warm-cache model lifecycle introduced as measurable optimization release.

Document hierarchy and source of truth:
1. This file (`docs/codex/system_desing_lock.md`) is the strategic architecture and product contract.
2. `DEPLOYMENT_PR_STACK.md` is the implementation execution plan (PR-by-PR detail).
3. `pipeline_instrumentation_migration.md` defines the interface-level migration from benchmark-coupled hooks to runtime telemetry adapters.
4. If there is a conflict, this file wins for architecture and scope, and `DEPLOYMENT_PR_STACK.md` must be updated to match.

Change Control:
1. Any architecture, scope, phase ordering, or release-strategy change must be made in this file first.
2. After this file changes, `DEPLOYMENT_PR_STACK.md` must be updated in the same PR to preserve phase-to-PR consistency.
3. Every PR touching either document must include a short `Docs Sync` note stating:
- what changed,
- which file is source-of-truth for that change,
- and whether the companion doc was updated.
4. If only execution detail changes (task breakdown, acceptance checks, file list), update `DEPLOYMENT_PR_STACK.md` and confirm this file remains valid.

Docs Sync (2026-02-19):
1. Added `docs/codex/pipeline_instrumentation_migration.md`.
2. Source-of-truth remains this file for architecture and scope.
3. PR stack doc updated to reference the instrumentation migration and temporary PR1 compatibility state.

Status snapshot (as of 2026-02-17):
1. Benchmark framework integration is complete and merged into `main`.
2. Deployment/operations work has not started yet and is governed by the phased plan below.

---

## Product Requirements (PRD Lock)

### Problem Statement
MiniFlow is technically functional but not yet presented as an operational ML system. The gap is not model quality alone; it is production execution: contract stability, packaging, deployment automation, observability, and maintenance workflows.

### Goals
1. Ship a deployable MiniFlow service with reproducible CI/CD.
2. Run it in managed runtime (ECS/Fargate) with health/readiness and basic telemetry.
3. Maintain benchmark/eval framework as release quality gate.
4. Establish a release ladder where each release improves latency and reliability.

### Non-Goals (for this phase)
1. Full auth platform (OAuth/RBAC).
2. Multi-region active-active deployment.
3. Streaming-first realtime architecture in v1.
4. External model serving microservice split in v1.

### Success Criteria
1. `main` branch always deployable (green CI).
2. Immutable image versions published and deployable.
3. Staging deployment accessible with `/health` and `/ready`.
4. One baseline release + one measurable latency improvement release documented.
5. Operational runbook includes rollback and incident steps.

---

## System Design

## 1) Component Inventory: Have / Refactor / Create / Defer / Delete

### A. Already Present (Keep)
1. Core pipeline stages:
- `src/stt/stt_pipeline.py`
- `src/llm/llm_pipeline.py`
- `src/tts/tts_pipelines.py`
- `src/sts_pipeline.py`
2. Benchmark framework:
- `src/benchmark/collectors/*`
- `src/benchmark/runner/experiment_runner.py`
- `src/benchmark/storage/*`
3. FastAPI service shell:
- `src/app.py`
4. Containerization assets:
- `Dockerfile`
- `docker-compose.yml`
5. Test base:
- `tests/unit_tests/*`
- `tests/integration_tests/benchmark/*`

### B. Refactor Needed
1. `src/app.py`
- Align to current `process_sample` contract.
- Remove stale response assumptions.
- Make `/ws` explicitly v2 or controlled stub.
2. `src/benchmark/runner.py`
- Mark legacy; prevent accidental use in deploy path.
3. Config paths
- Remove absolute local path dependency in benchmark config.
4. Docker runtime mode
- Split dev (`--reload`) from prod runtime command.
5. Logging/metrics endpoint
- Replace stub `/metrics` with real exported metrics.

### C. Must Create
1. CI workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/cd.yml`
- `.github/workflows/deploy-staging.yml`
2. Environment config model:
- `src/config/settings.py` (typed settings).
- `.env.example`
3. Infra-as-code:
- `infra/terraform/*` for ECS/Fargate staging stack.
4. API integration tests:
- `tests/integration_tests/api/*`
5. Ops docs:
- `docs/deploy.md`
- `docs/runbook.md`
- `docs/slo.md`
- `docs/release-notes-template.md`

### D. Defer (Future Releases)
1. Warm-cache inference lifecycle.
2. Streaming production endpoint.
3. Autoscaling and queueing architecture.
4. Multi-service model hosting split.

### E. Delete / De-emphasize
1. Legacy runner as primary entry (`src/benchmark/runner.py`) from active docs and deployment flows.
2. Ambiguous duplicate execution paths that bypass current collector contracts.

---

## 2) Target Architecture (v1 baseline + v2 improvement path)

### v1 Baseline (Release R1)
```text
Client -> ALB/API Gateway -> FastAPI (ECS task)
  -> process_sample()
     -> run_asr() -> run_llm() -> run_tts()
  -> response
  -> structured logs + basic metrics
Offline:
  -> ExperimentRunner + BenchmarkCollector
  -> raw_logs.jsonl + summary.json
  -> CI/staging eval gates
```

### v2 Improvement (Release R2+)
```text
Same runtime contract
+ model lifecycle enhancement:
  startup/load hooks + in-process warm model cache
+ latency comparison reports against R1 baseline
+ controlled rollout + rollback
```

---

## 3) Dev vs Prod Operating Model

### Dev Mode
1. `docker compose` with mounted source + reload.
2. Debug scripts under `src/debug_scripts/*`.
3. Local benchmark runs for rapid iteration.

### Prod-like Mode (Staging/Prod)
1. Immutable container image, no bind mount, no reload.
2. Env-driven config only.
3. ECS/Fargate task deployment via Terraform.
4. Health/readiness checks + log/metric monitoring.
5. CI gates + release promotion workflow.

### Cost Control (Free/Student Constraint)
1. Budget model:
- Treat cloud deployment as a credit-limited demo environment, not 24/7 production.
2. Runtime policy:
- Staging should be deployed on-demand and shut down when not actively testing.
3. Capacity policy:
- Keep single-service, minimum-size tasks for v1 demonstration.
4. Guardrails:
- Set budget alerts (`$10`, `$25`, `$50`) and enforce release freeze when threshold is exceeded.
5. Student-credit strategy:
- Use student/cloud credits first and maintain a provider fallback path in docs.
6. Release gating:
- Every release PR must include a brief cost-impact estimate and expected runtime window.

### Cloud Fallback Strategy
1. Primary deployment target:
- AWS ECS/Fargate (best alignment with target role).
2. Fallback order:
- Fallback A: Azure managed container runtime via student credits.
- Fallback B: DigitalOcean managed container runtime via student/github credits.
- Fallback C: Other low-cost managed container host for demo continuity.
3. Trigger conditions to switch providers:
- AWS credits unavailable or insufficient for staged demo runs.
- Cost forecast exceeds approved budget threshold for two consecutive release cycles.
- Required managed runtime features blocked under free/student constraints.
4. Architectural rule:
- Application/service contracts and benchmark architecture remain cloud-agnostic.
- Only infra/deploy layers are provider-specific.
5. Repository structure rule:
- Isolate provider assets under provider-specific infra paths (for example: `infra/aws`, `infra/azure`).
6. Migration readiness requirement:
- Maintain a provider switch checklist (image registry, secrets, networking, health checks, dashboards, rollback flow).

---

## 4) Benchmark Positioning Decision (Recommended Hybrid)

### Problem Definition
MiniFlow has two different observability needs that should not be conflated:
1. Runtime operations needs on live traffic:
- low-overhead metrics for latency, reliability, and saturation.
2. Deep experimentation and regression analysis needs:
- rich trial-scoped metrics (timing, lifecycle, hardware, quality, token analysis).

If full benchmark collection is executed on every live request, the API hot path becomes heavier, less predictable, and harder to operate.
If benchmarking is fully detached from service telemetry, release decisions become under-informed.

### Solution Options
1. Full live-path benchmarking:
- run `BenchmarkCollector` logic on every API request.
- strongest detail, highest overhead/risk.
2. Fully offline benchmarking:
- benchmark only in CI/staging jobs.
- safest runtime path, but weaker live operational visibility.
3. Hybrid split (chosen):
- lightweight runtime telemetry in live path.
- full benchmark runner in offline eval path.

### Final Decision
Use hybrid split to balance runtime safety and engineering rigor:
1. Live API path:
- record low-overhead operational telemetry only.
2. Offline benchmark path:
- keep `ExperimentRunner` + `BenchmarkCollector` as the deep evaluation authority.
3. Release gating:
- use offline benchmark artifacts in CI/staging to approve latency-impacting releases.

### Pros
1. Keeps request latency and failure surface controlled.
2. Preserves detailed benchmark depth for optimization work.
3. Demonstrates both production operations and ML evaluation maturity.

### Cons
1. Requires explicit mapping between runtime metrics and benchmark metrics.
2. Requires documentation discipline to keep both telemetry planes aligned.

---

## 5) Observability Standard (SOTA-Aligned, Pragmatic Scope)

### Standard Stack for This Project
1. Instrumentation:
- OpenTelemetry-compatible structured telemetry model (logs/metrics/traces ready).
2. Metrics backend:
- Prometheus for time-series scraping.
3. Dashboards and alerting:
- Grafana for visualization and SLO alerts.
4. Deep eval artifacts:
- benchmark storage outputs (`raw_logs.jsonl`, `summary.json`) remain offline analysis source.

### Where Prometheus and Grafana Fit
1. Prometheus:
- scrapes `/metrics` from deployed service.
- stores runtime operational metrics.
2. Grafana:
- reads Prometheus data.
- provides service health, latency, error, and release comparison dashboards.
3. They do not replace the benchmark runner:
- they serve operations and SLO management, not deep trial analytics.

### Required Runtime Metrics (v1)
1. Request-level:
- request count, error count, error rate, request latency (`p50/p95/p99`).
2. Stage-level summaries:
- ASR/LLM/TTS stage latency aggregates (lightweight).
3. Resource-level:
- GPU memory/utilization summaries, process-level health metrics.
4. Model/LLM economics:
- token counters and token-rate summaries needed for cost/latency trade-off tracking.

### Release Observability Requirements
1. Every deployment must carry release identity:
- `git_sha`, image tag, config/version metadata.
2. Dashboards must support pre/post release comparison.
3. Alerts must be SLO-driven (user-impact first), not noise-driven.

---

## 6) Model Loading Strategy Decision (Release Cadence Aware)

### Recommendation
Phase-gated strategy aligned to your release narrative:
1. R1 (baseline in prod): keep current sequential loading behavior.
2. R2 (improvement release): introduce warm cache lifecycle optimization.
3. R3+: selective stage-level cache policies and optional streaming.

### Why
1. Demonstrates controlled production change management.
2. Produces objective before/after latency evidence.
3. Avoids risky architecture shift before deployment pipeline is proven.

---

## Public APIs / Interfaces / Types (Changes to Lock)

## 1) API Layer Contract
1. `POST /s2s`
- Input: uploaded audio.
- Output: structured response based on `ProcessedSample` fields + telemetry envelope.
2. `GET /health`
- Liveness only.
3. `GET /ready`
- Readiness checks: config, model capability flags, dependencies.
4. `GET /metrics`
- Real metrics export (replace stub).
5. `WS /ws`
- Explicit status: deferred-v2 (or limited stub with clear contract).

## 2) Internal Interface Lock
1. `process_sample(...)` remains canonical pipeline orchestration entry.
2. `ExperimentRunner` remains canonical offline benchmark executor.
3. `BenchmarkCollector` remains trial-scoped metrics collector for offline benchmark path.

## 3) Configuration Interface
1. Typed settings object for runtime environment.
2. No absolute machine paths in config.
3. Distinct config overlays: `dev`, `staging`, `prod`.

---

## Chronological Execution Plan (Pre-implementation to Implementation-Ready)

Phase-to-PR mapping (must match `DEPLOYMENT_PR_STACK.md`):
1. Phase 1 -> PR1, PR2
2. Phase 2 -> PR3, PR4
3. Phase 3 -> PR5, PR6
4. Phase 4 -> PR7
5. Phase 5 -> PR8
6. Phase 6 -> Post-PR8 optimization releases (R2+)

## Phase 0: Design Lock Deliverables
1. Create/update:
- `DEPLOYMENT_PR_STACK.md` (already done)
- `docs/system-design.md`
- `docs/prd.md`
- `docs/dev-vs-prod-flow.md`
2. Include sequence diagrams for:
- `/s2s` request flow
- benchmark evaluation flow
- CI/CD deploy flow
3. Exit criteria:
- component map agreed,
- v1/v2 boundaries locked,
- acceptance criteria per PR locked.

## Phase 1: Runtime Contract Stabilization (PR1-PR2)
1. Align `src/app.py` to current pipeline interfaces.
2. Freeze `/ws` decision for v1.
3. Add API integration tests.
4. Exit criteria:
- API contract tests pass,
- no stale interface usage remains.

## Phase 2: Packaging and Config Portability (PR3-PR4)
1. Harden Docker for prod profile.
2. Add env-driven settings.
3. Remove absolute path dependencies.
4. Exit criteria:
- image runs without source mounts,
- same config works across machines.

## Phase 3: CI/CD Foundation (PR5-PR6)
1. Add CI quality gates.
2. Add image publish workflow with immutable tags.
3. Exit criteria:
- PRs blocked on failing checks,
- release artifacts reproducible.

## Phase 4: Managed Deployment (PR7)
1. Provision ECS/Fargate staging via Terraform.
2. Deploy from pipeline.
3. Exit criteria:
- staging endpoint healthy/ready,
- deployment repeatable.

## Phase 5: Operability and Proof (PR8)
1. Implement observability baseline.
2. Add runbooks + rollback drill.
3. Record baseline metrics.
4. Exit criteria:
- operational drill executed,
- evidence pack ready for interview.

## Phase 6: Latency Improvement Release (R2 narrative)
1. Introduce warm-cache model lifecycle optimization.
2. Compare against R1 baseline with benchmark evidence.
3. Exit criteria:
- measurable latency improvement documented,
- rollback-safe rollout pattern demonstrated.

---

## Test Plan and Scenarios

## 1) Unit and Benchmark Tests
1. Continue full unit suite (`pytest`).
2. Validate benchmark schema outputs:
- `raw_logs.jsonl`
- `summary.json`

## 2) API Integration Tests
1. `/health` returns stable liveness.
2. `/ready` fails when config/dependencies invalid.
3. `/s2s`:
- valid audio success path,
- invalid payload failure path,
- timeout/error path.

## 3) Container and Runtime Tests
1. Build image in CI.
2. Run container smoke test with health probe.
3. Verify prod command has no reload.

## 4) Deployment Tests
1. Staging deploy success.
2. Post-deploy health/readiness checks.
3. Rollback to previous image tag.

## 5) Release Comparison Tests
1. R1 baseline benchmark archived.
2. R2 benchmark archived.
3. delta report for p50/p95 latency and error rate.

---

## Risks and Mitigations

1. Risk: Contract drift between app and pipeline.
- Mitigation: freeze interfaces in Phase 1 and gate with integration tests.
2. Risk: Scope creep from streaming and cache redesign too early.
- Mitigation: hard boundary: streaming and warm cache are post-R1 releases.
3. Risk: Observability bolted on too late.
- Mitigation: enforce minimal telemetry in same PR as deployment path.
4. Risk: Demo looks local-only.
- Mitigation: prioritize ECS/Fargate staging and CI/CD artifacts in evidence pack.

---

## Assumptions and Defaults (Explicit)
1. Cloud target for demonstration is AWS ECS/Fargate.
2. v1 serving mode is synchronous `/s2s` only; websocket streaming deferred to v2.
3. Baseline release preserves current sequential model loading behavior.
4. Benchmark framework remains offline evaluation authority for release gating.
5. Security for demo scope is API key + secrets management + least-privilege IAM.
6. This phase optimizes for demonstrable production competency, not maximum feature completeness.
