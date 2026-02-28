# PR Stack Plan
## MiniFlow Deployment Demonstration (CI/CD + Deploy + Operate)

## Document Contract
1. Strategic source of truth: `docs/codex/system_desing_lock.md`.
2. This file is the execution layer: exact PR sequence, scope, and acceptance criteria.
3. `docs/codex/pipeline_instrumentation_migration.md` is the interface migration spec for decoupling offline benchmark collection from online runtime telemetry.
4. If architecture/scope changes, update `docs/codex/system_desing_lock.md` first, then sync this file.

## Change Control
1. If strategy, architecture, release phasing, or scope changes: update `docs/codex/system_desing_lock.md` first, then update this file in the same PR.
2. If only implementation detail changes: update this file and verify no strategic conflict with `docs/codex/system_desing_lock.md`.
3. For any PR modifying either document, add a `Docs Sync` note in the PR description:
   1. Source-of-truth file for the change.
   2. Companion file update status.
   3. Impacted phases/PR numbers.

## Docs Sync (2026-02-19)
1. Added instrumentation migration spec: `docs/codex/pipeline_instrumentation_migration.md`.
2. Locked direction:
   1. `BenchmarkCollector` remains offline-only.
   2. Request path moves to runtime telemetry adapters (OpenTelemetry + Prometheus), not benchmark collectors.
3. PR1 note:
   1. Temporary compatibility path may exist for synchronous API execution.
   2. It must be removed as part of instrumentation migration per the new spec.
4. Impacted phases:
   1. PR1 (temporary compatibility),
   2. PR2 (contract tests remain stable during migration),
   3. PR8 (runtime telemetry adapter becomes primary live-path instrumentation).

## 1. Objective (Why We’re Doing This)
You are not trying to build a fully hardened enterprise platform yet.
You are building a credible production-style demo that proves:

1. You can package ML inference code into a service.
2. You can run CI/CD with quality gates.
3. You can deploy to a managed target.
4. You can observe and operate the system.
5. You can manage change safely (versioning, rollback, runbooks).

This is exactly the signal hiring managers look for when they ask for production ML.

## 2. Scope and Constraints
1. Scope is deployment-readiness demonstration, not global architecture perfection.
2. Keep benchmark framework and API integration coherent.
3. Preserve fast iteration speed.
4. Avoid expensive infra until needed.
5. Prefer AWS-aligned patterns to match your target JD, but keep local-first dev loop.
6. Constrain all deployment work to free/student-credit-friendly operating patterns.

## 3. Current State (From Repo Assessment)
1. Benchmark/test baseline is strong (`144 passed`).
2. New benchmark framework is integrated and merged into `main`.
3. FastAPI and Docker are present.
4. API contract alignment with current pipeline is a planned PR1 validation/fix scope.
5. Legacy runner (`src/benchmark/runner.py`) conflicts conceptually with new runner (`src/benchmark/runner/experiment_runner.py`).
6. No CI workflows present.
7. No IaC present.
8. Container setup is dev-mode (`--reload`, bind mounts) and not production image flow.
9. Config has machine-absolute paths (`configs/3_TTS-to-vibevoice.yml`).

## 4. Definition of Demo Production-Ready
You are ready when all are true:

1. API works end-to-end in containerized mode with current pipeline contracts.
2. CI runs lint/tests/build on PR.
3. Image is versioned and published to registry.
4. Non-local deployment path exists (staging target).
5. Health/readiness, logs, and key metrics are visible.
6. Rollback is documented and executed once in practice.
7. Benchmark and API artifacts can be shown in interview.

---

## 5. Locked Design Decisions (Synced from System Design)

## 5.1 Benchmark Positioning (Hybrid)
Problem:
1. Full benchmark collection on every live request increases hot-path overhead and operational risk.
2. Offline-only benchmarking can leave a gap in live operational visibility.

Decision:
1. Live API path uses lightweight operational telemetry only.
2. Offline benchmark path (`ExperimentRunner` + `BenchmarkCollector`) remains the deep analysis authority.
3. Release gates use offline benchmark artifacts; runtime dashboards use live telemetry.

Implication for implementation:
1. Do not embed full trial collector logic in request handlers.
2. Add and maintain a runtime-to-benchmark metric mapping doc in PR8.

## 5.2 Observability Standard (SOTA-aligned, pragmatic)
Decision:
1. OpenTelemetry-aligned instrumentation model for logs/metrics/traces.
2. Prometheus for runtime metric scraping and storage.
3. Grafana for dashboards and SLO alerting.
4. Benchmark artifacts (`raw_logs.jsonl`, `summary.json`) remain offline eval data products.

Minimum runtime metrics for v1:
1. Request: count, error count/rate, latency (`p50/p95/p99`).
2. Stage latency summaries: ASR/LLM/TTS.
3. Resource summaries: GPU memory/utilization.
4. LLM economics: token counts/rates for cost-latency tracking.

Release observability requirements:
1. Every deployment tagged with release identity (`git_sha`, image tag, config version).
2. Dashboard supports pre/post release comparison.
3. Alerts are SLO-driven and user-impact oriented.

## 5.3 Cost Control Standard (Free/Student-Credit First)
Decision:
1. Staging is on-demand, not permanently online.
2. Deployments run with minimum viable capacity for demo scope.
3. Cost is a release gate input, not an afterthought.

Required controls:
1. Budget alerts configured at `$10`, `$25`, `$50`.
2. Scheduled or manual scale-down policy outside active demo/test windows.
3. Every deploy PR includes expected runtime duration and estimated cost impact.
4. If monthly threshold is exceeded, non-critical deploys pause until reviewed.

## 5.4 Cloud Fallback Strategy
Decision:
1. Primary provider is AWS for role alignment.
2. Keep provider switch as an execution fallback, not an architecture rewrite.

Fallback order:
1. Fallback A: Azure managed container runtime using student credits.
2. Fallback B: DigitalOcean managed container runtime using student/github credits.
3. Fallback C: Any low-cost managed container host for continuity.

Switch triggers:
1. AWS free/student credits unavailable or insufficient for planned demo cadence.
2. Forecasted spend exceeds approved threshold for two consecutive release windows.
3. Required managed-runtime capabilities are blocked under cost constraints.

Execution rule:
1. Service/API and benchmark contracts stay unchanged.
2. Only infra and deployment workflow steps are provider-specific.
3. Keep provider-specific assets isolated (for example: `infra/aws`, `infra/azure`).

---

## 6. PR Stack (Chronological)

## PR1: Runtime Contract Alignment
**Title:** `Align API and Runtime Entrypoints with Current Pipeline Contracts`

**Why first**
1. Deployment on top of broken runtime contracts creates noise and rework.
2. This removes the highest-risk mismatch before touching CI/CD.

**Scope**
1. Update `src/app.py` to call current `process_sample(...)` signature.
2. Update `/s2s` payload construction to match `ProcessedSample`.
3. Decide whether `/ws` stays stubbed or temporary-disabled with clear response.
4. Mark `src/benchmark/runner.py` as deprecated or remove from active docs.
5. Ensure README commands point to current paths (`src/debug_scripts/...`).

**Files**
1. `src/app.py`
2. `src/sts_pipeline.py` (only if needed for compatibility wrapper)
3. `src/benchmark/runner.py` or docs referencing it
4. `README.md`
5. `src/debug_scripts/debug_benchmark.py` if needed for contract usage

**Step-by-step**
1. Create a compatibility plan for `process_sample` invocation.
2. Implement API-side adaptation only, avoid pipeline behavior changes.
3. Add explicit error handling for invalid request payloads.
4. Validate `py_compile` and run tests.
5. Run one manual `/s2s` request locally.

**Acceptance criteria**
1. `/s2s` returns valid structured response without interface errors.
2. No stale call pattern remains in active entrypoints.
3. Full test suite still passes.

**Interview artifact**
1. Refactored service boundary after benchmark architecture change, with zero regression in test suite.

---

## PR2: API Contract Tests + Basic Reliability
**Title:** `Add API Integration Tests and Reliability Guards`

**Why now**
1. CI/CD without route-level tests is shallow.
2. You need confidence on the user-facing contract before deploying.

**Scope**
1. Add integration tests for `/health`, `/s2s`, `/ws` behavior.
2. Add request validation model(s) and predictable error responses.
3. Add input size guardrails and timeout defaults.
4. Add readiness endpoint `/ready`.

**Files**
1. `src/app.py`
2. `tests/integration_tests/api/test_health.py`
3. `tests/integration_tests/api/test_s2s.py`
4. `tests/integration_tests/api/test_ws.py` (or documented defer if stub)

**Step-by-step**
1. Define response schemas for each endpoint.
2. Add readiness logic with explicit checks.
3. Add tests for success/failure/timeouts.
4. Run targeted + full test suite.

**Acceptance criteria**
1. API integration tests pass consistently.
2. Failure modes return stable status codes/messages.
3. `/ready` is distinct from `/health`.

**Interview artifact**
1. Defined service contracts and added integration tests for runtime reliability.

---

## PR3: Container Hardening (Dev vs Prod)
**Title:** `Harden Docker Packaging and Split Dev/Prod Runtime Profiles`

**Why now**
1. You currently run dev-mode in container (`--reload`).
2. You need a deployable image artifact for registry and cloud run.

**Scope**
1. Add `.dockerignore`.
2. Remove prod default `--reload`.
3. Keep separate dev compose override for local reload workflow.
4. Ensure image does not include unnecessary local artifacts.

**Files**
1. `Dockerfile`
2. `.dockerignore`
3. `docker-compose.yml`
4. `docker-compose.dev.yml` (recommended)
5. `README.md`

**Step-by-step**
1. Add ignore patterns for caches, logs, benchmark outputs, notebooks.
2. Move reload behavior to dev-only compose.
3. Keep production command minimal and deterministic.
4. Build and run image locally with health check.

**Acceptance criteria**
1. Production image starts without bind mounts.
2. Dev profile still supports rapid iteration.
3. Image build context and size are reduced.

**Interview artifact**
1. Separated dev and prod container concerns, improved deployability and reproducibility.

---

## PR4: Config and Environment Standardization
**Title:** `Introduce Environment-Driven Configuration and Portable Paths`

**Why now**
1. Absolute local paths break deployment portability.
2. Env-driven config is expected in production-oriented systems.

**Scope**
1. Replace absolute `metrics` path usage with relative/env-resolved path.
2. Add `Settings` model (Pydantic or equivalent).
3. Add `.env.example`.
4. Define environment profiles (`dev`, `staging`, `prod`).

**Files**
1. `configs/3_TTS-to-vibevoice.yml`
2. `configs/metrics.yml`
3. `src/config/` (new settings module)
4. `.env.example`
5. `README.md`

**Step-by-step**
1. Introduce a single config loader entrypoint that resolves env values.
2. Remove machine-specific hardcoded path assumptions.
3. Add validation errors that fail fast.
4. Verify local + container runtime with only env inputs.

**Acceptance criteria**
1. Service runs on a new machine/container without path edits.
2. Missing critical env vars fail loudly.
3. Config behavior is deterministic across environments.

**Interview artifact**
1. Converted local-only config into deployable multi-env configuration model.

---

## PR5: CI Quality Gate
**Title:** `Add CI Workflow for Lint, Tests, and Build Verification`

**Why now**
1. This is the core production signal you want to demonstrate.
2. Must be in place before registry publish and deploy automation.

**Scope**
1. Add GitHub Actions workflow for PR/push:
   1. dependency sync
   2. lint/check
   3. tests
   4. Docker build smoke
2. Add status badges (optional but useful).

**Files**
1. `.github/workflows/ci.yml`
2. `README.md`

**Step-by-step**
1. Define CI matrix (single Python version is fine for demo).
2. Run `uv sync`, `pytest`, optional `ruff`.
3. Add Docker build step without push.
4. Enforce branch protection using CI status.

**Acceptance criteria**
1. CI runs automatically on PR.
2. CI fails on regressions.
3. Merge requires green checks.

**Interview artifact**
1. Implemented automated quality gate for ML service code and packaging.

---

## PR6: CD + Registry Publishing
**Title:** `Automate Versioned Image Publishing and Deployment Trigger`

**Why now**
1. CD shows operational maturity.
2. Versioned artifacts enable rollback and traceability.

**Scope**
1. Add workflow to publish tagged images (`git-sha`) to registry (`GHCR` or ECR).
2. Define release strategy (`main` push or tags).
3. Persist image metadata.

**Files**
1. `.github/workflows/cd.yml`
2. Deployment docs in `README.md` or `docs/deploy.md`

**Step-by-step**
1. Configure registry auth via repo secrets.
2. Build/push immutable tags.
3. Emit deployment artifact metadata.
4. Validate pull/run from fresh environment.

**Acceptance criteria**
1. Every release has immutable image tag.
2. Previous image remains deployable.
3. Release pipeline is repeatable.

**Interview artifact**
1. Built versioned deployment artifacts with reproducible releases and rollback path.

---

## PR7: IaC + Staging Deploy
**Title:** `Provision Staging Deployment via Infrastructure as Code`

**Why now**
1. This demonstrates actual deployment skill beyond local compose.
2. Aligns directly with JD expectations.

**Scope**
1. Add minimal IaC for staging service (prefer AWS ECS/Fargate for role fit).
2. Include service, task, networking, logs.
3. Wire deployment workflow to staging.
4. Implement cost guardrails (budget alarms + scale-down policy).
5. Add provider-switch checklist and fallback-ready infra layout.

**Files**
1. `infra/terraform/...`
2. `docs/deploy.md`
3. `.github/workflows/deploy-staging.yml`

**Step-by-step**
1. Define minimal module layout and variables.
2. Create staging stack.
3. Deploy from CI using published image tag.
4. Validate service endpoints and logs.
5. Validate budget alarms and off-hours shutdown behavior.
6. Document provider fallback triggers and switch steps.

**Acceptance criteria**
1. Staging URL responds on `/health` and `/ready`.
2. Deployed service uses registry image tag.
3. Deployment is reproducible from pipeline.
4. Budget alerts and scale-down policy are active and documented.
5. Provider fallback checklist exists and is runnable without app-code changes.

**Interview artifact**
1. Provisioned and deployed ML service on managed infrastructure via IaC.

---

## PR8: Observability + Operations Runbook
**Title:** `Add Runtime Observability, SLOs, and Rollback Runbook`

**Why now**
1. Deployment alone is not production; operation quality closes the loop.
2. This directly answers can you maintain ML systems in prod.

**Scope**
1. Add structured logs with request/trial IDs and release metadata.
2. Implement OpenTelemetry-aligned instrumentation in app/runtime paths.
3. Replace `/metrics` stub with Prometheus-exported runtime metrics.
4. Define Grafana dashboards and SLO alert thresholds.
5. Add runtime-vs-benchmark metric mapping documentation.
6. Add runbooks for incident/rollback.
7. Run one rollback drill and capture notes.
8. Add cost operations runbook section (budget actions + scale-down checklist).

**Files**
1. `src/app.py`
2. Logging/metrics modules under `src/logger/` or `src/observability/`
3. `docs/runbook.md`
4. `docs/slo.md`
5. `docs/postmortem-template.md`

**Step-by-step**
1. Instrument request lifecycle and stage timings with stable labels.
2. Expose metrics endpoint for Prometheus scraping.
3. Create Grafana dashboards for latency/error/saturation and release comparisons.
4. Document metric mapping between runtime telemetry and offline benchmark summaries.
5. Document operational procedures.
6. Execute rollback test and record evidence.

**Acceptance criteria**
1. You can trace one failed request end-to-end.
2. Core service metrics are visible in Prometheus/Grafana.
3. SLO alerts are defined and test-triggered at least once.
4. Runtime-vs-benchmark mapping is documented and reviewed.
5. Rollback tested and documented.
6. Cost-control runbook exists and is validated in one staging cycle.

**Interview artifact**
1. Implemented observability and operational procedures including rollback validation.

---

## 7. Branching and Stacked PR Workflow
1. Create linear stack:
   1. `pr1-runtime-contracts`
   2. `pr2-api-tests-reliability`
   3. `pr3-container-hardening`
   4. `pr4-config-env`
   5. `pr5-ci`
   6. `pr6-cd-registry`
   7. `pr7-iac-staging`
   8. `pr8-observability-runbook`
2. Each PR merges only after green CI and short demo evidence.
3. Keep each PR focused on one concern to make interview storytelling easy.

## 8. Evidence Pack for Hiring Manager
By the end of stack, prepare:

1. Architecture diagram (service + benchmark + deploy path).
2. CI run screenshots (green checks).
3. CD/release screenshot with image tags.
4. Staging deployment screenshot (`/health` and `/ready`).
5. Metrics/log dashboard screenshot.
6. Benchmark run artifact sample (`summary.json` + raw logs).
7. Rollback runbook + one executed rollback note.

## 9. Risk Controls During This Demo Build
1. Do not mix benchmark model redesign with deployment plumbing in same PR.
2. Keep runtime behavior changes minimal while adding deployment scaffolding.
3. Use fail-fast config validation to avoid silent environment drift.
4. Treat dev and prod commands as separate first-class workflows.

## 10. Suggested Execution Order in Time
1. Week 1: PR1 + PR2
2. Week 2: PR3 + PR4
3. Week 3: PR5 + PR6
4. Week 4: PR7 + PR8 + evidence pack
