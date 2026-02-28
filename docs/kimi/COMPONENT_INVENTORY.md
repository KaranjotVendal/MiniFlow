# Component Inventory

## Current File Structure

### Source Code (`src/`)

| File/Dir | Purpose | Status | Action |
|----------|---------|--------|--------|
| `sts_pipeline.py` | Main pipeline entry | ✅ Good | Keep |
| `prepare_data.py` | Data loading | ✅ Good | Keep |
| `app.py` | FastAPI service | ❌ Broken | **Fix PR1** |
| `utils.py` | Utilities | ✅ Good | Keep |
| `metrics.py` | Old metrics (legacy) | ⚠️ Legacy | Review if still used |
| `Untitled.ipynb` | Jupyter notebook | ⚠️ Dev artifact | Archive or delete |
| | | | |
| `llm/` | LLM pipeline | ✅ Good | Keep |
| `tts/` | TTS implementations | ✅ Good | Keep |
| `stt/` | ASR pipeline | ✅ Good | Keep |
| `config/` | Config loading | ✅ Good | Keep |
| `logger/` | Logging utilities | ✅ Good | Keep |
| `scripts/` | CLI scripts | ✅ Good | Keep |
| `report/` | Report generation | ✅ Good | Keep |
| | | | |
| `benchmark/` | Benchmark framework | ✅ Good | Keep |
| ├── runner/experiment_runner.py | Current runner | ✅ Good | Keep |
| ├── storage/jsonl_storage.py | Persistence | ✅ Good | Keep |
| ├── collectors/ | Data collection | ✅ Good | Keep |
| ├── metrics/ | Metric impls | ✅ Good | Keep |
| └── core/ | Base classes | ✅ Good | Keep |

### Configuration (`configs/`)

| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `baseline.yml` | XTTS baseline | ✅ Good | Keep |
| `2_TTS-to-vibevoice.yml` | VibeVoice config | ✅ Good | Keep |
| `3_TTS-to-vibevoice.yml` | Another config | ✅ Good | Keep |
| `1_TTS-to-cosyvoice.yml` | CosyVoice config | ✅ Good | Keep |
| `metrics.yml` | Metric definitions | ✅ Good | Keep |
| `sweep_example.yaml` | Sweep config | ✅ Good | Keep |

### Tests (`tests/`)

| Dir | Purpose | Status | Action |
|-----|---------|--------|--------|
| `unit_tests/` | Unit tests | ✅ 165 passing | Keep |
| `integration_tests/` | Integration tests | ✅ Created | Keep |
| `conftest.py` | Test fixtures | ✅ Good | Keep |

### Documentation

| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `README.md` | Main docs | ⚠️ Stale | **Update PR1** |
| `AGENTS.md` | Dev guide | ✅ Good | Keep |
| `DEPLOYMENT_PR_STACK.md` | Deployment plan | ✅ Good | Keep |
| `SYSTEM_DESIGN.md` | This document | ✅ New | Keep |
| `benchmark*.md` | Planning docs | ✅ Good | Archive later |

### Docker/Deployment

| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `Dockerfile` | Container build | ⚠️ Dev-only | **Update PR3** |
| `docker-compose.yml` | Local compose | ⚠️ Dev-only | **Update PR3** |
| `.dockerignore` | Docker ignore | ❌ Missing | **Add PR3** |

### CI/CD

| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `.github/workflows/ci.yml` | CI pipeline | ❌ Missing | **Add PR5** |
| `.github/workflows/cd.yml` | CD pipeline | ❌ Missing | **Add PR6** |

### Infrastructure

| Dir | Purpose | Status | Action |
|-----|---------|--------|--------|
| `infra/terraform/` | IaC | ❌ Missing | **Add PR7** |

### Environment/Config

| File | Purpose | Status | Action |
|------|---------|--------|--------|
| `.env.example` | Env template | ❌ Missing | **Add PR4** |
| `src/config/settings.py` | Pydantic settings | ❌ Missing | **Add PR4** |

---

## Required Changes Summary

### Phase 1: Foundation (PR1-2)

**Files to Modify:**
1. `src/app.py` - Fix API contract
2. `README.md` - Update commands

**Files to Add:**
1. `src/app/models.py` - Request/response schemas
2. `src/app/dependencies.py` - Dependency injection
3. `tests/integration_tests/api/test_health.py`
4. `tests/integration_tests/api/test_s2s.py`

### Phase 2: Packaging (PR3-4)

**Files to Modify:**
1. `Dockerfile` - Production hardening
2. `docker-compose.yml` - Add dev profile

**Files to Add:**
1. `Dockerfile.prod`
2. `.dockerignore`
3. `docker-compose.dev.yml`
4. `docker-compose.prod.yml`
5. `src/config/settings.py`
6. `.env.example`
7. `.env` (gitignored)

### Phase 3: Automation (PR5-6)

**Files to Add:**
1. `.github/workflows/ci.yml`
2. `.github/workflows/cd.yml`
3. `.github/workflows/deploy-staging.yml` (PR7)

### Phase 4: Deployment (PR7-8)

**Files to Add:**
1. `infra/terraform/main.tf`
2. `infra/terraform/variables.tf`
3. `infra/terraform/outputs.tf`
4. `infra/terraform/ecs.tf`
5. `infra/terraform/networking.tf`
6. `docs/runbook.md`
7. `docs/slo.md`
8. `docs/deploy.md`

---

## Critical Path Analysis

### Blockers for Production Deployment

1. **API Contract (PR1)** - Without this, service won't start
2. **Docker Prod (PR3)** - Without this, no deployable artifact
3. **CI/CD (PR5-6)** - Without this, no automated deployment

### Can Be Deferred

1. **Full IaC (PR7)** - Can manually deploy first
2. **Observability (PR8)** - Can add monitoring later
3. **Benchmark API** - Separate concern from core service

---

## Effort Estimates

| Phase | Stories | Estimated Effort |
|-------|---------|------------------|
| Phase 1 | 2 PRs | 3-4 days |
| Phase 2 | 2 PRs | 2-3 days |
| Phase 3 | 2 PRs | 2-3 days |
| Phase 4 | 2 PRs | 4-5 days |
| **Total** | **8 PRs** | **11-15 days** |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU availability in cloud | High | Test with CPU fallback, document requirements |
| Model download at startup | Medium | Bake models into image or use EFS |
| Cold start time | Medium | Keep models loaded, use warm pools |
| Concurrent request handling | Medium | Queue-based architecture, horizontal scaling |
| Cost overrun | Medium | Set limits, use spot instances, auto-shutdown |
