# Product Requirements Document (PRD)

## MiniFlow Production Deployment

---

## 1. Overview

### 1.1 Product Description
MiniFlow is a speech-to-speech AI service that processes audio input, generates AI responses, and returns synthesized audio output. It supports multiple TTS backends (XTTS, VibeVoice) and provides benchmarking capabilities.

### 1.2 Target Users
- End users via API
- ML engineers for benchmarking
- DevOps for operations

### 1.3 Success Criteria
1. API responds to requests with <5s latency (p95)
2. 99.9% uptime for health endpoint
3. Zero manual steps for deployment
4. Rollback capability <5 minutes

---

## 2. Functional Requirements

### 2.1 Core API

#### FR1: Speech-to-Speech Endpoint
**ID:** FR-API-001
**Priority:** P0
**Description:** Process audio and return AI response audio

**Request:**
```
POST /s2s
Content-Type: multipart/form-data

audio: [audio_file.wav]
```

**Response (200 OK):**
```json
{
  "transcript": "What is machine learning?",
  "response": "Machine learning is a subset of AI...",
  "audio": "<base64_encoded_wav>",
  "metadata": {
    "processing_time_ms": 2345,
    "model_version": "v1.2.3"
  }
}
```

**Error Response (400/500):**
```json
{
  "error": "Audio too long",
  "code": "AUDIO_LENGTH_EXCEEDED",
  "max_duration_seconds": 30
}
```

**Acceptance Criteria:**
- [ ] Accepts WAV, MP3, FLAC formats
- [ ] Returns base64 encoded audio
- [ ] Handles errors gracefully
- [ ] Response time <5s for 10s audio

---

#### FR2: Health Check
**ID:** FR-API-002
**Priority:** P0
**Description:** Liveness probe for load balancers

**Request:**
```
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Acceptance Criteria:**
- [ ] Returns 200 if process is running
- [ ] No dependencies checked (fast response)
- [ ] Response time <100ms

---

#### FR3: Readiness Check
**ID:** FR-API-003
**Priority:** P0
**Description:** Readiness probe for Kubernetes

**Request:**
```
GET /ready
```

**Response (200 OK):**
```json
{
  "status": "ready",
  "checks": {
    "gpu": "available",
    "models_loaded": true,
    "storage": "connected"
  }
}
```

**Response (503 Not Ready):**
```json
{
  "status": "not_ready",
  "checks": {
    "gpu": "initializing",
    "models_loaded": false
  }
}
```

**Acceptance Criteria:**
- [ ] Returns 200 only when models loaded
- [ ] Returns 503 during startup
- [ ] Used by K8s for traffic routing

---

#### FR4: WebSocket Streaming (Future)
**ID:** FR-API-004
**Priority:** P2
**Description:** Real-time streaming interface

**Status:** Stubbed in current implementation, full implementation deferred

---

### 2.2 Benchmark System

#### FR5: Run Experiment
**ID:** FR-BENCH-001
**Priority:** P1
**Description:** Execute benchmark experiment via CLI

**Command:**
```bash
python -m src.scripts.run_experiment --config configs/baseline.yml
```

**Acceptance Criteria:**
- [ ] Loads config from file
- [ ] Runs specified number of samples
- [ ] Saves results to Benchmark/ directory
- [ ] Generates summary.json

---

#### FR6: Run Sweep
**ID:** FR-BENCH-002
**Priority:** P1
**Description:** Run parameter sweep across configs

**Command:**
```bash
python -m src.scripts.run_sweep --sweep configs/sweep_example.yaml
```

**Acceptance Criteria:**
- [ ] Runs multiple experiments
- [ ] Compares results
- [ ] Generates sweep summary

---

## 3. Non-Functional Requirements

### 3.1 Performance

#### NFR1: Latency
**ID:** NFR-PERF-001
**Requirement:**
- p50 latency < 2s
- p95 latency < 5s
- p99 latency < 10s

**Measurement:** Time from request received to response sent

---

#### NFR2: Throughput
**ID:** NFR-PERF-002
**Requirement:**
- Handle 10 concurrent requests
- Queue depth > 0 triggers scaling

---

#### NFR3: Resource Usage
**ID:** NFR-PERF-003
**Requirement:**
- GPU memory < 16GB
- CPU usage < 80% under load

---

### 3.2 Reliability

#### NFR4: Availability
**ID:** NFR-REL-001
**Requirement:**
- 99.9% uptime for /health
- 99.5% uptime for /s2s

---

#### NFR5: Error Handling
**ID:** NFR-REL-002
**Requirement:**
- All errors logged with context
- No unhandled exceptions crash service
- Graceful degradation for partial failures

---

### 3.3 Observability

#### NFR6: Logging
**ID:** NFR-OBS-001
**Requirement:**
- Structured JSON logs
- Request ID tracing
- Stage timing logs
- Log level configurable via env

**Log Format:**
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO",
  "request_id": "uuid",
  "message": "Processing complete",
  "stage": "tts",
  "duration_ms": 1234,
  "component": "sts_pipeline"
}
```

---

#### NFR7: Metrics
**ID:** NFR-OBS-002
**Requirement:**
- Prometheus-compatible /metrics endpoint
- Request latency histograms
- Stage duration breakdown
- Error rate counters
- GPU utilization gauges

---

#### NFR8: Alerting
**ID:** NFR-OBS-003
**Requirement:**
- Alert on error rate > 1%
- Alert on latency p95 > 10s
- Alert on GPU memory > 90%

---

### 3.4 Security

#### NFR9: Input Validation
**ID:** NFR-SEC-001
**Requirement:**
- Validate audio format
- Enforce max file size (50MB)
- Enforce max duration (30s)
- Sanitize all inputs

---

#### NFR10: Secrets Management
**ID:** NFR-SEC-002
**Requirement:**
- No secrets in code
- Use environment variables or secrets manager
- Rotate credentials regularly

---

### 3.5 Portability

#### NFR11: Configuration
**ID:** NFR-PORT-001
**Requirement:**
- Single command to run anywhere
- Environment-driven configuration
- No hardcoded paths
- No machine-specific assumptions

---

#### NFR12: Containerization
**ID:** NFR-PORT-002
**Requirement:**
- Dockerfile produces runnable image
- Image runs without bind mounts
- Image works on CPU and GPU hosts

---

## 4. Deployment Requirements

### 4.1 CI/CD Pipeline

#### DR1: Continuous Integration
**ID:** DR-CI-001
**Trigger:** Pull request, push to main
**Steps:**
1. Install dependencies
2. Run linting (ruff/mypy)
3. Run unit tests
4. Build Docker image
5. Run smoke tests

**Acceptance Criteria:**
- [ ] Pipeline fails on test failure
- [ ] Pipeline fails on lint error
- [ ] Pipeline completes <10 minutes

---

#### DR2: Continuous Deployment
**ID:** DR-CD-001
**Trigger:** Push to main, tag creation
**Steps:**
1. Build production image
2. Tag with git SHA
3. Push to registry
4. Deploy to staging
5. Run integration tests
6. (Manual) Promote to production

**Acceptance Criteria:**
- [ ] Every merge produces image
- [ ] Images are immutable
- [ ] Rollback to previous image possible

---

### 4.2 Infrastructure

#### DR3: Staging Environment
**ID:** DR-INFRA-001
**Requirements:**
- Mirror production configuration
- Smaller instance size (cost)
- Same deployment process

---

#### DR4: Production Environment
**ID:** DR-INFRA-002
**Requirements:**
- Auto-scaling based on load
- Health checks for traffic routing
- Log aggregation
- Metrics dashboard

---

## 5. Operational Requirements

### 5.1 Monitoring

#### OR1: Health Dashboard
**ID:** OR-MON-001
**Requirements:**
- Real-time request volume
- Latency percentiles
- Error rates
- GPU utilization
- Queue depth

---

#### OR2: Alerting Runbook
**ID:** OR-MON-002
**Requirements:**
- Document what each alert means
- Document severity levels
- Document response steps
- Include escalation path

---

### 5.2 Incident Response

#### OR3: Rollback Procedure
**ID:** OR-INC-001
**Requirements:**
- Document rollback steps
- Test rollback procedure
- Time to rollback <5 minutes
- Verify rollback success

---

#### OR4: Post-Mortem Template
**ID:** OR-INC-002
**Requirements:**
- Standard template for incidents
- Root cause analysis section
- Action items with owners
- Timeline of events

---

## 6. User Stories

### 6.1 End User

**US1:** As a user, I want to send audio and receive AI response audio so that I can have voice conversations with AI.

**US2:** As a user, I want the service to be available 99.9% of the time so that I can rely on it.

---

### 6.2 ML Engineer

**US3:** As an ML engineer, I want to run benchmarks so that I can compare model performance.

**US4:** As an ML engineer, I want to parameter sweep so that I can find optimal configurations.

---

### 6.3 DevOps

**US5:** As a DevOps engineer, I want automated deployment so that I can deploy without manual steps.

**US6:** As a DevOps engineer, I want health checks so that I can monitor service health.

**US7:** As a DevOps engineer, I want rollback capability so that I can recover from bad deployments.

---

## 7. Out of Scope

The following are explicitly NOT in scope for initial production deployment:

1. **Multi-tenant isolation** - Single tenant only
2. **Authentication/Authorization** - Open API initially
3. **Rate limiting** - Add later if needed
4. **Full WebSocket implementation** - Stub only
5. **Model versioning** - Single model version
6. **A/B testing** - Not needed initially
7. **Edge deployment** - Cloud only

---

## 8. Appendix

### 8.1 Glossary

- **S2S:** Speech-to-Speech
- **ASR:** Automatic Speech Recognition
- **LLM:** Large Language Model
- **TTS:** Text-to-Speech
- **p95:** 95th percentile
- **IaC:** Infrastructure as Code
- **CI/CD:** Continuous Integration/Deployment

### 8.2 References

- System Design: `SYSTEM_DESIGN.md`
- Component Inventory: `COMPONENT_INVENTORY.md`
- Deployment Plan: `DEPLOYMENT_PR_STACK.md`
