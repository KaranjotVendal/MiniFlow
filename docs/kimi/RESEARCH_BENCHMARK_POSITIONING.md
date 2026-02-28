# Industry Standards & SOTA for ML Production Observability

## The Core Problem Pattern

This is a well-known problem in production ML called the **"Online vs Offline Metrics Gap"** or **"Training-Serving Skew Monitoring"**.

### Industry Terminology
- **Online Metrics**: Production telemetry (latency, throughput, errors)
- **Offline Metrics**: Evaluation metrics (accuracy, WER, UTMOS, F1)
- **Shadow/Backtesting**: Running offline evaluation on production traffic
- **Golden Dataset**: Controlled test set for consistent evaluation

---

## Industry Standard Approaches

### 1. The Standard: "Log Everything, Evaluate Selectively"

**Pattern**: Log lightweight features + predictions in production, run full evaluation offline

```
Production Request:
  Input -> Model -> Prediction
    |       |         |
    |       |         +---> Log: prediction, timestamp, request_id
    |       |
    |       +---> Log: model_version, latency, feature_vector (optional)
    |
    +---> Log: input_hash, metadata

Offline (Hourly/Daily):
  Sample logged predictions -> Ground truth labeling -> Full metrics computation
```

**Why this is standard:**
- ✅ Minimal production overhead
- ✅ Can add new metrics retroactively
- ✅ Supports A/B testing
- ✅ Regulatory compliance (audit trail)

**Companies using this:**
- Netflix (recommendation systems)
- Uber (ETAs, pricing)
- Spotify (recommendations)
- Any company with serious ML production

---

### 2. SOTA: Feature Store + Model Registry Integration

**Modern Architecture** (2023-2024):

```
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE STORE                            │
│  (Tecton, Feast, SageMaker Feature Store)                  │
├─────────────────────────────────────────────────────────────┤
│ Online Store (Low Latency)   │ Offline Store (Analytics)   │
│ - Real-time features         │ - Historical features       │
│ - Point lookups              │ - Batch training            │
│ - <10ms p99                  │ - Parquet/Delta Lake        │
└────────────────────┬─────────┴──────────┬────────────────────┘
                     │                    │
         ┌───────────▼────────┐  ┌───────▼──────────┐
         │  Production Model  │  │  Offline Eval    │
         │  (FastAPI/TFX/...) │  │  (Batch scoring) │
         └───────────┬────────┘  └───────┬──────────┘
                     │                   │
                     ▼                   ▼
         ┌────────────────────┐  ┌──────────────────┐
         │  Experiment Tracker│  │  Model Registry  │
         │  (W&B, MLflow)     │  │  (MLflow, ...)   │
         └────────────────────┘  └──────────────────┘
```

**Key Innovation**: Feature stores unify online/offline data, eliminating training-serving skew

---

## Standard Tools (2024)

### Production Observability

| Category | Standard Tools | Purpose |
|----------|----------------|---------|
| **Metrics Collection** | Prometheus, StatsD, Datadog | Latency, throughput, error rates |
| **Logging** | ELK Stack, Splunk, CloudWatch | Structured logs, tracing |
| **ML Monitoring** | Evidently AI, Fiddler, WhyLabs, Arize | Drift detection, performance |
| **Distributed Tracing** | Jaeger, Zipkin, OpenTelemetry | Request flow tracing |

### ML Experiment Tracking

| Tool | Strength | Production Integration |
|------|----------|------------------------|
| **Weights & Biases (W&B)** | Best-in-class viz, model registry | Real-time production metrics |
| **MLflow** | Open source, wide adoption | Model serving + tracking |
| **Neptune** | Fast experimentation | CI/CD integration |
| **DVC** | Data versioning | Pipeline reproducibility |

### Feature Stores

| Tool | Open Source | Best For |
|------|-------------|----------|
| **Feast** | ✅ Yes | Startups, flexibility |
| **Tecton** | ❌ No | Enterprise, managed |
| **AWS SageMaker Feature Store** | ❌ Managed | AWS-native stacks |

---

## SOTA Patterns (2024)

### Pattern 1: Continuous Evaluation (Most Advanced)

```
Production Traffic:
  ┌─────────────────────────────────────┐
  │ Shadow Model Evaluation             │
  │ - Run candidate model on prod data  │
  │ - Don't serve predictions           │
  │ - Compare vs current model          │
  └─────────────────────────────────────┘

  ┌─────────────────────────────────────┐
  │ Automated Canary Analysis           │
  │ - Gradual traffic shift             │
  │ - Automated rollback on degradation │
  │ - Statistical significance testing  │
  └─────────────────────────────────────┘
```

**Tools**: Argo Rollouts, Flagger, custom ML platforms

---

### Pattern 2: Real-time Feature Monitoring

```python
# SOTA: Monitor feature distributions in real-time
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Compare production features vs training features
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(
    reference_data=training_features,
    current_data=production_features_sample
)

# Alert if drift detected
if drift_report.as_dict()['metrics'][0]['result']['dataset_drift']:
    pagerduty.trigger_alert("Feature drift detected")
```

---

### Pattern 3: LLM-specific Observability (Emerging)

For LLM/GenAI systems (like MiniFlow's LLM stage):

| Tool | Purpose |
|------|---------|
| **LangSmith** (LangChain) | LLM tracing, eval, monitoring |
| **Langfuse** | Open-source LLM observability |
| **Helicone** | LLM metrics, caching, rate limiting |
| **Weights & Biases Prompts** | Prompt versioning + evaluation |
| **Traceloop** | OpenTelemetry for LLMs |

**Standard for LLMs:**
- Log prompts + completions
- Track token usage + cost
- Evaluate with LLM-as-judge
- A/B test prompts

---

## What Should MiniFlow Use?

### Recommended Stack (Industry-Aligned)

```
Production (Live API):
├── Prometheus Client (metrics export)
├── OpenTelemetry (distributed tracing)
├── Structured Logging (JSON)
└── Lightweight timing only

Offline (Benchmark):
├── MLflow or W&B (experiment tracking)
├── Evidently AI (drift/quality detection)
└── Custom BenchmarkCollector (keep current)

Integration:
├── /metrics endpoint (Prometheus format)
├── MLflow logging for benchmark runs
└── Grafana dashboard (optional, can defer)
```

### Simplified for Demo (Still Credible)

```
Production:
├── Prometheus /metrics endpoint
├── JSON structured logs
└── Simple timing decorator

Offline:
├── Keep current BenchmarkCollector
├── Export summary.json to MLflow
└── Version benchmark results with git SHA
```

---

## Key Takeaways

### ✅ Industry Standard (2024):
1. **Separate concerns**: Production telemetry ≠ offline evaluation
2. **Log predictions**: Store inputs/outputs for offline analysis
3. **Use standard tools**: Prometheus, structured logging
4. **Feature stores**: Unify online/offline if you have feature complexity

### 🚀 SOTA (Cutting Edge):
1. **Continuous evaluation**: Shadow models, automated canaries
2. **Real-time drift detection**: Catch issues before users complain
3. **LLM-specific observability**: Prompt tracking, LLM-as-judge
4. **Unified platforms**: Weights & Biases, MLflow bridging dev/prod

### 📊 For MiniFlow Specifically:

**Your Hybrid approach IS the industry standard.**

The only question is tooling sophistication:

| Approach | Tools | Effort | Interview Value |
|----------|-------|--------|-----------------|
| **Basic** | Custom timing + logs | Low | Shows understanding |
| **Standard** | Prometheus + MLflow | Medium | Industry-aligned |
| **Advanced** | W&B + Evidently + Feature Store | High | Cutting-edge |

**Recommendation**: Go with "Standard" - Prometheus + keep your BenchmarkCollector.
- Shows you know industry tools
- Not over-engineered for a demo
- Easy to explain in interviews

---

## References

1. **"ML Observability: A Survey"** - Chip Huyen, 2022
2. **"Designing ML Systems"** - Chip Huyen (Book, O'Reilly)
3. **Google's ML Testing Paper** - Sculley et al.
4. **Netflix Tech Blog** - "How Netflix Scales ML Infrastructure"
5. **Uber's Michelangelo Platform** - Production ML at scale
6. **Evidently AI State of ML Monitoring** - 2023 Report

---

## Bottom Line

**Your hybrid design is correct.**

The industry has converged on:
- ✅ Lightweight production telemetry (Prometheus/structured logs)
- ✅ Comprehensive offline evaluation (MLflow/W&B/BenchmarkCollector)
- ✅ Don't mix heavy evaluation into user-facing paths

You don't need a feature store or continuous evaluation for a credible demo.
Just show you understand the separation of concerns and use standard tools.

---

## Grafana in the Observability Stack

### What is Grafana?

**Grafana is a visualization and dashboarding layer**, not a data collection tool.

Think of it as:
- **Prometheus** = Collects and stores metrics (the database)
- **Grafana** = Visualizes metrics in dashboards (the UI)

```
┌─────────────────────────────────────────────────────────────┐
│                     GRAFANA LAYER                           │
│  - Dashboards                                               │
│  - Alerts (can alert from any data source)                  │
│  - Visualization (graphs, heatmaps, tables)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ Queries
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA SOURCES                              │
├─────────────────────────────────────────────────────────────┤
│ Prometheus  │  Logs (Loki)  │  Traces (Tempo)  │  MLflow   │
│ (metrics)   │  (logs)       │  (traces)        │  (experiments)
└─────────────────────────────────────────────────────────────┘
```

---

## The Standard Stack: Prometheus + Grafana

This is the industry-standard "Observability Stack" (often called PLG stack):

| Component | Purpose | Open Source |
|-----------|---------|-------------|
| **Prometheus** | Metrics collection & storage | ✅ Yes |
| **Grafana** | Visualization & dashboards | ✅ Yes |
| **Loki** | Log aggregation (like Prometheus for logs) | ✅ Yes |
| **Tempo** | Distributed tracing | ✅ Yes |

**Together they form the "PLG Stack" (Prometheus-Loki-Grafana)**

---

## Where Grafana Fits for MiniFlow

### Option 1: Minimal (No Grafana)

```
Production:
├── Prometheus /metrics endpoint  ◄── Your app exposes this
└── AWS CloudWatch (or similar)   ◄── Managed metrics UI

Result:
- ✅ Metrics collected
- ✅ Can view in CloudWatch
- ❌ No custom dashboards
- ❌ Vendor lock-in to AWS
```

**Use if:** You want simplest path, already on AWS/GCP/Azure with their monitoring.

---

### Option 2: Standard (Prometheus + Grafana) ⭐ Recommended

```
Production:
├── Prometheus /metrics endpoint  ◄── Your app
├── Prometheus Server             ◄── Scrapes metrics
└── Grafana                       ◄── Dashboards

Result:
- ✅ Custom dashboards
- ✅ ML-specific visualizations
- ✅ Alerting
- ✅ Open source, portable
```

**Dashboard Example:**
```json
{
  "title": "MiniFlow S2S Pipeline",
  "panels": [
    {
      "title": "Request Latency (p50/p95/p99)",
      "type": "graph",
      "targets": [
        {"expr": "histogram_quantile(0.50, rate(request_latency_bucket[5m]))"},
        {"expr": "histogram_quantile(0.95, rate(request_latency_bucket[5m]))"},
        {"expr": "histogram_quantile(0.99, rate(request_latency_bucket[5m]))"}
      ]
    },
    {
      "title": "Stage Breakdown",
      "type": "graph",
      "targets": [
        {"expr": "rate(asr_latency_sum[5m]) / rate(asr_latency_count[5m])"},
        {"expr": "rate(llm_latency_sum[5m]) / rate(llm_latency_count[5m])"},
        {"expr": "rate(tts_latency_sum[5m]) / rate(tts_latency_count[5m])"}
      ]
    },
    {
      "title": "GPU Utilization",
      "type": "gauge",
      "targets": [
        {"expr": "nvidia_gpu_utilization_gpu"}
      ]
    },
    {
      "title": "Error Rate",
      "type": "stat",
      "targets": [
        {"expr": "rate(request_errors_total[5m])"}
      ]
    }
  ]
}
```

**Use if:** You want nice dashboards for interview demo, open source stack.

---

### Option 3: Full Observability Stack (PLG)

```
Production:
├── App metrics → Prometheus
├── App logs    → Loki
├── App traces  → Tempo
└── Grafana dashboards (metrics + logs + traces)

Result:
- ✅ Correlation: Click on slow request → see logs → see trace
- ✅ "Single pane of glass"
- ⚠️ More complex setup
```

**Use if:** You want to show advanced observability skills.

---

## Grafana for MiniFlow: Recommended Setup

### Phase 1 (Baseline R1): Metrics Only

```yaml
# docker-compose.yml
services:
  app:
    build: .
    expose:
      - "8000"
    environment:
      - METRICS_ENABLED=true

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - ./grafana/provisioning:/etc/grafana/provisioning
```

**Grafana Dashboards to Create:**
1. **S2S Overview**: Request rate, latency, error rate
2. **Pipeline Stages**: ASR vs LLM vs TTS timing breakdown
3. **GPU Monitoring**: Utilization, memory, temperature
4. **Business Metrics**: Requests per model version

---

### Phase 2 (Improvement R2): Add Logs Integration

```yaml
# Add Loki for log aggregation
services:
  loki:
    image: grafana/loki
    ports:
      - "3100:3100"

  promtail:  # Log scraper
    image: grafana/promtail
    volumes:
      - /var/log/app:/var/log/app:ro
```

**Benefit:** Search logs in Grafana: `{app="miniflow"} |= "error"`

---

## Grafana vs CloudWatch vs Datadog

| Tool | Type | Cost | Best For |
|------|------|------|----------|
| **Grafana** | Visualization | Free (open source) | Custom dashboards, interviews |
| **Prometheus** | Metrics DB | Free | Time-series metrics |
| **CloudWatch** | AWS Managed | Pay per metric | AWS-native, simplest setup |
| **Datadog** | Commercial | $$$ | Enterprise, all-in-one |
| **Grafana Cloud** | Managed | Free tier + paid | Hosted Grafana |

**For MiniFlow Demo:**
- ✅ **Grafana + Prometheus** = Most impressive for interviews (open source, visual)
- ✅ **CloudWatch** = Simplest if staying AWS-only
- ❌ **Datadog** = Overkill and expensive for demo

---

## Interview Value of Grafana

### Without Grafana:
> "We export metrics to CloudWatch"
- Okay, but standard/AWS-specific

### With Grafana:
> "We use Prometheus for metrics collection and Grafana for visualization. Here's a dashboard showing p50/p95/p99 latency breakdown by pipeline stage, plus GPU utilization correlation with request volume."
- Shows:
  - Industry standard tools (Prometheus/Grafana)
  - Understanding of percentiles
  - Operational mindset (correlating metrics)
  - Can create custom dashboards
  - Open source stack (not vendor-locked)

**Stronger signal to hiring managers.**

---

## Implementation Effort

| Component | Effort | Value |
|-----------|--------|-------|
| Add Prometheus client to app | 30 min | High |
| Create /metrics endpoint | 1 hour | High |
| Setup Prometheus server | 30 min | Medium |
| Setup Grafana | 30 min | Medium |
| Create custom dashboards | 2-3 hours | **Very High** |
| **Total** | **~1 day** | **Interview gold** |

---

## Recommendation for MiniFlow

### For Interview Demo: Include Grafana ✅

**Why:**
1. Visual dashboards are impressive in screen shares
2. Shows you know the standard stack (Prometheus + Grafana)
3. Easy to explain latency breakdowns
4. Open source = shows engineering values

**What to build:**
```
1. Add prometheus-client to requirements
2. Create /metrics endpoint in FastAPI
3. Track: request_latency, stage_latency, error_count, gpu_utilization
4. Add Grafana to docker-compose
5. Create 2-3 dashboard JSON files
6. Screenshot dashboards for README/docs
```

**Result:**
- Live dashboards you can show
- Screenshots for evidence pack
- "We instrumented our service with Prometheus metrics and built Grafana dashboards for observability"

---

## Summary

| Question | Answer |
|----------|--------|
| What is Grafana? | Dashboard/visualization tool |
| Do we need it? | No, but **highly recommended** for demo |
| Effort? | ~1 day |
| Interview value? | **High** - visual, industry standard |
| Alternatives? | CloudWatch (simpler), Datadog (expensive) |
| Standard stack? | Prometheus (metrics) + Grafana (viz) |

**Bottom line:** Add Grafana for the demo. It costs 1 day of work but makes the observability story much stronger in interviews.

---

## Can Custom Benchmark Pipe to Grafana?

**Short answer:** Yes, but with important distinctions.

### The Challenge

Your BenchmarkCollector produces **batch/offline data**:
```json
// summary.json (batch result, post-experiment)
{
  "meta": {
    "experiment": "baseline",
    "num_trials": 20,
    "timestamp": "2024-01-15T10:00:00Z"
  },
  "pipeline": {
    "latency": {
      "trial_wall_time_seconds": {
        "mean": 5.23,
        "p95": 8.45,
        "p99": 12.1
      }
    }
  }
}
```

Grafana excels at **time-series metrics** (continuous, real-time):
```
request_latency{quantile="0.95"} 8.45 1705312800
request_latency{quantile="0.99"} 12.1 1705312800
```

**Different data models.** But you can bridge them.

---

## Option 1: Export Benchmark Results to Prometheus (Recommended)

Convert your summary.json to Prometheus metrics format:

```python
# src/benchmark/grafana_exporter.py
from prometheus_client import Gauge, push_to_gateway

# Define gauges for benchmark results
BENCHMARK_LATENCY_MEAN = Gauge(
    'benchmark_latency_mean_seconds',
    'Mean latency from benchmark run',
    ['experiment', 'stage']
)

BENCHMARK_LATENCY_P95 = Gauge(
    'benchmark_latency_p95_seconds',
    'P95 latency from benchmark run',
    ['experiment', 'stage']
)

def export_summary_to_prometheus(summary_dict: dict, experiment_name: str):
    """Export benchmark summary as Prometheus metrics."""

    # Export pipeline-level latency
    pipeline_latency = summary_dict['pipeline']['latency']['trial_wall_time_seconds']
    BENCHMARK_LATENCY_MEAN.labels(
        experiment=experiment_name,
        stage='pipeline'
    ).set(pipeline_latency['mean'])

    BENCHMARK_LATENCY_P95.labels(
        experiment=experiment_name,
        stage='pipeline'
    ).set(pipeline_latency['p95'])

    # Export stage-level latencies
    for stage in ['asr', 'llm', 'tts']:
        stage_latency = summary_dict[stage]['latency']['inference_seconds']
        BENCHMARK_LATENCY_MEAN.labels(
            experiment=experiment_name,
            stage=stage
        ).set(stage_latency['mean'])

        BENCHMARK_LATENCY_P95.labels(
            experiment=experiment_name,
            stage=stage
        ).set(stage_latency['p95'])

# Usage in ExperimentRunner
class ExperimentRunner:
    def _generate_summary(self) -> SummaryRecord:
        # ... existing code ...

        # Export to Prometheus for Grafana
        export_summary_to_prometheus(summary_dict, self.experiment)

        return summary_record
```

**Result:**
```
benchmark_latency_mean_seconds{experiment="baseline",stage="pipeline"} 5.23
benchmark_latency_mean_seconds{experiment="baseline",stage="asr"} 1.2
benchmark_latency_mean_seconds{experiment="baseline",stage="llm"} 2.8
benchmark_latency_mean_seconds{experiment="baseline",stage="tts"} 1.1
```

**Grafana Dashboard:**
- Show benchmark results alongside production metrics
- Compare: "Baseline mean latency: 5.23s" vs "Production p95: 8.45s"

---

## Option 2: Grafana Annotations (Benchmark Markers)

Mark when benchmarks ran on production metrics timeline:

```python
# Create annotation when benchmark completes
import requests

def create_grafana_annotation(experiment_name: str, results: dict):
    """Mark benchmark run on Grafana dashboards."""

    annotation = {
        "dashboardId": 1,  # Your dashboard ID
        "panelId": 0,      # 0 = all panels
        "time": int(time.time() * 1000),  # Unix millis
        "tags": ["benchmark", experiment_name],
        "text": f"Benchmark completed: {experiment_name}\n"
                f"Mean latency: {results['latency_mean']:.2f}s\n"
                f"P95: {results['latency_p95']:.2f}s"
    }

    requests.post(
        'http://grafana:3000/api/annotations',
        json=annotation,
        headers={'Authorization': 'Bearer YOUR_API_KEY'}
    )
```

**Grafana View:**
```
Request Latency Over Time
│
│     🏷️ Benchmark: baseline
│     Mean: 5.23s, P95: 8.45s
│         │
│     ╱╲  │
│    ╱  ╲ │
│ ──╱────╲│──────
└──────────────────
  10am  11am  12pm
```

**Value:** See correlation between benchmark runs and production performance.

---

## Option 3: Grafana JSON Data Source (Static Dashboards)

For one-off benchmark result visualization (not time-series):

```python
# Export summary as Grafana dashboard JSON
import json

def create_benchmark_dashboard(summary_dict: dict, experiment_name: str) -> dict:
    """Generate Grafana dashboard JSON from benchmark results."""

    dashboard = {
        "title": f"Benchmark Results: {experiment_name}",
        "panels": [
            {
                "title": "Latency Summary",
                "type": "stat",
                "targets": [
                    {
                        "rawSql": f"""
                        SELECT
                            {summary_dict['pipeline']['latency']['trial_wall_time_seconds']['mean']} as mean,
                            {summary_dict['pipeline']['latency']['trial_wall_time_seconds']['p95']} as p95,
                            {summary_dict['pipeline']['latency']['trial_wall_time_seconds']['p99']} as p99
                        """,
                        "format": "table"
                    }
                ]
            },
            {
                "title": "Stage Breakdown",
                "type": "bargauge",
                "targets": [
                    {
                        "rawSql": f"""
                        SELECT 'ASR' as stage, {summary_dict['asr']['latency']['inference_seconds']['mean']} as latency
                        UNION ALL
                        SELECT 'LLM', {summary_dict['llm']['latency']['inference_seconds']['mean']}
                        UNION ALL
                        SELECT 'TTS', {summary_dict['tts']['latency']['inference_seconds']['mean']}
                        """
                    }
                ]
            }
        ]
    }

    with open(f"grafana/dashboards/benchmark_{experiment_name}.json", 'w') as f:
        json.dump(dashboard, f, indent=2)
```

**Use Case:** Generate static dashboard for each benchmark run.

---

## Option 4: Push Gateway (One-off Metric Push)

For CI/CD pipeline integration:

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def push_benchmark_metrics(summary_dict: dict, experiment_name: str):
    """Push benchmark results to Prometheus Push Gateway."""

    registry = CollectorRegistry()

    latency_gauge = Gauge(
        'benchmark_latency_seconds',
        'Benchmark latency',
        ['experiment', 'stage', 'metric'],
        registry=registry
    )

    # Push all metrics
    for stage in ['pipeline', 'asr', 'llm', 'tts']:
        stage_data = summary_dict[stage]['latency']['inference_seconds']
        for metric in ['mean', 'p95', 'p99']:
            latency_gauge.labels(
                experiment=experiment_name,
                stage=stage,
                metric=metric
            ).set(stage_data[metric])

    # Push to gateway (Prometheus scrapes from here)
    push_to_gateway(
        'pushgateway:9091',
        job='miniflow_benchmark',
        registry=registry
    )
```

**Use Case:** CI/CD runs benchmark, pushes metrics, Grafana shows result immediately.

---

## Recommended Approach for MiniFlow

### Hybrid: Production Metrics + Benchmark Export

```
Production Runtime:
├── Real-time metrics → Prometheus → Grafana (live dashboards)
└── Continuous monitoring

Benchmark (CI/Staging):
├── Run ExperimentRunner
├── summary.json generated
├── Export to Prometheus (Option 1)
└── Create annotation (Option 2)

Grafana:
├── Dashboard: "Production Health" (live metrics)
├── Dashboard: "Benchmark Results" (exported summaries)
└── Timeline: Benchmark annotations on production metrics
```

### Implementation

**Add to ExperimentRunner:**

```python
class ExperimentRunner:
    def run(self) -> ExperimentSummary:
        # ... existing run logic ...

        summary = self._generate_summary()

        # NEW: Export to Grafana/Prometheus
        if self.config.get('export_metrics', False):
            export_summary_to_prometheus(
                summary.summary.to_dict(),
                self.experiment
            )
            create_grafana_annotation(
                self.experiment,
                summary.summary.to_dict()
            )

        return summary
```

**Grafana Dashboard Panels:**

1. **Live Production Metrics** (from /metrics endpoint)
   - Request rate
   - Latency p95/p99
   - Error rate

2. **Benchmark Comparison** (from exported summary)
   - Benchmark mean vs Production mean
   - Latency delta: (production - benchmark)
   - Historical benchmark trend

3. **Benchmark Timeline** (annotations)
   - When benchmarks ran
   - Results at that point in time

---

## Interview Value

### Without Benchmark-Grafana Integration:
> "We have benchmarks and we have production monitoring."
- Okay, but disconnected.

### With Integration:
> "We export benchmark results as Prometheus metrics so we can compare baseline performance against production in the same Grafana dashboard. We also annotate when benchmarks run to correlate with production events."
- Shows:
  - System thinking (connecting dev/prod metrics)
  - Tool proficiency (Prometheus ecosystem)
  - Operational maturity (comparing benchmarks to reality)

**Stronger signal.**

---

## Summary

| Approach | How | Use Case |
|----------|-----|----------|
| **Export as Prometheus metrics** | Convert summary.json to gauges | Compare benchmark vs production |
| **Grafana annotations** | API call to mark timestamp | See benchmark timing on timeline |
| **JSON dashboards** | Generate dashboard JSON | Static benchmark report |
| **Push Gateway** | Push to Prometheus | CI/CD integration |

**Recommended:** Use Option 1 (Prometheus export) + Option 2 (annotations) together.

**Result:** Your benchmark framework feeds into the same observability stack as production, showing you understand unified observability.

---

## OpenTelemetry: The Modern Observability Standard

### What is OpenTelemetry?

**OpenTelemetry (OTel)** is a **vendor-neutral observability framework** that unifies:
- **Traces** (distributed request flow)
- **Metrics** (numbers over time)
- **Logs** (structured events)

Think of it as the **"USB-C of observability"** - one standard, works with everything.

```
Before OpenTelemetry (Fragmented):
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Prometheus  │  │   Jaeger    │  │   Zipkin    │
│  (metrics)  │  │  (tracing)  │  │  (tracing)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
              Different APIs/SDKs

After OpenTelemetry (Unified):
┌─────────────────────────────────────────┐
│         OpenTelemetry SDK               │  ◄── One API in your code
│  (traces + metrics + logs)              │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┼───────────┐
       │           │           │
┌──────▼────┐ ┌────▼────┐ ┌────▼────┐
│Prometheus │ │ Grafana │ │  Datadog│  ◄── Export to any backend
│  (metrics)│ │ (traces)│ │ (cloud) │
└───────────┘ └─────────┘ └─────────┘
```

---

## OpenTelemetry vs Prometheus/Grafana

### Relationship (Not Replacement)

| Tool | Role | OpenTelemetry? |
|------|------|----------------|
| **OpenTelemetry** | Instrumentation SDK | ✅ Yes, the standard |
| **Prometheus** | Metrics database | ✅ Works with OTel |
| **Grafana** | Visualization | ✅ Works with OTel |
| **Jaeger** | Tracing backend | ✅ Works with OTel |
| **Datadog** | Commercial APM | ✅ Works with OTel |

**Key insight:** OpenTelemetry is **how you collect data**, not **where you store it**.

```python
# With OpenTelemetry
from opentelemetry import metrics, trace

# Create instruments (OTel terminology)
request_counter = metrics.get_meter(__name__).create_counter(
    "requests_total",
    description="Total requests"
)

latency_histogram = metrics.get_meter(__name__).create_histogram(
    "request_latency_seconds",
    description="Request latency"
)

# In your endpoint
@app.post("/s2s")
async def s2s_endpoint(audio):
    request_counter.add(1)

    with latency_histogram.record_time():
        result = await process_pipeline(audio)

    return result

# OTel exports to Prometheus (or any backend)
```

**Prometheus sees:**
```
requests_total 1
request_latency_seconds_bucket{le="0.5"} 0
request_latency_seconds_bucket{le="1.0"} 1
```

**Same metrics, modern collection method.**

---

## OpenTelemetry Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR APPLICATION                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         OpenTelemetry SDK                           │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │   │
│  │  │   Traces    │ │   Metrics   │ │    Logs     │  │   │
│  │  │   (spans)   │ │  (numbers)  │ │  (events)   │  │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘  │   │
│  │         │               │               │         │   │
│  │         └───────────────┼───────────────┘         │   │
│  │                         │                         │   │
│  │              ┌──────────▼──────────┐              │   │
│  │              │    OTLP Export      │              │   │
│  │              │  (OpenTelemetry     │              │   │
│  │              │   Protocol)         │              │   │
│  │              └──────────┬──────────┘              │   │
│  └─────────────────────────┼─────────────────────────┘   │
│                            │                              │
└────────────────────────────┼──────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼─────┐ ┌──────▼──────┐ ┌────▼─────┐
     │   Collector  │ │ Prometheus  │ │  Jaeger  │
     │  (optional)  │ │  (metrics)  │ │ (traces) │
     └──────────────┘ └─────────────┘ └──────────┘
```

**OTLP (OpenTelemetry Protocol):** Standard wire format for exporting telemetry.

---

## OpenTelemetry for MiniFlow

### Current State (Without OTel)

```python
# Custom metrics (what you have now)
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
latency = Histogram('request_latency_seconds', 'Request latency')

@app.post("/s2s")
async def s2s(audio):
    request_count.inc()
    with latency.time():
        return await process(audio)
```

**Pros:**
- ✅ Simple
- ✅ Direct Prometheus integration

**Cons:**
- ❌ Vendor-locked to Prometheus
- ❌ No tracing
- ❌ Manual correlation between metrics/logs

---

### With OpenTelemetry (Modern Approach)

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

# Setup once at startup
trace.set_tracer_provider(TracerProvider())
metrics.set_meter_provider(MeterProvider(
    metric_readers=[PrometheusMetricReader()]  # Export to Prometheus
))

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create instruments
request_counter = meter.create_counter(
    "miniflow.requests",
    description="Total S2S requests"
)

latency_histogram = meter.create_histogram(
    "miniflow.latency",
    description="Pipeline latency",
    unit="s"
)

@app.post("/s2s")
async def s2s(audio):
    # Start a trace span (distributed tracing)
    with tracer.start_as_current_span("s2s_request") as span:
        span.set_attribute("audio.duration", audio.duration)
        span.set_attribute("model.asr", "whisper-small")

        # Record metric
        request_counter.add(1, attributes={"status": "success"})

        # Time the pipeline with automatic span creation
        with tracer.start_span("pipeline_execution") as pipeline_span:
            start = time.time()

            with tracer.start_span("asr_stage"):
                transcript = await run_asr(audio)
                pipeline_span.set_attribute("transcript", transcript)

            with tracer.start_span("llm_stage"):
                response = await run_llm(transcript)
                pipeline_span.set_attribute("response", response[:100])

            with tracer.start_span("tts_stage"):
                audio_out = await run_tts(response)

            latency = time.time() - start
            latency_histogram.record(latency)
            pipeline_span.set_attribute("total_latency", latency)

        return audio_out
```

**What you get:**

```
Trace View (Jaeger/Grafana):
┌─────────────────────────────────────────────────────┐
│ s2s_request (5.23s)                                  │
│ ├── asr_stage (1.2s)                                 │
│ ├── llm_stage (2.8s)  ◄── Bottleneck identified!    │
│ └── tts_stage (1.1s)                                 │
└─────────────────────────────────────────────────────┘

Metrics (Prometheus/Grafana):
miniflow_latency_count{status="success"} 100
miniflow_latency_bucket{le="5.0"} 95
miniflow_latency_bucket{le="10.0"} 100

Correlation:
Trace ID: abc123 ──▶ Logs: "Processing audio"
                    ──▶ Metrics: latency=5.23s
                    ──▶ Benchmark: mean=5.1s
```

---

## Should MiniFlow Use OpenTelemetry?

### Option A: Keep Prometheus Client (Simpler)

**Use if:**
- Want simplest implementation
- Only need metrics (no tracing)
- Not demonstrating distributed systems

```python
from prometheus_client import Counter, Histogram
# Direct, simple, works
```

### Option B: OpenTelemetry (Modern Standard) ⭐ Recommended for Demo

**Use if:**
- Want to show modern practices
- Can demonstrate tracing (impressive in interviews)
- Future-proof the observability

```python
from opentelemetry import trace, metrics
# Industry standard, vendor-neutral
```

---

## OpenTelemetry vs Your Benchmark Collector

### Benchmark Collector (Current)

```python
class BenchmarkCollector:
    def start_trial(self, trial_id):
        self._trial_start = time.time()

    def end_trial(self):
        latency = time.time() - self._trial_start
        self.metrics.record("latency", latency)
```

**Purpose:** Offline experiment tracking with full ML metrics (WER, UTMOS)

### OpenTelemetry (Additional)

```python
with tracer.start_as_current_span("benchmark_trial"):
    result = run_pipeline()
    # Automatic trace + timing
```

**Purpose:** Online production observability with distributed tracing

### Relationship

```
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY STACK                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Production Runtime:                                        │
│  ├── OpenTelemetry ──▶ Traces + Metrics + Logs              │
│  │   └── Real-time pipeline observability                   │
│  │                                                          │
│  └── /metrics endpoint ──▶ Prometheus ──▶ Grafana           │
│      └── Live dashboards                                    │
│                                                             │
│  Offline Evaluation:                                        │
│  ├── BenchmarkCollector ──▶ summary.json                    │
│  │   └── Comprehensive ML metrics (WER, UTMOS)              │
│  │                                                          │
│  └── Export ──▶ Prometheus ──▶ Grafana                      │
│      └── Benchmark vs Production comparison                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**They complement each other:**
- OTel: Production runtime observability
- BenchmarkCollector: Offline ML evaluation rigor

---

## Industry Context (2024)

### Who's Adopting OpenTelemetry?

| Company | Status | Use Case |
|---------|--------|----------|
| **Google** | ✅ Core contributor | Cloud Trace, Monitoring |
| **Microsoft** | ✅ Core contributor | Azure Monitor, Application Insights |
| **AWS** | ✅ Adopted | AWS Distro for OpenTelemetry |
| **Meta** | ✅ Using | Internal observability |
| **Netflix** | ✅ Using | Distributed tracing at scale |
| **Uber** | ✅ Using | Microservices observability |

**Trend:** OpenTelemetry is becoming the **de facto standard**.

### Legacy Tools Being Replaced

| Legacy | Modern Replacement |
|--------|-------------------|
| Prometheus client (direct) | OpenTelemetry SDK |
| Jaeger client | OpenTelemetry tracing |
| StatsD | OpenTelemetry metrics |
| Custom logging | OpenTelemetry logs |

---

## Recommendation for MiniFlow

### Tier 1: Prometheus Only (If Keeping It Simple)

```python
# Just use prometheus_client
# Shows you understand metrics
# Simpler to explain in interviews
```

### Tier 2: OpenTelemetry + Prometheus (Recommended) ⭐

```python
# Use OpenTelemetry SDK
# Export to Prometheus backend
# Adds tracing capability
# Shows modern practices
```

### Tier 3: Full OpenTelemetry Stack

```python
# OpenTelemetry SDK
# OTLP to Collector
# Export to multiple backends (Prometheus, Jaeger)
# Full traces + metrics + logs correlation
```

**For Interview Demo: Tier 2 is sweet spot**
- Shows you know modern standards
- Can demonstrate tracing (impressive)
- Still integrates with Grafana
- Not over-engineered

---

## Implementation Path

### Option 1: Start with Prometheus, Add OTel Later

```python
# Phase 1 (PR8): Prometheus only
from prometheus_client import Counter

# Phase 2 (Future): Migrate to OTel
from opentelemetry import metrics
```

### Option 2: Start with OpenTelemetry Day 1

```python
# PR8: OpenTelemetry from start
from opentelemetry import trace, metrics
```

**Recommendation:** Start with Prometheus, mention OpenTelemetry as future improvement.

**Why:**
- Easier to get working quickly
- Can say "We started with Prometheus, planning to migrate to OpenTelemetry for distributed tracing"
- Shows awareness of industry trends without over-engineering

---

## Summary

| Question | Answer |
|----------|--------|
| What is OpenTelemetry? | Vendor-neutral observability SDK (traces + metrics + logs) |
| Does it replace Prometheus? | No, works with it |
| Does it replace Grafana? | No, feeds data to it |
| Should MiniFlow use it? | Optional but impressive |
| Effort? | Similar to Prometheus |
| Interview value? | **High** - shows modern practices |
| When to use? | If you want tracing or vendor flexibility |

**Bottom Line:**
- **Prometheus only** = Industry standard, totally acceptable
- **OpenTelemetry** = Modern standard, slight edge in interviews
- **Both together** = Best of both worlds (future-proof + practical)

For MiniFlow: **Prometheus is sufficient, OpenTelemetry is a bonus.**

If you want the extra interview points and have time, add OpenTelemetry. If not, Prometheus alone is still a strong signal.

---

## MiniFlow Observability Strategy: Production-Grade Design

### Executive Summary

Based on industry SOTA and production ML patterns, here is the comprehensive observability strategy for MiniFlow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PRINCIPLE: OpenTelemetry-first, vendor-neutral, SLO-driven        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LIVE PLANE (Production Runtime)                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ OpenTelemetry SDK                                           │   │
│  │ ├── Traces (distributed request flow)                      │   │
│  │ ├── Metrics (latency, throughput, errors)                  │   │
│  │ └── Logs (structured, contextual)                          │   │
│  │                                                             │   │
│  │ Exporters:                                                  │   │
│  │ ├── Prometheus (metrics storage)                           │   │
│  │ ├── Jaeger (trace visualization)                           │   │
│  │ └── Loki (log aggregation)                                 │   │
│  │                                                             │   │
│  │ Dashboards: Grafana                                        │   │
│  │ - S2S Overview (rate, latency, errors)                     │   │
│  │ - Pipeline Breakdown (ASR/LLM/TTS stages)                  │   │
│  │ - GPU Monitoring (utilization, memory, temp)               │   │
│  │ - Release Comparison (pre/post deployment)                 │   │
│  │                                                             │   │
│  │ SLOs:                                                       │   │
│  │ - Availability: 99.9%                                      │   │
│  │ - Latency p95: < 5s                                        │   │
│  │ - Error rate: < 0.1%                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              │ Release ID correlation               │
│                              ▼                                      │
│  EVAL PLANE (Offline Benchmark)                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ BenchmarkCollector                                          │   │
│  │ ├── WER (Word Error Rate)                                  │   │
│  │ ├── UTMOS (audio quality)                                  │   │
│  │ ├── Detailed latency profiling                             │   │
│  │ └── Hardware utilization                                   │   │
│  │                                                             │   │
│  │ Export to Prometheus:                                       │   │
│  │ ├── benchmark_latency_mean{experiment, stage}              │   │
│  │ ├── benchmark_latency_p95{experiment, stage}               │   │
│  │ └── benchmark_wer{experiment}                              │   │
│  │                                                             │   │
│  │ Grafana Integration:                                        │   │
│  │ ├── Benchmark vs Production comparison                     │   │
│  │ ├── Historical trend analysis                              │   │
│  │ └── Release annotations on timeline                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  LINKAGE: Every deployment tagged with                             │
│  - release_id (git SHA)                                            │
│  - model_version                                                   │
│  - config_name                                                     │
│  - timestamp                                                       │
│                                                                     │
│  Result: "Release v1.2.3-abc123 had benchmark WER 0.15 and         │
│          production p95 latency of 5.2s"                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Six Pillars

### 1. OpenTelemetry-First Instrumentation

**Why:** One standard for metrics, traces, and logs. Vendor-neutral and portable.

**Implementation:**
```python
from opentelemetry import trace, metrics

# Single SDK for all telemetry
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create instruments
request_counter = meter.create_counter("miniflow.requests")
latency_histogram = meter.create_histogram("miniflow.latency")

# Automatic context propagation
with tracer.start_as_current_span("s2s_request") as span:
    span.set_attribute("model.version", "qwen-2.5-3b")
    request_counter.add(1)
    latency_histogram.record(duration)
```

**Benefits:**
- One library instead of three (Prometheus client + logger + tracer)
- Automatic correlation between traces, metrics, and logs
- Can switch backends without code changes

---

### 2. Prometheus + Grafana for Core Operations

**Why:** Still the default standard in most teams. Battle-tested, open-source, huge ecosystem.

**Architecture:**
```
OpenTelemetry SDK
    │
    ├── Prometheus Exporter ──▶ Prometheus Server ──▶ Grafana
    │                              (metrics DB)        (dashboards)
    │
    ├── OTLP Exporter ────────▶ Jaeger (traces)
    │
    └── OTLP Exporter ────────▶ Loki (logs)
```

**Grafana Dashboards:**
1. **S2S Overview**
   - Request rate (req/sec)
   - Latency percentiles (p50/p95/p99)
   - Error rate (%)
   - Active requests

2. **Pipeline Breakdown**
   - ASR latency
   - LLM latency
   - TTS latency
   - Bottleneck identification

3. **GPU Monitoring**
   - Utilization %
   - Memory usage
   - Temperature
   - Power draw

4. **Release Comparison**
   - Current vs previous release
   - Benchmark overlay
   - Canary analysis

---

### 3. Distributed Tracing as First-Class Signal

**Why:** End-to-end traces with correlation IDs. Use exemplars to jump from metric spike → trace → logs quickly.

**What You Get:**
```
Trace: s2s_request (5.23s)
├── asr_stage (1.2s)
│   └── whisper_load (0.3s)
├── llm_stage (2.8s)  ◄── 53% of total time
│   ├── tokenization (0.1s)
│   ├── inference (2.5s)
│   └── detokenization (0.2s)
└── tts_stage (1.1s)
    └── voice_synthesis (0.9s)

Click on slow span → see logs → see metrics
```

**Benefits:**
- Identify bottlenecks instantly
- Understand request flow
- Correlate with logs and metrics
- Debug production issues faster

---

### 4. Structured Logs (JSON) with Context

**Why:** Logs are queryable, not free text. Include request ID, user/session ID, model/version, prompt/version, trial/release ID.

**Log Format:**
```json
{
  "timestamp": "2024-01-15T10:00:00.123Z",
  "level": "INFO",
  "message": "Pipeline completed successfully",
  "trace_id": "abc123def456",
  "span_id": "span789",
  "request_id": "req-xyz789",
  "release_id": "v1.2.3-abc123",
  "model_version": "qwen-2.5-3b-v1.2",
  "config": "baseline",
  "audio_duration": 5.2,
  "latency_total": 5.23,
  "latency_asr": 1.2,
  "latency_llm": 2.8,
  "latency_tts": 1.1,
  "gpu_utilization": 85.3,
  "tokens_input": 150,
  "tokens_output": 250,
  "cost_usd": 0.0045
}
```

**Query Examples:**
```bash
# Find all slow requests in release v1.2.3
{app="miniflow", release_id="v1.2.3-abc123"}
  | json
  | latency_total > 10

# Find errors for specific model version
{app="miniflow", model_version="qwen-2.5-3b-v1.2"}
  | json
  | level="ERROR"
```

---

### 5. SLO-Driven Alerting

**Why:** Alert on user-impacting indicators, not low-value infra noise.

**SLOs for MiniFlow:**

| SLO | Target | Alert When | Why |
|-----|--------|------------|-----|
| **Availability** | 99.9% | < 99.5% | Users can't use the service |
| **Latency p95** | < 5s | > 8s | Poor user experience |
| **Latency p99** | < 10s | > 15s | Critical degradation |
| **Error Rate** | < 0.1% | > 1% | Something is broken |
| **GPU Saturation** | < 80% | > 90% | About to hit capacity |

**NOT Alerted:**
- ❌ CPU usage (unless it causes latency)
- ❌ Memory usage (unless it causes OOM)
- ❌ Individual pod restarts (unless it causes errors)
- ❌ Log volume

**Principle:** Alert on **symptoms** (user pain), not **causes** (infrastructure).

---

### 6. Release-Aware Observability

**Why:** Every deployment tagged by version/commit/model config. Dashboards compare pre/post release behavior automatically.

**Implementation:**
```python
# All metrics tagged with release info
miniflow_latency_seconds{
  release_id="v1.2.3-abc123",
  commit_sha="abc123def",
  model_version="qwen-2.5-3b-v1.2",
  config="baseline",
  stage="production"
}
```

**Grafana Dashboard:**
```
Latency Comparison
│
│    v1.2.2 (baseline)    v1.2.3 (current)
│         │                    │
│    ━━━━━┷━━━━━            ╱╲  │
│               ╲          ╱  ╲ │
│                ╲________╱    ╲│
│
│    p95: 6.2s  ───────▶  p95: 5.2s ✅ Improvement!
└───────────────────────────────────────
```

**Automatic Analysis:**
- Compare current release vs previous
- Highlight regressions
- Correlate with benchmark results
- Inform rollback decisions

---

## LLM-Specific Observability

For MiniFlow (which uses LLMs), add these signals:

### 7. Token/Cost Telemetry

```python
# Track LLM costs
meter.create_counter("llm.tokens.input")
meter.create_counter("llm.tokens.output")
meter.create_counter("llm.cost_usd")

# Usage
llm_tokens_input.add(150)
llm_tokens_output.add(250)
llm_cost_usd.add(0.0045)  # Based on pricing
```

**Dashboard:**
- Cost per request
- Cost per successful task
- Daily/weekly spend
- Budget alerts

---

### 8. Model Behavior Tracing

```python
# Trace LLM-specific details
with tracer.start_span("llm_inference") as span:
    span.set_attribute("llm.model", "qwen-2.5-3b")
    span.set_attribute("llm.prompt_template_version", "v2.1")
    span.set_attribute("llm.temperature", 0.7)
    span.set_attribute("llm.max_tokens", 100)
    span.set_attribute("llm.tokens.input", 150)
    span.set_attribute("llm.tokens.output", 250)
    span.set_attribute("llm.retrieval.docs_count", 3)
    span.set_attribute("llm.retrieval.latency_ms", 45)
    span.set_attribute("llm.retry_count", 0)
```

**Benefits:**
- Debug prompt engineering issues
- Track RAG performance
- Monitor retry rates
- Correlate parameters with quality

---

### 9. Quality/Eval Telemetry

**Hybrid approach (as discussed):**

```
Online (Production):
├── Request latency
├── Error rate
├── Token usage
└── Proxy metrics (if available)

Offline (Benchmark):
├── WER (Word Error Rate)
├── UTMOS (audio quality)
├── Human evaluation
└── Detailed profiling

Linked by:
├── Release ID
├── Model version
└── Timestamp
```

**Result:**
> "Release v1.2.3 had benchmark WER of 0.15 and production p95 latency of 5.2s"

---

### 10. Safety Signals

```python
# Track responsible AI metrics
meter.create_counter("safety.moderation_flagged")
meter.create_counter("safety.fallback_triggered")
meter.create_gauge("safety.refusal_rate")

# Usage
if content_violation_detected:
    safety_moderation_flagged.add(1)

if fallback_to_simpler_model:
    safety_fallback_triggered.add(1)
```

**Dashboard:**
- Content policy violations
- Fallback rates
- Refusal rates
- Safety trend over time

---

## Recommended Stack for MiniFlow

### Core Stack (Must Have)

```
Instrumentation:  OpenTelemetry SDK
Metrics Storage:  Prometheus
Visualization:    Grafana
Tracing:          Jaeger (optional but recommended)
Logging:          Structured JSON + Loki (optional)
```

### Files to Create

```
src/observability/
├── __init__.py
├── telemetry.py          # OTel setup
├── metrics.py            # Metric definitions
├── tracing.py            # Tracing decorators
└── logging_config.py     # Structured logging

infra/observability/
├── prometheus.yml        # Prometheus config
├── grafana/
│   ├── dashboards/
│   │   ├── s2s-overview.json
│   │   ├── pipeline-breakdown.json
│   │   ├── gpu-monitoring.json
│   │   └── release-comparison.json
│   └── datasources.yml
└── docker-compose.obs.yml
```

---

## Implementation Priority

### Phase 1 (PR8 - Core): 2-3 days
1. OpenTelemetry SDK setup
2. Prometheus metrics export
3. Basic Grafana dashboard (S2S Overview)
4. Structured logging

### Phase 2 (PR8 - Advanced): 1-2 days
1. Distributed tracing (Jaeger)
2. Pipeline breakdown dashboard
3. GPU monitoring
4. Benchmark-to-Prometheus export

### Phase 3 (Post-PR8): 1-2 days
1. Release comparison dashboard
2. SLO alerting rules
3. Safety signals
4. Full PLG stack (Loki for logs)

---

## Interview Impact

### Without This Strategy:
> "We have logs and Prometheus metrics."
- Okay, but standard.

### With This Strategy:
> "We implemented OpenTelemetry-first observability with distributed tracing, SLO-driven alerting, and release-aware dashboards that automatically correlate production telemetry with offline benchmark results. We track LLM-specific signals like token usage and cost, and have safety monitoring for responsible AI."
- **Exceptional.** This is senior/staff-level thinking.

---

## Summary

| Pillar | Tool | Effort | Value |
|--------|------|--------|-------|
| OTel-first | OpenTelemetry SDK | 2-3 hrs | Modern standard |
| Prometheus/Grafana | Prometheus + Grafana | 2-3 hrs | Industry standard |
| Distributed Tracing | Jaeger | 2-3 hrs | Bottleneck identification |
| Structured Logs | JSON + context | 1-2 hrs | Debuggability |
| SLO Alerting | Prometheus rules | 2-3 hrs | Reliability engineering |
| Release-Aware | Tagged metrics | 2-3 hrs | Continuous deployment |
| LLM-Specific | Custom metrics | 2-3 hrs | Domain expertise |
| **Total** | | **~2-3 days** | **Interview-winning** |

**Recommendation:** Implement the full strategy. It's 2-3 days of work for a genuinely impressive observability story.
