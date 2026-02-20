# Runtime vs Benchmark Metric Mapping

## Purpose
Map low-overhead runtime telemetry to offline benchmark metrics used for release gating.

## Mapping
1. Runtime `miniflow_request_latency_seconds` -> Benchmark `pipeline.latency.total_latency_seconds`
2. Runtime `miniflow_stage_latency_seconds{stage="asr"}` -> Benchmark `asr.latency.inference_seconds`
3. Runtime `miniflow_stage_latency_seconds{stage="llm"}` -> Benchmark `llm.latency.inference_seconds`
4. Runtime `miniflow_stage_latency_seconds{stage="tts"}` -> Benchmark `tts.latency.inference_seconds`
5. Runtime `miniflow_requests_total{status="error"}` -> Benchmark `meta.status_counts.error`
6. Runtime `miniflow_tokens_generated_total` -> Benchmark `llm.inference.tokens_generated`

## Notes
1. Runtime metrics are operational and low overhead.
2. Benchmark metrics remain deep trial-scoped offline analysis artifacts.
