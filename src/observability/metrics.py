from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS_TOTAL = Counter(
    "miniflow_requests_total",
    "Total number of /s2s requests",
    ["status"],
)
REQUEST_LATENCY_SECONDS = Histogram(
    "miniflow_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["status"],
)
REQUESTS_IN_PROGRESS = Gauge(
    "miniflow_requests_in_progress",
    "Active /s2s requests",
)
STAGE_LATENCY_SECONDS = Histogram(
    "miniflow_stage_latency_seconds",
    "Stage latency in seconds",
    ["stage", "status"],
)
TOKENS_GENERATED_TOTAL = Counter(
    "miniflow_tokens_generated_total",
    "Total generated tokens",
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
