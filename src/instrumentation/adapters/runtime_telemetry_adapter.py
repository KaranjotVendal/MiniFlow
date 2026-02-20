import time
from typing import Any

from src.logger.logging import initialise_logger
from src.observability import metrics as m
from src.observability.tracing import span
from src.instrumentation.adapters.null_instrumentation import NullCollector

logger = initialise_logger(__name__)


class RuntimeTelemetryAdapter:
    def __init__(self, release_id: str) -> None:
        self.release_id = release_id
        self.collector = NullCollector()
        self._request_start: dict[str, float] = {}
        self._stage_start: dict[tuple[str, str], float] = {}

    def on_request_start(self, request_id: str, metadata: dict[str, Any] | None = None) -> None:
        self._request_start[request_id] = time.perf_counter()
        m.REQUESTS_IN_PROGRESS.inc()
        logger.info(f"request_start request_id={request_id} release_id={self.release_id}")

    def on_request_end(self, request_id: str, status: str, error: str | None = None) -> None:
        start = self._request_start.pop(request_id, None)
        if start is not None:
            latency = time.perf_counter() - start
            m.REQUEST_LATENCY_SECONDS.labels(status=status).observe(latency)
        m.REQUESTS_TOTAL.labels(status=status).inc()
        m.REQUESTS_IN_PROGRESS.dec()
        logger.info(
            f"request_end request_id={request_id} release_id={self.release_id} status={status} error={error}"
        )

    def on_stage_start(self, request_id: str, stage: str) -> None:
        self._stage_start[(request_id, stage)] = time.perf_counter()

    def on_stage_end(
        self,
        request_id: str,
        stage: str,
        status: str = "success",
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        key = (request_id, stage)
        start = self._stage_start.pop(key, None)
        if start is not None:
            m.STAGE_LATENCY_SECONDS.labels(stage=stage, status=status).observe(
                time.perf_counter() - start
            )
        logger.info(
            f"stage_end request_id={request_id} stage={stage} status={status} error={error}"
        )

    def on_token_metrics(
        self,
        request_id: str,
        *,
        tokens_generated: int,
        ttft: float | None,
        tokens_per_sec: float | None,
    ) -> None:
        if tokens_generated > 0:
            m.TOKENS_GENERATED_TOTAL.inc(tokens_generated)
        logger.info(
            f"token_metrics request_id={request_id} tokens={tokens_generated} ttft={ttft} tps={tokens_per_sec}"
        )

    def stage_span(self, request_id: str, stage: str):
        return span(name=f"{stage}_stage", request_id=request_id)
