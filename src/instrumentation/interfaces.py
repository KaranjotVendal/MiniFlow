from typing import Any, Protocol


class PipelineInstrumentation(Protocol):
    collector: Any

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
