from types import SimpleNamespace
from typing import Any


class _NoOpMetric:
    def start(self, *args, **kwargs) -> None:
        return None

    def end(self, *args, **kwargs) -> dict:
        return {}

    def record_stage_start(self, *args, **kwargs) -> None:
        return None

    def record_stage_end(self, *args, **kwargs) -> None:
        return None

    def record_load_start(self, *args, **kwargs) -> None:
        return None

    def record_load_end(self, *args, **kwargs) -> dict:
        return {}

    def add_tokens(self, *args, **kwargs) -> None:
        return None

    def evaluate(self, evaluator: str, *args, **kwargs) -> dict[str, float]:
        if evaluator == "wer":
            return {"wer": 0.0}
        if evaluator == "utmos":
            return {"utmos": 0.0}
        return {evaluator: 0.0}


class NullCollector:
    def __init__(self) -> None:
        self.context = None
        self.timing_metrics = _NoOpMetric()
        self.token_metrics = _NoOpMetric()
        self.lifecycle_metrics = _NoOpMetric()
        self.hardware_metrics = _NoOpMetric()
        self.quality_metrics = _NoOpMetric()
        self.current_trial = SimpleNamespace(
            quality=SimpleNamespace(wer=0.0, utmos=0.0)
        )

    def start_token_metrics(self) -> None:
        return None

    def finalize_token_metrics(self) -> None:
        return None

    def record_phase_metrics(self, phase_name: str, metrics: dict[str, Any]) -> None:
        return None


class NullInstrumentation:
    def __init__(self) -> None:
        self.collector = NullCollector()

    def on_request_start(self, request_id: str, metadata: dict[str, Any] | None = None) -> None:
        return None

    def on_request_end(self, request_id: str, status: str, error: str | None = None) -> None:
        return None

    def on_stage_start(self, request_id: str, stage: str) -> None:
        return None

    def on_stage_end(
        self,
        request_id: str,
        stage: str,
        status: str = "success",
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_token_metrics(
        self,
        request_id: str,
        *,
        tokens_generated: int,
        ttft: float | None,
        tokens_per_sec: float | None,
    ) -> None:
        return None
