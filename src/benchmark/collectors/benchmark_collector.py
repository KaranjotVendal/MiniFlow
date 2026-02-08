import time
from typing import Any

from src.benchmark.core.base import BaseMetric, MetricContext, Stage
from src.benchmark.metrics.hardware import HardwareMetrics
from src.benchmark.metrics.lifecycle import ModelLifecycleMetrics
from src.benchmark.metrics.quality import QualityMetrics
from src.benchmark.metrics.timing import TimingMetrics
from src.benchmark.metrics.tokens import TokenMetrics
from src.benchmark.collectors.trial_models import (
    HardwareSnapshot,
    TokenRecord,
    TrialRecord,
)


class BenchmarkCollector:
    """Trial-scoped holder for metric instances and trial state."""

    def __init__(
        self,
        metrics: dict[str, BaseMetric] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        metric_map = metrics or {}
        self.metrics: dict[str, BaseMetric] = metric_map
        self.config = config

        self.timing_metrics = self._find_metric(
            TimingMetrics, metric_name="TimingMetrics"
        )
        self.token_metrics = self._find_metric(
            TokenMetrics, metric_name="TokenMetrics"
        )
        self.lifecycle_metrics = self._find_metric(
            ModelLifecycleMetrics, metric_name="ModelLifecycleMetrics"
        )
        self.quality_metrics = self._find_metric(
            QualityMetrics, metric_name="QualityMetrics"
        )
        self.hardware_metrics = self._find_metric(
            HardwareMetrics, metric_name="HardwareMetrics"
        )

        benchmark_config = self.config["benchmark"] if self.config else {}
        self.is_streaming = bool(benchmark_config["enable_streaming_audio"])

        self._trial_context: MetricContext | None = None
        self._trial_started = False
        self._trial_start_perf: float | None = None
        self._current_trial: TrialRecord | None = None

    @property
    def context(self) -> MetricContext:
        self._require_active_trial()
        assert self._trial_context is not None
        return self._trial_context

    @property
    def current_trial(self) -> TrialRecord:
        self._require_active_trial()
        assert self._current_trial is not None
        return self._current_trial

    def start_trial(
        self,
        trial_id: str,
        sample_id: str | None = None,
        is_warmup: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._trial_started:
            raise RuntimeError("A trial is already active. Call end_trial() first.")

        self._current_trial = TrialRecord(
            trial_id=trial_id,
            sample_id=sample_id,
            is_warmup=is_warmup,
            metadata=(metadata or {}).copy(),
        )
        self._trial_start_perf = time.perf_counter()

        self._trial_context = MetricContext(
            stage=Stage.PIPELINE,
            trial_id=trial_id,
            config=self.config,
            timestamp=time.time(),
            metadata=self._current_trial.metadata.copy(),
        )

        for metric in self.metrics.values():
            metric.start(self._trial_context)

        self._trial_started = True

    def end_trial(self, status: str, error: str | None = None) -> dict[str, Any]:
        self._require_active_trial()
        assert self._trial_context is not None
        assert self._current_trial is not None

        end_time = time.perf_counter()
        self._current_trial.status = status
        self._current_trial.error = error

        if self._trial_start_perf is not None:
            self._current_trial.trial_wall_time_seconds = round(
                end_time - self._trial_start_perf, 6
            )

        timing_results = self.timing_metrics.end(self._trial_context)
        self._current_trial.latencies.populate(
            stage_latencies=timing_results.get("stage_latencies", {}),
            total_latency=timing_results.get("total_latency_seconds"),
        )

        token_results = self.token_metrics.end(self._trial_context)
        self._current_trial.tokens.tokens_generated = token_results.get(
            "tokens_generated"
        )
        self._current_trial.tokens.ttft = token_results.get("ttft")
        self._current_trial.tokens.tokens_per_sec = token_results.get("tokens_per_sec")
        self._current_trial.tokens.time_per_token = token_results.get("time_per_token")
        self._current_trial.tokens.total_generation_time = token_results.get(
            "total_generation_time"
        )

        self._apply_non_streaming_ttft_proxy(self._current_trial.tokens)

        lifecycle_results = self.lifecycle_metrics.end(self._trial_context)
        self._current_trial.lifecycle.populate(
            load_events=lifecycle_results.get("load_events", [])
        )
        self._current_trial.lifecycle.total_model_load_time = lifecycle_results.get(
            "total_model_load_time"
        )
        self._current_trial.lifecycle.cache_hits = lifecycle_results.get("cache_hits")
        self._current_trial.lifecycle.cache_misses = lifecycle_results.get(
            "cache_misses"
        )
        self._current_trial.lifecycle.models_loaded = lifecycle_results.get(
            "models_loaded"
        )

        hardware_results = self.hardware_metrics.end(self._trial_context)
        trial_snapshot = HardwareSnapshot.from_dict(hardware_results)
        self._current_trial.hardware.trial_snapshot = (
            trial_snapshot if trial_snapshot.has_any_value() else None
        )

        trial_metrics = self._current_trial.to_dict()

        self._reset_trial_state()
        return trial_metrics

    def record_phase_metrics(self, phase_name: str, metrics: dict[str, Any]) -> None:
        snapshot = HardwareSnapshot.from_dict(metrics)
        if not snapshot.has_any_value():
            raise ValueError(
                f"Phase '{phase_name}' must contain hardware metric keys. "
                "Use lifecycle/timing/token metric methods for non-hardware data."
            )

        assigned = self.current_trial.hardware.phases.assign(phase_name, snapshot)
        if not assigned:
            raise ValueError(f"Unknown hardware phase name: {phase_name}")

    def _apply_non_streaming_ttft_proxy(self, token_record: TokenRecord) -> None:
        if self.is_streaming:
            token_record.ttft_mode = "true_streaming"
            return

        if (
            isinstance(token_record.tokens_generated, int)
            and token_record.tokens_generated > 0
        ):
            if token_record.ttft is None and isinstance(
                token_record.total_generation_time, (int, float)
            ):
                token_record.ttft = token_record.total_generation_time
                token_record.ttft_mode = "proxy_non_streaming"
            else:
                token_record.ttft_mode = "true_streaming"

            if (
                isinstance(token_record.total_generation_time, (int, float))
                and token_record.total_generation_time > 0
            ):
                token_record.tokens_per_sec_total = round(
                    token_record.tokens_generated / token_record.total_generation_time,
                    4,
                )
            else:
                token_record.tokens_per_sec_total = 0.0

    def _find_metric(
        self, metric_type: type[BaseMetric], metric_name: str
    ) -> BaseMetric:
        for metric in self.metrics.values():
            if isinstance(metric, metric_type):
                return metric
        raise ValueError(f"{metric_name} is required for BenchmarkCollector.")

    def _require_active_trial(self) -> None:
        if not self._trial_started:
            raise RuntimeError("No active trial. Call start_trial() first.")

    def _reset_trial_state(self) -> None:
        self._trial_context = None
        self._trial_started = False
        self._trial_start_perf = None
        self._current_trial = None
