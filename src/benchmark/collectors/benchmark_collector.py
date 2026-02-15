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

        # TODO: I think we can delete the _find_metrics method and assign the metric classes
        # to the respective isntance variables using keys directly.
        self.timing_metrics: TimingMetrics = self._find_metric(
            TimingMetrics, metric_name="TimingMetrics"
        )
        self.token_metrics: TokenMetrics = self._find_metric(
            TokenMetrics, metric_name="TokenMetrics"
        )
        self.lifecycle_metrics: ModelLifecycleMetrics = self._find_metric(
            ModelLifecycleMetrics, metric_name="ModelLifecycleMetrics"
        )
        self.quality_metrics: QualityMetrics = self._find_metric(
            QualityMetrics, metric_name="QualityMetrics"
        )
        self.hardware_metrics: HardwareMetrics = self._find_metric(
            HardwareMetrics, metric_name="HardwareMetrics"
        )
        self.trial_hardware_metrics = HardwareMetrics(self.hardware_metrics.config.copy())

        benchmark_config = self.config["benchmark"]
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

    def _apply_non_streaming_ttft_proxy(self, token_record: TokenRecord) -> None:
        if self.is_streaming:
            token_record.ttft_mode = "true_streaming"
            return

        has_tokens = (
            isinstance(token_record.tokens_generated, int)
            and token_record.tokens_generated > 0
        )
        if not has_tokens:
            return

        total_time = token_record.total_generation_time
        has_total_time = isinstance(total_time, (int, float))
        has_positive_total_time = has_total_time and total_time > 0

        if token_record.ttft is None and has_total_time:
            # Non-streaming fallback: use end-to-end generation time as TTFT proxy.
            token_record.ttft = total_time
            token_record.ttft_mode = "proxy_non_streaming"
        else:
            token_record.ttft_mode = "true_streaming"

        if has_positive_total_time:
            token_record.tokens_per_sec_total = round(
                token_record.tokens_generated / total_time,
                4,
            )
            return

        token_record.tokens_per_sec_total = 0.0

    def _start_trial_metrics(self) -> None:
        assert self._trial_context is not None
        # Trial-scoped metrics only. Token and hardware phase metrics are explicit.
        self.timing_metrics.start(self._trial_context)
        self.lifecycle_metrics.start(self._trial_context)
        # Use a dedicated instance so per-phase resets do not invalidate trial snapshot.
        # TODO: can you explain the idea behind trial_hardware_metrics??
        self.trial_hardware_metrics.start(self._trial_context)

    def _finalize_trial_status(self, status: str, error: str | None) -> None:

        end_time = time.perf_counter()
        self._current_trial.status = status
        self._current_trial.error = error

        if self._trial_start_perf is not None:
            self._current_trial.trial_wall_time_seconds = round(
                end_time - self._trial_start_perf, 6
            )

    def _collect_timing_metrics(self) -> None:
        self.timing_metrics.end(self._trial_context)
        timing_result = self.timing_metrics.to_result()
        self._current_trial.latencies.populate(
            stage_latencies=timing_result.stage_latencies,
            total_latency=timing_result.total_latency_seconds,
        )

    def _collect_lifecycle_metrics(self) -> None:
        self.lifecycle_metrics.end(self._trial_context)
        lifecycle_result = self.lifecycle_metrics.to_result()
        self._current_trial.lifecycle.populate(
            load_events=[event.to_dict() for event in lifecycle_result.load_events]
        )
        self._current_trial.lifecycle.total_model_load_time = (
            lifecycle_result.total_model_load_time
        )
        self._current_trial.lifecycle.cache_hits = lifecycle_result.cache_hits
        self._current_trial.lifecycle.cache_misses = lifecycle_result.cache_misses
        self._current_trial.lifecycle.models_loaded = lifecycle_result.models_loaded

    def _collect_trial_hardware_metrics(self) -> None:
        self.trial_hardware_metrics.end(self._trial_context)
        hardware_result = self.trial_hardware_metrics.to_result()
        trial_snapshot = HardwareSnapshot.from_dict(hardware_result.to_dict())
        self._current_trial.hardware.trial_snapshot = (
            trial_snapshot if trial_snapshot.has_any_value() else None
        )

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

    def start_token_metrics(self) -> None:
        """Start token tracking explicitly, typically inside run_llm()."""
        self._require_active_trial()
        assert self._trial_context is not None
        self.token_metrics.start(self._trial_context)

    def finalize_token_metrics(self) -> None:
        """Finalize token metrics explicitly, typically inside run_llm()."""
        self._require_active_trial()
        assert self._trial_context is not None

        self.token_metrics.end(self._trial_context)
        token_result = self.token_metrics.to_result()
        self._current_trial.tokens.tokens_generated = token_result.tokens_generated
        self._current_trial.tokens.ttft = token_result.ttft
        self._current_trial.tokens.tokens_per_sec = token_result.tokens_per_sec
        self._current_trial.tokens.time_per_token = token_result.time_per_token
        self._current_trial.tokens.total_generation_time = (
            token_result.total_generation_time
        )
        self._apply_non_streaming_ttft_proxy(self._current_trial.tokens)

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

    def start_trial(
        self,
        trial_id: str,
        sample_id: str | None = None,
        is_warmup: bool = False,
        # TODO: what is a metadata for? what exactly are we trying to log with this?
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
            # TODO: again is this metadata really useful?
            metadata=self._current_trial.metadata.copy(),
        )

        self._start_trial_metrics()
        self._trial_started = True

    def end_trial(self, status: str, error: str | None = None) -> dict[str, Any]:
        self._require_active_trial()
        assert self._trial_context is not None
        assert self._current_trial is not None

        self._finalize_trial_status(status=status, error=error)
        self._collect_timing_metrics()
        self._collect_lifecycle_metrics()
        self._collect_trial_hardware_metrics()

        trial_metrics = self._current_trial.to_dict()

        self._reset_trial_state()
        return trial_metrics
