import time

from src.benchmark.core.base import BaseMetric, MetricContext
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.metrics.result_models import TimingResult
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@MetricRegistry.register("timing")
class TimingMetrics(BaseMetric):
    """Timing metric for tracking pipeline stage and total latency."""

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: Configuration dictionary with options:
                - stages: List of stage names to track (default: [])
        """
        super().__init__(config)
        self.stages: list[str] = config.get("stages", []) if config else []
        self._start_time: float | None = None
        self._stage_start_times: dict[str, float] = {}
        self._stage_latencies: dict[str, float] = {}
        self._last_result: TimingResult | None = None

    def start(self, context: MetricContext) -> None:
        """Capture the start timestamp and reset stage tracking.

        Args:
            context: The current metric context containing stage and trial info.
        """
        # TOOD: given the implmentation and current implementation we can probably delete _start_time in the future.
        self._start_time = time.perf_counter()
        self._stage_start_times = {}
        self._stage_latencies = {}
        self._last_result = None

    def record_stage_start(self, name: str) -> None:
        """Record when a stage begins.

        Args:
            name: The name of the stage to record.
        """
        if self._start_time is not None:
            self._stage_start_times[name] = time.perf_counter()

    def record_stage_end(self, name: str) -> None:
        """Record when a stage ends and calculate its duration.

        Args:
            name: The name of the stage to record.
        """
        if self._start_time is not None and name in self._stage_start_times:
            end_time = time.perf_counter()
            self._stage_latencies[name] = end_time - self._stage_start_times[name]

    def end(self, context: MetricContext) -> dict:
        """Calculate and return timing metrics.

        Args:
            context: The current metric context containing stage and trial info.

        Returns:
                - total_latency_seconds: Total elapsed time from start to end
                - stage_latencies: dict mapping stage names to their durations
        """
        total_latency = 0.0
        if self._start_time is not None:
            total_latency = time.perf_counter() - self._start_time

        self._last_result = TimingResult(
            total_latency_seconds=round(total_latency, 6),
            stage_latencies=self._stage_latencies.copy(),
        )
        return self._last_result.to_dict()

    def to_result(self) -> TimingResult:
        if self._last_result is None:
            raise RuntimeError("TimingMetrics result is unavailable before end().")
        return self._last_result
