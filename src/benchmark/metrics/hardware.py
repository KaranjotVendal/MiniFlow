import nvitop
import torch

from src.benchmark.core.base import BaseMetric, MetricContext
from src.benchmark.core.registry import MetricRegistry
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@MetricRegistry.register("hardware_basic")
@MetricRegistry.register("hardware_detailed")
class HardwareMetrics(BaseMetric):
    """Hardware monitoring metric for GPU memory and optional fragmentation analysis.

    This unified class supports two configuration modes:
    - hardware_basic: GPU memory allocation, reserved, peak, and efficiency
    - hardware_detailed: Includes memory fragmentation analysis (waste ratio, inactive blocks, etc.)

    Attributes:
        device: GPU device index to monitor.
        track_power: Whether to track GPU power draw in watts.
        track_fragmentation: Whether to track memory fragmentation metrics.
        waste_threshold: Threshold for considering memory as fragmented (0.0-1.0).

    Example:
        ```python
        # Basic mode - just memory metrics
        metric = HardwareMetrics({"device": 0, "track_power": False, "track_fragmentation": False})
        context = MetricContext(stage=Stage.PIPELINE, trial_id="trial_001", config={}, timestamp=time.time())
        metric.start(context)
        # ... run operation ...
        results = metric.end(context)
        # Returns: {"gpu_memory_allocated_mb": ..., "gpu_memory_reserved_mb": ..., ...}

        # Detailed mode - with fragmentation analysis
        metric = HardwareMetrics({"track_fragmentation": True, "waste_threshold": 0.3})
        metric.start(context)
        # ... run operation ...
        results = metric.end(context)
        # Returns: {...} + fragmentation metrics
        ```
    """

    GPU_MEM_BLOCK = 1024 * 1024

    def __init__(self, config: dict | None = None):
        """Initialize hardware metrics with configuration.

        Args:
            config: Configuration dictionary with options:
                - device: GPU device index (default: 0)
                - track_power: Track power draw in watts (default: False)
                - track_fragmentation: Track fragmentation metrics (default: False)
                - waste_threshold: Threshold for fragmented (default: 0.3)
        """
        super().__init__(config)
        self.device: int = config.get("device", 0) if config else 0
        self.track_power: bool = config.get("track_power", False) if config else False
        self.track_fragmentation: bool = (
            config.get("track_fragmentation", False) if config else False
        )
        # TODO: waste_ratio_threshold is a standard practice.
        self.waste_threshold: float = (
            config.get("waste_threshold", 0.3) if config else 0.3
        )
        self.nvitop_device_obj = nvitop.Device(index=self.device)
        self.is_cuda_available = torch.cuda.is_available()

    def _sync_cuda(self) -> None:
        """wait for GPU ops and ensure timing accuracy"""
        torch.cuda.synchronize(self.device)

    def _get_gpu_memory_reserved(self) -> int:
        """returns the current reserved GPU memory in MB"""
        return torch.cuda.memory_reserved(self.device) // self.GPU_MEM_BLOCK

    def _get_gpu_memory_allocated(self) -> int:
        """returns the currently allocated GPU memory in MB"""
        return torch.cuda.memory_allocated(self.device) // self.GPU_MEM_BLOCK

    def _get_gpu_memory_peak_allocated(self) -> int:
        """returns the currently allocated GPU memory in MB"""
        return torch.cuda.max_memory_allocated(self.device) // self.GPU_MEM_BLOCK

    def _get_gpu_memory_peak_reserved(self) -> int:
        """returns the peak reserved memory since the last reset (MB)"""
        return torch.cuda.max_memory_reserved(self.device) // self.GPU_MEM_BLOCK

    def _reset_peak_memory(self) -> None:
        """reset pytorch peak memory tracker. call before measuring each stage."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def _get_gpu_memory_stats(self) -> dict:
        """returns Detailed statistics | Segment counts, allocation counts, free/used blocks"""
        return torch.cuda.memory_stats(self.device)

    def start(self, context: MetricContext) -> None:
        """Capture baseline hardware state and reset peak memory tracking.

        This method is called at the beginning of a measured operation to:
        1. Reset peak memory statistics for accurate measurement
        2. Capture baseline memory state for delta calculations

        Args:
            context: The current metric context containing stage and trial info.
        """
        if self.is_cuda_available:
            try:
                self._reset_peak_memory()
                self._sync_cuda()

                self._baseline_allocated = self._get_gpu_memory_allocated()
                self._baseline_reserved = self._get_gpu_memory_reserved()
                if self.track_fragmentation:
                    self._start_stats = self._get_gpu_memory_stats()
            except Exception as e:
                logger.exception(e)

        # TODO: implement CPU and memory metrics as well.
        # [I will implement it manually later]

    def end(self, context: MetricContext) -> dict:
        """Collect hardware metrics based on configuration.
        Collects GPU memory metrics and optionally fragmentation and power metrics.
        Returns empty dict if CUDA is not available.

        Args:
            context: The current metric context containing stage and trial info.

        Returns:
            Dictionary containing hardware metric values. Basic mode returns:
            - gpu_memory_allocated_mb: Currently allocated memory in MB
            - gpu_memory_reserved_mb: Reserved/cached memory in MB
            - gpu_memory_peak_mb: Peak allocated memory during operation in MB
            - gpu_memory_efficiency: Ratio of allocated to reserved (0.0-1.0)

            Power tracking additionally returns:
            - gpu_power_draw_watts: Current power draw in watts
            - gpu_temperature_celsius: GPU temperature (if available via nvitop)
            - gpu_utilization_percent: GPU utilization percentage (if available)

            Detailed mode additionally returns:
            - fragmentation_waste_ratio: Unused memory ratio (0.0-1.0)
            - inactive_blocks: Number of inactive memory blocks
            - segment_count: Number of memory segments
            - pool_fraction: Memory pool utilization fraction (0.0-1.0)
            - is_fragmented: Boolean indicating if waste exceeds threshold
        """
        metrics = {}

        if not self.is_cuda_available:
            logger.warning("No cuda device found. returning empty dict")
            return metrics

        try:
            allocated = self._get_gpu_memory_allocated()
            reserved = self._get_gpu_memory_reserved()

            metrics["gpu_memory_allocated_mb"] = allocated
            metrics["gpu_memory_reserved_mb"] = reserved
            metrics["gpu_memory_peak_mb"] = self._get_gpu_memory_peak_allocated()
            metrics["gpu_memory_efficiency"] = (
                allocated / reserved if reserved > 0 else 0.0
            )

            if self.track_power:
                metrics["gpu_power_draw_watts"] = self._get_power_draw()
                temp = self._get_gpu_temperature()
                if temp is not None:
                    metrics["gpu_temperature_celsius"] = temp
                utilization = self._get_gpu_utilization()
                if utilization is not None:
                    metrics["gpu_utilization_percent"] = utilization

            if self.track_fragmentation:
                stats = self._get_gpu_memory_stats()
                # memory_stats() returns a dict with detailed allocation statistics:
                # - 'segment.count': Number of memory segments in the pool
                # - 'segment.all.current_bytes': Total bytes in all segments
                # - 'inactive_split.all.alloc_count': Number of inactive split blocks
                # - 'inactive_split.all.allocated_bytes': Bytes in inactive blocks
                # - 'active_split.all.alloc_count': Number of active split blocks
                # - 'pool_fraction': Ratio of actively used pool memory (0.0-1.0)
                # - 'reserved_bytes.all.peak': Peak reserved bytes
                waste_ratio = (reserved - allocated) / reserved if reserved > 0 else 0.0

                metrics["fragmentation_waste_ratio"] = waste_ratio
                metrics["inactive_blocks"] = stats.get(
                    "inactive_split.all.alloc_count", 0
                )
                metrics["segment_count"] = stats.get("segment.count", 0)
                metrics["pool_fraction"] = stats.get("pool_fraction", 0.0)
                metrics["is_fragmented"] = waste_ratio > self.waste_threshold
        except Exception as e:
            logger.exception(e)

        return metrics

    def _get_power_draw(self) -> float:
        """Get GPU power draw in watts.

        Returns:
            Power draw in watts, or 0.0 if unavailable.
        """
        power_mw = self.nvitop_device_obj.power_draw()
        # NOTE: power is returned in miliWatt
        return float(power_mw) / 1000.0

    def _get_gpu_temperature(self) -> float | None:
        """Get GPU temperature in Celsius.

        Returns:We are implementing a modular benchmark framework for MiniFlow (a speech-to-speech pipeline) based on the specifications in:
        - benchmark_implementation.md - Architecture design
        - benchmarking_plan.md - Metrics requirements
        - benchmark_tickets.md - Task breakdown (22 tasks across 5 milestones)

        Completed Work:
        Last task completed: Task 1.4 (Configuration Loading and Validation)
        - All 24 tests passing
        - Test structure reorganized into unit_tests/ and integration_tests/
        - Progress file updated with patterns and task logs
        Next task: Task 2.1 - Which we have implemented and just needs reviewing and sign off

        For New Session
        To continue work, start by:
        1. Reading benchmark_tickets.md to understand Task 1.4 requirements
        2. Reviewing existing patterns in src/benchmark/core/base.py and src/benchmark/config/
        3. Checking benchmark_progress.md for established patterns
        4. Running tests: python -m pytest tests/ -v
        5. Remember each task needs a review and sign off from me before you can move on to the  next task
            Temperature in Celsius, or None if unavailable.
        """
        temperature = self.nvitop_device_obj.temperature()
        return float(temperature) if temperature is not None else None

    def _get_gpu_utilization(self) -> int | None:
        """Get GPU utilization percentage.

        Returns:
            Utilization percentage (0-100), or None if unavailable.
        """
        utilization = self.nvitop_device_obj.utilization()
        return int(utilization) if utilization is not None else None
