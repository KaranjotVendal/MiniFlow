import time

from src.benchmark.core.base import BaseMetric, MetricContext
from src.benchmark.core.registry import MetricRegistry


@MetricRegistry.register("model_lifecycle")
class ModelLifecycleMetrics(BaseMetric):
    """Model lifecycle metric for tracking loading times and cache efficiency.

    This metric tracks model loading from disk, GPU transfer times, and cache
    hits/misses. It's useful for identifying loading bottlenecks and measuring
    cache efficiency across multiple trials.
    """
    # TODO: This class needs to be rethink and redesigned to make sure accurate model time taking

    def __init__(self, config: dict | None = None):
        """Initialize model lifecycle metrics with configuration.

        Args:
            config: Configuration dictionary with options:
                - track_gpu_transfer: Track time to transfer model to GPU (default: True)
        """
        super().__init__(config)
        self.track_gpu_transfer: bool = (
            config.get("track_gpu_transfer", True) if config else True
        )
        self._load_events: list[dict] = []
        self._current_load: dict | None = None

    def start(self, context: MetricContext) -> None:
        """Initialize tracking for a new trial.

        Args:
            context: The current metric context containing stage and trial info.
        """
        self._load_events = []
        self._current_load = None

    def record_load_start(self, model_name: str, source: str = "disk") -> None:
        """Record the start of a model load operation.

        Args:
            model_name: The name/identifier of the model being loaded.
            source: Where the model is loading from (e.g., "disk", "remote", "cache").
        """
        self._current_load = {
            "model_name": model_name,
            "source": source,
            "start_time": time.perf_counter(),
            "gpu_transfer_start": None,
            "end_time": None,
            "cached": False,
        }

    def record_gpu_transfer_start(self) -> None:
        """Record the start of GPU transfer for the current model load.

        Call this after disk load but before GPU transfer begins.
        """
        if self._current_load is not None and self.track_gpu_transfer:
            self._current_load["gpu_transfer_start"] = time.perf_counter()

    def record_load_end(self, cached: bool = False) -> dict | None:
        """Record the end of a model load operation and return the event.

        Args:
            cached: Whether the model was loaded from cache.

        Returns:
            The completed load event dict, or None if no load was in progress.
        """
        if self._current_load is None:
            return None

        end_time = time.perf_counter()
        self._current_load["end_time"] = end_time
        self._current_load["cached"] = cached
        self._current_load["total_time"] = end_time - self._current_load["start_time"]

        # TODO: can you please have a look at the current sequential loading and reconsider if this approach works?
        # for example do I have specify every call where the being loaded from as well as how do we make sure cached status?
        # [This todo has a plan that will be implemented by me manually]

        # Calculate GPU transfer time if tracked and applicable
        if (
            self.track_gpu_transfer
            and self._current_load.get("gpu_transfer_start") is not None
        ):
            gpu_start = self._current_load["gpu_transfer_start"]
            self._current_load["gpu_transfer_time"] = end_time - gpu_start
            self._current_load["disk_load_time"] = (
                gpu_start - self._current_load["start_time"]
            )
        else:
            self._current_load["gpu_transfer_time"] = 0.0
            self._current_load["disk_load_time"] = self._current_load["total_time"]

        event = self._current_load.copy()
        self._load_events.append(event)
        self._current_load = None
        return event

    def end(self, context: MetricContext) -> dict:
        """Calculate and return model lifecycle metrics for the trial.

        Args:
            context: The current metric context containing stage and trial info.

        Returns:
            Dictionary containing:
                - load_events: List of all model load events with timing details
                - total_model_load_time: Total time spent loading models
                - cache_hits: Number of models loaded from cache
                - cache_misses: Number of models loaded from source
                - models_loaded: Total number of models loaded
        """
        total_load_time = sum(event.get("total_time", 0) for event in self._load_events)
        cache_hits = sum(1 for event in self._load_events if event.get("cached", False))
        cache_misses = len(self._load_events) - cache_hits

        return {
            "load_events": self._load_events.copy(),
            "total_model_load_time": round(total_load_time, 6),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "models_loaded": len(self._load_events),
        }
