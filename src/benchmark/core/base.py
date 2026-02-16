from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Stage(str, Enum):
    """Pipeline stages for metric collection context.

    Attributes:
        ASR: Automatic speech recognition stage.
        LLM: Large language model generation stage.
        TTS: Text-to-speech synthesis stage.
        PIPELINE: Full pipeline execution context.
    """

    ASR = "asr"
    LLM = "llm"
    TTS = "tts"
    PIPELINE = "pipeline"


@dataclass
class MetricContext:
    """Context passed to all metrics during collection.

    This dataclass carries information about the current execution context,
    allowing metrics to understand which stage is being measured and access
    relevant configuration.

    Attributes:
        stage: Which pipeline stage is currently executing.
        trial_id: Unique identifier for the current trial/sample.
        config: Experiment configuration dictionary.
        timestamp: Wall-clock start timestamp for the operation.
        metadata: Additional context-specific information.
    """

    stage: Stage
    trial_id: str
    config: dict
    timestamp: float
    metadata: dict = field(default_factory=dict)


class BaseMetric(ABC):
    """Abstract base class for all benchmark metrics.

    All metric implementations must inherit from this class and implement
    the `start()` and `end()` methods. The framework calls these methods
    at the beginning and end of each measured operation.

    Attributes:
        config: Configuration dictionary for the metric.
        _data: Internal storage for collected metric values.
        _is_collecting: Flag indicating if metric collection is active.

    Example:
        ```python
        class CustomMetric(BaseMetric):
            def start(self, context: MetricContext) -> None:
                self._start_time = time.time()

            def end(self, context: MetricContext) -> dict[str, Any]:
                return {"duration": time.time() - self._start_time}
        ```
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._data: dict[str, Any] = {}
        self._is_collecting: bool = False

    @abstractmethod
    def start(self, context: MetricContext) -> None:
        """Called when metric collection starts.

        Subclasses should override this method to initialize any state
        needed for measuring the operation (e.g., capturing baseline
        memory usage, starting timers).

        Args:
            context: The current metric context containing stage and trial info.
        """
        pass

    @abstractmethod
    def end(self, context: MetricContext) -> dict[str, Any]:
        """Called when metric collection ends.

        Subclasses should override this method to finalize measurements
        and return the collected metric values.

        Args:
            context: The current metric context containing stage and trial info.

        Returns:
            Dictionary of collected metric values.
        """
        pass

    def get_value(self) -> dict[str, Any]:
        """Get current metric values.

        Returns:
            Copy of the internal data dictionary.
        """
        return self._data.copy()

    def is_enabled(self) -> bool:
        """Check if metric is enabled in configuration.

        Returns:
            True if the metric should be collected, False otherwise.
        """
        return self.config.get("enabled", True)


class BaseCollector(ABC):
    """Abstract base class for metric collectors.

    Collectors wrap operations and collect metrics from multiple metric
    instances. Subclasses implement the `collect()` method to define
    how metrics are gathered around an operation.

    Attributes:
        metrics: List of enabled metrics to collect.

    Example:
        ```python
        class PipelineCollector(BaseCollector):
            def collect(
                self, operation: Callable, context: MetricContext,
                *args, **kwargs
            ) -> dict[str, Any]:
                for metric in self.metrics:
                    metric.start(context)
                result = operation(*args, **kwargs)
                for metric in self.metrics:
                    result.update(metric.end(context))
                return result
        ```
    """

    def __init__(self, metrics: list["BaseMetric"]):
        self.metrics = [m for m in metrics if m.is_enabled()]

    @abstractmethod
    def collect(
        self, operation: Callable, context: MetricContext, *args, **kwargs
    ) -> dict[str, Any]:
        """Wrap an operation and collect metrics.

        Subclasses implement this method to define how metrics are
        collected around the execution of an operation.

        Args:
            operation: The callable function to wrap.
            context: The current metric context.
            *args: Positional arguments passed to the operation.
            **kwargs: Keyword arguments passed to the operation.

        Returns:
            Dictionary containing all collected metric values.
        """
        pass
