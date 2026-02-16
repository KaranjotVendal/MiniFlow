from typing import Callable, Type

from src.benchmark.core.base import BaseMetric
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


class MetricRegistry:
    """Registry for metric classes using decorator-based registration.

    This class provides a plugin architecture where metrics can be registered
    by name and retrieved at runtime. This enables configuration-driven
    metric selection.

    Attributes:
        _metrics: Dictionary mapping metric names to their classes.
    """

    _metrics: dict[str, Type[BaseMetric]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a metric class.

        Args:
            name: Unique identifier for the metric.

        Returns:
            Decorator function that registers the class.

        Example:
            ```python
            @MetricRegistry.register("hardware_basic")
            class HardwareMetrics(BaseMetric):
                pass
            ```
        """

        def decorator(metric_class: Type[BaseMetric]) -> Type[BaseMetric]:
            if name in cls._metrics:

                logger.warning(f"Metric '{name}' is already registered. Overwriting.")
            cls._metrics[name] = metric_class
            return metric_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseMetric]:
        """Retrieve a registered metric class by name.

        Args:
            name: The registered metric identifier.

        Returns:
            The metric class
        """
        if name not in cls._metrics:
            available = cls.list_sorted_metrics()
            raise KeyError(f"Metric '{name}' not found. Available metrics: {available}")
        return cls._metrics[name]

    @classmethod
    def list_sorted_metrics(cls) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of registered metric identifiers.
        """
        return sorted(cls._metrics.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics.

        Useful for testing to ensure a clean registry state.
        """
        cls._metrics.clear()

    @classmethod
    def count(cls) -> int:
        """Return the number of registered metrics.

        Returns:
            Number of registered metrics.
        """
        return len(cls._metrics)
