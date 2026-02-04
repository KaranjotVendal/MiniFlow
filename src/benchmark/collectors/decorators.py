from collections.abc import Callable
from functools import wraps
from typing import Any

from src.benchmark.collectors.context_managers import track_latency, track_memory


def track_latency_decorator(metric_name: str) -> Callable:
    """Decorator factory that wraps functions with latency tracking.

    The decorated function will have a `.metrics` attribute containing
    a dictionary of collected metrics. Each call to the decorated function
    updates this dictionary with the measured latency.

    Args:
        metric_name: Identifier for the operation being timed.

    Example:
        ```python
        @track_latency_decorator("inference")
        def inference(input_data):
            return model.generate(input_data)

        result = inference(data)
        # inference.metrics = {"inference_latency_seconds": 0.123456}
        ```

    Attributes:
        metrics: Dictionary containing accumulated metric values.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            metrics: dict[str, Any] = {}
            try:
                with track_latency(metric_name, metrics):
                    result = func(*args, **kwargs)
            finally:
                wrapper.metrics.update(metrics)
            return result

        wrapper.metrics = getattr(wrapper, "metrics", {})
        return wrapper

    return decorator


def track_memory_decorator(metric_name: str) -> Callable:
    """Decorator factory that wraps functions with GPU memory tracking.

    The decorated function will have a `.metrics` attribute containing
    a dictionary of collected metrics. Each call to the decorated function
    updates this dictionary with the measured memory delta.

    Args:
        metric_name: Identifier for the operation being measured.

    Example:
        ```python
        @track_memory_decorator("model_load")
        def load_model():
            return model.from_pretrained("path")

        model = load_model()
        # load_model.metrics = {"model_load_memory_delta_mb": 2048}
        ```

    Attributes:
        metrics: Dictionary containing accumulated metric values.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            metrics: dict[str, Any] = {}
            try:
                with track_memory(metric_name, metrics):
                    result = func(*args, **kwargs)
            finally:
                wrapper.metrics.update(metrics)
            return result

        wrapper.metrics = getattr(wrapper, "metrics", {})
        return wrapper

    return decorator
