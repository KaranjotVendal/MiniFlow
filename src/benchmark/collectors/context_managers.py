"""Context managers for convenient metric collection around code blocks."""

import time
from contextlib import contextmanager
from typing import Any

import torch

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@contextmanager
def track_latency(name: str, metrics_dict: dict[str, Any]) -> None:
    """Track execution time of a code block and store in metrics dictionary.

    This context manager measures the elapsed time of the wrapped code block
    and stores the result in the provided metrics dictionary under the key
    "{name}_latency_seconds".

    Args:
        name: Identifier for the operation being timed.
        metrics_dict: Dictionary to store the latency measurement.

    Example:
        ```python
        metrics = {}
        with track_latency("inference", metrics):
            result = model.inference(input_data)
        # metrics = {"inference_latency_seconds": 0.123456}
        ```
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        metrics_dict[f"{name}_latency_seconds"] = round(elapsed, 6)


@contextmanager
def track_memory(name: str, metrics_dict: dict[str, Any]) -> None:
    """Track memory delta of a code block and store in metrics dictionary.

    This context manager measures the change in GPU memory allocation during
    the wrapped code block and stores the result in the provided metrics
    dictionary under the key "{name}_memory_delta_mb".

    Args:
        name: Identifier for the operation being measured.
        metrics_dict: Dictionary to store the memory delta.

    Example:
        ```python
        metrics = {}
        with track_memory("model_load", metrics):
            model = load_model()
        # metrics = {"model_load_memory_delta_mb": 2048}
        ```
    """
    if not torch.cuda.is_available():
        try:
            yield
        finally:
            metrics_dict[f"{name}_memory_delta_mb"] = 0
        return

    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    start_allocated = torch.cuda.memory_allocated(device)
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        end_allocated = torch.cuda.memory_allocated(device)
        delta_mb = (end_allocated - start_allocated) // (1024 * 1024)
        metrics_dict[f"{name}_memory_delta_mb"] = delta_mb
