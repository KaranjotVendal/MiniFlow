import time
from contextlib import contextmanager

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@contextmanager
def span(name: str, request_id: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.debug(f"span={name} request_id={request_id} duration_s={duration:.6f}")
