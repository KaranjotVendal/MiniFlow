# Storage Module
# JSONL and other storage backends for benchmark results

from .base import BaseStorage
from .jsonl_storage import JSONLStorage

__all__ = [
    "BaseStorage",
    "JSONLStorage",
]
