# Metrics Module
# Individual metric implementations
#
# NOTE: The decorator @MetricRegistry.register() runs at import time, not at
# definition time. Importing this module triggers the registration of all
# metric classes with the MetricRegistry. This is why we import all metric
# modules here - to ensure they are registered when the package is loaded.

from .timing import TimingMetrics
from .hardware import HardwareMetrics
from .lifecycle import ModelLifecycleMetrics
from .quality import QualityMetrics
from .tokens import TokenMetrics

__all__ = [
    "TimingMetrics",
    "HardwareMetrics",
    "ModelLifecycleMetrics",
    "QualityMetrics",
    "TokenMetrics",
]
