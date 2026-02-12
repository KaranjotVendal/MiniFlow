from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimingResult:
    total_latency_seconds: float
    stage_latencies: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_latency_seconds": self.total_latency_seconds,
            "stage_latencies": self.stage_latencies.copy(),
        }


@dataclass
class TokenResult:
    tokens_generated: int
    ttft: float | None
    tokens_per_sec: float
    time_per_token: float
    total_generation_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_generated": self.tokens_generated,
            "ttft": self.ttft,
            "tokens_per_sec": self.tokens_per_sec,
            "time_per_token": self.time_per_token,
            "total_generation_time": self.total_generation_time,
        }


@dataclass
class ModelLoadResult:
    model_name: str | None = None
    source: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    total_time: float | None = None
    disk_load_time: float | None = None
    gpu_transfer_time: float | None = None
    cached: bool | None = None
    stage: str | None = None
    success: bool | None = None
    error_type: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelLoadResult":
        return cls(
            model_name=payload.get("model_name"),
            source=payload.get("source"),
            start_time=payload.get("start_time"),
            end_time=payload.get("end_time"),
            total_time=payload.get("total_time"),
            disk_load_time=payload.get("disk_load_time"),
            gpu_transfer_time=payload.get("gpu_transfer_time"),
            cached=payload.get("cached"),
            stage=payload.get("stage"),
            success=payload.get("success"),
            error_type=payload.get("error_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.model_name is not None:
            payload["model_name"] = self.model_name
        if self.source is not None:
            payload["source"] = self.source
        if self.start_time is not None:
            payload["start_time"] = self.start_time
        if self.end_time is not None:
            payload["end_time"] = self.end_time
        if self.total_time is not None:
            payload["total_time"] = self.total_time
        if self.disk_load_time is not None:
            payload["disk_load_time"] = self.disk_load_time
        if self.gpu_transfer_time is not None:
            payload["gpu_transfer_time"] = self.gpu_transfer_time
        if self.cached is not None:
            payload["cached"] = self.cached
        if self.stage is not None:
            payload["stage"] = self.stage
        if self.success is not None:
            payload["success"] = self.success
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        return payload


@dataclass
class LifecycleResult:
    load_events: list[ModelLoadResult] = field(default_factory=list)
    total_model_load_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    models_loaded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_events": [event.to_dict() for event in self.load_events],
            "total_model_load_time": self.total_model_load_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "models_loaded": self.models_loaded,
        }


@dataclass
class HardwareResult:
    gpu_memory_allocated_mb: float | None = None
    gpu_memory_reserved_mb: float | None = None
    gpu_memory_peak_mb: float | None = None
    gpu_memory_efficiency: float | None = None
    gpu_power_draw_watts: float | None = None
    gpu_temperature_celsius: float | None = None
    gpu_utilization_percent: float | None = None
    fragmentation_waste_ratio: float | None = None
    inactive_blocks: int | None = None
    segment_count: int | None = None
    pool_fraction: float | None = None
    is_fragmented: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.gpu_memory_allocated_mb is not None:
            payload["gpu_memory_allocated_mb"] = self.gpu_memory_allocated_mb
        if self.gpu_memory_reserved_mb is not None:
            payload["gpu_memory_reserved_mb"] = self.gpu_memory_reserved_mb
        if self.gpu_memory_peak_mb is not None:
            payload["gpu_memory_peak_mb"] = self.gpu_memory_peak_mb
        if self.gpu_memory_efficiency is not None:
            payload["gpu_memory_efficiency"] = self.gpu_memory_efficiency
        if self.gpu_power_draw_watts is not None:
            payload["gpu_power_draw_watts"] = self.gpu_power_draw_watts
        if self.gpu_temperature_celsius is not None:
            payload["gpu_temperature_celsius"] = self.gpu_temperature_celsius
        if self.gpu_utilization_percent is not None:
            payload["gpu_utilization_percent"] = self.gpu_utilization_percent
        if self.fragmentation_waste_ratio is not None:
            payload["fragmentation_waste_ratio"] = self.fragmentation_waste_ratio
        if self.inactive_blocks is not None:
            payload["inactive_blocks"] = self.inactive_blocks
        if self.segment_count is not None:
            payload["segment_count"] = self.segment_count
        if self.pool_fraction is not None:
            payload["pool_fraction"] = self.pool_fraction
        if self.is_fragmented is not None:
            payload["is_fragmented"] = self.is_fragmented
        return payload
