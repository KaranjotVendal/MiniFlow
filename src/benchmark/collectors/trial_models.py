from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyRecord:
    """Structured latency fields for known pipeline stages."""

    total_latency_seconds: float | None = None
    asr_seconds: float | None = None
    llm_seconds: float | None = None
    tts_seconds: float | None = None
    pipeline_seconds: float | None = None

    def populate(self, stage_latencies: dict[str, float], total_latency: float | None) -> None:
        self.total_latency_seconds = total_latency
        self.asr_seconds = stage_latencies.get("asr", stage_latencies.get("asr_inference_latency"))
        self.llm_seconds = stage_latencies.get("llm", stage_latencies.get("llm_inference_latency"))
        self.tts_seconds = stage_latencies.get("tts", stage_latencies.get("tts_inference_latency"))
        self.pipeline_seconds = stage_latencies.get("pipeline")

    def to_dict(self) -> dict[str, Any]:
        stage_dict: dict[str, float] = {}
        if self.asr_seconds is not None:
            stage_dict["asr"] = self.asr_seconds
        if self.llm_seconds is not None:
            stage_dict["llm"] = self.llm_seconds
        if self.tts_seconds is not None:
            stage_dict["tts"] = self.tts_seconds
        if self.pipeline_seconds is not None:
            stage_dict["pipeline"] = self.pipeline_seconds

        payload: dict[str, Any] = {}
        if self.total_latency_seconds is not None:
            payload["total_latency_seconds"] = self.total_latency_seconds
        if stage_dict:
            payload["latencies"] = stage_dict
        return payload


@dataclass
class ModelLoadEvent:
    """Typed model load event."""

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
    def from_dict(cls, payload: dict[str, Any]) -> "ModelLoadEvent":
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
class LifecycleRecord:
    """Typed lifecycle fields split by known stages."""

    asr_load: ModelLoadEvent | None = None
    llm_load: ModelLoadEvent | None = None
    llm_tokenizer_load: ModelLoadEvent | None = None
    llm_pipeline_load: ModelLoadEvent | None = None
    tts_load: ModelLoadEvent | None = None
    tts_processor_load: ModelLoadEvent | None = None
    unknown_load_events: list[ModelLoadEvent] = field(default_factory=list)
    total_model_load_time: float | None = None
    cache_hits: int | None = None
    cache_misses: int | None = None
    models_loaded: int | None = None

    @staticmethod
    def _append_or_keep_unknown(
        current: ModelLoadEvent | None,
        incoming: ModelLoadEvent,
        unknown_events: list[ModelLoadEvent],
    ) -> ModelLoadEvent:
        if current is None:
            return incoming
        unknown_events.append(current)
        return incoming

    def populate(self, load_events: list[dict[str, Any]]) -> None:
        self.asr_load = None
        self.llm_load = None
        self.llm_tokenizer_load = None
        self.llm_pipeline_load = None
        self.tts_load = None
        self.tts_processor_load = None
        self.unknown_load_events = []

        for event_dict in load_events:
            event = ModelLoadEvent.from_dict(event_dict)
            if event.stage == "asr":
                self.asr_load = self._append_or_keep_unknown(
                    self.asr_load, event, self.unknown_load_events
                )
            elif event.stage == "llm":
                model_name = (event.model_name or "").lower()
                if "tokenizer" in model_name:
                    self.llm_tokenizer_load = self._append_or_keep_unknown(
                        self.llm_tokenizer_load, event, self.unknown_load_events
                    )
                elif "pipeline" in model_name:
                    self.llm_pipeline_load = self._append_or_keep_unknown(
                        self.llm_pipeline_load, event, self.unknown_load_events
                    )
                else:
                    self.llm_load = self._append_or_keep_unknown(
                        self.llm_load, event, self.unknown_load_events
                    )
            elif event.stage == "tts":
                model_name = (event.model_name or "").lower()
                if "processor" in model_name or "process" in model_name:
                    self.tts_processor_load = self._append_or_keep_unknown(
                        self.tts_processor_load, event, self.unknown_load_events
                    )
                else:
                    self.tts_load = self._append_or_keep_unknown(
                        self.tts_load, event, self.unknown_load_events
                    )
            else:
                self.unknown_load_events.append(event)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        events: list[dict[str, Any]] = []
        if self.asr_load is not None:
            events.append(self.asr_load.to_dict())
            payload["asr_model_load"] = self.asr_load.to_dict()
        if self.llm_load is not None:
            events.append(self.llm_load.to_dict())
            payload["llm_model_load"] = self.llm_load.to_dict()
        if self.llm_tokenizer_load is not None:
            events.append(self.llm_tokenizer_load.to_dict())
            payload["llm_tokenizer_load"] = self.llm_tokenizer_load.to_dict()
        if self.llm_pipeline_load is not None:
            events.append(self.llm_pipeline_load.to_dict())
            payload["llm_pipeline_load"] = self.llm_pipeline_load.to_dict()
        if self.tts_load is not None:
            events.append(self.tts_load.to_dict())
            payload["tts_model_load"] = self.tts_load.to_dict()
        if self.tts_processor_load is not None:
            events.append(self.tts_processor_load.to_dict())
            payload["tts_processor_load"] = self.tts_processor_load.to_dict()
        if self.unknown_load_events:
            events.extend([event.to_dict() for event in self.unknown_load_events])

        if events:
            payload["load_events"] = events
        if self.total_model_load_time is not None:
            payload["total_model_load_time"] = self.total_model_load_time
        if self.cache_hits is not None:
            payload["cache_hits"] = self.cache_hits
        if self.cache_misses is not None:
            payload["cache_misses"] = self.cache_misses
        if self.models_loaded is not None:
            payload["models_loaded"] = self.models_loaded

        return payload


@dataclass
class QualityRecord:
    """Structured quality metric fields."""

    wer: float | None = None
    utmos: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.wer is not None:
            payload["wer"] = self.wer
        if self.utmos is not None:
            payload["utmos"] = self.utmos
        return payload


@dataclass
class TokenRecord:
    """Structured token metric fields."""

    tokens_generated: int | None = None
    ttft: float | None = None
    tokens_per_sec: float | None = None
    time_per_token: float | None = None
    total_generation_time: float | None = None
    tokens_per_sec_total: float | None = None
    ttft_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.tokens_generated is not None:
            payload["tokens_generated"] = self.tokens_generated
        if self.ttft is not None:
            payload["ttft"] = self.ttft
        if self.tokens_per_sec is not None:
            payload["tokens_per_sec"] = self.tokens_per_sec
        if self.time_per_token is not None:
            payload["time_per_token"] = self.time_per_token
        if self.total_generation_time is not None:
            payload["total_generation_time"] = self.total_generation_time
        if self.tokens_per_sec_total is not None:
            payload["tokens_per_sec_total"] = self.tokens_per_sec_total
        if self.ttft_mode is not None:
            payload["ttft_mode"] = self.ttft_mode
        return payload


@dataclass
class HardwareSnapshot:
    """Typed hardware metric snapshot."""

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HardwareSnapshot":
        return cls(
            gpu_memory_allocated_mb=payload.get("gpu_memory_allocated_mb"),
            gpu_memory_reserved_mb=payload.get("gpu_memory_reserved_mb"),
            gpu_memory_peak_mb=payload.get("gpu_memory_peak_mb"),
            gpu_memory_efficiency=payload.get("gpu_memory_efficiency"),
            gpu_power_draw_watts=payload.get("gpu_power_draw_watts"),
            gpu_temperature_celsius=payload.get("gpu_temperature_celsius"),
            gpu_utilization_percent=payload.get("gpu_utilization_percent"),
            fragmentation_waste_ratio=payload.get("fragmentation_waste_ratio"),
            inactive_blocks=payload.get("inactive_blocks"),
            segment_count=payload.get("segment_count"),
            pool_fraction=payload.get("pool_fraction"),
            is_fragmented=payload.get("is_fragmented"),
        )

    def has_any_value(self) -> bool:
        return any(
            value is not None
            for value in [
                self.gpu_memory_allocated_mb,
                self.gpu_memory_reserved_mb,
                self.gpu_memory_peak_mb,
                self.gpu_memory_efficiency,
                self.gpu_power_draw_watts,
                self.gpu_temperature_celsius,
                self.gpu_utilization_percent,
                self.fragmentation_waste_ratio,
                self.inactive_blocks,
                self.segment_count,
                self.pool_fraction,
                self.is_fragmented,
            ]
        )

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


@dataclass
class HardwarePhaseRecord:
    """Known hardware snapshots per stage/phase."""

    asr_model_load: HardwareSnapshot | None = None
    asr_inference: HardwareSnapshot | None = None
    llm_tokenizer: HardwareSnapshot | None = None
    llm_model_load: HardwareSnapshot | None = None
    llm_pipeline: HardwareSnapshot | None = None
    llm_inference: HardwareSnapshot | None = None
    tts_processor_load: HardwareSnapshot | None = None
    tts_model_load: HardwareSnapshot | None = None
    tts_inference: HardwareSnapshot | None = None

    def assign(self, phase_name: str, snapshot: HardwareSnapshot) -> bool:
        mapping = {
            "asr_model_load_gpu_metrics": "asr_model_load",
            "asr_inference_gpu_metrics": "asr_inference",
            "llm_tokenizer_gpu_metrics": "llm_tokenizer",
            "llm_model_load_gpu_metrics": "llm_model_load",
            "llm_pipeline_gpu_metrics": "llm_pipeline",
            "llm_inference_gpu_metrics": "llm_inference",
            "tts_processor_load_gpu_metrics": "tts_processor_load",
            "tts_model_load_gpu_metrics": "tts_model_load",
            "tts_inference_gpu_metrics": "tts_inference",
        }
        attribute = mapping.get(phase_name)
        if attribute is None:
            return False
        setattr(self, attribute, snapshot)
        return True

    def to_dict(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        if self.asr_model_load is not None:
            payload["asr_model_load"] = self.asr_model_load.to_dict()
        if self.asr_inference is not None:
            payload["asr_inference"] = self.asr_inference.to_dict()
        if self.llm_tokenizer is not None:
            payload["llm_tokenizer"] = self.llm_tokenizer.to_dict()
        if self.llm_model_load is not None:
            payload["llm_model_load"] = self.llm_model_load.to_dict()
        if self.llm_pipeline is not None:
            payload["llm_pipeline"] = self.llm_pipeline.to_dict()
        if self.llm_inference is not None:
            payload["llm_inference"] = self.llm_inference.to_dict()
        if self.tts_processor_load is not None:
            payload["tts_processor_load"] = self.tts_processor_load.to_dict()
        if self.tts_model_load is not None:
            payload["tts_model_load"] = self.tts_model_load.to_dict()
        if self.tts_inference is not None:
            payload["tts_inference"] = self.tts_inference.to_dict()
        return payload


@dataclass
class HardwareRecord:
    """Typed hardware record with trial-level and per-phase views."""

    trial_snapshot: HardwareSnapshot | None = None
    phases: HardwarePhaseRecord = field(default_factory=HardwarePhaseRecord)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.trial_snapshot is not None:
            payload.update(self.trial_snapshot.to_dict())
        phase_payload = self.phases.to_dict()
        if phase_payload:
            payload["hardware_phase_metrics"] = phase_payload
        return payload


@dataclass
class TrialRecord:
    """Explicit trial data for collector state."""

    trial_id: str
    sample_id: str | None
    is_warmup: bool
    status: str = "running"
    error: str | None = None
    trial_wall_time_seconds: float | None = None
    latencies: LatencyRecord = field(default_factory=LatencyRecord)
    quality: QualityRecord = field(default_factory=QualityRecord)
    tokens: TokenRecord = field(default_factory=TokenRecord)
    lifecycle: LifecycleRecord = field(default_factory=LifecycleRecord)
    hardware: HardwareRecord = field(default_factory=HardwareRecord)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trial_id": self.trial_id,
            "sample_id": self.sample_id,
            "is_warmup": self.is_warmup,
            "status": self.status,
            "error": self.error,
        }
        if self.trial_wall_time_seconds is not None:
            payload["trial_wall_time_seconds"] = self.trial_wall_time_seconds

        payload.update(self.latencies.to_dict())
        payload.update(self.quality.to_dict())
        payload.update(self.tokens.to_dict())
        payload.update(self.lifecycle.to_dict())
        payload.update(self.hardware.to_dict())
        return payload
