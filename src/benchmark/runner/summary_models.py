from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricStats:
    mean: float
    median: float
    min: float
    max: float
    std: float
    p95: float
    p99: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "p95": self.p95,
            "p99": self.p99,
            "count": self.count,
        }


@dataclass
class StatusCounts:
    success: int = 0
    error: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"success": self.success, "error": self.error}


@dataclass
class SummaryMeta:
    experiment: str
    run_id: str
    timestamp: str
    num_trials: int
    num_warmup_trials: int
    status_counts: StatusCounts

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "num_trials": self.num_trials,
            "num_warmup_trials": self.num_warmup_trials,
            "status_counts": self.status_counts.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SummaryMeta":
        counts = payload.get("status_counts", {})
        return cls(
            experiment=payload["experiment"],
            run_id=payload["run_id"],
            timestamp=payload["timestamp"],
            num_trials=payload["num_trials"],
            num_warmup_trials=payload["num_warmup_trials"],
            status_counts=StatusCounts(
                success=int(counts.get("success", 0)),
                error=int(counts.get("error", 0)),
            ),
        )


@dataclass
class SectionSummary:
    latency: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    load_times: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SectionSummary":
        return cls(
            latency=payload.get("latency", {}),
            hardware=payload.get("hardware", {}),
            load_times=payload.get("load_times", {}),
            inference=payload.get("inference", {}),
            quality=payload.get("quality", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": self.latency,
            "hardware": self.hardware,
            "load_times": self.load_times,
            "inference": self.inference,
            "quality": self.quality,
        }


@dataclass
class SummaryRecord:
    meta: SummaryMeta
    pipeline: SectionSummary
    asr: SectionSummary
    llm: SectionSummary
    tts: SectionSummary

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SummaryRecord":
        return cls(
            meta=SummaryMeta.from_dict(payload["meta"]),
            pipeline=SectionSummary.from_dict(payload["pipeline"]),
            asr=SectionSummary.from_dict(payload["asr"]),
            llm=SectionSummary.from_dict(payload["llm"]),
            tts=SectionSummary.from_dict(payload["tts"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "pipeline": self.pipeline.to_dict(),
            "asr": self.asr.to_dict(),
            "llm": self.llm.to_dict(),
            "tts": self.tts.to_dict(),
        }
