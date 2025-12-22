import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@dataclass
class Metrics:
    sample_id: int
    exp_name: str
    timestamp_start: float
    timestamp_end: float
    asr_wer: float
    tts_utmos: float
    asr_latency: float
    llm_latency: float
    tts_latency: float
    total_latency: float
    asr_gpu_peak_mem: float
    llm_gpu_peak_mem: float
    tts_gpu_peak_mem: float

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "exp_name": self.exp_name,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "asr_wer": self.asr_wer,
            "tts_utmos": self.tts_utmos,
            "asr_latency": self.asr_latency,
            "llm_latency": self.llm_latency,
            "tts_latency": self.tts_latency,
            "total_latency": self.total_latency,
            "asr_gpu_peak_mem": self.asr_gpu_peak_mem,
            "llm_gpu_peak_mem": self.llm_gpu_peak_mem,
            "tts_gpu_peak_mem": self.tts_gpu_peak_mem,
        }


def log_metrics(run_id: str, metrics: Metrics, folder: str = "./") -> None:
    filename = Path(f"{folder}/{run_id}.json")
    filename.parent.mkdir(exist_ok=True, parents=True)
    metrics_dict = asdict(metrics)  # .to_dict()
    metrics_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics_dict["run_id"] = run_id
    with open(filename, "a") as f:
        json.dump(metrics_dict, f)
        f.write("\n")
        logger.info(f"saved the metrics json at {filename}")


def log_jsonl(metrics_dict: dict, filepath: str):
    path = Path(filepath)
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics_dict) + "\n")
