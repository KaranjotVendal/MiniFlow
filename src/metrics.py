from dataclasses import dataclass
from pathlib import Path
import json
import time

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@dataclass
class Metrics:
    asr_wer: float
    tts_utmos: float
    asr_latency: float
    llm_latency: float
    tts_latency: float
    total_latency: float
    asr_gpu_util: float
    llm_gpu_util: float
    tts_gpu_util: float

    def to_dict(self):
        return {
            "asr_wer": self.asr_wer,
            "tts_utmos": self.tts_utmos,
            "asr_latency": self.asr_latency,
            "llm_latency": self.llm_latency,
            "tts_latency": self.tts_latency,
            "total_latency": self.total_latency,
            "asr_gpu_util": self.asr_gpu_util,
            "llm_gpu_util": self.llm_gpu_util,
            "tts_gpu_util": self.tts_gpu_util,
        }


def log_metrics(run_id: str, metrics: Metrics, folder: str = "./") -> None:
    filename = Path(f"{folder}/{run_id}.json")
    filename.parent.mkdir(exist_ok=True, parents=True)
    metrics_dict = metrics.to_dict()
    metrics_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics_dict["run_id"] = run_id
    with open(filename, "a") as f:
        json.dump(metrics_dict, f)
        f.write("\n")
        logger.info(f"saved the metrics json at {filename}")
