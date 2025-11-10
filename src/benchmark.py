import time

from src.baseline_pipeline import process_sample, ProcessedSample
from src.prepare_data import stream_dataset_samples
from src.metrics import log_metrics
from src.utils import clear_gpu_cache
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def run_benchmark(
    num_samples: int = 10, run_id: str = "benchmark", folder: str = "benchmark"
):
    results = []
    total_start = time.time()

    for i, sample in enumerate(stream_dataset_samples(num_samples=num_samples)):
        processed_sample, metrics = process_sample(
            sample, run_id=f"metrics_{run_id}_{i + 1}", folder=folder
        )
        results.append((processed_sample, metrics))

    # Averages
    n = len(results)
    avg_wer = sum(m.asr_wer for r, m in results) / n
    avg_mos = sum(m.tts_utmos for r, m in results) / n

    avg_total_latency = sum(m.total_latency for r, m in results) / n
    avg_asr_latency = sum(m.asr_latency for r, m in results) / n
    avg_llm_latency = sum(m.llm_latency for r, m in results) / n
    avg_tts_latency = sum(m.tts_latency for r, m in results) / n

    avg_asr_gpu = sum(m.asr_gpu_util for r, m in results) / n
    avg_llm_gpu = sum(m.llm_gpu_util for r, m in results) / n
    avg_tts_gpu = sum(m.tts_gpu_util for r, m in results) / n

    total_latency = time.time() - total_start

    summary = {
        "avg_asr_wer": avg_wer,
        "avg_mos": avg_mos,
        "avg_total_latency": avg_total_latency,
        "avg_asr_latency": avg_asr_latency,
        "avg_llm_latency": avg_llm_latency,
        "avg_tts_latency": avg_tts_latency,
        "total_latency": total_latency,
        "avg_asr_gpu": avg_asr_gpu,
        "avg_llm_gpu": avg_llm_gpu,
        "avg_tts_gpu": avg_tts_gpu,
    }

    log_metrics(folder="./Benchmark", run_id="avg_10", metrics=summary)
    logger.info(f"Benchmark Summary: {summary}")
    return summary


if __name__ == "__main__":
    summary = run_benchmark(num_samples=10)
