import torch
from src.prepare_data import stream_dataset_samples
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def get_test_sample():
    for sample in stream_dataset_samples(num_samples=1):
        return sample
    raise ValueError("No sample streamed from dataset")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available else "cpu")


def get_gpu_util() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    logger.info("No cuda available")
    # TODO: implement cpu util
    return None


def clear_gpu_cache():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
