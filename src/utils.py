import torch

from src.logger.logging import initialise_logger
from src.prepare_data import stream_dataset_samples

logger = initialise_logger(__name__)


def get_test_sample():
    for sample in stream_dataset_samples(num_samples=1):
        return sample
    raise ValueError("No sample streamed from dataset")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available else "cpu")
