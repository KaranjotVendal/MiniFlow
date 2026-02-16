from src.benchmark.collectors import BenchmarkCollector
import torch


def run_cosyvoice(config: dict, llm_response: str, device: str | torch.device, collector: BenchmarkCollector) -> tuple[torch.Tensor, float, int, float, float]:
    """CosyVoice TTS implementation - placeholder."""
    raise NotImplementedError("CosyVoice not yet implemented. Please use 'xtts' or 'vibevoice'.")
