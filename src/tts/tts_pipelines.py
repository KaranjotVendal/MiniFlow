from typing import TYPE_CHECKING

import torch

from src.tts.vibevoice import run_vibevoice
from src.tts.cosyvoice import run_cosyvoice
from src.tts.xtts import run_xtts
from src.logger.logging import initialise_logger
if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)


def run_tts(
    config: dict,
    llm_response: str,
    device: torch.device | str,
    collector: "BenchmarkCollector",
) -> tuple[torch.Tensor, int]:
    logger.info("executing TTS model")

    if config["model_name"] == "xtts":
        tts_waveform, output_sample_rate, = (
            run_xtts(config=config, llm_response=llm_response, device=device, collector=collector)
        )
    elif config["model_name"] == "vibevoice":
        tts_waveform, output_sample_rate = (
            run_vibevoice(config=config, llm_response=llm_response, device=device, collector=collector)
        )
    elif config["model_name"] == "cosyvoice":
        raise NotImplementedError("CosyVoice not yet implemented. Please use 'xtts' or 'vibevoice'.")

    else:
        raise ValueError(f"No valid TTS model name found. {config['model_name']}")


    return tts_waveform, output_sample_rate
