from typing import TYPE_CHECKING

import torch
from TTS.api import TTS

from src.utils import clear_gpu_cache
from src.logger.logging import initialise_logger

if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)


def run_xtts(
    config: dict,
    llm_response: str,
    device: torch.device | str,
    collector: "BenchmarkCollector",
    # reference_audio_tensor: torch.Tensor,
    # reference_sampling_rate: int,
) -> tuple[list, float]:
    # reference_wav = Path("./temp_reference_wav.wav")
    # torchaudio.save(reference_wav, reference_audio_tensor, reference_sampling_rate)

    # load model
    collector.hardware_metrics.start(collector.context)
    collector.lifecycle_metrics.record_load_start(
        model_name=config["model_name"],source="remote(HF)")

    model = TTS(config["model_id"]).to(device)

    load_event = collector.lifecycle_metrics.record_load_end(cached=False)
    collector.record_phase_metrics(
        "tts_model_load_gpu_metrics",
        collector.hardware_metrics.end(collector.context),
    )
    if load_event is not None:
        load_event["stage"] = "tts"
        load_event["cached"] = False
        load_event["success"] = True

    # inference
    # NOTE: can synthesize with the the original speaker's voice as well (need to test it)
    collector.hardware_metrics.start(collector.context)
    collector.timing_metrics.record_stage_start("tts_inference_latency")

    waveform: list = model.tts(
        text=llm_response, speaker=config["speaker"], language=config["language"]
    )

    collector.timing_metrics.record_stage_end("tts_inference_latency")
    collector.record_phase_metrics(
        "tts_inference_gpu_metrics",
        collector.hardware_metrics.end(collector.context),
    )

    # quality metric
    output_sampling_rate: int = model.synthesizer.output_sample_rate
    utmos_score: dict[str, float] = collector.quality_metrics.evaluate(evaluator="utmos",
        prediction=waveform, output_sample_rate=output_sampling_rate)
    collector.current_trial.quality.utmos = utmos_score["utmos"]


    del model
    clear_gpu_cache()
    return waveform, output_sampling_rate
