from typing import TYPE_CHECKING

import torch
from transformers import pipeline

from src.logger.logging import initialise_logger
from src.utils import clear_gpu_cache

if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)


def run_asr(
    config: dict,
    audio_tensor: torch.Tensor,
    sampling_rate: int,
    groundtruth: str,
    collector: "BenchmarkCollector",
    device: torch.device | str,
) -> str:
    """Run ASR and record stage/lifecycle/hardware metrics through collector."""
    logger.info("executing STT model")

    stage_name = "asr"
    load_closed = False
    pipe = None
    audio = {"array": audio_tensor.squeeze().numpy(), "sampling_rate": sampling_rate}

    # Try with requested device first, fall back to CPU on CUDA errors
    cuda_failed = False

    try:
        collector.hardware_metrics.start(collector.context)
        collector.lifecycle_metrics.record_load_start(
            model_name=config["model_name"],
            source="remote(HF)",
        )
        # Question: we are loading the whole pipeline here instead of model and processor.
        # should we load them separately?
        pipe = pipeline(
            "automatic-speech-recognition",
            model=config["model_id"],
            device=device,
            return_timestamps=False,
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
                "forced_decoder_ids": None,
            },
        )
        load_event = collector.lifecycle_metrics.record_load_end(cached=False)
        collector.record_phase_metrics(
            "asr_model_load_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )
        # Question: why are we returning load_event and modifying it?
        # we do not return it anywhere or use it anywhere??
        if load_event is not None:
            load_event["stage"] = stage_name
            load_event["cached"] = False
            load_event["success"] = True
        load_closed = True

        collector.hardware_metrics.start(collector.context)
        collector.timing_metrics.record_stage_start("asr_inference_latency")
        try:
            pred = pipe(audio)
        except RuntimeError as e:
            # Check if it's a CUDA error and we haven't tried CPU yet
            if "CUDA" in str(e) and device != "cpu" and not cuda_failed:
                logger.warning(f"CUDA error in ASR inference: {e}. Retrying with CPU.")
                cuda_failed = True
                # Recreate pipeline on CPU
                del pipe
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=config["model_id"],
                    device="cpu",
                    return_timestamps=False,
                    generate_kwargs={
                        "language": "en",
                        "task": "transcribe",
                        "forced_decoder_ids": None,
                    },
                )
                # Update audio to CPU format if needed
                audio = {
                    "array": audio_tensor.squeeze().cpu().numpy(),
                    "sampling_rate": sampling_rate,
                }
                pred = pipe(audio)
                device = "cpu"
            else:
                raise
        collector.timing_metrics.record_stage_end("asr_inference_latency")
        collector.record_phase_metrics(
            "asr_inference_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )

        transcription = pred["text"]
        wer_score: dict[str, float] = collector.quality_metrics.evaluate(
            evaluator="wer", prediction=transcription, reference=groundtruth
        )
        collector.current_trial.quality.wer = wer_score["wer"]

        return transcription

    except Exception:
        if not load_closed:
            load_event = collector.lifecycle_metrics.record_load_end(cached=False)
            if load_event is not None:
                load_event["stage"] = stage_name
                load_event["cached"] = False
                load_event["success"] = False
                load_event["error_type"] = "ASRLoadOrInferenceError"
        raise
    finally:
        if pipe is not None:
            del pipe
        clear_gpu_cache()
