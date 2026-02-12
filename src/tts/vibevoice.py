from typing import TYPE_CHECKING
import copy
from pathlib import Path

import torch

from src.utils import clear_gpu_cache
from src.logger.logging import initialise_logger
from src.tts.vibevoice_compatibility_utils import _fix_prefilled_outputs
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)


def _torch_compatibility_fix(model) -> None:
    # =============================================================================
    # TRANSFORMERS 4.57.x COMPATIBILITY FIX
    # =============================================================================
    # FIX: Add num_hidden_layers attribute to config for DynamicCache compatibility
    # This line is needed because transformers 4.57.x requires num_hidden_layers
    # but VibeVoiceStreamingConfig (originally for 4.x) doesn't expose it.
    # See: VIBEVOICE_FIXES_README.md
    if not hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = model.config.decoder_config.num_hidden_layers


def _device_attn_implementation(device: torch.device | str) -> tuple[str, torch.dtype, str]:
    # Determine dtype and attention implementation based on device
    device_str = str(device).split(":")[0] if ":" in str(device) else str(device)

    if device_str == "cuda":
        load_dtype = torch.bfloat16
        attn_impl = "flash_attention_2"
    elif device_str == "mps":
        load_dtype = torch.float32
        attn_impl = "sdpa"
    else:  # cpu
        load_dtype = torch.float32
        attn_impl = "sdpa"

    return device_str, load_dtype, attn_impl

def _handle_speaker_voice(config: dict, device_str: str) -> dict:
    # Speaker handling - use config-based path or default
    voice_name = config.get("voice_name", "en-Emma_woman")
    voice_base_dir = Path(__file__).parent.parent.parent / "vibevoice/demo/voices/streaming_model"
    voice_path = voice_base_dir / f"{voice_name}.pt"

    if not voice_path.exists():
        # Try alternative path from config or raise error
        if "voice_path" in config:
            voice_path = Path(config["voice_path"])
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found at: {voice_path}")

    all_prefilled_outputs = torch.load(voice_path, map_location=device_str, weights_only=False)
    # =============================================================================
    # TRANSFORMERS 4.57.x COMPATIBILITY FIX
    # =============================================================================
    # Fix old DynamicCache format issues from torch.load
    # Voice files created with transformers 4.51.3 must be converted to 4.57.3 format
    # This handles the key_cache → layers migration
    # See: VIBEVOICE_FIXES_README.md
    all_prefilled_outputs = _fix_prefilled_outputs(all_prefilled_outputs)

    return all_prefilled_outputs


def run_vibevoice(config: dict,
    llm_response: str,
    device: str | torch.device,
    collector: "BenchmarkCollector"
) -> tuple[torch.Tensor, int]:
    """Run VibeVoice realtime TTS model.

    Args:
        config: Config dict with model_id, cfg_scale, voice_name, etc.
        llm_response: Text to synthesize
        device: Target device for model

    Returns:
        (waveform, latency_sec, sample_rate, utmos_score, gpu_peak_mem_mb)
    """
    logger.info("executing VibeVoice TTS model")

    device_str, load_dtype, attn_impl = _device_attn_implementation(device)
    logger.info(f"Using dtype: {load_dtype}, attn_implementation: {attn_impl}")

    # Load Processor
    collector.hardware_metrics.start(collector.context)
    collector.lifecycle_metrics.record_load_start(
        model_name="vibevoice_processor",source="remote(HF)")

    processor = VibeVoiceStreamingProcessor.from_pretrained(config["model_id"])

    processor_load_event = collector.lifecycle_metrics.record_load_end(cached=False)
    collector.record_phase_metrics(
        "tts_processor_load_gpu_metrics",
        collector.hardware_metrics.end(collector.context),
    )
    if processor_load_event is not None:
        processor_load_event["stage"] = "tts"
        processor_load_event["cached"] = False
        processor_load_event["success"] = True

    # Load model with proper device handling
    try:

        collector.hardware_metrics.start(collector.context)
        collector.lifecycle_metrics.record_load_start(
            model_name=config["model_name"],source="remote(HF)")

        if device_str == "mps":
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                config["model_id"],
                dtype=load_dtype,
                attn_implementation=attn_impl,
                device_map=None,
            )
            model.to(device_str)
        else:  # cuda or cpu
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                config["model_id"],
                dtype=load_dtype,
                device_map=device_str,
                attn_implementation=attn_impl,
            )

        model_load_event = collector.lifecycle_metrics.record_load_end(cached=False)
        collector.record_phase_metrics(
            "tts_model_load_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )
        if model_load_event is not None:
            model_load_event["stage"] = "tts"
            model_load_event["cached"] = False
            model_load_event["success"] = True

    except Exception as e:
        if attn_impl == 'flash_attention_2':
            logger.warning(f"Flash attention failed, falling back to SDPA: {e}")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                config["model_id"],
                dtype=load_dtype,
                device_map=device_str,
                attn_implementation='sdpa'
            )
            model_load_event = collector.lifecycle_metrics.record_load_end(cached=False)
            collector.record_phase_metrics(
                "tts_model_load_gpu_metrics",
                collector.hardware_metrics.end(collector.context),
            )
            if model_load_event is not None:
                model_load_event["stage"] = "tts"
                model_load_event["cached"] = False
                model_load_event["success"] = True
        else:
            model_load_event = collector.lifecycle_metrics.record_load_end(cached=False)
            if model_load_event is not None:
                model_load_event["stage"] = "tts"
                model_load_event["cached"] = False
                model_load_event["success"] = False
                model_load_event["error_type"] = "TTSLoadOrInferenceError"
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    _torch_compatibility_fix(model)

    all_prefilled_outputs = _handle_speaker_voice(config, device_str)

    inputs = processor.process_input_with_cached_prompt(
        text=llm_response,
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device_str)

    # inference
    collector.hardware_metrics.start(collector.context)
    collector.timing_metrics.record_stage_start("tts_inference_latency")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=config["cfg_scale"],
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=True,
            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs)
        )

    collector.timing_metrics.record_stage_end("tts_inference_latency")
    collector.record_phase_metrics(
        "tts_inference_gpu_metrics",
        collector.hardware_metrics.end(collector.context),
    )

    tts_waveform = outputs.speech_outputs[0].squeeze().cpu()
    out_sample_rate = 24000

    # Calculate UTMOS score
    utmos_score: dict[str, float] = collector.quality_metrics.evaluate(evaluator="utmos",
        prediction=tts_waveform, output_sample_rate=out_sample_rate)
    collector.current_trial.quality.utmos = utmos_score["utmos"]

    # Cleanup
    del processor, model, inputs
    clear_gpu_cache()

    logger.info(f"VibeVoice: Generated {len(tts_waveform)/out_sample_rate:.2f}s of audio")
    return tts_waveform, out_sample_rate
