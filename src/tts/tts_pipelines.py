import tempfile
import time
from unittest.mock import patch
import copy
from pathlib import Path

import torch
import utmosv2
from TTS.api import TTS
from transformers import AutoModel, AutoProcessor
# =============================================================================
# TRANSFORMERS 4.57.x COMPATIBILITY: NEW IMPORTS
# =============================================================================
# Required for DynamicCache format conversion (transformers 4.51 → 4.57)
from transformers.cache_utils import DynamicCache, DynamicLayer

from src.benchmark.gpu_sampler import reset_peak_memory, get_gpu_memory_peak
from src.utils import clear_gpu_cache
from src.logger.logging import initialise_logger
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

logger = initialise_logger(__name__)


# =============================================================================
# TRANSFORMERS 4.57.x COMPATIBILITY FIXES
# =============================================================================
# VibeVoice was originally built for transformers==4.51.3.
# These functions bridge the gap when running transformers==4.57.3
# See VIBEVOICE_FIXES_README.md for full context.


def _fix_dynamic_cache_format(cache):
    """
    TRANSFORMERS 4.57.x COMPATIBILITY FIX:
    Converts old DynamicCache format to new format.

    The Problem:
    - Voice files (.pt) saved with transformers 4.x use: key_cache/value_cache
    - Transformers 4.57.x expects: layers attribute with DynamicLayer objects
    - This causes AttributeError: 'DynamicCache' object has no attribute 'layers'

    The Solution:
    - Detect old format (has key_cache/value_cache but no layers)
    - Create new DynamicCache with DynamicLayer objects
    - Migrate all cached tensors to new structure

    Voice files were created with transformers 4.51.3 but we're running 4.57.3
    """
    if not isinstance(cache, DynamicCache):
        return cache

    # Check if it's already in the new format (transformers 4.57.x compatible)
    if hasattr(cache, 'layers'):
        return cache

    # Old format has key_cache and value_cache attributes (transformers 4.x)
    if not (hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache')):
        raise ValueError(f"Unknown cache format: {cache}")

    # Create a new DynamicCache in new format
    new_cache = DynamicCache()

    # Convert old format to new format by creating DynamicLayer objects
    num_layers = len(cache.key_cache)
    for layer_idx in range(num_layers):
        layer = DynamicLayer()
        # Initialize layer with the data
        key_states = cache.key_cache[layer_idx]
        value_states = cache.value_cache[layer_idx]
        layer.lazy_initialization(key_states)
        layer.update(key_states, value_states)
        new_cache.layers.append(layer)

    return new_cache


def _fix_prefilled_outputs(all_prefilled_outputs):
    """
    TRANSFORMERS 4.57x COMPATIBILITY FIX:
    Recursively fixes all DynamicCache objects in loaded voice file data.

    Voice files contain prefilled outputs for:
    - lm (base language model cache)
    - tts_lm (TTS language model cache)
    - neg_lm (negative prompt cache)
    - neg_tts_lm (TTS negative prompt cache)

    Each may contain DynamicCache objects that need format conversion.
    """
    fixed = {}
    for key, value in all_prefilled_outputs.items():
        if hasattr(value, 'past_key_values') and value.past_key_values is not None:
            if isinstance(value.past_key_values, DynamicCache):
                fixed_cache = _fix_dynamic_cache_format(value.past_key_values)
                # Create new ModelOutput with fixed cache
                new_value = copy.deepcopy(value)
                new_value.past_key_values = fixed_cache
                fixed[key] = new_value
            else:
                fixed[key] = value
        else:
            fixed[key] = value
    return fixed


def utmos_model(waveform: torch.Tensor, output_sampling_rate: int) -> float:
    import torchaudio

    _org_dataloader = torch.utils.data.DataLoader

    def dataloader_no_workers(*args, **kwargs):
        """Patch dataloader to disable multiprocessing (avoids pickling error)"""
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return _org_dataloader(*args, **kwargs)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_f:
        temp_tts_wav = Path(tts_f.name)
    torchaudio.save(
        temp_tts_wav, torch.tensor(waveform).unsqueeze(0), output_sampling_rate
    )

    utmos_model = utmosv2.create_model(pretrained=True)

    # MOS scoring
    with patch("torch.utils.data.DataLoader", side_effect=dataloader_no_workers):
        mos = utmos_model.predict(input_path=temp_tts_wav)
    temp_tts_wav.unlink()
    # reference_wav.unlink()

    return mos


def run_xtts(
    config: dict,
    llm_response: str,
    device
    # reference_audio_tensor: torch.Tensor,
    # reference_sampling_rate: int,
) -> tuple[str, float, int, float, float]:
    # reference_wav = Path("./temp_reference_wav.wav")
    # torchaudio.save(reference_wav, reference_audio_tensor, reference_sampling_rate)

    reset_peak_memory()
    # model_id = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(config["model_id"]).to(device)

    start = time.time()
    # NOTE: can synthesize with the the original speaker's voice as well (need to test it)
    waveform = tts.tts(
        text=llm_response, speaker=config["speaker"], language=config["language"]
    )
    latency = time.time() - start
    output_sampling_rate: int = tts.synthesizer.output_sample_rate

    tts_gpu_util = get_gpu_memory_peak()

    mos = utmos_model(waveform, output_sampling_rate)

    del tts

    # clean up
    clear_gpu_cache()
    # torch.cuda.empty_cache()
    return waveform, latency, output_sampling_rate, mos, tts_gpu_util



def run_vibevoice(config: dict, llm_response: str, device: str | torch.device) -> tuple[torch.Tensor, float, int, float, float]:
    """Run VibeVoice realtime TTS model.

    Args:
        config: Config dict with model_id, cfg_scale, voice_name, etc.
        llm_response: Text to synthesize
        device: Target device for model

    Returns:
        (waveform, latency_sec, sample_rate, utmos_score, gpu_peak_mem_mb)
    """
    logger.info("executing VibeVoice TTS model")
    logger.info(f"DEVICE: {device}")

    reset_peak_memory()

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

    logger.info(f"Using dtype: {load_dtype}, attn_implementation: {attn_impl}")

    # Load Processor and Model
    processor = VibeVoiceStreamingProcessor.from_pretrained(config["model_id"])

    # Load model with proper device handling
    try:
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
    except Exception as e:
        if attn_impl == 'flash_attention_2':
            logger.warning(f"Flash attention failed, falling back to SDPA: {e}")
            model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                config["model_id"],
                dtype=load_dtype,
                device_map=device_str,
                attn_implementation='sdpa'
            )
        else:
            raise

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    # =============================================================================
    # TRANSFORMERS 4.57.x COMPATIBILITY FIX
    # =============================================================================
    # FIX: Add num_hidden_layers attribute to config for DynamicCache compatibility
    # This line is needed because transformers 4.57.x requires num_hidden_layers
    # but VibeVoiceStreamingConfig (originally for 4.x) doesn't expose it.
    # See: VIBEVOICE_FIXES_README.md
    if not hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = model.config.decoder_config.num_hidden_layers

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

    # Generate
    start = time.time()
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
    latency = time.time() - start

    tts_waveform = outputs.speech_outputs[0].squeeze().cpu()
    out_sample_rate = 24000
    tts_gpu_peak_mem = get_gpu_memory_peak()

    # Calculate UTMOS score
    mos = utmos_model(tts_waveform, out_sample_rate)

    # Cleanup
    del processor, model, inputs
    clear_gpu_cache()

    logger.info(f"VibeVoice: Generated {len(tts_waveform)/out_sample_rate:.2f}s of audio in {latency:.2f}s")

    return tts_waveform, latency, out_sample_rate, mos, tts_gpu_peak_mem


def run_cosyvoice(config: dict, llm_response: str, device: str | torch.device) -> tuple[torch.Tensor, float, int, float, float]:
    """CosyVoice TTS implementation - placeholder."""
    raise NotImplementedError("CosyVoice not yet implemented. Please use 'xtts' or 'vibevoice'.")


def run_tts(
    config: dict, llm_response: str, device
) -> tuple[torch.Tensor, float, int, float, float]:
    logger.info("executing TTS model")
    logger.info(f"DEVICE: {device}")

    if config["model_name"] == "xtts":
        tts_waveform, tts_latency, output_sample_rate, mos, tts_gpu_peak_mem = (
            run_xtts(config, llm_response, device)
        )
    elif config["model_name"] == "cosyvoice":
        tts_waveform, tts_latency, output_sample_rate, mos, tts_gpu_peak_mem = (
            run_cosyvoice(config, llm_response, device)
        )
    elif config["model_name"] == "vibevoice":
        tts_waveform, tts_latency, output_sample_rate, mos, tts_gpu_peak_mem = (
            run_vibevoice(config, llm_response, device)
        )
    else:
        raise f"No valid TTS model name found. {config['model_name']}"

    return tts_waveform, tts_latency, output_sample_rate, mos, tts_gpu_peak_mem
