import copy

# =============================================================================
# TRANSFORMERS 4.57.x COMPATIBILITY: NEW IMPORTS
# =============================================================================
# Required for DynamicCache format conversion (transformers 4.51 → 4.57)
from transformers.cache_utils import DynamicCache, DynamicLayer

from src.logger.logging import initialise_logger


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
