def get_default_metrics_config() -> dict:
    """Get default metrics configuration.

    Returns:
        Dictionary with default metric settings.
    """
    return {
        "enabled": [],
        "configurations": {},
    }


def get_default_hardware_config() -> dict:
    """Get default hardware metrics configuration."""
    return {
        "device": 0,
        "track_power": False,
        "track_fragmentation": False,
        "waste_threshold": 0.3,
    }


def get_default_timing_config() -> dict:
    """Get default timing metrics configuration."""
    return {
        "stages": ["asr", "llm", "tts", "pipeline"],
    }


def get_default_tokens_config() -> dict:
    """Get default token metrics configuration."""
    return {
        "track_ttft": True,
    }


def get_default_quality_config() -> dict:
    """Get default quality metrics configuration."""
    return {
        "evaluators": ["wer", "utmos"],
    }
