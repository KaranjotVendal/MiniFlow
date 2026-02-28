"""Shared pytest configuration and fixtures."""

import pytest
import torch


def _is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available for torchaudio operations.

    Returns:
        True if FFmpeg libraries are available, False otherwise.
    """
    try:
        # Try to use torchaudio save which requires FFmpeg
        from io import BytesIO

        import torchaudio

        waveform = torch.zeros(16000)
        buffer = BytesIO()
        torchaudio.save(buffer, waveform.unsqueeze(0), 16000, format="wav")
        return True
    except Exception:
        return False


def _is_nvml_available() -> bool:
    """Check if NVML (NVIDIA Management Library) is available.

    Returns:
        True if NVML is available, False otherwise.
    """
    try:
        import nvitop

        nvitop.Device(index=0)
        return True
    except Exception:
        return False


# pytest markers for conditional test skipping
requires_ffmpeg = pytest.mark.skipif(
    not _is_ffmpeg_available(),
    reason="FFmpeg not available - required for audio processing",
)

requires_nvml = pytest.mark.skipif(
    not _is_nvml_available(),
    reason="NVML not available - required for GPU hardware metrics",
)
