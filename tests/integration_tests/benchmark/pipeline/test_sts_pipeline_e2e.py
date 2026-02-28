"""
End-to-End Integration Tests for STS Pipeline.

These tests run the actual STS pipeline (ASR -> LLM -> TTS) with minimal configuration.
They test the real code path without mocks.

Note: These tests require GPU and may take several minutes to run.
Use: pytest -m integration to run only these tests.
"""

import pytest
import torch

from src.benchmark.collectors import BenchmarkCollector
from src.benchmark.metrics.hardware import HardwareMetrics
from src.benchmark.metrics.lifecycle import ModelLifecycleMetrics
from src.benchmark.metrics.quality import QualityMetrics
from src.benchmark.metrics.timing import TimingMetrics
from src.benchmark.metrics.tokens import TokenMetrics
from src.prepare_data import AudioSample, stream_dataset_samples
from src.sts_pipeline import process_sample


class TestSTSPipelineE2E:
    """End-to-end integration tests for the STS pipeline."""

    @pytest.fixture
    def minimal_config(self) -> dict:
        """Minimal configuration for E2E testing."""
        return {
            "asr": {
                "model_id": "openai/whisper-small",  # Smallest Whisper model
                "model_name": "whisper-small",
            },
            "llm": {
                # Use smallest Qwen model for fast testing
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_name": "Qwen2.5-0.5B",
                "quantization": {
                    "enabled": True,
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "use_double_quant": True,
                    "compute_dtype": "bf16",
                },
                "kv_cache": False,
                "max_new_tokens": 20,  # Minimal tokens for fast testing
            },
            "tts": {
                "model_name": "vibevoice",
                "model_id": "microsoft/VibeVoice-Realtime-0.5B",
                "cfg_scale": 1.5,
                "voice_name": "en-Emma_woman",
            },
            "benchmark": {
                "enable_streaming_audio": False,
            },
        }

    @pytest.fixture
    def sample_audio(self) -> AudioSample:
        """Create a simple test audio sample."""
        # Generate a simple sine wave
        duration = 2.0  # 2 seconds
        sampling_rate = 16000
        t = torch.linspace(0, duration, int(sampling_rate * duration))
        # Simple speech-like signal (multiple frequencies)
        audio = (
            0.3 * torch.sin(2 * 440 * 3.14159 * t)  # 440 Hz
            + 0.2 * torch.sin(2 * 880 * 3.14159 * t)  # 880 Hz
            + 0.1 * torch.sin(2 * 1320 * 3.14159 * t)  # 1320 Hz
        )
        audio_tensor = audio.unsqueeze(0)  # Add channel dimension

        return AudioSample(
            audio_tensor=audio_tensor,
            transcript="Hello world",
            accent="US English",
            duration=duration,
            sample_to_noise_ratio=30.0,
            utmos=4.0,
            sampling_rate=sampling_rate,
        )

    @pytest.fixture
    def collector(self) -> BenchmarkCollector:
        """Create a real BenchmarkCollector with minimal metrics."""
        metrics = {
            "timing": TimingMetrics({"stages": ["asr", "llm", "tts", "pipeline"]}),
            "hardware": HardwareMetrics({"device": 0, "track_power": False}),
            "lifecycle": ModelLifecycleMetrics({}),
            "quality": QualityMetrics({"evaluators": []}),  # Skip quality eval for speed
            "tokens": TokenMetrics({"track_ttft": False}),
        }

        config = {
            "benchmark": {"enable_streaming_audio": False},
            "dataset": {"num_samples": 1, "warmup_samples": 0},
        }

        return BenchmarkCollector(metrics=metrics, config=config)

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_sts_pipeline_single_sample(self, minimal_config, sample_audio, collector):
        """Test running the full STS pipeline on a single sample."""
        device = torch.device("cuda:0")

        collector.start_trial(
            trial_id="e2e_test_1",
            sample_id="e2e_test_1",
            is_warmup=False,
        )

        try:
            result = process_sample(
                config=minimal_config,
                sample=sample_audio,
                run_id="e2e_test",
                collector=collector,
                device=device,
            )

            # Verify result
            assert result is not None
            assert result.groundtruth == "Hello world"
            assert result.asr_transcript is not None
            assert result.llm_response is not None
            assert result.tts_waveform is not None
            assert result.tts_waveform_output_sr > 0

            # Get trial metrics
            trial_metrics = collector.end_trial(status="success")

            # Verify metrics were collected
            assert trial_metrics["status"] == "success"
            assert "latencies" in trial_metrics

            print(f"\nASR transcript: {result.asr_transcript}")
            print(f"LLM response: {result.llm_response}")
            print(f"TTS waveform shape: {result.tts_waveform.shape}")
            print(f"TTS sample rate: {result.tts_waveform_output_sr}")

        except Exception as e:
            collector.end_trial(status="error", error=str(e))
            raise

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_sts_pipeline_with_dataset_sample(self, minimal_config, collector):
        """Test running STS pipeline with a real dataset sample."""
        device = torch.device("cuda:0")

        # Get one sample from dataset
        samples = list(stream_dataset_samples(num_samples=1, split="test"))
        sample = samples[0]

        collector.start_trial(
            trial_id="e2e_dataset_test",
            sample_id="e2e_dataset_test",
            is_warmup=False,
        )

        try:
            result = process_sample(
                config=minimal_config,
                sample=sample,
                run_id="e2e_dataset_test",
                collector=collector,
                device=device,
            )

            # Verify result
            assert result is not None
            assert result.groundtruth == sample.transcript
            assert result.asr_transcript is not None
            assert result.llm_response is not None

            trial_metrics = collector.end_trial(status="success")
            assert trial_metrics["status"] == "success"

            print(f"\nDataset sample transcript: {sample.transcript}")
            print(f"ASR output: {result.asr_transcript}")
            print(f"LLM response: {result.llm_response[:100]}...")

        except Exception as e:
            collector.end_trial(status="error", error=str(e))
            raise
