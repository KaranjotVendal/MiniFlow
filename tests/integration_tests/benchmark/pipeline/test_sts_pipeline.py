"""
Integration tests for the STS (Speech-to-Speech) pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.prepare_data import AudioSample
from src.sts_pipeline import ProcessedSample, process_sample


class TestPipelineIntegration:
    """Integration tests for the full STS pipeline."""

    @pytest.fixture
    def sample_audio(self) -> AudioSample:
        """Create a sample audio input."""
        duration = 1.0
        sampling_rate = 16000
        t = torch.linspace(0, duration, int(sampling_rate * duration))
        audio_tensor = torch.sin(2 * 440 * 3.14159 * t).unsqueeze(0)

        return AudioSample(
            audio_tensor=audio_tensor,
            transcript="Hello world",
            accent="British English",
            duration=duration,
            sample_to_noise_ratio=30.0,
            utmos=4.0,
            sampling_rate=sampling_rate,
        )

    @pytest.fixture
    def mock_config(self) -> dict:
        """Create a mock configuration for the pipeline."""
        return {
            "asr": {"model_id": "openai/whisper-small", "model_name": "whisper-small"},
            "llm": {
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_name": "Qwen2.5-0.5B",
                "quantization": {
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "use_double_quant": True,
                    "compute_dtype": "bf16",
                },
                "kv_cache": False,
                "max_new_tokens": 50,
            },
            "tts": {
                "model_name": "vibevoice",
                "model_id": "microsoft/VibeVoice-Realtime-0.5B",
                "cfg_scale": 1.5,
                "voice_name": "en-Emma_woman",
            },
            "benchmark": {"enable_streaming_audio": False},
        }

    @patch("src.sts_pipeline.run_tts")
    @patch("src.sts_pipeline.run_llm")
    @patch("src.sts_pipeline.run_asr")
    def test_full_pipeline_with_mocks(
        self, mock_run_asr, mock_run_llm, mock_run_tts, sample_audio, mock_config
    ):
        """Test the full pipeline with mocked model components."""
        mock_run_asr.return_value = "transcribed text"
        mock_run_llm.return_value = (
            "mock response",
            [{"role": "system", "content": "You are helpful."}],
        )
        mock_run_tts.return_value = (torch.randn(1, 1000), 24000)

        mock_collector = MagicMock()
        mock_collector.context = MagicMock()
        mock_collector.current_trial = MagicMock()

        result = process_sample(
            config=mock_config,
            sample=sample_audio,
            run_id="test_run",
            collector=mock_collector,
            device="cpu",
        )

        assert isinstance(result, ProcessedSample)
        assert result.groundtruth == "Hello world"
        mock_run_asr.assert_called_once()
        mock_run_llm.assert_called_once()
        mock_run_tts.assert_called_once()

    def test_processed_sample_to_dict(self, sample_audio):
        """Test ProcessedSample serialization to dict."""
        waveform = torch.randn(1, 1000)

        processed = ProcessedSample(
            groundtruth="Hello world",
            asr_transcript="transcribed text",
            llm_response="mock response",
            tts_waveform=waveform,
            tts_waveform_output_sr=24000,
            new_history=[{"role": "system", "content": "You are helpful."}],
        )

        result_dict = processed.to_dict()

        assert result_dict["groundtruth"] == "Hello world"
        assert result_dict["asr_transcript"] == "transcribed text"
        assert result_dict["llm_response"] == "mock response"
        assert result_dict["tts_waveform_output_sr"] == 24000


class TestPipelineErrorHandling:
    """Integration tests for pipeline error handling."""

    @pytest.fixture
    def sample_audio(self) -> AudioSample:
        return AudioSample(
            audio_tensor=torch.randn(1, 16000),
            transcript="Test",
            accent="US English",
            duration=1.0,
            sample_to_noise_ratio=30.0,
            utmos=4.0,
            sampling_rate=16000,
        )

    @patch("src.sts_pipeline.run_asr")
    def test_pipeline_raises_on_asr_error(self, mock_run_asr, sample_audio):
        """Test that pipeline raises when ASR fails."""
        mock_run_asr.side_effect = RuntimeError("ASR model failed")

        mock_collector = MagicMock()
        mock_collector.context = MagicMock()

        mock_config = {
            "asr": {"model_id": "test", "model_name": "test"},
            "llm": {
                "model_id": "test",
                "model_name": "test",
                "quantization": {},
                "kv_cache": False,
                "max_new_tokens": 10,
            },
            "tts": {"model_name": "vibevoice", "model_id": "test"},
            "benchmark": {},
        }

        with pytest.raises(RuntimeError, match="ASR model failed"):
            process_sample(
                config=mock_config,
                sample=sample_audio,
                run_id="test",
                collector=mock_collector,
                device="cpu",
            )


class TestPipelineWithHistory:
    """Integration tests for pipeline with conversation history."""

    @pytest.fixture
    def sample_audio(self) -> AudioSample:
        return AudioSample(
            audio_tensor=torch.randn(1, 16000),
            transcript="Second question",
            accent="US English",
            duration=1.0,
            sample_to_noise_ratio=30.0,
            utmos=4.0,
            sampling_rate=16000,
        )

    @patch("src.sts_pipeline.run_tts")
    @patch("src.sts_pipeline.run_llm")
    @patch("src.sts_pipeline.run_asr")
    def test_pipeline_passes_history_to_llm(
        self, mock_run_asr, mock_run_llm, mock_run_tts, sample_audio
    ):
        """Test that pipeline correctly passes history to LLM."""
        mock_run_asr.return_value = "What is AI?"

        initial_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        expected_new_history = initial_history + [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
        ]

        mock_run_llm.return_value = ("AI stands for Artificial Intelligence.", expected_new_history)
        mock_run_tts.return_value = (torch.randn(1, 1000), 24000)

        mock_collector = MagicMock()
        mock_collector.context = MagicMock()
        mock_collector.current_trial = MagicMock()

        mock_config = {
            "asr": {"model_id": "test", "model_name": "test"},
            "llm": {
                "model_id": "test",
                "model_name": "test",
                "quantization": {},
                "kv_cache": False,
                "max_new_tokens": 10,
            },
            "tts": {"model_name": "vibevoice", "model_id": "test"},
            "benchmark": {},
        }

        result = process_sample(
            config=mock_config,
            sample=sample_audio,
            run_id="test",
            collector=mock_collector,
            device="cpu",
            history=initial_history,
        )

        mock_run_llm.assert_called_once()
        call_kwargs = mock_run_llm.call_args.kwargs
        assert call_kwargs["history"] == initial_history

        assert len(result.new_history) == 5
        assert result.new_history == expected_new_history
