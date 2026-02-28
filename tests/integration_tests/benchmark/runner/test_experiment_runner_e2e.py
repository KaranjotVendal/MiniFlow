"""
End-to-End Integration Tests for ExperimentRunner.

These tests run the actual ExperimentRunner with mocked dataset samples but
real collector, storage, and metric collection.

Note: These tests require GPU and may take several minutes.
Use: pytest -m integration to run only these tests.
"""

from pathlib import Path

import pytest
import torch
import yaml

from src.benchmark.runner.experiment_runner import ExperimentRunner
from src.prepare_data import AudioSample


class TestExperimentRunnerE2E:
    """End-to-end integration tests for ExperimentRunner."""

    @pytest.fixture
    def temp_benchmark_dir(self, tmp_path: Path) -> Path:
        """Temporary benchmark output root for this test module."""
        output_dir = tmp_path / "benchmark_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def metrics_config_file(self, tmp_path: Path) -> Path:
        """Create a minimal metrics config file."""
        config = {
            "enabled": ["timing", "hardware_basic", "model_lifecycle"],
            "configurations": {
                "timing": {"stages": ["asr", "llm", "tts", "pipeline"]},
                "hardware_basic": {"device": 0, "track_power": False},
                "model_lifecycle": None,
            },
        }
        metrics_path = tmp_path / "metrics.yml"
        with open(metrics_path, "w") as f:
            yaml.dump(config, f)
        return metrics_path

    @pytest.fixture
    def experiment_config(self, tmp_path: Path, metrics_config_file: Path) -> dict:
        """Create a minimal experiment config."""
        return {
            "experiment": {"name": "e2e_test", "description": "E2E integration test"},
            "dataset": {"name": "test", "split": "test", "num_samples": 1, "warmup_samples": 0},
            "metrics": str(metrics_config_file),
            "asr": {"model_id": "openai/whisper-small", "model_name": "whisper-small"},
            "llm": {
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
                "max_new_tokens": 20,
            },
            "tts": {
                "model_name": "vibevoice",
                "model_id": "microsoft/VibeVoice-Realtime-0.5B",
                "cfg_scale": 1.5,
                "voice_name": "en-Emma_woman",
            },
            "benchmark": {"enable_streaming_audio": False},
        }

    @pytest.fixture
    def mock_sample_generator(self):
        """Create a generator that yields mock audio samples."""

        def generate_samples(num_samples: int):
            for i in range(num_samples):
                # Create simple test audio
                duration = 1.0
                sampling_rate = 16000
                t = torch.linspace(0, duration, int(sampling_rate * duration))
                audio = 0.3 * torch.sin(2 * 440 * 3.14159 * t)

                yield AudioSample(
                    audio_tensor=audio.unsqueeze(0),
                    transcript=f"Test sentence {i}",
                    accent="US English",
                    duration=duration,
                    sample_to_noise_ratio=30.0,
                    utmos=4.0,
                    sampling_rate=sampling_rate,
                )

        return generate_samples

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_experiment_runner_e2e(
        self, experiment_config, temp_benchmark_dir, mock_sample_generator
    ):
        """Test running a real mini-experiment with mocked dataset but real components."""

        # Patch the dataset streaming to use our mock generator
        from unittest.mock import patch

        with patch("src.benchmark.runner.experiment_runner.stream_dataset_samples") as mock_stream:
            # Setup mock to return our generator
            mock_stream.return_value = mock_sample_generator(1)

            # Create and run experiment
            runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

            summary = runner.run()

            # Verify results
            assert summary is not None
            assert summary.num_trials == 1
            assert summary.experiment == "e2e_test"

            # Verify storage files exist
            assert runner.output_dir.exists()

            # Load and verify saved data
            trials = runner.storage.load_trials()
            assert len(trials) == 1

            summary_data = runner.storage.load_summary()
            assert summary_data is not None
            assert "meta" in summary_data
            assert summary_data["meta"]["num_trials"] == 1

            print("\nExperiment run completed!")
            print(f"Run ID: {summary.run_id}")
            print(f"Output dir: {runner.output_dir}")
            print(f"Trial count: {summary.num_trials}")
            print(f"Trials saved: {len(trials)}")

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_experiment_runner_with_warmup(
        self, experiment_config, temp_benchmark_dir, mock_sample_generator
    ):
        """Test running experiment with warmup samples."""

        # Configure with warmup
        experiment_config["dataset"]["warmup_samples"] = 1
        experiment_config["dataset"]["num_samples"] = 1

        from unittest.mock import patch

        # Create a call counter to track warmup vs main
        call_count = [0]

        def sample_generator_wrapper(num_samples: int, split: str = "test"):
            for i in range(num_samples + 1):  # +1 for warmup
                call_count[0] += 1
                duration = 1.0
                sampling_rate = 16000
                t = torch.linspace(0, duration, int(sampling_rate * duration))
                audio = 0.3 * torch.sin(2 * 440 * 3.14159 * t)

                yield AudioSample(
                    audio_tensor=audio.unsqueeze(0),
                    transcript=f"Test {i}",
                    accent="US English",
                    duration=duration,
                    sample_to_noise_ratio=30.0,
                    utmos=4.0,
                    sampling_rate=sampling_rate,
                )

        with patch("src.benchmark.runner.experiment_runner.stream_dataset_samples") as mock_stream:
            mock_stream.return_value = sample_generator_wrapper(1)

            runner = ExperimentRunner.from_config(experiment_config, output_dir=temp_benchmark_dir)

            summary = runner.run()

            # Verify warmup was executed
            assert summary.num_trials == 1

            # Verify storage
            trials = runner.storage.load_trials()
            summary_data = runner.storage.load_summary()

            assert len(trials) == 1
            assert summary_data["meta"]["num_warmup_trials"] == 1

            print("\nExperiment with warmup completed!")
            print(f"Main trials: {summary.num_trials}")
            print(f"Warmup trials: {summary_data['meta']['num_warmup_trials']}")
