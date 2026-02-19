from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import sounddevice as sd
from numpy.typing import NDArray
import torch

from src.stt.stt_pipeline import run_asr
from src.tts.tts_pipelines import run_tts
from src.llm.llm_pipeline import run_llm
from src.prepare_data import AudioSample
from src.logger.logging import initialise_logger

if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)


class _NoOpMetric:
    """No-op metric adapter for synchronous API path."""

    def start(self, *args, **kwargs) -> None:
        return None

    def end(self, *args, **kwargs) -> dict:
        return {}

    def record_stage_start(self, *args, **kwargs) -> None:
        return None

    def record_stage_end(self, *args, **kwargs) -> None:
        return None

    def record_load_start(self, *args, **kwargs) -> None:
        return None

    def record_load_end(self, *args, **kwargs) -> dict:
        # Must be mutable because stage code annotates this event.
        return {}

    def add_tokens(self, *args, **kwargs) -> None:
        return None

    def evaluate(self, evaluator: str, *args, **kwargs) -> dict[str, float]:
        if evaluator == "wer":
            return {"wer": 0.0}
        if evaluator == "utmos":
            return {"utmos": 0.0}
        return {evaluator: 0.0}


class NoOpCollector:
    """No-op collector used only for API synchronous inference path."""

    def __init__(self) -> None:
        self.context = None
        self.timing_metrics = _NoOpMetric()
        self.token_metrics = _NoOpMetric()
        self.lifecycle_metrics = _NoOpMetric()
        self.hardware_metrics = _NoOpMetric()
        self.quality_metrics = _NoOpMetric()
        self.current_trial = SimpleNamespace(quality=SimpleNamespace(wer=0.0, utmos=0.0))

    def start_token_metrics(self) -> None:
        return None

    def finalize_token_metrics(self) -> None:
        return None

    def record_phase_metrics(self, phase_name: str, metrics: dict) -> None:
        return None


@dataclass
class ProcessedSample:
    groundtruth: str
    asr_transcript: str
    llm_response: str
    tts_waveform: NDArray
    tts_waveform_output_sr: int
    new_history: list[str]

    def to_dict(self):
        return {
            "groundtruth": self.groundtruth,
            "asr_transcript": self.asr_transcript,
            "llm_response": self.llm_response,
            "tts_waveform": self.tts_waveform,
            "tts_waveform_output_sr": self.tts_waveform_output_sr,
            "new_history": self.new_history,
        }


def process_sample(
    config: dict,
    sample: AudioSample,
    run_id: str,
    collector: "BenchmarkCollector | None" = None,
    device: torch.device | str = "cuda",
    history: list[dict] | None = None,
    stream_audio: bool = False
) -> ProcessedSample:
    """End-to-End processing for one audio sample."""
    if collector is None:
        collector = NoOpCollector()

    # ASR
    transcription = run_asr(
        config=config["asr"],
        audio_tensor=sample.audio_tensor,
        sampling_rate=sample.sampling_rate,
        groundtruth=sample.transcript,
        collector=collector,
        device=device,
    )

    # LLM
    response, new_history = run_llm(
        config=config["llm"],
        transcription=transcription,
        history=history,
        collector=collector,
        device=device
    )

    # TTS (can optionally use input audio as reference for voice)
    tts_waveform, output_sample_rate = run_tts(
        config=config["tts"],
        llm_response=response,
        device=device,
        collector=collector,
        # , audio_tensor, sampling_rate
    )
    # NOTE: xtts returns a list while vibevoice will need to be verified.
    tts_waveform: list

    # optionally stream the generated audio
    if stream_audio:
        print("Playing audio now...")
        sd.play(tts_waveform, output_sample_rate)  # Usually 24000 Hz for XTTS
        sd.wait()  # Block until playback finishes
        print("Playback complete.")

    processed_sample = ProcessedSample(
        groundtruth=sample.transcript,
        asr_transcript=transcription,
        llm_response=response,
        tts_waveform=tts_waveform,
        tts_waveform_output_sr=output_sample_rate,
        new_history=new_history)

        # TODO: implement multi turn system. might need to find a dataset with
        # multi turn conversation
        # if i == (num_samples - 1):
        #     logger.info(f"Simulating 2nd turn...")
        #

    return processed_sample
