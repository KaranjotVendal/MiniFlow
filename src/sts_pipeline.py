from dataclasses import dataclass
from typing import TYPE_CHECKING

import sounddevice as sd
from numpy.typing import NDArray
import torch

from src.benchmark.collectors import BenchmarkCollector
from src.stt.stt_pipeline import run_asr
from src.tts.tts_pipelines import run_tts
from src.llm.llm_pipeline import run_llm
from src.prepare_data import AudioSample
from src.logger.logging import initialise_logger

if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)


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
    collector: "BenchmarkCollector",
    device: torch.device | str,
    history: list[dict] | None = None,
    stream_audio: bool = False
) -> ProcessedSample:
    """End-to-End processing for one audio sample"""
    if collector is None:
        raise ValueError("BenchmarkCollector is required for process_sample().")

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
