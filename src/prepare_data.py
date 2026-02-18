import time
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data.dataset import IterableDataset

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


@dataclass
class AudioSample:
    audio_tensor: Path
    transcript: str
    accent: str
    duration: float
    sample_to_noise_ratio: float
    utmos: float
    sampling_rate: int


def stream_dataset_samples(
    num_samples: int = 10, split: str = "train", accent_filter: str | None = None
) -> AudioSample:
    """returns sample"""
    logger.info("streaming samples GLOBE dataset ...")

    try:
        dataset: IterableDataset = load_dataset(
            "MushanW/GLOBE_V3", split=split, streaming=True
        )
    except Exception as e:
        logger.warning(f"Error loading GLOBE dataset: {e}")
        raise

    sample_accents = set()
    processed_samples = 0
    logger.info(f"filtering for English samples and extracting {num_samples} samples")

    # TODO: process the whole dataset to collect all accent classes and reimplement filter logic
    # filtered_dataset = dataset.filter(
    #     lambda x: x["predicted_accent"] != "United States English"
    # )

    for i, sample in enumerate(dataset):
        if processed_samples >= num_samples:
            break

        try:
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            accent = sample["predicted_accent"]
            transcript = sample["whisper_transcription_large_v3"]
            sample_to_noise_ratio = sample["snr"]
            utmos = sample["utmos"]

            duration = len(audio_array) / sampling_rate if len(audio_array) > 0 else 0.0
            # sample_accents.add(accent)

            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

            # check audio format (channels, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            audio_sample = AudioSample(
                audio_tensor=audio_tensor,
                transcript=transcript,
                accent=accent,
                duration=duration,
                sample_to_noise_ratio=sample_to_noise_ratio,
                utmos=utmos,
                sampling_rate=sampling_rate,
            )

            yield audio_sample
            processed_samples += 1
            logger.info(
                f"yielded sample {processed_samples}: Accent: {audio_sample.accent}"
            )
        except Exception as e:
            logger.warning(f"Error saving audio file: {e}")
            continue

    logger.info(f"Finished Streaming {processed_samples} samples.")


def main():
    start_time = time.time()
    for idx, sample in enumerate(stream_dataset_samples()):
        print(f"sample {idx + 1}:")
        print(f"waveform: {sample.audio_tensor.shape}")
        print(f"transcript: {sample.transcript}")
        print(f"accent: {sample.accent}")
        print(f" duration: {sample.duration}")
        print(f"snr: {sample.sample_to_noise_ratio}")

        if idx == 2:
            break
    end_time = time.time() - start_time

    logger.info(f"Data preparation completed in {end_time:.3f} seconds.")
    logger.info(f"streamed {idx + 1} audio samples")


if __name__ == "__main__":
    main()
