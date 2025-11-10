import time
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data.dataset import IterableDataset
import torchaudio
from datasets import load_dataset

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def create_dirs():
    data_dir = Path("data_assets/test_samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@dataclass
class AudioSample:
    audio_tensor: Path
    transcript: str
    accent: str
    duration: float
    sample_to_noise_ratio: float
    utmos: float
    sampling_rate: int


def download_globe_samples(num_samples=10) -> list[AudioSample]:
    """returns list of dict with audio parts and transcripts"""
    logger.info("Loading GLOBE dataset with streaming...")
    data_dir = create_dirs()
    # sampling_rate = 25000  # 25kHz

    try:
        dataset: IterableDataset = load_dataset(
            "MushanW/GLOBE_V3", split="train", streaming=True
        )
    except Exception as e:
        logger.warning(f"Error loading GLOBE dataset: {e}")
        logger.info("Falling back to manual sample download")
        raise

    english_samples = []
    sample_accents = set()
    logger.info(f"filtering for English samples and extracting {num_samples} samples")

    # TODO: process the whole dataset to collect all accent classes and reimplement filter logic
    # filtered_dataset = dataset.filter(
    #     lambda x: x["predicted_accent"] != "United States English"
    # )

    for i, sample in enumerate(dataset):
        if len(english_samples) >= num_samples:
            break

        try:
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            accent = sample["predicted_accent"]
            transcript = sample["whisper_transcription_large_v3"]

            duration = len(audio_array) / sampling_rate if len(audio_array) > 0 else 0.0
            sample_accents.add(accent)

            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

            # check audio format (channels, samples)
            if audio_tensor.dim() == 1:
                # add channel dim
                audio_tensor = audio_tensor.unsqueeze(0)

            filename = f"sample_{len(english_samples) + 1}.wav"
            filepath = data_dir / filename

            torchaudio.save(filepath, audio_tensor, sampling_rate)

            # store sample information
            english_samples.append(
                {
                    "audio_path": filepath,
                    "transcript": transcript,
                    "accent": accent,
                    "duration": duration,
                    "sampling_rate": sampling_rate,
                }
            )
        except Exception as e:
            logger.warning(f"Error saving audio file: {e}")
            continue

    # save transcripts to file
    transcript_file = data_dir / "transcripts.txt"
    with open(transcript_file, "w") as f:
        for i, sample in enumerate(english_samples):
            f.write(f"{i + 1}. {sample['transcript']}\n")

    logger.info(f"Downloaded {len(english_samples)} english samples to {data_dir}")
    return english_samples


def stream_dataset_samples(
    num_samples: int = 10, accent_filter: str | None = None
) -> AudioSample:
    """returns sample"""
    logger.info("streaming samples GLOBE dataset ...")

    try:
        dataset: IterableDataset = load_dataset(
            "MushanW/GLOBE_V3", split="train", streaming=True
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
    samples = download_globe_samples(num_samples=6)
    end_time = time.time() - start_time

    logger.info(f"Data preparation completed in {end_time:.3f} seconds.")
    logger.info(f"Downloaded {len(samples)} audio samples")

    for i, sample in enumerate(samples[:2]):
        print(f"sample {i + 1}:")
        # print(f"Path: {sample.audio_path}")
        # print(f"transcript: {sample.transcript}")
        # print(f"accent: {sample.accent}")
        # print(f" duration: {sample.duration}")


def main_stream():
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
    # main()
    main_stream()
