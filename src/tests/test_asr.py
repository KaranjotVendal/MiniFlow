import time

import torch
from transformers import pipeline
import jiwer

from src.utils import get_test_sample

if __name__ == "__main__":
    sample = get_test_sample()
    audio_tensor = sample.audio_tensor.squeeze().numpy()
    groundtruth: str = sample.transcript
    sampling_rate = sample.sampling_rate

    audio = {"array": audio_tensor, "sampling_rate": sampling_rate}

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=torch.device("cuda" if torch.cuda.is_available else "cpu"),
    )

    start = time.time()
    pred = pipe(audio)
    latency = time.time() - start

    transcription = pred["text"]
    wer = jiwer.wer(groundtruth.lower(), transcription.lower())

    print(f"Trascription: {transcription}")
    print(f"GT: {groundtruth}")
    print(f"Latency: {latency:.3f}s")
    print(f"WER: {wer:.3%}")

    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
