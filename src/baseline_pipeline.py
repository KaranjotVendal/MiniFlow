import time
import tempfile
from typing import Any, Callable
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torchaudio
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from TTS.api import TTS
import jiwer
import sounddevice as sd
import utmosv2

from numpy.typing import NDArray

from src.utils import get_device, get_gpu_util, clear_gpu_cache
from src.prepare_data import AudioSample, stream_dataset_samples
from src.metrics import log_metrics, Metrics
from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)
DEVICE = get_device()
utmos_model = utmosv2.create_model(pretrained=True)
_org_dataloader = torch.utils.data.DataLoader


@dataclass
class ProcessedSample:
    groundtruth: str
    asr_transcript: str
    llm_response: str
    tts_wavform: NDArray
    tts_wavform_output_sr: int
    new_history: list[str]

    def to_dict(self):
        return {
            "groundtruth": self.groundtruth,
            "asr_transcript": self.asr_transcript,
            "llm_response": self.llm_response,
            "tts_wavform": self.tts_wavform,
            "tts_wavform_output_sr": self.tts_waveform_output_sr,
            "new_history": self.new_history,
        }


def run_asr(audio_tensor: torch.Tensor, sampling_rate: int) -> tuple[str, float, float]:
    model_id = "openai/whisper-small"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        # dtype=torch.float16,
        device=DEVICE,
    )
    # 🔑 Critical: Set language & task to avoid detection + translation
    pipe.model.generation_config.language = "en"
    pipe.model.generation_config.task = "transcribe"

    audio = {"array": audio_tensor.squeeze().numpy(), "sampling_rate": sampling_rate}

    start = time.time()
    pred = pipe(audio)
    latency = time.time() - start
    transcription = pred["text"]

    asr_gpu_util = get_gpu_util()

    # cleanup
    del pipe
    clear_gpu_cache()
    return transcription, latency, asr_gpu_util


def run_llm(transcription: str, history: str | None = None) -> tuple[dict, float, list]:
    """loads an llm, generates response to transcription and offloads."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    if history is None:
        history = [{"role": "system", "content": "You are a helpful AI assistant."}]

    messages = history + [{"role": "user", "content": transcription}]

    prompt = (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        + "\nassistant:"
    )

    start = time.time()
    response = pipe(prompt, max_new_tokens=50, do_sample=False)
    latency = time.time() - start

    generated_text = response[0]["generated_text"]
    assistant_reply = generated_text[len(prompt) :].strip()

    llm_gpu_util = get_gpu_util()
    # offload the model
    del pipe
    clear_gpu_cache()
    return assistant_reply, latency, history, llm_gpu_util


def run_tts(
    text: str, reference_audio_tensor: torch.Tensor, reference_sampling_rate: int
) -> tuple[str, float, int, float, float]:
    from pathlib import Path

    def dataloader_no_workers(*args, **kwargs):
        """Patch dataloader to disable multiprocessing (avoids pickling error)"""
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return _org_dataloader(*args, **kwargs)

    reference_wav = Path("./temp_reference_wav.wav")
    torchaudio.save(reference_wav, reference_audio_tensor, reference_sampling_rate)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

    start = time.time()
    waveform = tts.tts(text=text, speaker="Gracie Wise", language="en")
    latency = time.time() - start

    output_sampling_rate: int = tts.synthesizer.output_sample_rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_f:
        temp_tts_wav = Path(tts_f.name)
    torchaudio.save(
        temp_tts_wav, torch.tensor(waveform).unsqueeze(0), output_sampling_rate
    )

    with patch("torch.utils.data.DataLoader", side_effect=dataloader_no_workers):
        mos = utmos_model.predict(input_path=temp_tts_wav)
    temp_tts_wav.unlink()

    tts_gpu_util = get_gpu_util()

    # clean up
    # reference_wav.unlink()
    # clear_gpu_cache()
    return waveform, latency, output_sampling_rate, mos, tts_gpu_util


def process_sample(
    sample: AudioSample,
    history: list[dict] | None = None,
    stream_audio: bool = False,
    run_id: str = "baseline",
    folder: str = "./",
) -> tuple[ProcessedSample, Metrics]:
    """End-to-End processing for one audio sample"""
    audio_tensor = sample.audio_tensor
    sampling_rate = sample.sampling_rate
    groundtruth = sample.transcript

    # ASR
    transcript, asr_latency, asr_gpu_util = run_asr(audio_tensor, sampling_rate)
    asr_wer = jiwer.wer(groundtruth.lower(), transcript.lower())

    # LLM
    response, llm_latency, new_history, llm_gpu_util = run_llm(transcript, history)

    # TTS (can optionally use input audio as reference for voice)
    tts_waveform, tts_latency, output_sample_rate, mos, tts_gpu_util = run_tts(
        response, audio_tensor, sampling_rate
    )
    tts_waveform: list

    # optionally stream the generated audio
    if stream_audio:
        print("Playing audio now...")
        sd.play(tts_waveform, output_sample_rate)  # Usually 24000 Hz for XTTS
        sd.wait()  # Block until playback finishes
        print("Playback complete.")

    total_latency = asr_latency + llm_latency + tts_latency

    metrics = Metrics(
        asr_wer=asr_wer,
        tts_utmos=mos,
        asr_latency=asr_latency,
        llm_latency=llm_latency,
        tts_latency=tts_latency,
        total_latency=total_latency,
        asr_gpu_util=asr_gpu_util,
        llm_gpu_util=llm_gpu_util,
        tts_gpu_util=tts_gpu_util,
    )
    log_metrics(run_id=run_id, metrics=metrics.to_dict(), folder=folder)

    logger.info(f"Asr latency: {asr_latency}")
    logger.info(f"LLM latency: {llm_latency}")
    logger.info(f"TTS latency: {tts_latency}")
    logger.info(f"Total latency: {total_latency}")

    processed_sample = ProcessedSample(
        groundtruth=groundtruth,
        asr_transcript=transcript,
        llm_response=response,
        tts_wavform=tts_waveform,
        tts_wavform_output_sr=output_sample_rate,
        new_history=new_history,
    )

    return processed_sample, metrics


def main(num_samples: int = 3):
    results: list[ProcessedSample] = []
    total_wer: float = 0
    total_latency: float = 0

    try:
        start = time.time()
        for i, sample in enumerate(stream_dataset_samples(num_samples=num_samples)):
            logger.info(f"Processing sample {i + 1}")
            _start = time.time()
            result = process_sample(sample)
            _latency = time.time() - _start
            results.append(result)

            logger.info(f"Turn 1 - Transcript: {result.groundtruth}")
            logger.info(f"Response: {result.llm_response}")
            logger.info(f"ASR WER: {result.asr_wer}")
            logger.info(f"Total latency to run sample: {i}: {_latency}s")

            # TODO: implement multi turn system. might need to find a dataset with
            # multi turn conversation
            # if i == (num_samples - 1):
            #     logger.info(f"Simulating 2nd turn...")
            #
        latency = time.time() - start
        logger.info(f"Total latency to run the whole loop: {latency}s")
    except Exception as e:
        logger.exception(f"encountered an error: {e}")
        raise


if __name__ == "__main__":
    try:
        start = time.time()
        main(num_samples=3)
        latency = time.time() - start
        logger.info(f"Total latency to run the whole loop: {latency}s")
    except Exception as e:
        logger.warning(f"encountered an error: {e}")
