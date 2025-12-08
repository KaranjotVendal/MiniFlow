import time
from dataclasses import dataclass
from unittest.mock import patch
from pathlib import Path

import jiwer
import sounddevice as sd
import torch
from numpy.typing import NDArray
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.metrics import Metrics, log_metrics
from src.tts.tts_pipelines import run_tts
from src.prepare_data import AudioSample, stream_dataset_samples
from src.utils import clear_gpu_cache, get_device
from src.benchmark.gpu_sampler import reset_peak_memory, get_gpu_memory_peak
from src.logger.logging import initialise_logger


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "none": None,
    None: None,
}


logger = initialise_logger(__name__)
DEVICE = get_device()
# utmos_model = utmosv2.create_model(pretrained=True)
# _org_dataloader = torch.utils.data.DataLoader


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


def run_asr(
    config: dict | None, audio_tensor: torch.Tensor, sampling_rate: int
) -> tuple[str, float, float]:
    logger.info("executing STT model")
    logger.info(f"DEVICE: {DEVICE}")
    # model_id = "openai/whisper-small"

    reset_peak_memory()
    # # set language & task to avoid detection + translation
    # pipe.model.generation_config.language = "en"
    # pipe.model.generation_config.task = "transcribe"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=config["model_id"],
        device=DEVICE,
        return_timestamps=False,
        generate_kwargs={
            "language": "en",
            "task": "transcribe",
            "forced_decoder_ids": None,  # 👈 Explicitly disable legacy mechanism
        },
    )

    audio = {"array": audio_tensor.squeeze().numpy(), "sampling_rate": sampling_rate}

    start = time.time()
    pred = pipe(audio)
    latency = time.time() - start
    transcription = pred["text"]

    asr_gpu_util = get_gpu_memory_peak()

    # cleanup
    del pipe
    clear_gpu_cache()
    return transcription, latency, asr_gpu_util


def _quant_config(config: dict):
    q = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["quant_type"],
        bnb_4bit_use_double_quant=q["use_double_quant"],
        bnb_4bit_compute_dtype=DTYPE_MAP[q["compute_dtype"]],  # torch.bfloat16,
    )
    return quant_config


def _build_history(history: list[dict] | None, transcription):
    if history is None:
        history = [{"role": "system", "content": "You are a helpful AI assistant."}]

    messages = history + [{"role": "user", "content": transcription}]

    prompt = (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        + "\nassistant:"
    )
    return prompt


def run_llm(
    config: dict | None, transcription: str, history: str | None = None
) -> tuple[dict, float, list, float]:
    """loads an llm, generates response to transcription and offloads."""
    logger.info("executing LLM model")
    logger.info(f"DEVICE: {DEVICE}")
    reset_peak_memory()

    quant_config = _quant_config(config)

    # model_id = "Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map=DEVICE,
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=config["kv_cache"],
    )

    prompt = _build_history(history=history, transcription=transcription)

    start = time.time()
    response = pipe(prompt, max_new_tokens=config["max_new_tokens"], do_sample=False)
    latency = time.time() - start

    generated_text = response[0]["generated_text"]
    assistant_reply = generated_text[len(prompt) :].strip()

    llm_gpu_util = get_gpu_memory_peak()
    # offload the model
    del pipe, model, tokenizer
    clear_gpu_cache()
    return assistant_reply, latency, history, llm_gpu_util


# def run_tts(
#     config: dict,
#     llm_response: str,
#     reference_audio_tensor: torch.Tensor,
#     reference_sampling_rate: int,
# ) -> tuple[str, float, int, float, float]:
#     logger.info("executing TTS model")
#     logger.info(f"DEVICE: {DEVICE}")

#     from pathlib import Path
#     def dataloader_no_workers(*args, **kwargs):
#         """Patch dataloader to disable multiprocessing (avoids pickling error)"""
#         kwargs["num_workers"] = 0
#         kwargs["pin_memory"] = False
#         return _org_dataloader(*args, **kwargs)

#     reference_wav = Path("./temp_reference_wav.wav")
#     torchaudio.save(reference_wav, reference_audio_tensor, reference_sampling_rate)

#     reset_peak_memory()
#     # model_id = "tts_models/multilingual/multi-dataset/xtts_v2"
#     tts = TTS(config["model_id"]).to(DEVICE)

#     start = time.time()
#     # NOTE: can synthesize with the the original speaker's voice as well (need to test it)
#     waveform = tts.tts(text=llm_response, speaker=config["speaker"], language=config["language"])
#     latency = time.time() - start

#     output_sampling_rate: int = tts.synthesizer.output_sample_rate

#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tts_f:
#         temp_tts_wav = Path(tts_f.name)
#     torchaudio.save(
#         temp_tts_wav, torch.tensor(waveform).unsqueeze(0), output_sampling_rate
#     )

#     # MOS scoring
#     with patch("torch.utils.data.DataLoader", side_effect=dataloader_no_workers):
#         mos = utmos_model.predict(input_path=temp_tts_wav)
#     temp_tts_wav.unlink()
#     reference_wav.unlink()

#     tts_gpu_util = get_gpu_memory_peak()

#     # clean up
#     clear_gpu_cache()
#     return waveform, latency, output_sampling_rate, mos, tts_gpu_util


def process_sample(
    sample: AudioSample,
    config: dict | None = None,
    history: list[dict] | None = None,
    stream_audio: bool = False,
    run_id: str = "baseline",
    folder: str = "./",
    jsonl_file: str | None = None,
) -> tuple[ProcessedSample, Metrics]:
    """End-to-End processing for one audio sample"""
    audio_tensor = sample.audio_tensor
    sampling_rate = sample.sampling_rate
    groundtruth = sample.transcript

    exp_name = Path(folder).name if folder else "experiment"

    timestamp_start = time.time()
    # ASR
    transcript, asr_latency, asr_gpu_peak_mem = run_asr(
        config["asr"], audio_tensor, sampling_rate
    )
    asr_wer = jiwer.wer(groundtruth.lower(), transcript.lower())

    # LLM
    response, llm_latency, new_history, llm_gpu_peak_mem = run_llm(
        config["llm"], transcript, history
    )

    # TTS (can optionally use input audio as reference for voice)
    tts_waveform, tts_latency, output_sample_rate, mos, tts_gpu_peak_mem = run_tts(
        config=config["tts"],
        llm_response=response,
        device=DEVICE,
        # , audio_tensor, sampling_rate
    )
    tts_waveform: list

    # optionally stream the generated audio
    if stream_audio:
        print("Playing audio now...")
        sd.play(tts_waveform, output_sample_rate)  # Usually 24000 Hz for XTTS
        sd.wait()  # Block until playback finishes
        print("Playback complete.")

    total_latency = asr_latency + llm_latency + tts_latency
    timestamp_end = time.time() - timestamp_start

    logger.info(f"Asr latency: {asr_latency}")
    logger.info(f"LLM latency: {llm_latency}")
    logger.info(f"TTS latency: {tts_latency}")
    logger.info(f"Total latency: {total_latency}")

    processed_sample = ProcessedSample(
        groundtruth=groundtruth,
        asr_transcript=transcript,
        llm_response=response,
        tts_waveform=tts_waveform,
        tts_waveform_output_sr=output_sample_rate,
        new_history=new_history,
    )

    metrics = Metrics(
        sample_id=run_id,
        exp_name=exp_name,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
        asr_wer=asr_wer,
        tts_utmos=mos,
        asr_latency=asr_latency,
        llm_latency=llm_latency,
        tts_latency=tts_latency,
        total_latency=total_latency,
        asr_gpu_peak_mem=asr_gpu_peak_mem,
        llm_gpu_peak_mem=llm_gpu_peak_mem,
        tts_gpu_peak_mem=tts_gpu_peak_mem,
    )

    if jsonl_file:
        from src.metrics import log_jsonl

        log_jsonl(metrics.to_dict(), jsonl_file)
    else:
        log_metrics(run_id=run_id, metrics=metrics, folder=folder)

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
    main(num_samples=3)
