from typing import TYPE_CHECKING

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.utils import clear_gpu_cache
from src.logger.logging import initialise_logger

if TYPE_CHECKING:
    from src.benchmark.collectors import BenchmarkCollector

logger = initialise_logger(__name__)

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


def _quant_config(config: dict):
    # TODO: write a good docstring to understand
    #  how BitsAndBytesConfig works and what each param does.
    q = config["quantization"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["quant_type"],
        bnb_4bit_use_double_quant=q["use_double_quant"],
        bnb_4bit_compute_dtype=DTYPE_MAP[q["compute_dtype"]],  # torch.bfloat16,
    )
    return quant_config


def _build_history(history: list[dict] | None, transcription) -> str:
    if history is None:
        history = [{"role": "system", "content": "You are a helpful AI assistant."}]

    messages = history + [{"role": "user", "content": transcription}]

    prompt = (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        + "\nassistant:"
    )
    return prompt


def run_llm(
    config: dict | None,
    transcription: str,
    device: torch.device | str,
    collector: "BenchmarkCollector",
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """loads an llm, generates response to transcription and offloads."""
    logger.info("executing LLM model")

    model = None
    tokenizer = None
    pipe = None
    load_closed = False
    load_event = None

    try:
        # model
        quant_config = _quant_config(config)
        collector.hardware_metrics.start(collector.context)
        collector.lifecycle_metrics.record_load_start(model_name=config["model_name"], source="remote(HF)")

        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            device_map=device,
            quantization_config=quant_config,
        )

        model_load_event = collector.lifecycle_metrics.record_load_end(cached=False)
        collector.record_phase_metrics(
            "llm_model_load_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )
        if model_load_event is not None:
            model_load_event["stage"] = "llm"
            model_load_event["cached"] = False
            model_load_event["success"] = True

        # tokenizer
        collector.hardware_metrics.start(collector.context)
        collector.lifecycle_metrics.record_load_start(model_name="tokenizer", source="remote(HF)")

        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])

        tokenizer_load_event = collector.lifecycle_metrics.record_load_end(cached=False)
        collector.record_phase_metrics(
            "llm_tokenizer_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )
        if tokenizer_load_event is not None:
            tokenizer_load_event["stage"] = "llm"
            tokenizer_load_event["cached"] = False
            tokenizer_load_event["success"] = True

        # pipeline
        collector.hardware_metrics.start(collector.context)
        collector.lifecycle_metrics.record_load_start(
            model_name="text_generation_pipeline", source="in_memory"
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            use_cache=config["kv_cache"],
        )

        load_event = collector.lifecycle_metrics.record_load_end(cached=False)
        collector.record_phase_metrics(
            "llm_pipeline_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )
        if load_event is not None:
            load_event["stage"] = "llm"
            load_event["cached"] = False
            load_event["success"] = True
        load_closed = True

        # inference
        prompt = _build_history(history=history, transcription=transcription)
        collector.start_token_metrics()
        collector.hardware_metrics.start(collector.context)
        collector.timing_metrics.record_stage_start("llm_inference_latency")

        response = pipe(prompt, max_new_tokens=config["max_new_tokens"], do_sample=False)

        collector.timing_metrics.record_stage_end("llm_inference_latency")
        collector.record_phase_metrics(
            "llm_inference_gpu_metrics",
            collector.hardware_metrics.end(collector.context),
        )

        # token metrics
        # TODO: We will add streaming soon.
        generated_text = response[0]["generated_text"]
        assistant_reply = generated_text[len(prompt) :].strip()
        token_count = len(tokenizer.encode(assistant_reply, add_special_tokens=False))
        collector.token_metrics.add_tokens(token_count)
        collector.finalize_token_metrics()

        if history is None:
            history = [{"role": "system", "content": "You are a helpful AI assistant."}]

        updated_history = history + [
            {"role": "user", "content": transcription},
            {"role": "assistant", "content": assistant_reply},
        ]
        return assistant_reply, updated_history


    except Exception:
        if collector is not None and not load_closed:
            load_event = collector.lifecycle_metrics.record_load_end(cached=False)
            if load_event is not None:
                load_event["stage"] = "llm"
                load_event["cached"] = False
                load_event["success"] = False
                load_event["error_type"] = "LLMLoadOrInferenceError"
        raise
    finally:
        if pipe is not None:
            del pipe
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_gpu_cache()
