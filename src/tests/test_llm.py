import time

import torch
from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.utils import get_test_sample, get_device


if __name__ == "__main__":
    sample = get_test_sample()
    sample_transcript = sample.transcript

    # Simulate conversation
    history = [{"role": "system", "content": "You are a helpful AI assistant."}]
    messages = history + [{"role": "user", "content": sample_transcript}]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=get_device(),
        quantization_config=quant_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # Format prompt
    prompt = (
        "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        + "\nassistant:"
    )
    start = time.time()
    response = pipe(prompt, max_new_tokens=50, do_sample=False)
    latency = time.time() - start

    print(f"Input: {sample_transcript}")
    print(f"Response: {response}")
    print(f"Latency: {latency:.3f}s")
