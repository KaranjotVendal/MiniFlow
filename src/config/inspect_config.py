from pprint import pprint
from pathlib import Path

from src.config.load_config import load_experiment_config


def inspect_config(config_path: Path) -> dict:
    config: dict = load_experiment_config(config_path)

    print("\n==================== EXPERIMENT SUMMARY ====================")
    exp = config.get("experiment", {})
    print(f"Experiment name: {exp.get('name')}")
    print(f"Description: {exp.get('description')}\n")

    # Dataset
    ds = config.get("dataset", {})
    print("Dataset:")
    print(f"  - name: {ds.get('name')}")
    print(f"  - split: {ds.get('split')}")
    print(f"  - num_samples: {ds.get('num_samples')}")
    print(f"  - warmup: {ds.get('warmup')}\n")

    # ASR
    asr = config.get("asr", {})
    print("ASR:")
    print(f"  - model_id: {asr.get('model_id')}\n")

    # LLM
    llm = config.get("llm", {})
    print("LLM:")
    print(f"  - model_id: {llm.get('model_id')}")
    print(f"  - kv_cache: {llm.get('kv_cache')}")
    print(f"  - max_new_tokens: {llm.get('max_new_tokens')}")
    if "quantization" in llm:
        print("  Quantization:")
        pprint(llm["quantization"], indent=4)
    print()

    # TTS
    tts = config.get("tts", {})
    print("TTS:")
    print(f"  - model_id: {tts.get('model_id')}")
    print(f"  - speaker: {tts.get('speaker')}")
    print(f"  - language: {tts.get('language')}\n")

    # Benchmark flags
    bench = config.get("benchmark", {})
    print("Benchmark Flags:")
    pprint(bench, indent=4)
    print("\n======================================================\n")

    return config
