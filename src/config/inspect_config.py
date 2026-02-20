from pprint import pprint
from pathlib import Path

from src.config.load_config import load_yaml_config


def _resolve_metrics_config_path(main_config_path: Path, metrics_value: str | Path) -> Path:
    path = Path(metrics_value)
    if path.is_absolute():
        return path
    return (main_config_path.parent / path).resolve()


def inspect_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    config: dict = load_yaml_config(config_path)
    config["__config_dir"] = str(config_path.parent)

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
    print(f"  - warmup_samples: {ds.get('warmup_samples')}\n")

    # ASR
    asr = config.get("asr", {})
    print("ASR:")
    print(f"  - model_name: {asr.get('model_name')}")
    print(f"  - model_id: {asr.get('model_id')}\n")

    # LLM
    llm = config.get("llm", {})
    print("LLM:")
    print(f"  - model_name: {llm.get('model_name')}")
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
    print(f"  - model_name: {tts.get('model_name')}")
    print(f"  - model_id: {tts.get('model_id')}")
    print(f"  - speaker: {tts.get('speaker')}")
    print(f"  - language: {tts.get('language')}\n")

    # Benchmark runtime options
    benchmark = config.get("benchmark", {})
    print("Benchmark:")
    pprint(benchmark, indent=4)
    print()

    # Metrics config (external file)
    metrics_path_value = config.get("metrics")
    if metrics_path_value:
        metrics_path = _resolve_metrics_config_path(config_path, metrics_path_value)
        config["metrics"] = str(metrics_path)
        print(f"Metrics config path: {metrics_path}")
        if metrics_path.exists():
            metrics_config = load_yaml_config(metrics_path)
            enabled = metrics_config.get("enabled", [])
            configurations = metrics_config.get("configurations", {})
            print(f"Enabled metrics: {enabled}")
            print("Metric configurations:")
            pprint(configurations, indent=4)
        else:
            print("Metrics config file not found.")
    else:
        print("Metrics config path: None")

    print("\n======================================================\n")

    return config
