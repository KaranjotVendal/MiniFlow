#!/usr/bin/env python3
"""
Compare benchmark results from multiple summary.json files.

Usage:
    # Compare 2 benchmarks
    python src/scripts/compare_benchmarks.py \
        Benchmark/baseline/baseline_67ecd4/summary.json \
        Benchmark/baseline/baseline_a0dd05/summary.json

    # Compare 3+ benchmarks (matrix format)
    python src/scripts/compare_benchmarks.py Benchmark/baseline/

    # With custom title
    python src/scripts/compare_benchmarks.py \
        --title "XTTS vs VibeVoice Benchmark" \
        Benchmark/baseline/baseline_67ecd4/summary.json \
        Benchmark/baseline/baseline_a0dd05/summary.json
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

from src.logger.logging import initialise_logger

logger = initialise_logger(__name__)


def parse_config(config_path: Path) -> dict[str, str]:
    """Parse config.yml to extract readable component names."""
    result = {"asr": "", "llm": "", "tts": ""}

    content = config_path.read_text()
    config = yaml.safe_load(content)

    result["asr"] = config["asr"]["model_name"]
    result["llm"] = config["llm"]["model_name"]
    result["tts"] = config["tts"]["model_name"]
    return result


def extract_short_name(model_id: str) -> str:
    """Extract short name from model ID."""
    if not model_id:
        return ""

    # Handle common patterns
    name = model_id.strip()

    # Remove registry prefix
    if "/" in name:
        parts = name.split("/")
        name = parts[-1]

    # Remove common prefixes
    prefixes = ["tts_models/", "speecht5_v2/", "ms_hub/"]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]

    # Shorten common model names
    replacements = [
        ("Qwen/Qwen2.5-", "qwen-"),
        ("Qwen2.5-", "qwen-"),
        ("microsoft/", ""),
        ("openai/", ""),
        ("-Instruct", ""),
        ("-instruct", ""),
        ("-Base", ""),
        ("-base", ""),
    ]

    for old, new in replacements:
        name = name.replace(old, new)

    # Handle specific patterns
    if name.startswith("Phi-4"):
        name = name.replace("Phi-4-mini-instruct", "phi-4").replace("Phi-4-", "phi-4-")

    return name


def generate_benchmark_name(config: dict[str, str], folder_name: str) -> str:
    """Generate readable benchmark name from config or folder."""
    if config.get("asr") and config.get("llm") and config.get("tts"):
        return f"{config['asr']}_{config['llm']}_{config['tts']}"
    return folder_name


def compute_memory_stats(logs_path: Path) -> dict[str, float]:
    """Compute memory statistics from raw_logs.jsonl.
    # TODO: this will be deleted once we move this logic."""
    stats = {
        "asr_peak_mean": 0.0,
        "llm_peak_mean": 0.0,
        "tts_peak_mean": 0.0,
        "max_asr": 0.0,
        "max_llm": 0.0,
        "max_tts": 0.0
    }

    asr_peaks = []
    llm_peaks = []
    tts_peaks = []
    max_asr = 0.0
    max_llm = 0.0
    max_tts = 0.0
    with logs_path.open("r") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            asr_peaks.append(record["asr_gpu_peak_mem"])
            llm_peaks.append(record["llm_gpu_peak_mem"])
            tts_peaks.append(record["tts_gpu_peak_mem"])

            max_asr = record["asr_gpu_peak_mem"] if max_asr < record["asr_gpu_peak_mem"] else max_asr
            max_llm = record["llm_gpu_peak_mem"] if max_llm < record["llm_gpu_peak_mem"] else max_llm
            max_tts = record["tts_gpu_peak_mem"] if max_tts < record["tts_gpu_peak_mem"] else max_tts

    if asr_peaks:
        stats["asr_peak_mean"] = sum(asr_peaks) / len(asr_peaks)
    if llm_peaks:
        stats["llm_peak_mean"] = sum(llm_peaks) / len(llm_peaks)
    if tts_peaks:
        stats["tts_peak_mean"] = sum(tts_peaks) / len(tts_peaks)
    stats["max_asr"] = max_asr
    stats["max_llm"] = max_llm
    stats["max_tts"] = max_tts

    return stats


def load_summary(summary_path: Path) -> dict:
    """Load summary.json file."""
    if not summary_path.exists():
        logger.warning("summary path doesn't exist, returning empty dict")
        return {}

    return json.loads(summary_path.read_text())


def format_percent(value: float) -> str:
    """Format percentage with sign."""
    if value > 0:
        return f"+{value:.1f}%"
    return f"{value:.1f}%"


def format_seconds(value: float) -> str:
    """Format seconds with 3 decimal places."""
    return f"{value:.3f}s"


def format_mb(value: float) -> str:
    """Format megabytes with 0 decimal places."""
    return f"{value:.0f} MB"


def compute_change(baseline: float, current: float) -> float:
    """Compute percentage change: (current - baseline) / baseline * 100."""
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100


def compare_two(
    data: list,
    title: str | None = None,
) -> str:
    """Generate comparison table for 2 benchmarks."""
    name1: str = data[0]["name"]
    data1: dict = data[0]["data"]
    mem1: dict = data[0]["mem"]

    name2: str = data[1]["name"]
    data2: dict = data[1]["data"]
    mem2: dict = data[1]["mem"]

    lines = []
    if title:
        lines.append(f"# Benchmark Comparison: {title}")
    else:
        lines.append(f"# Benchmark Comparison: {name1} vs {name2}")

    lines.append("")
    lines.append(f"**Baseline:** {name1}")
    lines.append(f"**Comparison:** {name2}")
    lines.append("")

    # Latency table (mean)
    lines.append("## Latency (mean, seconds)")
    lines.append("")
    lines.append("| Metric        | " + f"{name1} | {name2} | Change |")
    lines.append(
        "|---------------|" + f" {'-' * len(name1)} | {'-' * len(name2)} | -------- |"
    )

    metrics = [
        ("asr_latency", "ASR Latency"),
        ("llm_latency", "LLM Latency"),
        ("tts_latency", "TTS Latency"),
        ("total_latency", "Total Latency"),
    ]

    for key, label in metrics:
        baseline_val = data1.get(key, {}).get("mean", 0)
        current_val = data2.get(key, {}).get("mean", 0)
        change = compute_change(baseline_val, current_val)
        lines.append(
            f"| {label:<13} | {format_seconds(baseline_val):^15} | {format_seconds(current_val):^15} | {format_percent(change):^8} |"
        )

    lines.append("")

    # Latency (p95)
    lines.append("## Latency Percentiles (p95, seconds)")
    lines.append("")
    lines.append("| Metric        | " + f"{name1} | {name2} | Change |")
    lines.append(
        "|---------------|" + f" {'-' * len(name1)} | {'-' * len(name2)} | -------- |"
    )

    for key, label in metrics:
        baseline_val = data1.get(key, {}).get("p95", 0)
        current_val = data2.get(key, {}).get("p95", 0)
        change = compute_change(baseline_val, current_val)
        lines.append(
            f"| {label:<13} | {format_seconds(baseline_val):^15} | {format_seconds(current_val):^15} | {format_percent(change):^8} |"
        )

    lines.append("")

    # Latency (p99)
    lines.append("## Latency Percentiles (p99, seconds)")
    lines.append("")
    lines.append("| Metric        | " + f"{name1} | {name2} | Change |")
    lines.append(
        "|---------------|" + f" {'-' * len(name1)} | {'-' * len(name2)} | -------- |"
    )

    for key, label in metrics:
        baseline_val = data1.get(key, {}).get("p99", 0)
        current_val = data2.get(key, {}).get("p99", 0)
        change = compute_change(baseline_val, current_val)
        lines.append(
            f"| {label:<13} | {format_seconds(baseline_val):^15} | {format_seconds(current_val):^15} | {format_percent(change):^8} |"
        )

    lines.append("")

    # Memory (peak, MB)
    lines.append("## Memory (peak, MB) - computed from raw_logs.jsonl")
    lines.append("")
    lines.append("| Metric         | " + f"{name1} | {name2} | Change |")
    lines.append(
        "|----------------|" + f" {'-' * len(name1)} | {'-' * len(name2)} | -------- |"
    )

    mem_metrics = [
        ("asr_peak_mean", "ASR Peak Mem"),
        ("llm_peak_mean", "LLM Peak Mem"),
        ("tts_peak_mean", "TTS Peak Mem"),
        ("max_asr", "Max ASR"),
        ("max_llm", "Max LLM"),
        ("max_tts", "Max TTS"),

    ]

    for key, label in mem_metrics:
        baseline_val = mem1.get(key, 0)
        current_val = mem2.get(key, 0)
        change = compute_change(baseline_val, current_val)
        lines.append(
            f"| {label:<14} | {format_mb(baseline_val):^15} | {format_mb(current_val):^15} | {format_percent(change):^8} |"
        )

    lines.append("")

    # Quality
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Metric   | " + f"{name1} | {name2} | Change |")
    lines.append(
        "|----------|" + f" {'-' * len(name1)} | {'-' * len(name2)} | -------- |"
    )

    qual_metrics = [
        ("asr_wer_mean", "ASR WER"),
        ("tts_utmos_mean", "TTS UTMOS"),
    ]

    for key, label in qual_metrics:
        baseline_val = data1.get(key, 0)
        current_val = data2.get(key, 0)
        change = compute_change(baseline_val, current_val)
        lines.append(
            f"| {label:<8} | {baseline_val:^15.3f} | {current_val:^15.3f} | {format_percent(change):^8} |"
        )

    lines.append("")

    return "\n".join(lines)


def compare_multiple(
    benchmarks: list[dict],
    title: str | None = None,
) -> str:
    """Generate comparison table for 3+ benchmarks."""
    lines = []

    if title:
        lines.append(f"# Benchmark Comparison: {title}")
    else:
        lines.append("# Benchmark Comparison: All Experiments")
    lines.append("")

    names = [b["name"] for b in benchmarks]

    # Latency (mean)
    lines.append("## Latency (mean, seconds)")
    lines.append("")
    header = "| Metric        | " + " | ".join(f"{n:<20}" for n in names) + " |"
    separator = "|---------------|" + " | ".join("-" * 20 for _ in names) + " |"
    lines.append(header)
    lines.append(separator)

    metrics = [
        ("asr_latency", "ASR Latency"),
        ("llm_latency", "LLM Latency"),
        ("tts_latency", "TTS Latency"),
        ("total_latency", "Total Latency"),
    ]

    for key, label in metrics:
        values = [
            format_seconds(b["data"].get(key, {}).get("mean", 0)) for b in benchmarks
        ]
        lines.append(
            f"| {label:<13} | " + " | ".join(f"{v:<20}" for v in values) + " |"
        )

    lines.append("")

    # Latency (p95)
    lines.append("## Latency Percentiles (p95, seconds)")
    lines.append("")
    lines.append(header)
    lines.append(separator)

    for key, label in metrics:
        values = [
            format_seconds(b["data"].get(key, {}).get("p95", 0)) for b in benchmarks
        ]
        lines.append(
            f"| {label:<13} | " + " | ".join(f"{v:<20}" for v in values) + " |"
        )

    lines.append("")

    # Latency (p99)
    lines.append("## Latency Percentiles (p99, seconds)")
    lines.append("")
    lines.append(header)
    lines.append(separator)

    for key, label in metrics:
        values = [
            format_seconds(b["data"].get(key, {}).get("p99", 0)) for b in benchmarks
        ]
        lines.append(
            f"| {label:<13} | " + " | ".join(f"{v:<20}" for v in values) + " |"
        )

    lines.append("")

    # Memory (peak, MB)
    lines.append("## Memory (peak, MB) - computed from raw_logs.jsonl")
    lines.append("")
    lines.append("| Metric         | " + " | ".join(f"{n:<20}" for n in names) + " |")
    lines.append("|----------------|" + " | ".join("-" * 20 for _ in names) + " |")

    mem_metrics = [
        ("asr_peak_mean", "ASR Peak Mem"),
        ("llm_peak_mean", "LLM Peak Mem"),
        ("tts_peak_mean", "TTS Peak Mem"),
        ("max_overall", "Max Overall"),
    ]

    for key, label in mem_metrics:
        values = [format_mb(b["mem"].get(key, 0)) for b in benchmarks]
        lines.append(
            f"| {label:<14} | " + " | ".join(f"{v:<20}" for v in values) + " |"
        )

    lines.append("")

    # Quality
    lines.append("## Quality Metrics")
    lines.append("")
    lines.append("| Metric   | " + " | ".join(f"{n:<20}" for n in names) + " |")
    lines.append("|----------|" + " | ".join("-" * 20 for _ in names) + " |")

    qual_metrics = [
        ("asr_wer_mean", "ASR WER"),
        ("tts_utmos_mean", "TTS UTMOS"),
    ]

    for key, label in qual_metrics:
        values = [f"{b['data'].get(key, 0):.3f}" for b in benchmarks]
        lines.append(f"| {label:<8} | " + " | ".join(f"{v:<20}" for v in values) + " |")

    lines.append("")

    return "\n".join(lines)


def find_summary_files(paths: list[str]) -> list[Path]:
    """Find all summary.json files from given paths."""
    result = []
    for path_str in paths:
        path = Path(path_str)

        if path.is_file():
            if path.name == "summary.json":
                result.append(path)
                continue

        # TODO: this dir logic is broken and needs to be fixed
        assert path.is_dir(), f"{path} is not dir nor is a file"
        result.extend(list(path.glob("*/summary.json")))

    return sorted(result)


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results from summary.json files."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="""Paths to summary.json files or directories containing them.
        for example: Benchmark/baseline/baseline_uuid or Benchmark/baseline/baseline_uuid""",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the comparison",
    )
    args = parser.parse_args()

    summary_files = find_summary_files(args.paths)

    if len(summary_files) < 2:
        logger.error("Error: At least 2 benchmark files required")
        sys.exit(1)

    benchmarks = []
    for summary_path in summary_files:
        folder = summary_path.parent
        folder_name = folder.name

        summary_data = load_summary(summary_path)
        config = parse_config(folder / "config.yml")
        memory_data = compute_memory_stats(folder / "raw_logs.jsonl")

        name = generate_benchmark_name(config, folder_name)
        run_id = summary_data.get("run_id", folder_name)

        benchmarks.append(
            {
                "path": summary_path,
                "folder": folder,
                "name": name,
                "run_id": run_id,
                "data": summary_data,
                "mem": memory_data,
            }
        )

    if len(benchmarks) == 2:
        output = compare_two(data=benchmarks, title=args.title)
    else:
        output = compare_multiple(benchmarks, args.title)

    print(output)


if __name__ == "__main__":
    main()
