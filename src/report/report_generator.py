import json
from pathlib import Path
from datetime import datetime

from src.report.plotting import (
    save_bar_plot,
    save_line_plot,
    save_latency_percentiles_plot,
)

# TODO: needs to be refactored with newer benchmark framework
def generate_experiment_report(exp_folder: str | Path):
    exp_folder = Path(exp_folder)
    summary_path = exp_folder / "summary.json"
    raw_logs_path = exp_folder / "raw_logs.jsonl"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json found in {exp_folder}")

    summary = json.load(open(summary_path))

    plots_dir = exp_folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    latency_vals = {
        "ASR": summary["asr_latency"]["mean"],
        "LLM": summary["llm_latency"]["mean"],
        "TTS": summary["tts_latency"]["mean"],
    }

    # latency plot
    save_bar_plot(
        values=latency_vals,
        title="Mean Latency Breakdown",
        ylabel="Latency (s)",
        output_path=plots_dir / "latency_breakdown.png",
    )

    # memory plot
    memory_vals = []
    turn = []
    latency_samples = []
    if raw_logs_path.exists():
        with open(raw_logs_path) as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                memory_vals.append(d["llm_gpu_peak_mem"])
                latency_samples.append(d["total_latency"])
                turn.append(i + 1)

    if memory_vals:
        save_line_plot(
            xs=turn,
            ys=memory_vals,
            title="LLM GPU Peak Memory Per Trial",
            xlabel="Trial",
            ylabel="Peak Memory (MB)",
            output_path=plots_dir / "memory_llm.png",
        )

    if latency_samples:
        save_latency_percentiles_plot(
            samples=latency_samples,
            title="End-to-End latency CDF",
            output_path=plots_dir / "latency_cdf.png",
        )

    # Markdown report
    md = []

    md.append(f"# Experiment Report: {summary['experiment']}")
    md.append("")
    md.append(f"**Genenrated at: {datetime.now()}**")
    md.append("")
    md.append("## Summary Metrics")
    md.append("```json")
    md.append(json.dumps(summary, indent=2))
    md.append("```")
    md.append("")

    md.append("## Latency Breakdown (Mean)")
    md.append("![](plots/latency_breakdown.png)")
    md.append("")

    if memory_vals:
        md.append("## LLM GPU Memory Usage Per Trial")
        md.append("![](plots/memory_llm.png)")
        md.append("")

    if latency_samples:
        md.append("## Latency CDF (p50/p95/p99)")
        md.append("![](plots/latency_cdf.png)")
        md.append("")

    report_path = exp_folder / "report.md"
    report_path.write_text("\n".join(md))

    return report_path
