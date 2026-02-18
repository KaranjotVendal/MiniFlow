# src/report/sweep_report_generator.py

import json
from pathlib import Path
from datetime import datetime

from src.report.plotting import save_bar_plot


# TODO: needs to be refactored with newer benchmark or it could also be replaced by grafana
def generate_sweep_report(sweep_folder: str | Path):
    sweep_folder = Path(sweep_folder)
    sweep_summary_path = sweep_folder / "sweep_summary.json"

    if not sweep_summary_path.exists():
        raise FileNotFoundError(f"No sweep_summary.json at {sweep_folder}")

    sweep_summary = json.load(open(sweep_summary_path))

    plots_dir = sweep_folder / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Latency comparison plot
    experiments = sweep_summary["experiments"]
    exp_names = [e["experiment"] for e in experiments]
    total_latency_means = [e["total_latency_mean"] for e in experiments]

    save_bar_plot(
        values=dict(zip(exp_names, total_latency_means)),
        title="Total Latency Mean Across Experiments",
        ylabel="Latency (s)",
        output_path=plots_dir / "sweep_latency.png",
    )

    # -------------------------
    # Markdown report
    # -------------------------
    md = []
    md.append(f"# Sweep Report: {sweep_summary['sweep_name']}")
    md.append("")
    md.append(f"**Generated at:** {datetime.now()}")
    md.append("")
    md.append("## Sweep Summary")
    md.append("```json")
    md.append(json.dumps(sweep_summary, indent=2))
    md.append("```")
    md.append("")
    md.append("## Total Latency Comparison")
    md.append("![](plots/sweep_latency.png)")
    md.append("")

    report_path = sweep_folder / "sweep_report.md"
    report_path.write_text("\n".join(md))

    return report_path
