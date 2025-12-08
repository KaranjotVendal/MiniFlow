import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np


def save_bar_plot(values: dict, title: str, ylabel: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(values.keys())
    vals = list(values.values())

    plt.figure(figsize=(8, 5))
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_line_plot(xs, ys, title: str, xlabel: str, ylabel: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_latency_percentiles_plot(samples: list, title: str, output_path: Path):
    """draw empirical CDF and mark p50, p95, p90.
    samples: list or numpy arr fo latency values"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(samples)
    arr = arr[~np.isnan(arr)]
    arr = np.sort(arr)
    if arr.size == 0:
        return None

    # empirical CDF
    y = np.arange(1, len(arr) + 1) / len(arr)
    plt.figure(figsize=(8, 5))
    plt.step(arr, y, where="post")
    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    # percentiles
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    p99 = np.percentile(arr, 99)

    for val, label, color in [
        (p50, "p50", "green"),
        (p95, "p95", "orange"),
        (p99, "p99", "red"),
    ]:
        plt.axvline(val, linestyle="--", color=color, linewidth=1)
        plt.text(
            val,
            0.05,
            f"{label}={val:.3f}s",
            rotation=90,
            color=color,
            va="bottom",
            ha="right",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
