from pathlib import Path

import torch

from src.benchmark.collectors import BenchmarkCollector
from src.benchmark.core.base import BaseMetric
from src.benchmark.core.registry import MetricRegistry
from src.config.load_config import load_yaml_config
from src.utils import get_device

# Import side effects: register all metric classes in MetricRegistry.
import src.benchmark.metrics  # noqa: F401


def build_metric_instances(
    metrics_config_path: str | Path = "configs/metrics.yml",
    device: torch.device | str | None = None,
) -> dict[str, BaseMetric]:
    metrics_cfg = load_yaml_config(metrics_config_path)
    enabled_metrics: list[str] = metrics_cfg["enabled"]
    metric_configurations: dict[str, dict] = metrics_cfg["configurations"]

    runtime_device = get_device() if device is None else device
    if "hardware_basic" in metric_configurations:
        metric_configurations["hardware_basic"]["device"] = runtime_device
    if "hardware_detailed" in metric_configurations:
        metric_configurations["hardware_detailed"]["device"] = runtime_device

    metrics: dict[str, BaseMetric] = {}
    for metric_name in enabled_metrics:
        metric_class = MetricRegistry.get(metric_name)
        metrics[metric_name] = metric_class(metric_configurations[metric_name])
    return metrics


def build_collector(
    config_path: str | Path,
    metrics_config_path: str | Path = "configs/metrics.yml",
) -> tuple[dict, BenchmarkCollector, torch.device | str]:
    config = load_yaml_config(config_path)
    device = get_device()
    metrics = build_metric_instances(metrics_config_path=metrics_config_path, device=device)
    collector = BenchmarkCollector(metrics=metrics, config=config)
    return config, collector, device
