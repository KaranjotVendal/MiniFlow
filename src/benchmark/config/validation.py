from dataclasses import dataclass, field


@dataclass
class MetricConfig:
    """Configuration for individual metrics.

    Attributes:
        enabled: Whether the metric is enabled.
        parameters: Additional metric-specific parameters.
    """

    enabled: bool = True
    parameters: dict = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Metrics configuration section.

    Attributes:
        enabled: List of metric names to enable.
        configurations: Per-metric configuration dictionaries.
    """

    enabled: list[str] = field(default_factory=list)
    configurations: dict[str, dict] = field(default_factory=dict)

    def get_config(self, metric_name: str) -> dict | None:
        """Get configuration for a specific metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Configuration dictionary for the metric.
        """
        return self.configurations.get(metric_name, None)


@dataclass
class BenchmarkConfig:
    """Root benchmark configuration.

    Attributes:
        experiment_name: Name of the experiment.
        num_samples: Number of samples to run.
        warmup_samples: Number of warmup samples before measurement.
        output_dir: Directory for output files.
        metrics: Metrics configuration.
        pipeline_config: Pipeline-specific configuration.
    """

    experiment_name: str
    num_samples: int = 20
    warmup_samples: int = 3
    output_dir: str = "./Benchmark"
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    pipeline_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "experiment_name": self.experiment_name,
            "num_samples": self.num_samples,
            "warmup_samples": self.warmup_samples,
            "output_dir": self.output_dir,
            "metrics": {
                "enabled": self.metrics.enabled,
                "configurations": self.metrics.configurations,
            },
            "pipeline_config": self.pipeline_config,
        }
