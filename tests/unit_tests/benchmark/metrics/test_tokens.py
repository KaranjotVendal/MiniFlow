import pytest
import time

from src.benchmark.core.base import MetricContext, Stage
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.metrics.tokens import TokenMetrics


def test_registers_tokens():
    """Verify registration with registry."""
    metric_class = MetricRegistry.get("tokens")
    assert metric_class is TokenMetrics


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"track_ttft": True}, True),
        ({"track_ttft": False}, False),
    ],
)
def test_config_options(config, expected):
    """Test configuration options with parametrization."""
    metric = TokenMetrics(config)
    assert metric.track_ttft is expected


class TestTokenMetricsBasic:
    """Test basic token counting and metrics."""

    def test_token_counting(self):
        """Test that tokens are counted correctly."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        metric.on_token_generated("hello")
        metric.on_token_generated(" ")
        metric.on_token_generated("world")

        results = metric.end(context)

        assert results["tokens_generated"] == 3

    def test_add_tokens_non_streaming(self):
        """Test add_tokens for non-streaming mode."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        metric.add_tokens(10)

        results = metric.end(context)

        assert results["tokens_generated"] == 10

    def test_ttft_calculation(self):
        """Test TTFT is calculated correctly."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        time.sleep(0.01)
        metric.on_token_generated("first")

        results = metric.end(context)

        # TTFT should be ~0.01s (sleep time), bounded reasonably
        assert 0.01 <= results["ttft"] <= 0.1

    def test_tps_calculation(self):
        """Test tokens per second is calculated correctly (excludes TTFT)."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        time.sleep(0.01)
        metric.on_token_generated("first")
        time.sleep(0.04)
        metric.on_token_generated("second")
        metric.on_token_generated("third")
        metric.on_token_generated("fourth")

        results = metric.end(context)

        assert results["tokens_generated"] == 4
        assert results["ttft"] >= 0.01
        assert results["ttft"] <= 0.02
        assert results["tokens_per_sec"] > 0
        assert results["tokens_per_sec"] < 100


class TestTokenMetricsEdgeCases:
    """Test edge cases and error handling."""

    def test_no_tokens_generated(self):
        """Test metrics when no tokens are generated."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)
        time.sleep(0.01)

        results = metric.end(context)

        assert results["tokens_generated"] == 0
        assert results["ttft"] is None
        assert results["tokens_per_sec"] == 0.0
        assert results["time_per_token"] == 0.0
        assert results["total_generation_time"] >= 0.01

    def test_ttft_disabled(self):
        """Test with TTFT tracking disabled."""
        metric = TokenMetrics({"track_ttft": False})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        time.sleep(0.01)
        metric.on_token_generated("first")

        results = metric.end(context)

        assert results["ttft"] is None
        assert results["tokens_generated"] == 1

    def test_single_token(self):
        """Test with only one token generated."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        time.sleep(0.01)
        metric.on_token_generated("only")

        results = metric.end(context)

        assert results["tokens_generated"] == 1
        assert results["ttft"] is not None
        assert results["ttft"] >= 0.01
        assert results["tokens_per_sec"] > 0

    # Some tokenizers yield empty strings or None for special tokens
    def test_empty_token_handling(self):
        """Test that empty tokens and None values are handled gracefully."""
        metric = TokenMetrics({"track_ttft": True})
        context = MetricContext(
            stage=Stage.LLM,
            trial_id="trial_001",
            config={},
            timestamp=0.0,
        )
        metric.start(context)

        metric.on_token_generated("")
        metric.on_token_generated(None)

        results = metric.end(context)

        assert results["tokens_generated"] == 2


def test_streaming_token_generation():
    """Test typical streaming token generation scenario."""
    metric = TokenMetrics({"track_ttft": True})
    context = MetricContext(
        stage=Stage.LLM,
        trial_id="trial_001",
        config={},
        timestamp=0.0,
    )
    metric.start(context)

    tokens = [
        "The",
        " quick",
        " brown",
        " fox",
        " jumps",
        " over",
        " the",
        " lazy",
        " dog",
    ]
    for token in tokens:
        metric.on_token_generated(token)
        time.sleep(0.005)

    results = metric.end(context)

    assert results["tokens_generated"] == len(tokens)
    assert results["ttft"] is not None
    assert results["ttft"] < 0.05
    assert results["tokens_per_sec"] > 0
    assert results["total_generation_time"] >= 0.04


def test_multiple_trials_independent():
    """Verify trials have independent tracking."""
    metric = TokenMetrics({"track_ttft": True})
    context1 = MetricContext(
        stage=Stage.LLM,
        trial_id="trial_001",
        config={},
        timestamp=0.0,
    )
    context2 = MetricContext(
        stage=Stage.LLM,
        trial_id="trial_002",
        config={},
        timestamp=0.0,
    )

    metric.start(context1)
    metric.on_token_generated("a")
    metric.on_token_generated("b")
    results1 = metric.end(context1)

    metric.start(context2)
    metric.add_tokens(5)
    results2 = metric.end(context2)

    assert results1["tokens_generated"] == 2
    assert results2["tokens_generated"] == 5


def test_combined_streaming_and_non_streaming():
    """Test combining on_token_generated with add_tokens."""
    metric = TokenMetrics({"track_ttft": True})
    context = MetricContext(
        stage=Stage.LLM,
        trial_id="trial_001",
        config={},
        timestamp=0.0,
    )
    metric.start(context)

    metric.on_token_generated("first")
    time.sleep(0.01)
    metric.add_tokens(5)

    results = metric.end(context)

    assert results["tokens_generated"] == 6
    assert results["ttft"] is not None
