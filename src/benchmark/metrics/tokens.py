import time

from src.benchmark.core.base import BaseMetric, MetricContext
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.metrics.result_models import TokenResult


@MetricRegistry.register("tokens")
class TokenMetrics(BaseMetric):
    """LLM token-level metrics for tracking token generation performance.

    This metric tracks tokens generated, time to first token (TTFT), and
    tokens per second (TPS). It supports both streaming and non-streaming
    token generation scenarios.

    Attributes:
        track_ttft: Whether to track time to first token (default: True).
        _token_count: Internal counter for tokens generated.
        _first_token_time: Time when first token was generated (None if not yet).
        _start_time: High-precision start time for measurement.
    """

    def __init__(self, config: dict | None = None):
        """Initialize token metrics with configuration.

        Args:
            config: Configuration dictionary with options:
                - track_ttft: Track time to first token (default: True)
        """
        super().__init__(config)
        self.track_ttft: bool = config["track_ttft"]
        self._token_count: int = 0
        self._first_token_time: float | None = None
        self._start_time: float | None = None
        self._last_result: TokenResult | None = None

    def start(self, context: MetricContext) -> None:
        """Initialize tracking for a new generation.

        Args:
            context: The current metric context containing stage and trial info.
        """
        self._token_count = 0
        self._first_token_time = None
        self._start_time = time.perf_counter()
        self._last_result = None

    def on_token_generated(self, token: str | None = None) -> None:
        """Record the generation of a single token.

        This method should be called for each token generated during streaming
        inference. It increments the token count and captures the time to first
        token if TTFT tracking is enabled.

        Args:
            token: The token that was generated (optional, for compatibility).
        """
        self._token_count += 1
        if self.track_ttft and self._first_token_time is None:
            self._first_token_time = time.perf_counter() - self._start_time

    def add_tokens(self, count: int) -> None:
        """Add token count for non-streaming mode.

        In non-streaming scenarios where tokens are generated in a batch,
        use this method to add the total token count after generation completes.

        Args:
            count: Number of tokens to add.
        """
        self._token_count += count

    def end(self, context: MetricContext) -> dict:
        """Calculate and return token metrics for the generation.

        Args:
            context: The current metric context containing stage and trial info.

        Returns:
            Dictionary containing:
                - tokens_generated: Total number of tokens generated
                - ttft: Time to first token in seconds (None if not tracked or no tokens)
                - tokens_per_sec: Tokens generated per second (excludes TTFT time)
                - time_per_token: Average time per token in seconds
                - total_generation_time: Total generation time in seconds
        """
        end_time = time.perf_counter()

        if self._start_time is None:
            self._last_result = TokenResult(
                tokens_generated=0,
                ttft=None,
                tokens_per_sec=0.0,
                time_per_token=0.0,
                total_generation_time=0.0,
            )
            return self._last_result.to_dict()

        total_time = end_time - self._start_time
        if self._token_count == 0:
            self._last_result = TokenResult(
                tokens_generated=0,
                ttft=None,
                tokens_per_sec=0.0,
                time_per_token=0.0,
                total_generation_time=round(total_time, 6),
            )
            return self._last_result.to_dict()

        generation_time = total_time - (self._first_token_time or 0)

        if generation_time <= 0:
            tokens_per_sec = 0.0
            time_per_token = 0.0
        else:
            tokens_per_sec = self._token_count / generation_time
            time_per_token = generation_time / self._token_count

        self._last_result = TokenResult(
            tokens_generated=self._token_count,
            ttft=self._first_token_time,
            tokens_per_sec=round(tokens_per_sec, 4),
            time_per_token=round(time_per_token, 6),
            total_generation_time=round(total_time, 6),
        )
        return self._last_result.to_dict()

    def to_result(self) -> TokenResult:
        if self._last_result is None:
            raise RuntimeError("TokenMetrics result is unavailable before end().")
        return self._last_result
