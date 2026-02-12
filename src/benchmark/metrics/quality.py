from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jiwer
import utmosv2
import torch
import torchaudio

from src.benchmark.core.base import BaseMetric, MetricContext
from src.benchmark.core.registry import MetricRegistry


class BaseEvaluator(ABC):
    """Abstract base class for quality evaluators.

    All evaluators must implement the `name` attribute and `evaluate()` method.
    This allows pluggable quality metrics that can be added without modifying
    the core QualityMetrics class.

    Attributes:
        name: Unique identifier for the evaluator.
    """

    NAME: str

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        """Evaluate prediction against reference.

        Args:
            prediction: The predicted/output text or value.
            reference: The ground truth/reference text or value.

        Returns:
            Quality score (lower is better for WER, higher is better for UTMOS).
        """
        pass


class WEREvaluator(BaseEvaluator):
    """WER is the standard metric for ASR quality, calculated as:
    WER = (Substitutions + Insertions + Deletions) / (Words in Reference)

    Attributes:
        NAME: Unique identifier ("wer").

    NOTE: jiwer.wer case-sensitivity question: jiwer.wer IS case-sensitive by default (1.0 WER).
    Our WEREvaluator applies .lower() making it case-insensitive.
    """

    NAME: str = "wer"

    def evaluate(self, prediction: str, reference: str | None = None) -> float:
        """Calculate WER between prediction and reference.

        Args:
            prediction: The predicted ASR output.
            reference: The ground truth transcription.

        Returns:
            Word Error Rate as a float (0.0 = perfect, higher = more errors).
        """
        if reference is None:
            reference = ""
        wer = jiwer.wer(reference.lower(), prediction.lower())
        return round(wer, 4)


class UTMOSEvaluator(BaseEvaluator):
    """UTMOS is a neural network-based speech quality prediction model.

    UTMOS was trained on 16kHz audio. Input sample rate must be >= 16kHz.

    Attributes:
        NAME: Unique identifier ("utmos").
    """

    NAME: str = "utmos"
    # Minimum sample rate UTMOS was trained on
    MIN_SAMPLE_RATE: int = 16000

    def _utmos_evaluate(
        self, waveform: torch.Tensor, output_sampling_rate: int
    ) -> float:
        """Calculate UTMOS score for a synthesized waveform.

        Args:
            waveform: The synthesized audio waveform.
            output_sampling_rate: Sample rate of the audio (must be >= 16kHz).

        Returns:
            MOS score (typically in range 1.0-5.0).

        Raises:
            ValueError: If output_sampling_rate is less than 16kHz.
        """
        if output_sampling_rate < self.MIN_SAMPLE_RATE:
            raise ValueError(
                f"UTMOS requires sample rate >= {self.MIN_SAMPLE_RATE}Hz, "
                f"got {output_sampling_rate}Hz"
            )

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "temp_audio.wav"
            torchaudio.save(
                str(temp_path),
                waveform.unsqueeze(0) if waveform.dim() == 1 else waveform,
                output_sampling_rate,
            )

            utmos = utmosv2.create_model(pretrained=True)
            result = utmos.predict(input_path=str(temp_path))
            mos = result if isinstance(result, (int, float)) else 0.0

        return round(float(mos), 2)

    def evaluate(self, waveform: torch.Tensor, output_sample_rate: int) -> float:
        """Calculate UTMOS score for synthesized speech.

        Args:
            prediction: The synthesized speech waveform tensor.
            reference: Optional reference for comparison (not used).

        Returns:
            MOS score in range 1.0-5.0 (higher is better),
            or 0.0 when utmosv2 is unavailable.
        """
        return self._utmos_evaluate(waveform, output_sample_rate)


_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "wer": WEREvaluator,
    "utmos": UTMOSEvaluator,
}


def get_evaluator(name: str) -> type[BaseEvaluator]:
    """Get an evaluator class by name."""
    return _EVALUATOR_REGISTRY[name]


@MetricRegistry.register("quality")
class QualityMetrics(BaseMetric):
    """Quality evaluation metric with pluggable evaluator system.

    This metric evaluates output quality using domain-specific evaluators
    like WER for ASR and UTMOS for speech quality (TTS). The evaluator system
    is pluggable, allowing custom evaluators to be added.

    Attributes:
        evaluators: List of evaluator instances to run.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize QualityMetrics with configuration.

        Args:
            config: Configuration dictionary with options:
                - evaluators: List of evaluator names to enable (default: ["wer"])
        """
        super().__init__(config)
        evaluators_list = self.config["evaluators"]
        self.evaluators: dict[str, BaseEvaluator] = self._create_evaluators(
            names=evaluators_list
        )

    def _create_evaluators(self, names: list[str]) -> dict[str, BaseEvaluator]:
        """Create evaluator instances from names."""
        evaluators = {}
        for name in names:
            evaluator_class = get_evaluator(name)
            evaluators[name] = evaluator_class()

        return evaluators

    def evaluate(
        self,
        evaluator: str,
        prediction: Any,
        reference: Any = None,
        output_sample_rate: int | None = None,
    ) -> dict[str, float]:
        """Evaluate prediction against reference using a specific evaluator.

        Args:
            evaluator: Name of the evaluator to use (e.g., "wer", "utmos").
            prediction: The predicted/output text or value.
            reference: The ground truth/reference text or value (optional).

        Returns:
            Dictionary mapping evaluator name to score.
        """
        if evaluator == WEREvaluator.NAME:
            assert output_sample_rate is None, (
                "WER Evaluator does not use output_sample_rate"
            )
            score = self.evaluators[evaluator].evaluate(prediction, reference)

        elif evaluator == UTMOSEvaluator.NAME:
            assert reference is None and output_sample_rate is not None, (
                "UTMOS Evaluator requires output_sample_rate and not reference (GT)"
            )
            score = self.evaluators[evaluator].evaluate(prediction, output_sample_rate)
        else:
            raise ValueError(f"Unknown evaluator: {evaluator}")

        return {evaluator: score}

    def start(self, context: MetricContext) -> None:
        pass

    def end(self, context: MetricContext) -> None:
        pass
