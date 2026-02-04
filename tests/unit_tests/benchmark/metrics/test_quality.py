import pytest
import torch

from src.benchmark.core.base import MetricContext, Stage
from src.benchmark.core.registry import MetricRegistry
from src.benchmark.metrics.quality import (
    QualityMetrics,
    WEREvaluator,
    UTMOSEvaluator,
    get_evaluator,
)


def test_registers_quality():
    """Verify registration with registry."""
    metric_class = MetricRegistry.get("quality")
    assert metric_class is QualityMetrics


@pytest.mark.parametrize(
    "prediction,reference,expected_wer",
    [
        ("hello world", "hello world", 0.0),
        ("hello planet", "hello world", 0.5),
        ("foo bar", "hello world", 1.0),
        ("hello world", "", 2.0),
        ("", "hello world", 1.0),
        ("HELLO WORLD", "hello world", 0.0),
        ("hello world.", "hello world", 0.5),
    ],
)
def test_wer_evaluator(prediction, reference, expected_wer):
    """Test WER evaluation with various inputs.

    WER (Word Error Rate) = (Substitutions + Insertions + Deletions) / Words in Reference

    Cases:
    - Perfect match: 0 errors / 2 words = 0.0
    - One error: 1 substitution / 2 words = 0.5
    - Complete mismatch: 2 substitutions / 2 words = 1.0
    - Empty reference: 2 insertions / 0 words = 2.0 (jiwer behavior)
    - Empty prediction: 2 deletions / 2 words = 1.0
    - Case insensitive: .lower() applied = 0.0
    - Punctuation: "." treated as separate token = 0.5
    """
    evaluator = WEREvaluator()
    wer = evaluator.evaluate(prediction, reference)
    assert wer == expected_wer


def test_wer_evaluator_name():
    """Test evaluator has correct name."""
    evaluator = WEREvaluator()
    assert evaluator.NAME == "wer"


@pytest.mark.parametrize(
    "waveform,sample_rate",
    [
        (torch.randn(16000), 16000),
        (torch.randn(24000), 24000),
    ],
)
def test_utmos_evaluator(waveform, sample_rate):
    """Test UTMOS evaluation returns a valid score.

    Sample rates tested:
    - 16000: Native UTMOS training sample rate
    - 24000: XTTS output sample rate used in MiniFlow

    Note: UTMOS may produce values slightly outside [1.0, 5.0] for synthetic
    inputs (e.g., random noise). We check that the score is in a reasonable
    range for production audio quality assessment.
    """
    evaluator = UTMOSEvaluator()
    score = evaluator.evaluate(waveform, sample_rate)
    assert isinstance(score, float)
    # UTMOS scores are typically in [1.0, 5.0], but may vary slightly
    # for synthetic test inputs. Allow small tolerance for edge cases.
    assert 0.5 <= score <= 5.5, f"UTMOS score {score} is out of expected range"


def test_utmos_evaluator_name():
    """Test evaluator has correct name."""
    evaluator = UTMOSEvaluator()
    assert evaluator.NAME == "utmos"


@pytest.mark.parametrize(
    "name,expected_class",
    [
        ("wer", WEREvaluator),
        ("utmos", UTMOSEvaluator),
    ],
)
def test_get_evaluator(name, expected_class):
    """Test getting evaluator classes from registry."""
    evaluator_class = get_evaluator(name)
    assert evaluator_class is expected_class


def test_get_nonexistent_evaluator_raises_key_error():
    """Test getting non-existent evaluator raises KeyError."""
    with pytest.raises(KeyError):
        get_evaluator("nonexistent")


@pytest.mark.parametrize(
    "config,expected_count,expected_keys",
    [
        (None, 1, {"wer"}),
        ({}, 1, {"wer"}),
        ({"evaluators": ["wer", "utmos"]}, 2, {"wer", "utmos"}),
        ({"evaluators": ["wer"]}, 1, {"wer"}),
    ],
)
def test_quality_metrics_config(config, expected_count, expected_keys):
    """Test QualityMetrics configuration with various configs."""
    metric = QualityMetrics(config)
    assert len(metric.evaluators) == expected_count
    assert set(metric.evaluators.keys()) == expected_keys


def test_unknown_evaluator_raises_key_error():
    """Test that unknown evaluators raise KeyError."""
    config = {"evaluators": ["wer", "nonexistent"]}
    with pytest.raises(KeyError):
        QualityMetrics(config)


@pytest.mark.parametrize(
    "prediction,reference,expected_score",
    [
        ("hello world", "hello world", 0.0),
        ("hello planet", "hello world", 0.5),
    ],
)
def test_wer_evaluation(prediction, reference, expected_score):
    """Test evaluating WER through QualityMetrics."""
    metric = QualityMetrics({"evaluators": ["wer"]})
    scores = metric.evaluate("wer", prediction, reference=reference)
    assert scores["wer"] == expected_score


def test_utmos_evaluation(tmp_path):
    """Test evaluating UTMOS through QualityMetrics."""
    metric = QualityMetrics({"evaluators": ["utmos"]})
    waveform = torch.randn(16000)
    scores = metric.evaluate("utmos", waveform, output_sample_rate=16000)
    assert "utmos" in scores
    assert isinstance(scores["utmos"], float)


def test_multiple_evaluators(tmp_path):
    """Test evaluating with multiple evaluators."""
    metric = QualityMetrics({"evaluators": ["wer", "utmos"]})
    waveform = torch.randn(16000)

    wer_scores = metric.evaluate("wer", "hello world", reference="hello world")
    utmos_scores = metric.evaluate("utmos", waveform, output_sample_rate=16000)

    assert "wer" in wer_scores
    assert "utmos" in utmos_scores


def test_evaluate_returns_scores_dict():
    """Test evaluate() returns individual scores as dict."""
    metric = QualityMetrics({"evaluators": ["wer"]})
    scores = metric.evaluate("wer", "hello world", reference="hello world")
    assert isinstance(scores, dict)
    assert "wer" in scores


def test_wer_with_none_reference():
    """Test evaluate() handles None reference for WER.

    WEREvaluator treats None as empty string which results in WER = 2.0.
    """
    metric = QualityMetrics({"evaluators": ["wer"]})
    scores = metric.evaluate("wer", "hello world", reference=None)
    assert scores["wer"] == 2.0


def test_unknown_evaluator_raises_value_error():
    """Test evaluate() raises ValueError for unknown evaluator."""
    metric = QualityMetrics({"evaluators": ["wer"]})
    with pytest.raises(ValueError, match="Unknown evaluator"):
        metric.evaluate("unknown", "test")


def test_wer_wrong_params_raises_assertion():
    """Test WER requires reference and not output_sample_rate."""
    metric = QualityMetrics({"evaluators": ["wer"]})
    with pytest.raises(AssertionError):
        metric.evaluate("wer", "hello", reference="world", output_sample_rate=16000)


def test_utmos_wrong_params_raises_assertion():
    """Test UTMOS requires output_sample_rate and not reference."""
    metric = QualityMetrics({"evaluators": ["utmos"]})
    waveform = torch.randn(16000)
    with pytest.raises(AssertionError):
        metric.evaluate("utmos", waveform, reference="should not have this")
