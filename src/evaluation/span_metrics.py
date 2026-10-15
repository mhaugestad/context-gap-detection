"""Shared relation-level metrics for antecedent span extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random
from typing import Any


LENGTH_BANDS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9+", 9, None),
)


def length_band(token_length: int) -> str:
    if token_length < 1:
        raise ValueError("token_length must be positive")
    for name, lower, upper in LENGTH_BANDS:
        if token_length >= lower and (upper is None or token_length <= upper):
            return name
    raise AssertionError("unreachable length band")


def _overlap_f1(predicted: tuple[int, int], gold: tuple[int, int]) -> float:
    overlap = max(0, min(predicted[1], gold[1]) - max(predicted[0], gold[0]))
    predicted_length = predicted[1] - predicted[0]
    gold_length = gold[1] - gold[0]
    if predicted_length < 1 or gold_length < 1:
        return 0.0
    precision = overlap / predicted_length
    recall = overlap / gold_length
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_span_prediction(
    predicted_span: tuple[int, int],
    gold_spans: Sequence[tuple[int, int]],
    predicted_char_span: tuple[int, int] | None = None,
    gold_char_spans: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Score one half-open token span against all alternative gold spans."""
    if predicted_span[0] < 0 or predicted_span[0] >= predicted_span[1]:
        raise ValueError(f"Invalid predicted span {predicted_span}")
    if not gold_spans:
        raise ValueError("At least one gold antecedent span is required")
    if any(start < 0 or start >= end for start, end in gold_spans):
        raise ValueError("Gold antecedent spans must be non-empty half-open spans")

    f1s = [_overlap_f1(predicted_span, gold) for gold in gold_spans]
    exact = any(predicted_span == gold for gold in gold_spans)
    localization = any(
        max(predicted_span[0], gold[0]) < min(predicted_span[1], gold[1])
        for gold in gold_spans
    )
    start_correct = any(predicted_span[0] == gold[0] for gold in gold_spans)
    end_correct = any(predicted_span[1] == gold[1] for gold in gold_spans)
    character_exact = None
    if predicted_char_span is not None and gold_char_spans is not None:
        character_exact = predicted_char_span in gold_char_spans

    return {
        "exact_match": exact,
        "token_overlap_f1": max(f1s),
        "localization_hit": localization,
        "start_boundary_correct": start_correct,
        "end_boundary_correct": end_correct,
        "character_exact_match": character_exact,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total")
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def aggregate_span_metrics(
    scored: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate relation-level records, including uncertainty for rates."""
    total = len(scored)
    result: dict[str, Any] = {"total_relations": total}
    for key, output_key in (
        ("exact_match", "exact_span_match"),
        ("localization_hit", "localization_hit"),
        ("start_boundary_correct", "start_boundary_accuracy"),
        ("end_boundary_correct", "end_boundary_accuracy"),
        ("character_exact_match", "character_exact_match"),
    ):
        applicable = [
            bool(record[key])
            for record in scored
            if record.get(key) is not None
        ]
        correct = sum(applicable)
        denominator = len(applicable)
        result[output_key] = {
            "correct": correct,
            "total": denominator,
            "value": correct / denominator if denominator else None,
            "ci95": wilson_interval(correct, denominator),
        }
    f1_values = [float(record["token_overlap_f1"]) for record in scored]
    f1_interval = None
    if f1_values:
        rng = random.Random(42)
        bootstrap_means = sorted(
            sum(f1_values[rng.randrange(len(f1_values))] for _ in f1_values)
            / len(f1_values)
            for _ in range(5000)
        )
        f1_interval = [
            bootstrap_means[int(0.025 * (len(bootstrap_means) - 1))],
            bootstrap_means[int(0.975 * (len(bootstrap_means) - 1))],
        ]
    result["token_overlap_f1"] = {
        "mean": sum(f1_values) / len(f1_values) if f1_values else None,
        "total": len(f1_values),
        "bootstrap_ci95": f1_interval,
    }
    return result


def paired_comparison(
    first: Sequence[bool],
    second: Sequence[bool],
    first_name: str,
    second_name: str,
    seed: int = 42,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    """Summarize a paired binary comparison with a bootstrap interval."""
    if len(first) != len(second):
        raise ValueError("Paired comparison inputs must have equal lengths")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    total = len(first)
    first_wins = sum(a and not b for a, b in zip(first, second))
    second_wins = sum(b and not a for a, b in zip(first, second))
    both_correct = sum(a and b for a, b in zip(first, second))
    both_incorrect = sum(not a and not b for a, b in zip(first, second))
    differences = [int(a) - int(b) for a, b in zip(first, second)]
    observed = sum(differences) / total if total else None
    interval = None
    if total:
        rng = random.Random(seed)
        bootstrap_means = []
        for _ in range(bootstrap_samples):
            sample = [differences[rng.randrange(total)] for _ in range(total)]
            bootstrap_means.append(sum(sample) / total)
        bootstrap_means.sort()
        low = bootstrap_means[int(0.025 * (bootstrap_samples - 1))]
        high = bootstrap_means[int(0.975 * (bootstrap_samples - 1))]
        interval = [low, high]
    return {
        "first": first_name,
        "second": second_name,
        "total_relations": total,
        "first_only_correct": first_wins,
        "second_only_correct": second_wins,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "exact_match_difference": observed,
        "bootstrap_ci95": interval,
        "interpretation": "descriptive paired comparison; no significance claim",
    }
