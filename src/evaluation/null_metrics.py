"""Metrics for LINK/NULL classification and antecedent extraction."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def classification_metrics(
    gold_is_null: Sequence[bool],
    predicted_is_null: Sequence[bool],
) -> dict[str, Any]:
    if len(gold_is_null) != len(predicted_is_null):
        raise ValueError("Gold and predicted labels must have equal lengths")
    if not gold_is_null:
        return {
            "count": 0,
            "confusion_matrix": {"gold_link": {"pred_link": 0, "pred_null": 0},
                                 "gold_null": {"pred_link": 0, "pred_null": 0}},
            "link": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "null": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
        }

    matrix = {"gold_link": {"pred_link": 0, "pred_null": 0},
              "gold_null": {"pred_link": 0, "pred_null": 0}}
    for gold, predicted in zip(gold_is_null, predicted_is_null):
        matrix["gold_null" if gold else "gold_link"][
            "pred_null" if predicted else "pred_link"
        ] += 1

    def one_class(is_null: bool) -> dict[str, float]:
        true_positive = matrix["gold_null" if is_null else "gold_link"][
            "pred_null" if is_null else "pred_link"
        ]
        false_positive = matrix["gold_link" if is_null else "gold_null"][
            "pred_null" if is_null else "pred_link"
        ]
        false_negative = matrix["gold_null" if is_null else "gold_link"][
            "pred_link" if is_null else "pred_null"
        ]
        precision = true_positive / (true_positive + false_positive) if (
            true_positive + false_positive
        ) else 0.0
        recall = true_positive / (true_positive + false_negative) if (
            true_positive + false_negative
        ) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (
            precision + recall
        ) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    link = one_class(False)
    null = one_class(True)
    return {
        "count": len(gold_is_null),
        "confusion_matrix": matrix,
        "link": link,
        "null": null,
        "macro_f1": (link["f1"] + null["f1"]) / 2,
        "balanced_accuracy": (link["recall"] + null["recall"]) / 2,
    }


def evaluate_null_predictions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classification = classification_metrics(
        [bool(row["gold_is_null"]) for row in records],
        [bool(row["predicted_is_null"]) for row in records],
    )
    positive = [row for row in records if not row["gold_is_null"]]
    negative = [row for row in records if row["gold_is_null"]]
    forced_exact = sum(bool(row["forced_link_exact"]) for row in positive)
    actual_exact = sum(bool(row["end_to_end_correct"]) for row in positive)
    forced_f1 = (
        sum(float(row["forced_link_token_f1"]) for row in positive) / len(positive)
        if positive else 0.0
    )
    actual_f1 = (
        sum(
            0.0 if row["predicted_is_null"] else float(row["forced_link_token_f1"])
            for row in positive
        ) / len(positive)
        if positive else 0.0
    )
    strategy_groups: dict[str, Any] = {}
    for strategy in sorted({str(row.get("negative_strategy")) for row in negative}):
        group = [row for row in negative if row.get("negative_strategy") == strategy]
        metrics = classification_metrics(
            [True] * len(group),
            [bool(row["predicted_is_null"]) for row in group],
        )
        strategy_groups[strategy] = {
            "count": len(group),
            "null_recall": metrics["null"]["recall"],
            "false_link_rate": 1.0 - metrics["null"]["recall"],
        }
    return {
        **classification,
        "end_to_end_exact_accuracy": (
            sum(bool(row["end_to_end_correct"]) for row in records) / len(records)
            if records else 0.0
        ),
        "positive_relations": len(positive),
        "negative_relations": len(negative),
        "forced_link_span_em": forced_exact / len(positive) if positive else 0.0,
        "forced_link_token_f1": forced_f1,
        "actual_positive_span_em_after_null": actual_exact / len(positive) if positive else 0.0,
        "actual_positive_token_f1_after_null": actual_f1,
        "false_null_rate": (
            sum(bool(row["predicted_is_null"]) for row in positive) / len(positive)
            if positive else 0.0
        ),
        "negative_false_link_rate": (
            sum(not bool(row["predicted_is_null"]) for row in negative) / len(negative)
            if negative else 0.0
        ),
        "negative_strategy_metrics": strategy_groups,
    }


def calibration_metrics(records: Sequence[Mapping[str, Any]], bins: int = 10) -> dict[str, Any]:
    if bins < 1:
        raise ValueError("bins must be positive")
    if not records:
        return {"count": 0, "brier_score": 0.0, "reliability_bins": []}
    rows = sorted(records, key=lambda row: float(row["null_probability"]))
    result = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        group = [
            row for row in rows
            if lower <= float(row["null_probability"]) < upper
            or (index == bins - 1 and float(row["null_probability"]) == 1.0)
        ]
        if not group:
            continue
        result.append({
            "lower": lower,
            "upper": upper,
            "count": len(group),
            "mean_predicted_null_probability": sum(
                float(row["null_probability"]) for row in group
            ) / len(group),
            "observed_null_rate": sum(bool(row["gold_is_null"]) for row in group) / len(group),
        })
    return {
        "count": len(records),
        "brier_score": sum(
            (float(row["null_probability"]) - float(bool(row["gold_is_null"]))) ** 2
            for row in records
        ) / len(records),
        "reliability_bins": result,
    }


def stratified_performance(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    context_bands: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    negative_context_bands: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    candidate_counts: list[tuple[int, Mapping[str, Any]]] = []
    for row in records:
        count = int(row["context_token_count"])
        band = "1-32" if count <= 32 else (
            "33-64" if count <= 64 else ("65-128" if count <= 128 else "129+")
        )
        context_bands[band].append(row)
        if row["gold_is_null"]:
            negative_context_bands[band].append(row)
        candidate_counts.append((int(row["candidate_count"]), row))
    by_context_length = {}
    negative_by_context_length = {}
    for band in ("1-32", "33-64", "65-128", "129+"):
        group = context_bands.get(band, [])
        if group:
            by_context_length[band] = {
                "count": len(group),
                **classification_metrics(
                    [bool(row["gold_is_null"]) for row in group],
                    [bool(row["predicted_is_null"]) for row in group],
                ),
                "predicted_null_rate": sum(
                    bool(row["predicted_is_null"]) for row in group
                ) / len(group),
            }
        negative_group = negative_context_bands.get(band, [])
        if negative_group:
            negative_by_context_length[band] = {
                "count": len(negative_group),
                "null_recall": sum(
                    bool(row["predicted_is_null"]) for row in negative_group
                ) / len(negative_group),
                "false_link_rate": sum(
                    not bool(row["predicted_is_null"]) for row in negative_group
                ) / len(negative_group),
            }

    by_candidate_count = {}
    if candidate_counts:
        candidate_counts.sort(key=lambda item: item[0])
        for q in range(4):
            start = q * len(candidate_counts) // 4
            end = (q + 1) * len(candidate_counts) // 4
            group = [row for _, row in candidate_counts[start:end]]
            if group:
                by_candidate_count[f"q{q + 1}"] = {
                    "count": len(group),
                    "candidate_count_min": min(int(row["candidate_count"]) for row in group),
                    "candidate_count_max": max(int(row["candidate_count"]) for row in group),
                    **classification_metrics(
                        [bool(row["gold_is_null"]) for row in group],
                        [bool(row["predicted_is_null"]) for row in group],
                    ),
                    "predicted_null_rate": sum(
                        bool(row["predicted_is_null"]) for row in group
                    ) / len(group),
                }
    return {
        "by_context_length": by_context_length,
        "negative_only_by_context_length": negative_by_context_length,
        "by_candidate_count_quantile": by_candidate_count,
    }


def paired_contrast_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in records:
        pair_id = row.get("pair_id")
        if pair_id:
            pairs[str(pair_id)][
                "negative" if row["gold_is_null"] else "positive"
            ] = row
    complete = [
        pair for pair in pairs.values()
        if "positive" in pair and "negative" in pair
    ]
    counts = Counter()
    for pair in complete:
        positive_null = bool(pair["positive"]["predicted_is_null"])
        negative_null = bool(pair["negative"]["predicted_is_null"])
        if not positive_null and negative_null:
            counts["both_correct"] += 1
        elif not positive_null and not negative_null:
            counts["both_link"] += 1
        elif positive_null and negative_null:
            counts["both_null"] += 1
        else:
            counts["positive_null_negative_link"] += 1
    count = len(complete)
    return {
        "pair_count": count,
        "both_decisions_correct": counts["both_correct"],
        "paired_contrast_accuracy": counts["both_correct"] / count if count else 0.0,
        "both_predicted_link": counts["both_link"],
        "both_predicted_null": counts["both_null"],
        "positive_null_negative_link": counts["positive_null_negative_link"],
    }


def bootstrap_interval(
    values: Sequence[Mapping[str, Any]],
    metric: str,
    seed: int = 42,
    samples: int = 1000,
) -> dict[str, float]:
    if not values:
        return {"lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [values[rng.randrange(len(values))] for _ in values]
        metrics = evaluate_null_predictions(selected)
        estimates.append(float(metrics[metric]))
    estimates.sort()
    return {
        "lower": estimates[int(0.025 * (samples - 1))],
        "upper": estimates[int(0.975 * (samples - 1))],
    }


def paired_bootstrap_difference(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    metric: str,
    seed: int = 42,
    samples: int = 1000,
) -> dict[str, Any]:
    """Bootstrap the paired difference, aligning rows by example and relation."""
    def key(row: Mapping[str, Any]) -> tuple[str, int]:
        return (str(row["example_id"]), int(row["relation_id"]))

    second_by_key = {key(row): row for row in second}
    common = [
        (row, second_by_key[key(row)])
        for row in first
        if key(row) in second_by_key
    ]
    if not common:
        return {
            "count": 0,
            "difference": None,
            "ci95": None,
        }
    first_rows = [pair[0] for pair in common]
    second_rows = [pair[1] for pair in common]
    observed = (
        evaluate_null_predictions(first_rows)[metric]
        - evaluate_null_predictions(second_rows)[metric]
    )
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        indices = [rng.randrange(len(common)) for _ in common]
        first_sample = [first_rows[index] for index in indices]
        second_sample = [second_rows[index] for index in indices]
        estimates.append(
            evaluate_null_predictions(first_sample)[metric]
            - evaluate_null_predictions(second_sample)[metric]
        )
    estimates.sort()
    return {
        "count": len(common),
        "difference": observed,
        "ci95": [
            estimates[int(0.025 * (samples - 1))],
            estimates[int(0.975 * (samples - 1))],
        ],
        "interpretation": "paired bootstrap; descriptive, not a significance test",
    }

