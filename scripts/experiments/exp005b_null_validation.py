"""Validate the frozen EXP-005 conditioned NULL model without retraining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.experiments.exp005_null_antecedent import (  # noqa: E402
    _data_loader,
    _enrich_predictions,
    _load_span_model,
    _preprocess_split,
    _span_to_char,
    _tensor_batch,
)
from src.data.null_antecedent import _encode_null_relation  # noqa: E402
from src.evaluation.null_metrics import evaluate_null_predictions  # noqa: E402
from src.models.null_antecedent import NullAntecedentModel  # noqa: E402
from src.training.null_module import NullAntecedentLightningModule  # noqa: E402


SOURCE_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "exp005_null_antecedent"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "exp005b_null_validation"
DEFAULT_MODEL_CHECKPOINT = (
    SOURCE_OUTPUT_DIR / "conditioned" / "checkpoints" / "best.ckpt"
)
REVIEW_COLUMNS = (
    "example_id",
    "source_example_id",
    "relation_id",
    "negative_generation_strategy",
    "context",
    "chunk",
    "gold_anaphor",
    "predicted_output",
    "manual_label",
    "review_status",
    "justification",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, default=SOURCE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--accelerator", choices=("auto", "cpu", "mps", "gpu"), default="auto"
    )
    return parser.parse_args(argv)


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"EXP-005 configuration not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in EXP-005 configuration: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU was requested but CUDA is not available")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_conditioned_model(
    model_checkpoint: Path,
    base_checkpoint: Path,
    dropout: float,
    max_span_length: int,
) -> tuple[NullAntecedentModel, str, str]:
    if not model_checkpoint.is_file():
        raise FileNotFoundError(
            f"Frozen EXP-005 conditioned checkpoint not found: {model_checkpoint}"
        )
    span_model = _load_span_model(base_checkpoint)
    if span_model.max_span_length != max_span_length:
        raise ValueError(
            "The EXP-004 checkpoint max_span_length does not match EXP-005 config"
        )
    model = NullAntecedentModel(
        span_model=span_model,
        approach="conditioned",
        dropout=dropout,
        freeze_span_model=True,
    )
    module = NullAntecedentLightningModule.load_from_checkpoint(
        model_checkpoint,
        map_location="cpu",
        model=model,
    )
    if module.model.approach != "conditioned":
        raise ValueError("The supplied EXP-005 checkpoint is not conditioned")
    if any(parameter.requires_grad for parameter in module.model.span_model.parameters()):
        raise RuntimeError("The EXP-004 span model must remain frozen")
    module.eval()
    return module.model, _sha256(model_checkpoint), _sha256(base_checkpoint)


def _evaluate_with_competitor_scores(
    model: NullAntecedentModel,
    loader: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval().to(device)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            device_batch = _tensor_batch(batch, device)
            output = model(device_batch)
            batch_records = NullAntecedentLightningModule.records_from_batch(
                output, device_batch
            )
            by_location = {
                (int(item["batch_index"]), int(item["relation_index"])): item
                for item in output["relations"]
            }
            for record in batch_records:
                relation = by_location[
                    (record["batch_index"], record["relation_index"])
                ]
                ordered = relation["scores"].detach().float().sort(descending=True).values
                second_score = (
                    float(ordered[1].cpu().item()) if ordered.numel() > 1 else None
                )
                second_index = (
                    int(relation["scores"].detach().float().argmax().item())
                    if ordered.numel() == 1 else int(
                        relation["scores"].detach().float().argsort(descending=True)[1].item()
                    )
                )
                second_token_span = relation["candidate_spans"][second_index].tolist()
                second_char_span = _span_to_char(
                    batch["offset_mapping"][int(record["batch_index"])],
                    second_token_span,
                )
                context = batch["context"][int(record["batch_index"])]
                record["second_highest_span_score"] = second_score
                record["second_highest_span_text"] = (
                    context[second_char_span[0]:second_char_span[1]]
                    if second_char_span else None
                )
                record["top_two_span_score_gap"] = (
                    float(ordered[0].cpu().item()) - second_score
                    if second_score is not None else None
                )
            records.extend(
                _enrich_predictions(batch_records, [batch], "conditioned")
            )
    return records


def _rebuild_original_dev_negatives(
    archived_records: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """Recreate the exact EXP-005 negative contexts, not a new synthetic sample."""
    relation_by_key = {
        (str(example["id"]), int(relation["relation_id"])): relation
        for example in positives
        for relation in example["relations"]
    }
    example_by_id = {str(example["id"]): example for example in positives}
    positive_records = [row for row in archived_records if not row["gold_is_null"]]
    if len(positive_records) != len(relation_by_key):
        raise ValueError(
            "The archived EXP-005 and current dev positive-relation counts differ"
        )
    for row in positive_records:
        key = (str(row["source_example_id"]), int(row["relation_id"]))
        example = example_by_id.get(key[0])
        if example is None:
            raise ValueError(f"Archived positive source is missing from dev data: {key}")
        if str(example["context"]) != str(row["context_text"]) or str(
            example["chunk"]
        ) != str(row["chunk_text"]):
            raise ValueError(
                "Current dev text differs from the archived EXP-005 predictions"
            )
        if key not in relation_by_key:
            raise ValueError(
                f"Archived positive relation is missing from dev data: {key}"
            )

    negatives: list[dict[str, Any]] = []
    negative_keys: set[tuple[str, int]] = set()
    for row in archived_records:
        if not row["gold_is_null"]:
            continue
        key = (str(row["source_example_id"]), int(row["relation_id"]))
        if key in negative_keys:
            raise ValueError(f"Duplicate archived synthetic negative relation: {key}")
        negative_keys.add(key)
        relation = relation_by_key.get(key)
        if relation is None:
            raise ValueError(f"Archived synthetic negative has no source relation: {key}")
        encoded = _encode_null_relation(
            str(row["context_text"]),
            str(row["chunk_text"]),
            relation,
            tokenizer,
            max_length,
            str(row["source_example_id"]),
            str(row["replacement_source_id"]),
            str(row["negative_strategy"]),
            "dev",
        )
        if encoded is None:
            raise ValueError(
                f"Archived synthetic negative no longer aligns to its anaphor: {key}"
            )
        negatives.append(encoded)
    return negatives


def _strategy_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(
        str(row["negative_strategy"])
        for row in rows
        if row.get("gold_is_null") and row.get("negative_strategy")
    )
    if not counts:
        raise ValueError("Each source group must include a synthetic negative")
    return "|".join(f"{name}:{counts[name]}" for name in sorted(counts))


def source_grouped_stratified_split(
    records: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split source examples, stratifying by their negative-strategy mix."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_id = record.get("source_example_id")
        if source_id is None:
            raise ValueError("Every prediction requires source_example_id")
        groups[str(source_id)].append(dict(record))

    strata: dict[str, list[str]] = defaultdict(list)
    for source_id, rows in groups.items():
        strata[_strategy_signature(rows)].append(source_id)

    rng = random.Random(seed)
    calibration_ids: set[str] = set()
    stratum_counts: dict[str, dict[str, int]] = {}
    for signature in sorted(strata):
        source_ids = sorted(strata[signature])
        rng.shuffle(source_ids)
        calibration_count = len(source_ids) // 2
        calibration_ids.update(source_ids[:calibration_count])
        stratum_counts[signature] = {
            "source_groups": len(source_ids),
            "calibration_source_groups": calibration_count,
            "evaluation_source_groups": len(source_ids) - calibration_count,
        }

    calibration = [
        row for source_id, rows in groups.items()
        if source_id in calibration_ids for row in rows
    ]
    evaluation = [
        row for source_id, rows in groups.items()
        if source_id not in calibration_ids for row in rows
    ]
    calibration_sources = {str(row["source_example_id"]) for row in calibration}
    evaluation_sources = {str(row["source_example_id"]) for row in evaluation}
    if calibration_sources & evaluation_sources:
        raise RuntimeError("Calibration and evaluation source groups overlap")
    if not calibration or not evaluation:
        raise ValueError("The source-grouped split produced an empty subset")

    details = {
        "method": "deterministic 50% split within source-group negative-strategy strata",
        "seed": seed,
        "strata": stratum_counts,
        "calibration_source_group_count": len(calibration_sources),
        "evaluation_source_group_count": len(evaluation_sources),
        "calibration_relation_count": len(calibration),
        "evaluation_relation_count": len(evaluation),
        "calibration_source_ids": sorted(calibration_sources),
        "evaluation_source_ids": sorted(evaluation_sources),
        "source_disjoint": True,
    }
    return calibration, evaluation, details


def _with_threshold(
    records: Sequence[Mapping[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    adjusted = []
    for row in records:
        item = dict(row)
        predicted_null = (
            float(row["null_score"]) - float(row["highest_span_score"])
        ) > threshold
        item["predicted_is_null"] = predicted_null
        item["end_to_end_correct"] = (
            predicted_null if bool(row["gold_is_null"])
            else not predicted_null and bool(row["forced_link_exact"])
        )
        item["decision_threshold"] = threshold
        item["decision_margin"] = (
            float(row["null_score"]) - float(row["highest_span_score"])
        )
        adjusted.append(item)
    return adjusted


def select_threshold(
    calibration_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not calibration_records:
        raise ValueError("Threshold selection requires calibration records")
    margins = [
        float(row["null_score"]) - float(row["highest_span_score"])
        for row in calibration_records
    ]
    if not all(math.isfinite(value) for value in margins):
        raise ValueError("NULL decision margins must be finite")
    candidates = {0.0, *margins}
    candidates.add(math.nextafter(min(margins), -math.inf))
    candidates.add(math.nextafter(max(margins), math.inf))

    evaluated = []
    for threshold in sorted(candidates):
        metrics = evaluate_null_predictions(
            _with_threshold(calibration_records, threshold)
        )
        evaluated.append({
            "threshold": threshold,
            "metrics": metrics,
        })
    selected = max(
        evaluated,
        key=lambda item: (
            item["metrics"]["end_to_end_exact_accuracy"],
            item["metrics"]["actual_positive_span_em_after_null"],
            -abs(item["threshold"]),
            item["threshold"],
        ),
    )
    return {
        "selection_objective": [
            "end_to_end_exact_accuracy",
            "actual_positive_span_em_after_null",
        ],
        "threshold_candidates_evaluated": len(evaluated),
        "selected_threshold": selected["threshold"],
        "calibration_metrics_at_selected_threshold": selected["metrics"],
    }


def _sample_manual_review(
    records: Sequence[Mapping[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    strategies = ("context_removal", "hard_context_replacement")
    negatives = [row for row in records if bool(row["gold_is_null"])]
    for strategy in strategies:
        rows = [row for row in negatives if row.get("negative_strategy") == strategy]
        if len(rows) < 25:
            raise ValueError(
                f"Need at least 25 {strategy} development negatives; found {len(rows)}"
            )
        false_link = sorted(
            (row for row in rows if not bool(row["predicted_is_null"])),
            key=lambda row: (str(row["source_example_id"]), int(row["relation_id"])),
        )
        correct_null = sorted(
            (row for row in rows if bool(row["predicted_is_null"])),
            key=lambda row: (str(row["source_example_id"]), int(row["relation_id"])),
        )
        if not false_link or not correct_null:
            raise ValueError(
                f"{strategy} must include both false-LINK and correct-NULL predictions"
            )
        false_link_quota = min(len(false_link), 24)
        correct_quota = min(len(correct_null), 25 - false_link_quota)
        remaining = 25 - false_link_quota - correct_quota
        false_link_quota += min(remaining, len(false_link) - false_link_quota)
        remaining = 25 - false_link_quota - correct_quota
        correct_quota += min(remaining, len(correct_null) - correct_quota)
        if false_link_quota < 1 or correct_quota < 1:
            raise ValueError("Manual review sample must include both prediction outcomes")
        chosen = rng.sample(false_link, false_link_quota)
        chosen.extend(rng.sample(correct_null, correct_quota))
        rng.shuffle(chosen)
        selected.extend(dict(row) for row in chosen)
    return selected


def _write_manual_review(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "example_id": row["example_id"],
                "source_example_id": row["source_example_id"],
                "relation_id": row["relation_id"],
                "negative_generation_strategy": row["negative_strategy"],
                "context": row["context_text"],
                "chunk": row["chunk_text"],
                "gold_anaphor": row.get("anaphor_text") or "",
                "predicted_output": row.get("predicted_target") or "",
                "manual_label": "",
                "review_status": "PENDING",
                "justification": "",
            })


def _manual_review_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for strategy in ("context_removal", "hard_context_replacement"):
        group = [row for row in rows if row["negative_strategy"] == strategy]
        counts[strategy] = {
            "sampled": len(group),
            "correct_null_predictions": sum(
                bool(row["predicted_is_null"]) for row in group
            ),
            "false_link_predictions": sum(
                not bool(row["predicted_is_null"]) for row in group
            ),
            "pending_human_review": len(group),
        }
    return {
        "sampled_count": len(rows),
        "validated_count": 0,
        "valid_null_count": 0,
        "invalid_null_count": 0,
        "ambiguous_count": 0,
        "pending_count": len(rows),
        "review_status": "PENDING_HUMAN_REVIEW",
        "labels_assigned_automatically": False,
        "label_definitions": {
            "VALID_NULL": "No valid antecedent exists in the provided context.",
            "INVALID_NULL": "A valid antecedent exists in the provided context.",
            "AMBIGUOUS": "Insufficient evidence to determine whether a valid antecedent exists.",
        },
        "sampling_note": (
            "Within each generation strategy, all available false-LINK cases are "
            "included up to 24, then correct-NULL cases are sampled to fill 25; "
            "this error-enriched sample must not be treated as representative."
        ),
        "by_strategy": counts,
    }


def _context_band(token_count: int) -> str:
    if token_count <= 32:
        return "1-32"
    if token_count <= 64:
        return "33-64"
    if token_count <= 128:
        return "65-128"
    return "129+"


def _summarize_group(
    group: Sequence[Mapping[str, Any]],
    false_nulls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "positive_link_count": len(group),
        "false_null_count": len(false_nulls),
        "false_null_rate": len(false_nulls) / len(group) if group else 0.0,
    }


def false_null_analysis(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positives = [row for row in records if not bool(row["gold_is_null"])]
    false_nulls = [
        row for row in positives if bool(row["predicted_is_null"])
    ]

    def group_by(
        key_fn: Any,
    ) -> dict[str, list[Mapping[str, Any]]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in positives:
            grouped[str(key_fn(row))].append(row)
        return grouped

    by_localization = group_by(
        lambda row: "forced_link_exact" if row["forced_link_exact"]
        else "forced_link_incorrect"
    )
    by_context = group_by(
        lambda row: _context_band(int(row["context_token_count"]))
    )
    by_candidate_count = group_by(
        lambda row: (
            "1" if int(row["candidate_count"]) == 1
            else "2-5" if int(row["candidate_count"]) <= 5
            else "6+"
        )
    )
    localization = {
        name: _summarize_group(
            group, [row for row in group if bool(row["predicted_is_null"])]
        )
        for name, group in sorted(by_localization.items())
    }
    context_length = {
        name: _summarize_group(
            group, [row for row in group if bool(row["predicted_is_null"])]
        )
        for name, group in sorted(by_context.items())
    }
    competition = {
        name: _summarize_group(
            group, [row for row in group if bool(row["predicted_is_null"])]
        )
        for name, group in sorted(by_candidate_count.items())
    }

    def numeric_summary(
        rows: Sequence[Mapping[str, Any]],
        key: str,
    ) -> dict[str, float | int | None]:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return {
            "count": len(values),
            "mean": sum(values) / len(values) if values else None,
            "median": median(values) if values else None,
        }

    false_null_details = []
    for row in false_nulls:
        false_null_details.append({
            "example_id": row["example_id"],
            "source_example_id": row["source_example_id"],
            "relation_id": row["relation_id"],
            "context": row["context_text"],
            "chunk": row["chunk_text"],
            "gold_anaphor": row.get("anaphor_text"),
            "gold_antecedent_spans": row["gold_character_spans"],
            "gold_antecedent_texts": row["gold_target"],
            "highest_scoring_predicted_span": row.get("highest_scoring_span_text"),
            "highest_scoring_character_span": row.get("highest_scoring_character_span"),
            "second_highest_predicted_span": row.get("second_highest_span_text"),
            "null_score": row["null_score"],
            "best_span_score": row["highest_span_score"],
            "margin_null_minus_best_span": (
                float(row["null_score"]) - float(row["highest_span_score"])
            ),
            "forced_link_prediction_correct": bool(row["forced_link_exact"]),
            "forced_link_token_f1": row["forced_link_token_f1"],
            "context_token_count": row["context_token_count"],
            "candidate_count": row["candidate_count"],
            "second_highest_span_score": row.get("second_highest_span_score"),
            "top_two_span_score_gap": row.get("top_two_span_score_gap"),
        })

    def false_vs_accepted(key: str) -> dict[str, Any]:
        accepted = [row for row in positives if not bool(row["predicted_is_null"])]
        return {
            "false_nulls": numeric_summary(false_nulls, key),
            "accepted_links": numeric_summary(accepted, key),
        }

    close_competitor_counts = {
        str(gap): sum(
            row.get("top_two_span_score_gap") is not None
            and float(row["top_two_span_score_gap"]) <= gap
            for row in false_nulls
        )
        for gap in (0.25, 0.5, 1.0)
    }
    return {
        "positive_relation_count": len(positives),
        "false_null_count": len(false_nulls),
        "false_null_rate": len(false_nulls) / len(positives) if positives else 0.0,
        "analysis_definition": {
            "weak_span_localization": "forced_link_exact is false; token F1 is also reported per case.",
            "context_length_bands": "1-32, 33-64, 65-128, and 129+ context tokens.",
            "competing_antecedents": (
                "Candidate counts and top-two score gaps are descriptive proxies; "
                "they do not establish semantic competition."
            ),
        },
        "false_null_predictions": false_null_details,
        "by_forced_link_localization": localization,
        "by_context_length": context_length,
        "by_candidate_count": competition,
        "false_null_vs_accepted_link_comparison": {
            "context_token_count": false_vs_accepted("context_token_count"),
            "candidate_count": false_vs_accepted("candidate_count"),
            "top_two_span_score_gap": false_vs_accepted("top_two_span_score_gap"),
            "forced_link_token_f1": false_vs_accepted("forced_link_token_f1"),
        },
        "false_nulls_with_close_runner_up_count": close_competitor_counts,
    }


def _calibration_result(
    records: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    calibration, evaluation, split_details = source_grouped_stratified_split(
        records, seed
    )
    selection = select_threshold(calibration)
    selected_threshold = float(selection["selected_threshold"])
    baseline_rows = _with_threshold(evaluation, 0.0)
    calibrated_rows = _with_threshold(evaluation, selected_threshold)
    baseline_metrics = evaluate_null_predictions(baseline_rows)
    calibrated_metrics = evaluate_null_predictions(calibrated_rows)
    metrics_to_report = (
        "macro_f1",
        "null",
        "false_null_rate",
        "actual_positive_span_em_after_null",
        "end_to_end_exact_accuracy",
    )
    return {
        "decision_rule": "predict NULL iff null_score - highest_span_score > threshold",
        "split": split_details,
        "threshold_selection": selection,
        "held_out_comparison": {
            "same_evaluation_examples_for_both_thresholds": True,
            "original_threshold": {
                "threshold": 0.0,
                "metrics": baseline_metrics,
            },
            "calibrated_threshold": {
                "threshold": selected_threshold,
                "metrics": calibrated_metrics,
            },
            "metric_differences_calibrated_minus_original": {
                name: (
                    calibrated_metrics[name]["recall"]
                    - baseline_metrics[name]["recall"]
                    if name == "null"
                    else calibrated_metrics[name] - baseline_metrics[name]
                )
                for name in metrics_to_report
            },
            "reported_metric_names": list(metrics_to_report),
        },
        "calibration_subset_is_not_used_as_independent_evaluation": True,
    }


def _report_markdown(
    config: Mapping[str, Any],
    review: Mapping[str, Any],
    false_nulls: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> str:
    held_out = calibration["held_out_comparison"]
    original = held_out["original_threshold"]
    adjusted = held_out["calibrated_threshold"]
    original_metrics = original["metrics"]
    adjusted_metrics = adjusted["metrics"]
    localization = false_nulls["by_forced_link_localization"]
    weak_localization = localization.get("forced_link_incorrect", {
        "false_null_count": 0,
        "positive_link_count": 0,
        "false_null_rate": 0.0,
    })
    correct_localization = localization.get("forced_link_exact", {
        "false_null_count": 0,
        "positive_link_count": 0,
        "false_null_rate": 0.0,
    })
    context_rates = false_nulls["by_context_length"]
    gap_comparison = false_nulls["false_null_vs_accepted_link_comparison"][
        "top_two_span_score_gap"
    ]
    lines = [
        "# EXP-005B — NULL Prediction Validation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "",
        "- Evaluation-only: reloaded the frozen conditioned EXP-005 checkpoint; no training or fine-tuning was run.",
        "- Loaded only the named `dev` dataset split. The reserved test split was not requested.",
        "- Manual labels are left blank and marked pending; no synthetic label was automatically validated.",
        "- Calibration/evaluation are source-disjoint and stratified by each source group's synthetic-negative strategy mix.",
        "",
        "## Manual negative review",
        "",
        f"- Sampled {review['sampled_count']} negatives; reviewed {review['validated_count']}.",
        f"- Pending: {review['pending_count']}; VALID_NULL: {review['valid_null_count']}; "
        f"INVALID_NULL: {review['invalid_null_count']}; AMBIGUOUS: {review['ambiguous_count']}.",
        "- Per-strategy sample counts and prediction-outcome composition are in `manual_review_summary.json`.",
        "",
        "## False-NULL analysis on original positive development relations",
        "",
        f"- False NULLs: {false_nulls['false_null_count']}/{false_nulls['positive_relation_count']} "
        f"({false_nulls['false_null_rate']:.4f}).",
        "- Every false-NULL record, its gold span(s), highest-scoring span, scores, margin, "
        "forced-LINK correctness, context length, and candidate competition proxies are in "
        "`false_null_analysis.json`.",
        (
            f"- Localization pattern: false-NULL rate was "
            f"{weak_localization['false_null_rate']:.4f} for forced-LINK span errors "
            f"({weak_localization['false_null_count']}/{weak_localization['positive_link_count']}) "
            f"versus {correct_localization['false_null_rate']:.4f} when forced-LINK Span EM "
            f"was correct ({correct_localization['false_null_count']}/"
            f"{correct_localization['positive_link_count']})."
        ),
        "- Context-length false-NULL rates: " + "; ".join(
            f"{band}: {values['false_null_rate']:.4f} "
            f"({values['false_null_count']}/{values['positive_link_count']})"
            for band, values in sorted(context_rates.items())
        ) + ".",
        (
            f"- Candidate competition proxy: median top-two span-score gap was "
            f"{gap_comparison['false_nulls']['median']:.4f} for false NULLs versus "
            f"{gap_comparison['accepted_links']['median']:.4f} for accepted links; "
            f"{false_nulls['false_nulls_with_close_runner_up_count']['0.5']} false NULLs "
            "had a runner-up within 0.5 score points. Candidate-count bins are weak "
            "proxies because they count enumerated spans, not semantic antecedents."
        ),
        "- These are descriptive associations, not causal claims; the comparatively "
        "higher false-NULL rate when forced span localization is wrong suggests that "
        "span uncertainty and NULL rejection co-occur.",
        "",
        "## Held-out threshold comparison",
        "",
        f"- Calibration selected threshold: `{adjusted['threshold']:.8g}` using only "
        f"{calibration['split']['calibration_relation_count']} calibration relations.",
        f"- Held-out source groups: {calibration['split']['evaluation_source_group_count']}; "
        f"held-out relations: {calibration['split']['evaluation_relation_count']}.",
        "",
        "| Threshold | Macro F1 | NULL recall | False-NULL rate | Positive Span EM | E2E exact |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {original['threshold']:.8g} | {original_metrics['macro_f1']:.4f} | "
            f"{original_metrics['null']['recall']:.4f} | {original_metrics['false_null_rate']:.4f} | "
            f"{original_metrics['actual_positive_span_em_after_null']:.4f} | "
            f"{original_metrics['end_to_end_exact_accuracy']:.4f} |"
        ),
        (
            f"| {adjusted['threshold']:.8g} | {adjusted_metrics['macro_f1']:.4f} | "
            f"{adjusted_metrics['null']['recall']:.4f} | {adjusted_metrics['false_null_rate']:.4f} | "
            f"{adjusted_metrics['actual_positive_span_em_after_null']:.4f} | "
            f"{adjusted_metrics['end_to_end_exact_accuracy']:.4f} |"
        ),
        "",
        "Both thresholds were evaluated once on the same held-out source groups. "
        "Calibration-subset performance is used for threshold selection only.",
        (
            f"On held-out data, calibration changed false-NULL rate by "
            f"{calibration['held_out_comparison']['metric_differences_calibrated_minus_original']['false_null_rate']:+.4f}, "
            f"positive Span EM by "
            f"{calibration['held_out_comparison']['metric_differences_calibrated_minus_original']['actual_positive_span_em_after_null']:+.4f}, "
            f"and E2E exact by "
            f"{calibration['held_out_comparison']['metric_differences_calibrated_minus_original']['end_to_end_exact_accuracy']:+.4f}."
        ),
        "",
        "## Conclusion and limitations",
        "",
        "Human review is pending, so the validity of synthetic NULL labels and discrimination "
        "on human-confirmed valid negatives are not established. The held-out threshold result "
        "is preliminary and applies to this balanced synthetic development set only. Do not "
        "proceed to EXP-006 on the basis of these pending labels alone.",
        "",
        f"Frozen conditioned checkpoint SHA-256: `{config['conditioned_checkpoint_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def _append_experiment_log(
    report: str,
    log_path: Path,
) -> None:
    heading = "# EXP-005B — NULL Prediction Validation"
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if heading in existing:
        existing = existing[:existing.index(heading)].rstrip() + "\n\n"
    else:
        existing = existing.rstrip() + "\n\n"
    log_path.write_text(existing + report, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    exp005_config = _read_config(args.source_output_dir / "config.yaml")
    if exp005_config.get("approach") != "all":
        raise ValueError("Expected the completed EXP-005 all-approach configuration")
    base_checkpoint = Path(exp005_config["base_checkpoint"])
    if not base_checkpoint.is_absolute():
        base_checkpoint = REPOSITORY_ROOT / base_checkpoint
    base_hash_expected = exp005_config.get("base_checkpoint_sha256")
    if base_hash_expected and _sha256(base_checkpoint) != base_hash_expected:
        raise ValueError("EXP-004 checkpoint hash differs from the EXP-005 configuration")
    source_predictions_path = (
        args.source_output_dir / "conditioned" / "predictions.json"
    )
    if not source_predictions_path.is_file():
        raise FileNotFoundError(
            f"Canonical EXP-005 conditioned predictions not found: {source_predictions_path}"
        )
    archived_records = json.loads(source_predictions_path.read_text(encoding="utf-8"))
    if not isinstance(archived_records, list):
        raise ValueError("EXP-005 conditioned predictions must be a JSON array")
    model, conditioned_hash, base_hash = _load_conditioned_model(
        args.model_checkpoint,
        base_checkpoint,
        float(exp005_config["dropout"]),
        int(exp005_config["max_span_length"]),
    )

    load_kwargs: dict[str, Any] = {
        "download_config": DownloadConfig(local_files_only=args.offline),
    }
    if exp005_config.get("dataset_revision") is not None:
        load_kwargs["revision"] = exp005_config["dataset_revision"]
    raw_dev = list(load_dataset(
        exp005_config["dataset_name"],
        split="dev",
        **load_kwargs,
    ))
    tokenizer = AutoTokenizer.from_pretrained(
        exp005_config["encoder"],
        use_fast=True,
        local_files_only=args.offline,
    )
    positives, audit = _preprocess_split(
        raw_dev,
        tokenizer,
        int(exp005_config["max_length"]),
        "dev",
    )
    negatives = _rebuild_original_dev_negatives(
        archived_records,
        positives,
        tokenizer,
        int(exp005_config["max_length"]),
    )
    evaluation_examples = positives + negatives
    loader = _data_loader(
        evaluation_examples,
        tokenizer,
        int(exp005_config["batch_size"]),
        False,
        int(exp005_config["seed"]),
    )
    device = _resolve_device(args.accelerator)
    records = _evaluate_with_competitor_scores(model, loader, device)
    del model

    positive_relation_count = sum(len(example["relations"]) for example in positives)
    negative_relation_count = sum(len(example["relations"]) for example in negatives)
    if len(records) != positive_relation_count + negative_relation_count:
        raise RuntimeError("Inference did not return one prediction per relation")
    if any(row["negative_strategy"] not in {
        None,
        "context_removal",
        "hard_context_replacement",
    } for row in records):
        raise ValueError("Unexpected synthetic-negative generation strategy")
    review_rows = _sample_manual_review(records, int(exp005_config["seed"]))
    review_summary = _manual_review_summary(review_rows)
    _write_manual_review(args.output_dir / "manual_review.csv", review_rows)
    _write_json(args.output_dir / "manual_review_summary.json", review_summary)

    false_nulls = false_null_analysis(records)
    _write_json(args.output_dir / "false_null_analysis.json", false_nulls)
    calibration = _calibration_result(records, int(exp005_config["seed"]))
    _write_json(args.output_dir / "calibration_results.json", calibration)
    source_split = {
        source_id: "calibration"
        for source_id in calibration["split"]["calibration_source_ids"]
    }
    source_split.update({
        source_id: "evaluation"
        for source_id in calibration["split"]["evaluation_source_ids"]
    })
    threshold = float(calibration["threshold_selection"]["selected_threshold"])
    original_rows = _with_threshold(records, 0.0)
    calibrated_rows = _with_threshold(records, threshold)
    original_by_key = {
        (row["example_id"], row["relation_id"]): row
        for row in original_rows
    }
    calibrated_by_key = {
        (row["example_id"], row["relation_id"]): row
        for row in calibrated_rows
    }
    for row in records:
        row["calibration_partition"] = source_split[str(row["source_example_id"])]
        key = (row["example_id"], row["relation_id"])
        row["margin_null_minus_best_span"] = (
            float(row["null_score"]) - float(row["highest_span_score"])
        )
        row["threshold_zero_predicted_is_null"] = original_by_key[key][
            "predicted_is_null"
        ]
        row["calibrated_predicted_is_null"] = calibrated_by_key[key][
            "predicted_is_null"
        ]

    output_config = {
        "experiment": "EXP-005B",
        "objective": "evaluation-only NULL prediction validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": exp005_config["dataset_name"],
        "dataset_revision": exp005_config.get("dataset_revision"),
        "loaded_splits": ["dev"],
        "reserved_test_split_loaded": False,
        "conditioned_checkpoint": str(args.model_checkpoint),
        "conditioned_checkpoint_sha256": conditioned_hash,
        "exp004_base_checkpoint": str(base_checkpoint),
        "exp004_base_checkpoint_sha256": base_hash,
        "frozen_backbone": True,
        "retrained": False,
        "inference": {
            "encoder": exp005_config["encoder"],
            "batch_size": exp005_config["batch_size"],
            "max_length": exp005_config["max_length"],
            "max_span_length": exp005_config["max_span_length"],
            "dropout": exp005_config["dropout"],
            "seed": exp005_config["seed"],
            "device": str(device),
            "negative_examples": (
                "Reconstructed from the exact saved EXP-005 dev predictions to "
                "preserve the original synthetic sample."
            ),
        },
        "development_data": {
            "positive_relations": positive_relation_count,
            "synthetic_negative_relations": negative_relation_count,
            "synthetic_negatives_by_strategy": dict(Counter(
                str(example["relations"][0]["strategy"])
                for example in negatives
            )),
            "canonical_exp005_predictions": str(source_predictions_path),
            "canonical_exp005_predictions_sha256": _sha256(source_predictions_path),
            "audit": audit,
        },
        "manual_review_sampling_seed": exp005_config["seed"],
        "calibration_split_seed": exp005_config["seed"],
        "git": exp005_config.get("git"),
    }
    _write_json(args.output_dir / "config.yaml", output_config)
    _write_json(args.output_dir / "predictions.json", records)
    report = _report_markdown(output_config, review_summary, false_nulls, calibration)
    (args.output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    _append_experiment_log(
        report,
        REPOSITORY_ROOT / "EXPERIMENTLOG.md",
    )
    return {
        "output_dir": str(args.output_dir),
        "prediction_count": len(records),
        "manual_review_count": len(review_rows),
        "false_null_count": false_nulls["false_null_count"],
        "selected_threshold": threshold,
        "evaluation_metrics": calibration["held_out_comparison"],
        "checkpoint_sha256": conditioned_hash,
        "retrained": False,
        "test_split_loaded": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
