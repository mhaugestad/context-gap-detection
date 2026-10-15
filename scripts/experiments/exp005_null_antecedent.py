"""Evaluate explicit NULL antecedent scoring against the EXP-004 span scorer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
from datasets import DownloadConfig, load_dataset
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.null_antecedent import (
    NullAntecedentCollator,
    audit_source_split,
    construct_split_negatives,
)
from src.evaluation.null_metrics import (
    bootstrap_interval,
    calibration_metrics,
    evaluate_null_predictions,
    paired_bootstrap_difference,
    paired_contrast_metrics,
    stratified_performance,
)
from src.models.null_antecedent import NullAntecedentModel
from src.training.null_module import NullAntecedentLightningModule
from src.training.span_module import SpanExtractionLightningModule


OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "exp005_null_antecedent"
BASE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "experiments"
    / "exp004_span_extraction"
    / "span_pair"
    / "checkpoints"
    / "best.ckpt"
)
DEFAULTS = {
    "dataset_name": "mhaugestad/contextualise-context",
    "dataset_revision": None,
    "encoder_name": "distilbert-base-uncased",
    "seed": 42,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "max_epochs": 20,
    "patience": 4,
    "gradient_clip_val": 1.0,
    "precision": "32-true",
    "max_length": 512,
    "max_span_length": 32,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach",
        choices=("constant", "learned_bias", "conditioned", "all"),
        default="all",
    )
    parser.add_argument("--dataset-name", default=DEFAULTS["dataset_name"])
    parser.add_argument("--dataset-revision", default=DEFAULTS["dataset_revision"])
    parser.add_argument("--encoder-name", default=DEFAULTS["encoder_name"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--gradient-clip-val", type=float, default=DEFAULTS["gradient_clip_val"])
    parser.add_argument("--precision", default=DEFAULTS["precision"])
    parser.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--max-span-length", type=int, default=DEFAULTS["max_span_length"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    positive_ints = ("batch_size", "max_epochs", "patience", "max_length", "max_span_length")
    for name in positive_ints:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate must be positive and --weight-decay non-negative")
    if args.num_workers < 0 or args.gradient_clip_val < 0:
        parser.error("--num-workers and --gradient-clip-val cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0, 1)")
    return args


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json_yaml(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty_working_tree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_working_tree": None}


def _dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "download_config": DownloadConfig(local_files_only=args.offline),
    }
    if args.dataset_revision is not None:
        kwargs["revision"] = args.dataset_revision
    return kwargs


def _load_raw_splits(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = _dataset_kwargs(args)
    # Load named splits individually. EXP-003's reserved test split is never requested.
    return {
        split: list(load_dataset(args.dataset_name, split=split, **kwargs))
        for split in ("train", "dev")
    }


def _preprocess_split(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.data.preprocessor import CoreferencePreprocessor

    preprocessor = CoreferencePreprocessor(
        tokenizer,
        max_length=max_length,
        truncation_policy="drop",
        annotation_policy="drop",
    )
    positives, audit = audit_source_split(rows, preprocessor)
    positives = [item for item in positives if item["relations"]]
    for item in positives:
        item["source_example_id"] = str(item["id"])
        for relation in item["relations"]:
            relation["source_split"] = split_name
    return positives, audit


def _collate_fn(tokenizer: Any):
    return NullAntecedentCollator(tokenizer.pad_token_id)


def _data_loader(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn(tokenizer),
        generator=generator,
    )


def _tensor_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _span_to_char(offsets: torch.Tensor, span: Sequence[int]) -> tuple[int, int] | None:
    start, end = int(span[0]), int(span[1])
    if start < 0 or end <= start or end > offsets.shape[0]:
        return None
    start_char = int(offsets[start, 0].item())
    end_char = int(offsets[end - 1, 1].item())
    return (start_char, end_char) if end_char > start_char else None


def _enrich_predictions(
    records: list[dict[str, Any]],
    batches: list[Mapping[str, Any]],
    approach: str,
) -> list[dict[str, Any]]:
    by_location = {
        (row["batch_index"], row["relation_index"]): row
        for row in records
    }
    for batch in batches:
        for batch_index, relation_index in batch["relation_mask"].nonzero(
            as_tuple=False
        ).tolist():
            row = by_location.get((batch_index, relation_index))
            if row is None:
                continue
            anaphor_token_span = batch["relations"][batch_index, relation_index].tolist()
            offsets = batch["offset_mapping"][batch_index]
            anaphor_char_span = _span_to_char(offsets, anaphor_token_span)
            chunk = batch["chunk"][batch_index]
            row["anaphor_text"] = (
                chunk[anaphor_char_span[0]:anaphor_char_span[1]]
                if anaphor_char_span else None
            )
            context = batch["context"][batch_index]
            predicted_char = _span_to_char(offsets, row["predicted_span"])
            gold_char_spans = [
                mapped for span in row["gold_spans"]
                if (mapped := _span_to_char(offsets, span)) is not None
            ]
            row["highest_scoring_span_text"] = (
                context[predicted_char[0]:predicted_char[1]]
                if predicted_char else None
            )
            row["highest_scoring_character_span"] = (
                list(predicted_char) if predicted_char else None
            )
            row["gold_character_spans"] = [list(span) for span in gold_char_spans]
            metadata = batch["relation_metadata"][batch_index][relation_index] or {}
            row["gold_target"] = (
                "NULL" if row["gold_is_null"] else [
                    context[start:end] for start, end in gold_char_spans
                ]
            )
            row["predicted_target"] = (
                "NULL" if row["predicted_is_null"] else row["highest_scoring_span_text"]
            )
            row["negative_strategy"] = metadata.get("strategy")
            row["replacement_source_id"] = metadata.get("replacement_source_id")
            row["approach"] = approach
            del row["batch_index"]
            del row["relation_index"]
    return records


def _evaluate(
    model: NullAntecedentModel,
    loader: DataLoader,
    device: torch.device,
    approach: str,
) -> list[dict[str, Any]]:
    model.eval().to(device)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(_tensor_batch(batch, device))
            batch_records = NullAntecedentLightningModule.records_from_batch(
                output, _tensor_batch(batch, device)
            )
            records.extend(_enrich_predictions(batch_records, [batch], approach))
    return records


def _always_link(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        clone = dict(row)
        clone["predicted_is_null"] = False
        clone["end_to_end_correct"] = (
            not clone["gold_is_null"] and bool(clone["forced_link_exact"])
        )
        clone["null_probability"] = 0.0
        clone["approach"] = "always_link"
        result.append(clone)
    return result


def _adjust_for_candidate_count(
    records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Fit a simple log(candidate-count) offset on pairs held out from evaluation."""
    pair_ids = sorted({
        str(row["pair_id"]) for row in records if row.get("pair_id")
    })
    calibration_pairs = set(pair_ids[::5])
    calibration = [row for row in records if row.get("pair_id") in calibration_pairs]
    held_out = [row for row in records if row.get("pair_id") not in calibration_pairs]
    if not calibration or not held_out:
        return {
            "status": "insufficient paired examples",
            "calibration_pair_count": len(calibration_pairs),
            "held_out_pair_count": max(0, len(pair_ids) - len(calibration_pairs)),
        }

    def adjusted(rows: list[dict[str, Any]], gamma: float) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            clone = dict(row)
            offset = gamma * math.log(max(int(row["candidate_count"]), 1))
            adjusted_null_score = float(row["null_score"]) + offset
            predicted_null = adjusted_null_score > float(row["highest_span_score"])
            clone["predicted_is_null"] = predicted_null
            clone["end_to_end_correct"] = (
                predicted_null if row["gold_is_null"]
                else not predicted_null and bool(row["forced_link_exact"])
            )
            clone["null_probability"] = 1.0 / (
                1.0 + math.exp(max(-60.0, min(
                    60.0,
                    float(row["highest_span_score"]) - adjusted_null_score,
                )))
            )
            result.append(clone)
        return result

    candidates = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    selected_gamma = 0.0
    selected_metrics: dict[str, Any] | None = None
    for gamma in candidates:
        metrics = evaluate_null_predictions(adjusted(calibration, gamma))
        if selected_metrics is None or (
            metrics["macro_f1"],
            metrics["end_to_end_exact_accuracy"],
        ) > (
            selected_metrics["macro_f1"],
            selected_metrics["end_to_end_exact_accuracy"],
        ):
            selected_gamma = gamma
            selected_metrics = metrics
    unadjusted_test = evaluate_null_predictions(held_out)
    adjusted_test = evaluate_null_predictions(adjusted(held_out, selected_gamma))
    return {
        "status": "completed exploratory diagnostic",
        "method": "NULL score + gamma * log(candidate_count)",
        "gamma_grid": list(candidates),
        "selected_gamma": selected_gamma,
        "calibration_pair_count": len(calibration_pairs),
        "calibration_relation_count": len(calibration),
        "held_out_pair_count": len(pair_ids) - len(calibration_pairs),
        "held_out_relation_count": len(held_out),
        "calibration_metrics_at_selected_gamma": selected_metrics,
        "held_out_unadjusted_metrics": unadjusted_test,
        "held_out_adjusted_metrics": adjusted_test,
        "held_out_macro_f1_difference": (
            adjusted_test["macro_f1"] - unadjusted_test["macro_f1"]
        ),
        "held_out_e2e_exact_difference": (
            adjusted_test["end_to_end_exact_accuracy"]
            - unadjusted_test["end_to_end_exact_accuracy"]
        ),
        "seed": seed,
        "note": (
            "This diagnostic uses the highest span score rather than the full "
            "candidate softmax and is not part of the primary comparison."
        ),
    }


class BestNullCheckpoint(Callback):
    """Select by dev macro F1, breaking ties with end-to-end exact accuracy."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.best_macro_f1: float | None = None
        self.best_e2e: float | None = None
        self.best_epoch: int | None = None

    def on_validation_end(
        self,
        trainer: L.Trainer,
        _module: L.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        macro_f1 = float(metrics["val/macro_f1"].detach().cpu().item())
        e2e = float(metrics["val/e2e_exact"].detach().cpu().item())
        if self.best_macro_f1 is not None and (
            macro_f1 < self.best_macro_f1
            or (macro_f1 == self.best_macro_f1 and e2e <= (self.best_e2e or 0.0))
        ):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.path), weights_only=True)
        self.best_macro_f1 = macro_f1
        self.best_e2e = e2e
        self.best_epoch = trainer.current_epoch + 1


def _load_span_model(checkpoint: Path) -> Any:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EXP-004 span-pair checkpoint not found: {checkpoint}")
    base_module = SpanExtractionLightningModule.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    if getattr(base_module.model, "approach", None) != "span_pair":
        raise ValueError("The supplied EXP-004 checkpoint is not a span-pair model")
    return base_module.model


def _train_approach(
    approach: str,
    span_model: Any,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[NullAntecedentModel, dict[str, Any]]:
    L.seed_everything(args.seed, workers=True)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)
    checkpoint_path = output_dir / approach / "checkpoints" / "best.ckpt"
    checkpoint_callback = BestNullCheckpoint(checkpoint_path)
    early_stopping = EarlyStopping(
        monitor="val/macro_f1",
        mode="max",
        patience=args.patience,
        min_delta=0.0,
    )
    model = NullAntecedentModel(
        span_model=span_model,
        approach=approach,
        dropout=args.dropout,
        freeze_span_model=True,
    )
    module = NullAntecedentLightningModule(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback, early_stopping],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    start = time.perf_counter()
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    runtime = time.perf_counter() - start
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Training did not produce a best checkpoint for {approach}")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    module.load_state_dict(saved["state_dict"])
    module.eval()
    return module.model, {
        "best_epoch": checkpoint_callback.best_epoch,
        "best_development_macro_f1": checkpoint_callback.best_macro_f1,
        "best_development_e2e_exact": checkpoint_callback.best_e2e,
        "epochs_completed": trainer.current_epoch + 1,
        "training_runtime_seconds": runtime,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "frozen_backbone": True,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        "device": str(trainer.strategy.root_device),
    }


def _error_diagnostics(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    failures = [row for row in records if not row["end_to_end_correct"]]
    result = []
    for row in failures[:max(limit, len(failures))]:
        if row["gold_is_null"] and not row["predicted_is_null"]:
            category = (
                "False LINK: plausible but incorrect distractor"
                if row.get("negative_strategy") == "hard_context_replacement"
                else "False LINK: unrelated antecedent selected"
            )
        elif not row["gold_is_null"] and row["predicted_is_null"]:
            category = "False NULL: valid antecedent rejected"
        elif not row["gold_is_null"]:
            category = "Correct LINK decision, incorrect antecedent span"
        else:
            category = "Synthetic negative annotation ambiguity"
        result.append({
            "category": category,
            "example_id": row["example_id"],
            "source_example_id": row.get("source_example_id"),
            "context": row["context_text"],
            "chunk": row["chunk_text"],
            "anaphor": row.get("anaphor_text"),
            "gold_target": row.get("gold_target"),
            "predicted_target": row.get("predicted_target"),
            "highest_scoring_span": row.get("highest_scoring_span_text"),
            "highest_span_score": row["highest_span_score"],
            "null_score": row["null_score"],
            "candidate_count": row["candidate_count"],
            "context_token_count": row["context_token_count"],
            "negative_generation_strategy": row.get("negative_strategy"),
            "replacement_source_id": row.get("replacement_source_id"),
            "null_probability": row["null_probability"],
        })
    return result


def _summary_rows(
    records_by_approach: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    metrics = {
        name: evaluate_null_predictions(rows)
        for name, rows in records_by_approach.items()
    }
    return metrics


def _report_markdown(
    comparison: Mapping[str, Any],
    training: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    negative_reports: Mapping[str, Any],
) -> str:
    names = ("always_link", "constant", "learned_bias", "conditioned")
    labels = {
        "always_link": "Always-LINK",
        "constant": "Constant NULL",
        "learned_bias": "Learned NULL bias",
        "conditioned": "Conditioned NULL",
    }
    lines = [
        "# EXP-005 — Null Antecedent Prediction",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Dataset and construction",
        "",
        "| Split | Positive relations | Synthetic negatives | Context-removal | Hard replacement |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev"):
        report = negative_reports[split]
        accepted = report["accepted_by_strategy"]
        lines.append(
            f"| {split} | {report['positive_relation_count']} | "
            f"{report['negative_relation_count']} | "
            f"{accepted.get('context_removal', 0)} | "
            f"{accepted.get('hard_context_replacement', 0)} |"
        )
    lines += [
        "",
        "Unmatched annotations and examples without valid relations were audited, not treated as NULL. "
        "Synthetic labels are heuristic and the manual review sets are explicitly marked unreviewed.",
        "",
        "## Development comparison",
        "",
        "| Approach | Macro F1 | Balanced Accuracy | E2E Exact | Forced-LINK Span EM | Actual positive Span EM | NULL Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = comparison.get(name)
        if row is None:
            continue
        lines.append(
            f"| {labels[name]} | {row['macro_f1']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | "
            f"{row['end_to_end_exact_accuracy']:.4f} | "
            f"{row['forced_link_span_em']:.4f} | "
            f"{row['actual_positive_span_em_after_null']:.4f} | "
            f"{row['null']['recall']:.4f} |"
        )
    lines += ["", "## Training", ""]
    for approach, details in training.items():
        lines.append(
            f"- **{approach}:** best epoch {details['best_epoch']}; "
            f"{details['epochs_completed']} epochs; "
            f"{details['training_runtime_seconds']:.2f}s; "
            f"best dev Macro F1 {details['best_development_macro_f1']:.4f}; "
            f"best dev E2E exact {details['best_development_e2e_exact']:.4f}."
        )
    lines += [
        "",
        "## Audit notes",
        "",
        f"- Train valid positive relations: {dataset_audit['train']['valid_aligned_positive_relations']}.",
        f"- Dev valid positive relations: {dataset_audit['dev']['valid_aligned_positive_relations']}.",
        "- The reserved test split was not loaded.",
        "- Forced-LINK span quality uses the unchanged frozen EXP-004 span scorer; post-NULL quality also includes abstention decisions.",
        "- Read `comparison_metrics.json`, `calibration_metrics.json`, `contrast_metrics.json`, and prediction files for stratified metrics and error analysis.",
        "",
    ]
    return "\n".join(lines)


def _append_experiment_log(
    comparison: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    negative_reports: Mapping[str, Any],
    training: Mapping[str, Any],
    configuration: Mapping[str, Any],
    git_info: Mapping[str, Any],
    checkpoint_sha256: str,
    log_path: Path,
) -> None:
    def metric(name: str, key: str) -> str:
        values: Mapping[str, Any] = comparison.get(name, {})
        value = (
            values.get("null", {}).get("recall")
            if key == "null.recall"
            else values.get(key)
        )
        return "not run" if value is None else f"{float(value):.4f}"

    rows = []
    for name, label in (
        ("always_link", "Always-LINK"),
        ("constant", "Constant NULL"),
        ("learned_bias", "Learned NULL bias"),
        ("conditioned", "Conditioned NULL"),
    ):
        rows.append(
            f"| {label} | {metric(name, 'macro_f1')} | "
            f"{metric(name, 'balanced_accuracy')} | "
            f"{metric(name, 'end_to_end_exact_accuracy')} | "
            f"{metric(name, 'forced_link_span_em')} | "
            f"{metric(name, 'actual_positive_span_em_after_null')} | "
            f"{metric(name, 'null.recall')} |"
        )
    baseline = comparison["always_link"]
    learned = comparison.get("learned_bias", {})
    best_null = max(
        (
            comparison[name]
            for name in ("constant", "learned_bias", "conditioned")
            if name in comparison
        ),
        key=lambda row: (row["macro_f1"], row["balanced_accuracy"]),
        default={},
    )
    conditioned = comparison.get("conditioned", {})
    h1 = (
        "Supported" if best_null
        and best_null.get("macro_f1", 0) > baseline["macro_f1"]
        and best_null.get("balanced_accuracy", 0) > baseline["balanced_accuracy"]
        else "Not supported"
    )
    h2 = (
        "Supported" if conditioned
        and conditioned.get("macro_f1", -1) > max(
            comparison.get("constant", {}).get("macro_f1", -1),
            learned.get("macro_f1", -1),
        )
        and conditioned.get("end_to_end_exact_accuracy", -1) > max(
            comparison.get("constant", {}).get("end_to_end_exact_accuracy", -1),
            learned.get("end_to_end_exact_accuracy", -1),
        )
        else "Mixed or inconclusive"
    )
    positive_base_em = comparison["always_link"]["forced_link_span_em"]
    conditioned_actual_em = conditioned.get("actual_positive_span_em_after_null")
    conditioned_forced_em = conditioned.get("forced_link_span_em")
    if positive_base_em is None or conditioned_actual_em is None:
        h3 = "Inconclusive"
    elif (
        conditioned_forced_em is not None
        and abs(conditioned_forced_em - positive_base_em) <= 0.05
        and conditioned_actual_em < positive_base_em - 0.05
    ):
        h3 = "Mixed (forced-LINK span quality is preserved, but post-NULL positive EM drops beyond 5pp)"
    else:
        h3 = "Supported" if (
            conditioned_forced_em is not None
            and abs(conditioned_forced_em - positive_base_em) <= 0.05
        ) else "Not supported"
    pair_acc = conditioned.get("paired_contrast", {}).get(
        "paired_contrast_accuracy", 0.0
    )
    always_pair_acc = baseline.get("paired_contrast", {}).get(
        "paired_contrast_accuracy", 0.0
    )
    h4 = (
        "Descriptively supported"
        if pair_acc > always_pair_acc else "Inconclusive"
    )

    def configured(name: str, fallback: Any) -> Any:
        value = configuration.get(name, fallback)
        return fallback if value is None else value

    paired_differences = comparison.get("paired_differences_vs_always_link", {})
    difference_lines = []
    for name in ("learned_bias", "conditioned"):
        for metric_name in ("macro_f1", "balanced_accuracy", "end_to_end_exact_accuracy"):
            estimate = paired_differences.get(name, {}).get(metric_name, {})
            if estimate.get("difference") is not None:
                low, high = estimate["ci95"]
                difference_lines.append(
                    f"  - {name} vs Always-LINK {metric_name}: "
                    f"{estimate['difference']:+.4f} (paired bootstrap 95% CI "
                    f"[{low:+.4f}, {high:+.4f}])."
                )

    training_lines = [
        f"- {name}: best epoch {info.get('best_epoch')}; "
        f"{info.get('epochs_completed')} epochs; "
        f"{info.get('training_runtime_seconds', 0.0):.2f}s on "
        f"{info.get('device', 'unspecified device')}."
        for name, info in training.items()
    ]
    train_rejected = sum(negative_reports["train"]["rejected_by_reason"].values())
    dev_rejected = sum(negative_reports["dev"]["rejected_by_reason"].values())

    content = f"""# EXP-005 — Null Antecedent Prediction

**Description:** Extended the frozen EXP-004 span-pair scorer with a fixed NULL candidate, a learned global NULL bias, and a context-conditioned NULL head. Gold anaphor spans remain supplied; automatic anaphor detection is out of scope.

**Hypotheses:**
- H1 — NULL prediction is learnable: {h1}. Confidence intervals are in `comparison_metrics.json`.
- H2 — Context-conditioned NULL scoring is beneficial: {h2}.
- H3 — Span extraction can be preserved: {h3}; frozen forced-LINK metrics are reported separately from post-NULL correctness.
- H4 — The model uses antecedent availability: {h4}; paired controlled-context results are in `contrast_metrics.json`.

**Dataset:** `{configured('dataset_name', DEFAULTS['dataset_name'])}` (revision `{configured('dataset_revision', 'unspecified')}`); {dataset_audit['train']['valid_aligned_positive_relations']} aligned train positive relations and {negative_reports['train']['negative_relation_count']} synthetic train negatives; {dataset_audit['dev']['valid_aligned_positive_relations']} aligned development positive relations and {negative_reports['dev']['negative_relation_count']} synthetic development negatives. Context-removal and hard context replacement are split-local. No zero-relation examples were labeled NULL. Annotation-filtered and token-alignment exclusions are in `dataset_audit.json`; the test split was not loaded. Synthetic labels are heuristic, not human-verified; review subsets are included in `negative_generation_report.json`. Rejected negative candidates: {train_rejected} train and {dev_rejected} dev.

**Configuration:** `{configured('encoder', DEFAULTS['encoder_name'])}` span-pair model; EXP-004 checkpoint SHA-256 `{checkpoint_sha256}`; seed {configured('seed', DEFAULTS['seed'])}; batch size {configured('batch_size', DEFAULTS['batch_size'])}; NULL head learning rate {configured('learning_rate', DEFAULTS['learning_rate'])}; weight decay {configured('weight_decay', DEFAULTS['weight_decay'])}; dropout {configured('dropout', DEFAULTS['dropout'])}; up to {configured('max_epochs', DEFAULTS['max_epochs'])} epochs; early-stopping patience {configured('patience', DEFAULTS['patience'])}; max span length {configured('max_span_length', DEFAULTS['max_span_length'])}; frozen encoder/span scorer in the primary comparison. Git commit `{git_info.get('commit')}`, dirty working tree `{git_info.get('dirty_working_tree')}`.
{chr(10).join(training_lines)}

**Results:**

| Approach | Macro F1 | Balanced Accuracy | E2E Exact | Forced-LINK Span EM | Actual positive Span EM | NULL Recall |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Forced-LINK Span EM measures span selection with NULL ignored; actual positive Span EM includes NULL decisions.

**Observations:**
- The always-LINK baseline has balanced accuracy {baseline['balanced_accuracy']:.4f} on this balanced evaluation.
- The best NULL-enabled result has Macro F1 {best_null.get('macro_f1', 0.0):.4f}, balanced accuracy {best_null.get('balanced_accuracy', 0.0):.4f}, and E2E exact {best_null.get('end_to_end_exact_accuracy', 0.0):.4f}.
- Paired differences versus Always-LINK, including 95% bootstrap intervals:
{chr(10).join(difference_lines) if difference_lines else '  - No learned variant was run.'}
- Learned-bias results: Macro F1 {metric('learned_bias', 'macro_f1')}, balanced accuracy {metric('learned_bias', 'balanced_accuracy')}, E2E exact {metric('learned_bias', 'end_to_end_exact_accuracy')}; bootstrap intervals are recorded for Macro F1 and E2E exact.
- Conditioned-head results: Macro F1 {metric('conditioned', 'macro_f1')}, balanced accuracy {metric('conditioned', 'balanced_accuracy')}, E2E exact {metric('conditioned', 'end_to_end_exact_accuracy')}.
- Paired contrast accuracy: {pair_acc:.4f} for conditioned NULL versus {always_pair_acc:.4f} for Always-LINK; category counts are saved in `contrast_metrics.json`.
- Forced-LINK Span EM is held constant across NULL approaches because the span scorer is frozen; post-NULL Span EM reflects False-NULL decisions.
- Candidate-count/context-length effects, the held-out candidate-count-adjustment diagnostic, NULL probability Brier score, and calibration bins are available in the exported diagnostics.

**Hypothesis assessment:** H1 {h1}; H2 {h2}; H3 {h3}; H4 {h4}.

**Limitations:** Synthetic negatives may contain implicit valid referents despite lexical exclusion; balanced LINK/NULL proportions do not represent deployment prevalence; candidate-count effects can alter softmax probabilities; calibration is exploratory; supplied gold anaphors mean anaphor detection and referentiality remain unsolved.

**Conclusion:** This comparison measures whether explicit NULL scoring improves conditional antecedent resolution on a small controlled evaluation. Interpret all findings in light of the synthetic-label uncertainty and finite development sample.

**Next steps:** Carry forward the best frozen NULL scorer only if its macro F1 and end-to-end exact gains are consistent; otherwise retain the simpler span-pair baseline and obtain human-reviewed NULL examples before additional training.
"""
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    marker = "# EXP-005 — Null Antecedent Prediction"
    if marker in existing:
        existing = existing[:existing.index(marker)].rstrip() + "\n\n"
    log_path.write_text(existing + content, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_name,
        use_fast=True,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("The selected tokenizer must define a pad token")

    raw_splits = _load_raw_splits(args)
    positives: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    negatives: dict[str, list[dict[str, Any]]] = {}
    generation_reports: dict[str, dict[str, Any]] = {}
    manifests: dict[str, list[dict[str, Any]]] = {}
    examples_by_split: dict[str, list[dict[str, Any]]] = {}
    for index, split in enumerate(("train", "dev")):
        positives[split], audits[split] = _preprocess_split(
            raw_splits[split], tokenizer, args.max_length, split
        )
        negatives[split], generation_reports[split], manifests[split] = (
            construct_split_negatives(
                raw_splits[split],
                positives[split],
                tokenizer,
                args.max_length,
                split,
                args.seed + index,
            )
        )
        trainable = positives[split] + negatives[split]
        examples_by_split[split] = trainable

    train_source_ids = {
        str(row["id"]) for row in raw_splits["train"]
    }
    dev_source_ids = {
        str(row["id"]) for row in raw_splits["dev"]
    }
    if train_source_ids & dev_source_ids:
        raise ValueError("Source example IDs overlap between train and dev")
    for split in ("train", "dev"):
        if any(
            row["source_example_id"] not in {
                str(item["id"]) for item in raw_splits[split]
            }
            for row in negatives[split]
        ):
            raise ValueError(f"Negative target source leaked into {split}")
    audit_out = {
        "dataset": args.dataset_name,
        "revision": args.dataset_revision,
        "loaded_splits": ["train", "dev"],
        "reserved_test_split_loaded": False,
        "train": audits["train"],
        "dev": audits["dev"],
    }
    atomic_write_json(args.output_dir / "dataset_audit.json", audit_out)
    atomic_write_json(
        args.output_dir / "negative_generation_report.json",
        generation_reports,
    )
    atomic_write_json(
        args.output_dir / "synthetic_negative_manifest.json",
        manifests,
    )

    train_loader = _data_loader(
        examples_by_split["train"], tokenizer, args.batch_size, True, args.seed,
        args.num_workers,
    )
    dev_loader = _data_loader(
        examples_by_split["dev"], tokenizer, args.batch_size, False, args.seed,
        args.num_workers,
    )
    all_dev_loader = _data_loader(
        examples_by_split["dev"], tokenizer, args.batch_size, False, args.seed,
        args.num_workers,
    )
    checkpoint_hash = file_sha256(args.base_checkpoint)
    run_config = {
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "loaded_splits": ["train", "dev"],
        "encoder": args.encoder_name,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": checkpoint_hash,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "gradient_clip_val": args.gradient_clip_val,
        "precision": args.precision,
        "max_length": args.max_length,
        "max_span_length": args.max_span_length,
        "accelerator": args.accelerator,
        "approach": args.approach,
        "constant_null_score": 0.0,
        "negative_strategy": "split-local context removal / hard replacement",
        "positive_relation_count": {
            split: sum(len(row["relations"]) for row in positives[split])
            for split in ("train", "dev")
        },
        "negative_relation_count": {
            split: len(negatives[split]) for split in ("train", "dev")
        },
        "git": git_metadata(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_yaml(args.output_dir / "config.yaml", run_config)

    approaches = (
        ("constant", "learned_bias", "conditioned")
        if args.approach == "all" else (args.approach,)
    )
    records_by_approach: dict[str, list[dict[str, Any]]] = {}
    training_info: dict[str, dict[str, Any]] = {}
    device = torch.device(
        "mps" if args.accelerator in {"auto", "mps"} and torch.backends.mps.is_available()
        else "cuda" if args.accelerator in {"auto", "gpu"} and torch.cuda.is_available()
        else "cpu"
    )

    for approach in approaches:
        span_model = _load_span_model(args.base_checkpoint)
        if span_model.max_span_length != args.max_span_length:
            raise ValueError(
                "EXP-004 checkpoint max_span_length "
                f"{span_model.max_span_length} does not match --max-span-length "
                f"{args.max_span_length}"
            )
        if approach == "constant":
            model = NullAntecedentModel(
                span_model,
                approach="constant",
                freeze_span_model=True,
            )
            records = _evaluate(model, all_dev_loader, device, approach)
            info = {
                "constant_null_score": 0.0,
                "frozen_backbone": True,
                "trainable_parameter_count": 0,
            }
            approach_dir = args.output_dir / approach
        else:
            model, info = _train_approach(
                approach,
                span_model,
                train_loader,
                dev_loader,
                args,
                args.output_dir,
            )
            records = _evaluate(model, all_dev_loader, device, approach)
            approach_dir = args.output_dir / approach
            training_info[approach] = info

        metrics = evaluate_null_predictions(records)
        metrics.update(stratified_performance(records))
        metrics["calibration"] = calibration_metrics(records)
        metrics["paired_contrast"] = paired_contrast_metrics(records)
        if approach != "constant":
            metrics["bootstrap_95_ci"] = {
                name: bootstrap_interval(records, name, args.seed)
                for name in ("macro_f1", "end_to_end_exact_accuracy")
            }
        metrics["training"] = info
        records_by_approach[approach] = records
        atomic_write_json(approach_dir / "predictions.json", records)
        atomic_write_json(approach_dir / "metrics.json", metrics)
        if approach in {"learned_bias", "conditioned"}:
            atomic_write_json(
                approach_dir / "failure_diagnostics.json",
                _error_diagnostics(records),
            )
        del model, span_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline_records = _always_link(
        records_by_approach.get("constant")
        or next(iter(records_by_approach.values()))
    )
    records_by_approach["always_link"] = baseline_records
    adjustment_source = records_by_approach.get("conditioned") or next(
        rows for name, rows in records_by_approach.items()
        if name != "always_link"
    )
    count_adjustment = _adjust_for_candidate_count(adjustment_source, args.seed)
    comparison = _summary_rows(records_by_approach)
    for name, records in records_by_approach.items():
        comparison[name]["stratified"] = stratified_performance(records)
        comparison[name]["paired_contrast"] = paired_contrast_metrics(records)
        comparison[name]["calibration"] = calibration_metrics(records)
        if name in {"learned_bias", "conditioned"}:
            comparison[name]["bootstrap_95_ci"] = {
                metric: bootstrap_interval(records, metric, args.seed)
                for metric in ("macro_f1", "balanced_accuracy", "end_to_end_exact_accuracy")
            }
    comparison["paired_differences_vs_always_link"] = {
        name: {
            metric: paired_bootstrap_difference(
                records,
                baseline_records,
                metric,
                seed=args.seed,
            )
            for metric in ("macro_f1", "balanced_accuracy", "end_to_end_exact_accuracy")
        }
        for name, records in records_by_approach.items()
        if name in {"constant", "learned_bias", "conditioned"}
    }
    comparison["candidate_count_adjustment"] = count_adjustment
    atomic_write_json(args.output_dir / "comparison_metrics.json", comparison)
    atomic_write_json(
        args.output_dir / "calibration_metrics.json",
        {name: calibration_metrics(rows) for name, rows in records_by_approach.items()},
    )
    atomic_write_json(
        args.output_dir / "contrast_metrics.json",
        {name: paired_contrast_metrics(rows) for name, rows in records_by_approach.items()},
    )
    atomic_write_json(
        args.output_dir / "candidate_count_adjustment.json",
        count_adjustment,
    )
    report = _report_markdown(
        comparison, training_info, audit_out, generation_reports
    )
    (args.output_dir / "comparison_report.md").write_text(report, encoding="utf-8")

    if args.approach == "all":
        _append_experiment_log(
            comparison,
            audit_out,
            generation_reports,
            training_info,
            run_config,
            run_config["git"],
            checkpoint_hash,
            REPOSITORY_ROOT / "EXPERIMENTLOG.md",
        )
    return comparison


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
