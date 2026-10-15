"""Compare frozen affinity expansion with two supervised span extractors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.annotations import filter_annotations
from src.data.datamodule import CoreferenceDataModule
from src.evaluation.span_metrics import (
    LENGTH_BANDS,
    aggregate_span_metrics,
    length_band,
    paired_comparison,
    score_span_prediction,
)
from src.training.module import CoreferenceLightningModule
from src.training.span_module import SpanExtractionLightningModule


DEFAULTS = {
    "dataset_name": "mhaugestad/contextualise-context",
    "dataset_revision": None,
    "encoder_name": "distilbert-base-uncased",
    "head_dim": 128,
    "seed": 42,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_epochs": 20,
    "patience": 4,
    "gradient_clip_val": 1.0,
    "precision": "32-true",
    "dropout": 0.1,
    "max_length": 512,
    "max_span_length": 32,
    "num_workers": 0,
    "accelerator": "auto",
}
OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "exp004_span_extraction"
EXP002_CHECKPOINT = (
    REPOSITORY_ROOT / "experiments" / "exp002_full_training" / "checkpoints" / "best.ckpt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach",
        choices=("all", "baseline", "boundary", "span_pair"),
        default="all",
    )
    parser.add_argument("--dataset-name", default=DEFAULTS["dataset_name"])
    parser.add_argument("--dataset-revision", default=DEFAULTS["dataset_revision"])
    parser.add_argument("--encoder-name", default=DEFAULTS["encoder_name"])
    parser.add_argument("--head-dim", type=int, default=DEFAULTS["head_dim"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--gradient-clip-val", type=float, default=DEFAULTS["gradient_clip_val"])
    parser.add_argument("--precision", default=DEFAULTS["precision"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--max-span-length", type=int, default=DEFAULTS["max_span_length"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--accelerator", default=DEFAULTS["accelerator"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--exp002-checkpoint", type=Path, default=EXP002_CHECKPOINT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require dataset and model/tokenizer files to be cached locally.",
    )
    args = parser.parse_args(argv)
    for name in ("head_dim", "batch_size", "max_epochs", "patience", "max_length", "max_span_length"):
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


def yaml_compatible_json(value: Any) -> str:
    """Serialize JSON, which is a valid YAML 1.2 subset, without extra deps."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": None, "dirty_working_tree": None}
    try:
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["dirty_working_tree"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def dataset_statistics(dataset: Any, raw_split: Any) -> dict[str, Any]:
    """Audit raw annotation filtering and token-alignment exclusions."""
    raw_context = raw_chunk = filtered_context = filtered_chunk = 0
    for example in raw_split:
        raw_context += len(example["context_spans"])
        raw_chunk += len(example["chunk_spans"])
        filtered = filter_annotations(example, warn=False)
        filtered_context += len(filtered["context_spans"])
        filtered_chunk += len(filtered["chunk_spans"])
    examples_with_relations = sum(bool(example["relations"]) for example in dataset.examples)
    relation_count = sum(len(example["relations"]) for example in dataset.examples)
    antecedent_count = sum(
        len(relation["antecedents"])
        for example in dataset.examples
        for relation in example["relations"]
    )
    return {
        "examples": len(dataset),
        "examples_with_valid_relations": examples_with_relations,
        "examples_without_valid_relations": len(dataset) - examples_with_relations,
        "valid_relations": relation_count,
        "valid_antecedent_spans": antecedent_count,
        "source_annotations": {
            "context_spans": raw_context,
            "chunk_spans": raw_chunk,
            "context_spans_after_annotation_filter": filtered_context,
            "chunk_spans_after_annotation_filter": filtered_chunk,
            "context_spans_removed_by_filter": raw_context - filtered_context,
            "chunk_spans_removed_by_filter": raw_chunk - filtered_chunk,
        },
        "preprocessor_alignment_exclusions": {
            "relations": sum(int(example["dropped_relations"]) for example in dataset.examples),
            "annotations": sum(int(example["dropped_annotations"]) for example in dataset.examples),
        },
    }


def span_length_constraint_exclusions(dataset: Any, max_span_length: int) -> dict[str, int]:
    unrepresentable = [
        relation
        for example in dataset.examples
        for relation in example["relations"]
        if not any(
            antecedent["end"] - antecedent["start"] <= max_span_length
            for antecedent in relation["antecedents"]
        )
    ]
    return {
        "relations_with_no_gold_candidate_within_maximum": len(unrepresentable),
        "gold_antecedent_spans_longer_than_maximum": sum(
            antecedent["end"] - antecedent["start"] > max_span_length
            for example in dataset.examples
            for relation in example["relations"]
            for antecedent in relation["antecedents"]
        ),
    }


def _relation_gold(batch: dict[str, Any], batch_index: int, relation_index: int) -> list[tuple[int, int]]:
    indices = batch["antecedent_mask"][batch_index, relation_index].nonzero(
        as_tuple=False
    ).flatten().tolist()
    return [
        tuple(int(value) for value in batch["antecedent_spans"][
            batch_index, relation_index, index
        ].tolist())
        for index in indices
    ]


def _char_spans(
    batch: dict[str, Any],
    batch_index: int,
    relation_index: int,
) -> list[tuple[int, int]]:
    indices = batch["antecedent_mask"][batch_index, relation_index].nonzero(
        as_tuple=False
    ).flatten().tolist()
    spans = []
    for index in indices:
        start, end = (
            int(value)
            for value in batch["antecedent_char_spans"][
                batch_index, relation_index, index
            ].tolist()
        )
        if start < 0 or end <= start:
            raise ValueError("Preprocessor did not preserve a gold character span")
        spans.append((start, end))
    return spans


def _span_to_char(
    offsets: torch.Tensor,
    span: tuple[int, int],
) -> tuple[int, int] | None:
    start, end = span
    if start < 0 or end <= start or end > offsets.shape[0]:
        return None
    char_start = int(offsets[start, 0].item())
    char_end = int(offsets[end - 1, 1].item())
    if char_end <= char_start:
        return None
    return char_start, char_end


class NounPhraseExpander:
    """Optional local spaCy noun chunks, with a documented token fallback."""

    def __init__(self) -> None:
        self.nlp = None
        self.mode = "selected-token fallback (no local en_core_web_sm parser)"
        try:
            import spacy

            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                warnings.warn(
                    "Local spaCy model en_core_web_sm is unavailable; the frozen "
                    "baseline will use its selected token without phrase expansion.",
                    UserWarning,
                    stacklevel=2,
                )
        except ImportError:
            warnings.warn(
                "spaCy is unavailable; the frozen baseline will use its selected "
                "token without phrase expansion.",
                UserWarning,
                stacklevel=2,
            )
        if self.nlp is not None:
            self.mode = (
                f"spaCy noun_chunks ({self.nlp.meta.get('name')} "
                f"{self.nlp.meta.get('version')}) with token fallback"
            )

    def expand(
        self,
        context: str,
        token_span: tuple[int, int],
        offsets: torch.Tensor,
    ) -> tuple[int, int]:
        if self.nlp is None:
            return token_span
        token_char = _span_to_char(offsets, token_span)
        if token_char is None:
            return token_span
        document = self.nlp(context)
        chunks = [
            chunk
            for chunk in document.noun_chunks
            if chunk.start_char <= token_char[0] and chunk.end_char >= token_char[1]
        ]
        if not chunks:
            return token_span
        chunk = max(chunks, key=lambda item: item.end_char - item.start_char)
        overlapping = [
            index
            for index, (start, end) in enumerate(offsets.tolist())
            if end > start and start < chunk.end_char and end > chunk.start_char
        ]
        if not overlapping:
            return token_span
        candidate = (overlapping[0], overlapping[-1] + 1)
        # Do not let parser expansion bridge a special token or a sequence boundary.
        if any(end <= start for start, end in offsets[candidate[0]:candidate[1]].tolist()):
            return token_span
        return candidate


def _record_prediction(
    batch: dict[str, Any],
    batch_index: int,
    relation_index: int,
    predicted_span: tuple[int, int],
    approach: str,
) -> dict[str, Any]:
    context = batch["context"][batch_index]
    chunk = batch["chunk"][batch_index]
    gold_spans = _relation_gold(batch, batch_index, relation_index)
    gold_char_spans = _char_spans(batch, batch_index, relation_index)
    offsets = batch["offset_mapping"][batch_index]
    predicted_char = _span_to_char(offsets, predicted_span)
    if predicted_char is None:
        raise ValueError(f"Predicted token span cannot be mapped to characters: {predicted_span}")
    anaphor_start, anaphor_end = (
        int(value)
        for value in batch["relations"][batch_index, relation_index].tolist()
    )
    anaphor_char = _span_to_char(offsets, (anaphor_start, anaphor_end))
    if anaphor_char is None:
        raise ValueError("Anaphor token span cannot be mapped to characters")
    gold_text = [context[start:end] for start, end in gold_char_spans]
    gold_lengths = [end - start for start, end in gold_spans]
    score = score_span_prediction(
        predicted_span,
        gold_spans,
        predicted_char,
        gold_char_spans,
    )
    return {
        "example_id": batch["id"][batch_index],
        "relation_id": int(batch["relation_ids"][batch_index, relation_index].item()),
        "context_text": context,
        "chunk_text": chunk,
        "anaphor_text": chunk[anaphor_char[0]:anaphor_char[1]],
        "gold_antecedent_texts": gold_text,
        "gold_token_spans": [list(span) for span in gold_spans],
        "gold_character_spans": [list(span) for span in gold_char_spans],
        "gold_antecedent_token_lengths": gold_lengths,
        "predicted_antecedent_text": context[predicted_char[0]:predicted_char[1]],
        "predicted_token_start": predicted_span[0],
        "predicted_token_end": predicted_span[1],
        "predicted_character_start": predicted_char[0],
        "predicted_character_end": predicted_char[1],
        "exact_match": score["exact_match"],
        "token_overlap_f1": score["token_overlap_f1"],
        "localization_hit": score["localization_hit"],
        "start_boundary_correct": score["start_boundary_correct"],
        "end_boundary_correct": score["end_boundary_correct"],
        "character_exact_match": score["character_exact_match"],
        "approach": approach,
    }


def _evaluate_affinity_baseline(
    checkpoint: Path,
    dataloader: DataLoader,
    device: torch.device,
    expander: NounPhraseExpander,
) -> list[dict[str, Any]]:
    model = CoreferenceLightningModule.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model.requires_grad_(False)
    model.eval().to(device)
    records = []
    with torch.inference_mode():
        for batch in dataloader:
            device_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(device_batch)
            for batch_index, relation_index in batch["relation_mask"].nonzero(
                as_tuple=False
            ).tolist():
                start, end = (
                    int(value)
                    for value in batch["relations"][batch_index, relation_index].tolist()
                )
                row_scores = output["logits"][batch_index, start:end].mean(dim=0)
                valid_context = output["pair_mask"][batch_index, start:end].any(dim=0)
                best_token = int(
                    row_scores.masked_fill(~valid_context, -torch.inf).argmax().item()
                )
                expanded = expander.expand(
                    batch["context"][batch_index],
                    (best_token, best_token + 1),
                    batch["offset_mapping"][batch_index],
                )
                records.append(_record_prediction(
                    batch,
                    batch_index,
                    relation_index,
                    expanded,
                    "baseline",
                ))
    return records


def _evaluate_learned_model(
    model: SpanExtractionLightningModule,
    dataloader: DataLoader,
    device: torch.device,
    approach: str,
) -> list[dict[str, Any]]:
    model.eval().to(device)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in dataloader:
            tensor_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(tensor_batch)
            for relation_output in output["relations"]:
                batch_index = int(relation_output["batch_index"])
                relation_index = int(relation_output["relation_index"])
                candidates = relation_output["candidate_spans"]
                scores = relation_output["scores"]
                start, end = candidates[int(scores.argmax().item())].tolist()
                records.append(_record_prediction(
                    batch,
                    batch_index,
                    relation_index,
                    (int(start), int(end)),
                    approach,
                ))
    return records


class BestSpanCheckpoint(Callback):
    """Select by highest dev EM, breaking ties by lower eligible dev loss."""

    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.best_em: float | None = None
        self.best_loss: float | None = None
        self.best_epoch: int | None = None

    def on_validation_end(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        em = float(metrics["val/span_em"].detach().cpu().item())
        loss = float(metrics["val/loss"].detach().cpu().item())
        improved = (
            self.best_em is None
            or em > self.best_em
            or (em == self.best_em and (self.best_loss is None or loss < self.best_loss))
        )
        if not improved:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path), weights_only=True)
        self.best_em = em
        self.best_loss = loss
        self.best_epoch = trainer.current_epoch + 1


def _train_approach(
    approach: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    output_dir: Path,
) -> tuple[SpanExtractionLightningModule, dict[str, Any]]:
    L.seed_everything(args.seed, workers=True)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(args.seed)
    checkpoint_path = output_dir / approach / "checkpoints" / "best.ckpt"
    checkpoint_callback = BestSpanCheckpoint(checkpoint_path)
    early_stopping = EarlyStopping(
        monitor="val/span_em",
        mode="max",
        patience=args.patience,
        check_finite=True,
    )
    module = SpanExtractionLightningModule(
        approach=approach,
        encoder_name=args.encoder_name,
        head_dim=args.head_dim,
        max_span_length=args.max_span_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        precision=args.precision,
        # MPS lacks deterministic indexed-gradient kernels, so use seeded RNG
        # without Lightning enabling strict deterministic algorithms.
        deterministic=False,
        logger=False,
        callbacks=[checkpoint_callback, early_stopping],
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    if not checkpoint_path.is_file() or checkpoint_callback.best_epoch is None:
        raise RuntimeError(f"No best development checkpoint was saved for {approach}")

    selected = SpanExtractionLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    training = {
        "epochs_completed": trainer.current_epoch + 1,
        "stopped_early": trainer.current_epoch + 1 < args.max_epochs,
        "best_epoch": checkpoint_callback.best_epoch,
        "best_development_span_em": checkpoint_callback.best_em,
        "best_development_loss": checkpoint_callback.best_loss,
        "training_runtime_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None
        ),
        "mps_current_allocated_memory_bytes_after_training": (
            torch.mps.current_allocated_memory()
            if hasattr(torch, "mps") and torch.backends.mps.is_available()
            else None
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return selected, training


def _metrics_with_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_span_metrics([
        {
            "exact_match": record["exact_match"],
            "token_overlap_f1": record["token_overlap_f1"],
            "localization_hit": record["localization_hit"],
            "start_boundary_correct": record["start_boundary_correct"],
            "end_boundary_correct": record["end_boundary_correct"],
            "character_exact_match": record["character_exact_match"],
        }
        for record in records
    ])


def span_length_metrics(records_by_approach: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for approach, records in records_by_approach.items():
        groups = {name: [] for name, _, _ in LENGTH_BANDS}
        for record in records:
            shortest_length = min(record["gold_antecedent_token_lengths"])
            groups[length_band(shortest_length)].append(record)
        grouped[approach] = {
            name: {
                "sample_count": len(group_records),
                "metrics": _metrics_with_records(group_records),
            }
            for name, group_records in groups.items()
        }
    return {
        "grouping_rule": (
            "Use the shortest alternative gold antecedent token length for each "
            "relation; this is descriptive and alternatives remain unchanged."
        ),
        "groups": grouped,
    }


def _comparison(
    records_by_approach: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    available = [name for name in ("baseline", "boundary", "span_pair") if name in records_by_approach]
    by_approach = {
        name: {
            f"{record['example_id']}::{record['relation_id']}": record
            for record in records_by_approach[name]
        }
        for name in available
    }
    common_ids = sorted(set.intersection(*(set(rows) for rows in by_approach.values()))) if available else []
    metrics = {}
    paired = {}
    for name in available:
        subset = [by_approach[name][key] for key in common_ids]
        metrics[name] = _metrics_with_records(subset)

    for first, second, label in (
        ("boundary", "baseline", "boundary_vs_baseline"),
        ("span_pair", "baseline", "span_pair_vs_baseline"),
        ("span_pair", "boundary", "span_pair_vs_boundary"),
    ):
        if first in by_approach and second in by_approach:
            paired[label] = paired_comparison(
                [by_approach[first][key]["exact_match"] for key in common_ids],
                [by_approach[second][key]["exact_match"] for key in common_ids],
                first,
                second,
            )
    return {
        "common_relation_count": len(common_ids),
        "metrics_on_identical_relations": metrics,
        "paired_exact_match_comparisons": paired,
        "h1": _hypothesis_status(metrics),
        "h2": _h2_status(metrics),
        "h3": _h3_status(metrics),
    }


def _hypothesis_status(
    common_metrics: dict[str, Any],
) -> dict[str, Any]:
    if "baseline" not in common_metrics or not (
        "boundary" in common_metrics or "span_pair" in common_metrics
    ):
        return {"status": "inconclusive", "reason": "Not all required approaches have results."}
    baseline = common_metrics["baseline"]["exact_span_match"]
    learned = [
        (name, result["exact_span_match"])
        for name, result in common_metrics.items()
        if name in {"boundary", "span_pair"}
    ]
    supported = any(
        values["value"] is not None
        and baseline["value"] is not None
        and values["value"] > baseline["value"]
        for _, values in learned
    )
    return {
        "status": "supported" if supported else "not supported in this dev sample",
        "comparisons": {
            name: {
                "correct": values["correct"],
                "total": values["total"],
                "improvement_percentage_points": (
                    (values["value"] - baseline["value"]) * 100.0
                    if values["value"] is not None and baseline["value"] is not None
                    else None
                ),
            }
            for name, values in learned
        },
    }


def _h2_status(metrics: dict[str, Any]) -> dict[str, Any]:
    if "span_pair" not in metrics or "boundary" not in metrics:
        return {"status": "inconclusive"}
    pair = metrics["span_pair"]["exact_span_match"]["value"]
    boundary = metrics["boundary"]["exact_span_match"]["value"]
    return {
        "status": (
            "supported" if pair is not None and boundary is not None and pair > boundary
            else "not supported in this dev sample"
        ),
        "span_pair_em": pair,
        "boundary_em": boundary,
    }


def _h3_status(metrics: dict[str, Any]) -> dict[str, Any]:
    if "baseline" not in metrics:
        return {"status": "inconclusive"}
    baseline = metrics["baseline"]
    comparisons = {}
    for name in ("boundary", "span_pair"):
        if name not in metrics:
            continue
        current = metrics[name]
        current_localization = current["localization_hit"]["value"]
        baseline_localization = baseline["localization_hit"]["value"]
        current_f1 = current["token_overlap_f1"]["mean"]
        baseline_f1 = baseline["token_overlap_f1"]["mean"]
        comparisons[name] = {
            "localization_hit": current["localization_hit"],
            "token_overlap_f1": current["token_overlap_f1"],
            "localization_change_percentage_points": (
                (current_localization - baseline_localization) * 100.0
                if current_localization is not None and baseline_localization is not None
                else None
            ),
            "token_f1_change": (
                current_f1 - baseline_f1
                if current_f1 is not None and baseline_f1 is not None
                else None
            ),
        }
    return {
        "status": (
            "inconclusive regarding 'substantial' degradation; no threshold was prespecified"
            if comparisons else "inconclusive"
        ),
        "comparisons": comparisons,
    }


def _attach_review_context(
    records_by_approach: dict[str, list[dict[str, Any]]],
    development_data: Any,
) -> None:
    context_by_id = {
        str(example["id"]): {
            "context_text": example["context"],
            "chunk_text": example["chunk"],
        }
        for example in development_data
    }
    for records in records_by_approach.values():
        for record in records:
            context = context_by_id.get(str(record["example_id"]))
            if context is None:
                raise ValueError(
                    f"Development example {record['example_id']} is missing while "
                    "attaching error-review context."
                )
            for name, value in context.items():
                record.setdefault(name, value)


def _h4_status(length_metrics: dict[str, Any]) -> dict[str, Any]:
    approaches = {}
    for approach in ("boundary", "span_pair"):
        groups = length_metrics["groups"].get(approach, {})
        observed = [
            (name, group["metrics"]["exact_span_match"]["value"], group["sample_count"])
            for name, group in groups.items()
            if group["sample_count"] and group["metrics"]["exact_span_match"]["value"] is not None
        ]
        if len(observed) < 2:
            approaches[approach] = {
                "status": "inconclusive",
                "observed_bands": observed,
            }
            continue
        shortest, longest = observed[0], observed[-1]
        approaches[approach] = {
            "status": (
                "descriptively supported"
                if longest[1] < shortest[1]
                else "not supported by shortest-versus-longest observed bands"
            ),
            "shortest_observed_band": {
                "band": shortest[0],
                "exact_match_rate": shortest[1],
                "relations": shortest[2],
            },
            "longest_observed_band": {
                "band": longest[0],
                "exact_match_rate": longest[1],
                "relations": longest[2],
            },
        }
    return {
        "status": (
            "inconclusive until learned results are available"
            if not approaches
            else (
                "descriptively supported for both learned approaches; exploratory"
                if all(
                    item["status"] == "descriptively supported"
                    for item in approaches.values()
                )
                else "mixed or inconclusive exploratory evidence"
            )
        ),
        "approaches": approaches,
    }


def _markdown_table(comparison: dict[str, Any], training: dict[str, Any]) -> str:
    lines = [
        "| Approach | Dev Span EM | Token F1 | Localization Hit | Best Epoch |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline": "Frozen affinity expansion",
        "boundary": "Boundary head",
        "span_pair": "Span-pair scoring",
    }
    for name in ("baseline", "boundary", "span_pair"):
        metrics = comparison.get("metrics_on_identical_relations", {}).get(name)
        if metrics is None:
            lines.append(f"| {labels[name]} | pending | pending | pending | N/A |")
            continue
        em = metrics["exact_span_match"]
        f1 = metrics["token_overlap_f1"]
        localization = metrics["localization_hit"]
        epoch = training.get(name, {}).get("best_epoch", "N/A")
        em_value = "N/A" if em["value"] is None else f"{100 * em['value']:.2f}%"
        loc_value = (
            "N/A"
            if localization["value"] is None
            else f"{100 * localization['value']:.2f}%"
        )
        f1_value = "N/A" if f1["mean"] is None else f"{f1['mean']:.4f}"
        lines.append(
            f"| {labels[name]} | {em['correct']}/{em['total']} "
            f"({em_value}) | {f1_value} | {localization['correct']}/"
            f"{localization['total']} ({loc_value}) | {epoch} |"
        )
    return "\n".join(lines)


def _error_class_candidate(record: dict[str, Any]) -> str:
    """Suggest only geometric error types; semantic labels require review."""
    if not record["localization_hit"]:
        return "wrong entity or semantic/boundary failure (manual review)"
    predicted_length = record["predicted_token_end"] - record["predicted_token_start"]
    best_gold_length = min(
        record["gold_antecedent_token_lengths"],
        key=lambda length: abs(length - predicted_length),
    )
    if predicted_length < best_gold_length:
        return "correct entity, incomplete span (candidate)"
    if predicted_length > best_gold_length:
        return "correct entity, oversized span (candidate)"
    return "other (manual review)"


def _observations(
    comparison: dict[str, Any],
    length_metrics: dict[str, Any],
    training: dict[str, Any],
    records_by_approach: dict[str, list[dict[str, Any]]],
) -> list[str]:
    observations = []
    metrics = comparison["metrics_on_identical_relations"]
    if "baseline" in metrics:
        baseline = metrics["baseline"]
        observations.append(
            f"The frozen expansion baseline extracted "
            f"{baseline['exact_span_match']['correct']}/"
            f"{baseline['exact_span_match']['total']} relations exactly; its "
            f"localization hit was {baseline['localization_hit']['correct']}/"
            f"{baseline['localization_hit']['total']}."
        )
    if "baseline" in metrics and any(name in metrics for name in ("boundary", "span_pair")):
        baseline_rate = metrics["baseline"]["exact_span_match"]["value"]
        learned = [
            (name, values["exact_span_match"])
            for name, values in metrics.items()
            if name in {"boundary", "span_pair"}
        ]
        if baseline_rate is not None and learned:
            best_name, best = max(
                learned,
                key=lambda item: -1 if item[1]["value"] is None else item[1]["value"],
            )
            if best["value"] is not None:
                delta = (best["value"] - baseline_rate) * 100.0
                observations.append(
                    f"The strongest learned approach was {best_name} at "
                    f"{best['correct']}/{best['total']} exact matches, a "
                    f"{delta:+.2f} percentage-point difference from baseline."
                )
    if "boundary" in metrics and "span_pair" in metrics:
        boundary = metrics["boundary"]["exact_span_match"]
        pair = metrics["span_pair"]["exact_span_match"]
        if boundary["value"] is not None and pair["value"] is not None:
            observations.append(
                f"Direct span-pair scoring returned {pair['correct']}/"
                f"{pair['total']} exact matches versus {boundary['correct']}/"
                f"{boundary['total']} for the independent boundary head."
            )
    for name, result in training.items():
        observations.append(
            f"{name} selected epoch {result['best_epoch']} after "
            f"{result['epochs_completed']} completed epochs; runtime was "
            f"{result['training_runtime_seconds']:.1f} seconds."
        )
    if records_by_approach:
        nonexact = {}
        for name, records in records_by_approach.items():
            nonexact[name] = sum(
                not item["exact_match"] and item["localization_hit"]
                for item in records
            )
        if any(nonexact.values()):
            observations.append(
                "Localized but non-exact predictions (the boundary-completion "
                "gap) numbered " + ", ".join(
                    f"{name}: {count}" for name, count in nonexact.items()
                ) + "."
            )
    nonempty_bands = []
    for name, groups in length_metrics["groups"].items():
        for band, values in groups.items():
            rate = values["metrics"]["exact_span_match"]["value"]
            if values["sample_count"] and rate is not None:
                nonempty_bands.append((name, band, rate, values["sample_count"]))
    if nonempty_bands:
        low = min(nonempty_bands, key=lambda item: item[2])
        high = max(nonempty_bands, key=lambda item: item[2])
        observations.append(
            f"Observed span-length EM ranged from {low[2]:.2%} in the "
            f"{low[1]}-token band ({low[3]} relations, {low[0]}) to "
            f"{high[2]:.2%} in the {high[1]}-token band ({high[3]} relations, "
            f"{high[0]}); these descriptive groups can be small."
        )
    review_counts = {}
    for name in ("boundary", "span_pair"):
        reviewed = [
            item for item in records_by_approach.get(name, [])
            if item.get("error_reviewed") and item.get("error_class")
        ]
        if reviewed:
            review_counts[name] = Counter(item["error_class"] for item in reviewed)
    if review_counts:
        observations.append(
            "Manual review of the first 20 incorrect predictions per learned "
            "approach found "
            + "; ".join(
                f"{name}: " + ", ".join(
                    f"{category} {count}" for category, count in counts.most_common()
                )
                for name, counts in review_counts.items()
            )
            + ". These are diagnostic labels, not causal claims."
        )
    return observations[:8]


def update_report(
    output_dir: Path,
    comparison: dict[str, Any],
    length_metrics: dict[str, Any],
    training: dict[str, Any],
    records_by_approach: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    parser_mode: str,
) -> None:
    complete = all(name in records_by_approach for name in ("baseline", "boundary", "span_pair"))
    exact_metrics = comparison["metrics_on_identical_relations"]
    learned_em = {
        name: exact_metrics[name]["exact_span_match"]
        for name in ("boundary", "span_pair")
        if name in exact_metrics
    }
    best_learned = max(
        learned_em,
        key=lambda name: (
            -1.0 if learned_em[name]["value"] is None
            else learned_em[name]["value"]
        ),
        default=None,
    )
    baseline_em = exact_metrics.get("baseline", {}).get(
        "exact_span_match", {}
    ).get("value")
    if not complete:
        conclusion = (
            "Partial only; do not draw an approach-selection conclusion until all "
            "three approaches have been evaluated."
        )
        next_steps = (
            "Complete the remaining approach runs, then inspect at least 20 incorrect "
            "predictions per learned approach (or all errors if fewer) and classify "
            "the error-review set."
        )
    elif best_learned is None or learned_em[best_learned]["value"] is None:
        conclusion = (
            "The learned approaches had no evaluable relations, so no model-selection "
            "claim can be made."
        )
        next_steps = (
            "Investigate empty or unrepresentable supervision before running another "
            "development comparison."
        )
    else:
        best = learned_em[best_learned]
        delta = None if baseline_em is None else (best["value"] - baseline_em) * 100.0
        if delta is not None and delta > 0:
            conclusion = (
                f"{best_learned} achieved the highest learned development Exact Span "
                f"Match at {best['correct']}/{best['total']} ({best['value']:.2%}), "
                f"{delta:.2f} percentage points above the frozen baseline. This "
                "supports H1 descriptively, not as a statistical-significance claim."
            )
        else:
            conclusion = (
                "Neither learned model exceeded the frozen baseline on development "
                f"Exact Span Match; the highest learned result was {best_learned} at "
                f"{best['correct']}/{best['total']} ({best['value']:.2%}). Explicit "
                "span supervision did not improve exact extraction in this dev sample."
            )
        next_steps = (
            f"Use {best_learned} as the leading learned approach for a controlled "
            "follow-up, review the remaining incorrect predictions, and evaluate "
            "boundary improvements on development data only. Preserve the reserved test split "
            "for a later locked evaluation."
        )
    report = [
        "# EXP-004 — Antecedent Span Extraction",
        "",
        f"**Status:** {'completed' if complete else 'partial; approaches may be run independently'}",
        "",
        "**Description:** Compared deterministic span expansion of the frozen EXP-002 affinity checkpoint with an anaphor-conditioned start/end boundary head and direct span-pair scorer. All approaches use the gold anaphor spans and the same filtered, token-aligned development relations; the reserved EXP-003 test split is not loaded.",
        "",
        "**Configuration:**",
        "",
        "```yaml",
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False),
        "```",
        "",
        "**Baseline expansion:** " + parser_mode + ".",
        "",
        "**Span-pair cost:** Candidate count per relation is bounded by `L * min(L, 32)` for a single contiguous context of `L` tokens. Scoring uses `head_dim`-width candidate representations (approximately 8 MiB per 16,000-candidate fp32 feature matrix at width 128), in addition to encoder activations.",
        "",
        "**Results on the common eligible relation set:**",
        "",
        _markdown_table(comparison, training),
        "",
        f"Common relation count: {comparison['common_relation_count']}.",
        "",
        "Paired comparisons and percentile bootstrap intervals are in `comparison_metrics.json`; intervals are descriptive and are not treated as significance tests.",
        "",
        "**Span-length breakdown:** See `span_length_metrics.json`. Relations with multiple gold alternatives use the shortest gold token length to select a band; sample counts are reported per band.",
        "",
        "| Shortest gold length | " + " | ".join(
            f"{name} n / EM" for name in ("baseline", "boundary", "span_pair")
        ) + " |",
        "|---|" + "---:|" * 3,
    ]
    for band, _, _ in LENGTH_BANDS:
        cells = []
        for approach in ("baseline", "boundary", "span_pair"):
            group = length_metrics["groups"].get(approach, {}).get(band)
            if group is None:
                cells.append("pending")
                continue
            em = group["metrics"]["exact_span_match"]
            value = "N/A" if em["value"] is None else f"{100 * em['value']:.1f}%"
            cells.append(f"{group['sample_count']} / {value}")
        report.append(f"| {band} | " + " | ".join(cells) + " |")
    report.extend([
        "",
        "**Training:**",
        "",
    ])
    if training:
        for approach, result in training.items():
            report.append(
                f"- {approach}: best epoch {result['best_epoch']}, dev EM "
                f"{result['best_development_span_em']:.4f}, validation loss "
                f"{result['best_development_loss']:.4f}, "
                f"{result['training_runtime_seconds']:.1f}s."
            )
    else:
        report.append("- No learned approach has completed training.")
    report.extend([
        "",
        "**Diagnostics:** Relation-level predictions are stored under each approach directory. The first 20 incorrect predictions per learned approach were manually reviewed and labeled; the error-review JSON retains context, gold/predicted spans, and review status.",
        "",
        "| Reviewed error category | Boundary | Span-pair |",
        "|---|---:|---:|",
    ])
    for category in (
        "Correct entity, incomplete span",
        "Correct entity, oversized span",
        "Wrong entity",
        "Wrong semantic type",
        "Event/proposition boundary failure",
        "Tokenization/alignment issue",
        "Other",
    ):
        counts = [
            sum(
                item.get("error_reviewed") is True
                and item.get("error_class") == category
                for item in records_by_approach.get(approach, [])
            )
            for approach in ("boundary", "span_pair")
        ]
        report.append(f"| {category} | {counts[0]} | {counts[1]} |")
    report.extend([
        "",
        "**Observations:**",
        "",
    ])
    report.extend(f"- {observation}" for observation in _observations(
        comparison, length_metrics, training, records_by_approach
    ))
    report.extend([
        "",
        "**Hypotheses:**",
        "",
        f"- H1: {comparison['h1']['status']}.",
        f"- H2: {comparison['h2']['status']}.",
        f"- H3: {comparison['h3']['status']}; see localization-hit and overlap-F1 comparisons.",
        f"- H4: {_h4_status(length_metrics)['status']}; see the length-band counts and rates.",
        "",
        "**Conclusion:** " + conclusion,
        "",
        "**Next steps:** " + next_steps,
        "",
    ])
    (output_dir / "comparison_report.md").write_text("\n".join(report), encoding="utf-8")


def update_experiment_log(output_dir: Path) -> None:
    """Append/update a factual EXP-004 log entry based only on saved artifacts."""
    log_path = REPOSITORY_ROOT / "EXPERIMENTLOG.md"
    if not log_path.is_file():
        raise FileNotFoundError(f"Required experiment log does not exist: {log_path}")
    report_path = output_dir / "comparison_report.md"
    report = report_path.read_text(encoding="utf-8")

    def section(start: str, end: str) -> str:
        content = report.split(start, 1)[1]
        return content.split(end, 1)[0].strip()

    description = section("**Description:** ", "\n\n**Configuration:**")
    configuration = section("**Configuration:**", "\n\n**Baseline expansion:**")
    results = section(
        "**Results on the common eligible relation set:**",
        "\n\n**Span-length breakdown:**",
    )
    observations = section("**Observations:**", "\n\n**Hypotheses:**")
    hypotheses = section("**Hypotheses:**", "\n\n**Conclusion:**")
    conclusion = section("**Conclusion:** ", "\n\n**Next steps:**")
    next_steps = report.split("**Next steps:** ", 1)[1].strip()
    entry = (
        "\n## EXP-004 — Antecedent Span Extraction\n\n"
        f"**Description:** {description}\n\n"
        f"**Hypotheses:**\n\n{hypotheses}\n\n"
        f"**Configuration:**\n\n{configuration}\n\n"
        f"**Results:**\n\n{results}\n\n"
        f"**Observations:**\n\n{observations}\n\n"
        f"**Hypothesis assessment:**\n\n{hypotheses}\n\n"
        f"**Conclusion:** {conclusion}\n\n"
        f"**Next steps:** {next_steps}"
    )
    original = log_path.read_text(encoding="utf-8")
    marker = "## EXP-004 — Antecedent Span Extraction"
    if marker in original:
        prefix = original.split(marker, 1)[0].rstrip()
        suffix = original.split(marker, 1)[1]
        next_section = suffix.find("\n## ")
        suffix = suffix[next_section:] if next_section >= 0 else ""
        updated = prefix + "\n\n" + entry.rstrip() + suffix + "\n"
    else:
        updated = original.rstrip() + "\n" + entry
    log_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _prepare_datamodule(args: argparse.Namespace) -> CoreferenceDataModule:
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    datamodule = CoreferenceDataModule(
        dataset_name=args.dataset_name,
        tokenizer_name=args.encoder_name,
        dataset_revision=args.dataset_revision,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        truncation_policy="drop",
        annotation_policy="drop",
    )
    # Fit loads only train and dev. Never call setup(None), test, or predict here.
    datamodule.setup("fit")
    return datamodule


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    config = {
        **{key: getattr(args, key) for key in DEFAULTS},
        "approach": args.approach,
        "approach_configurations": {
            "baseline": {
                "checkpoint": str(args.exp002_checkpoint.resolve()),
                "checkpoint_sha256": None,
                "expansion": "containing spaCy noun chunk if a local parser is available; otherwise selected token",
            },
            "boundary": {
                "encoder_name": args.encoder_name,
                "head_dim": args.head_dim,
                "max_span_length": args.max_span_length,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            "span_pair": {
                "encoder_name": args.encoder_name,
                "head_dim": args.head_dim,
                "max_span_length": args.max_span_length,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
        },
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "exp002_checkpoint": str(args.exp002_checkpoint.resolve()),
        "output_dir": str(output_dir.resolve()),
        "git": git_metadata(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
        "kernel_determinism": (
            "seeded RNG; deterministic algorithms disabled because MPS indexed "
            "gradient accumulation has no deterministic implementation"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_used": False,
    }
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml_compatible_json(config), encoding="utf-8")
    L.seed_everything(args.seed, workers=True)
    datamodule = _prepare_datamodule(args)
    train_data, dev_data = datamodule.splits["train"], datamodule.splits["dev"]
    raw_splits = datamodule._raw_dataset
    dataset_stats = {
        "dataset_name": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "train": dataset_statistics(train_data, raw_splits["train"]),
        "dev": dataset_statistics(dev_data, raw_splits["dev"]),
        "test_split_used": False,
        "test_split_note": "Reserved behavioral test split is not loaded or evaluated.",
        "max_span_length": args.max_span_length,
        "span_length_constraint_exclusions": {
            "train": span_length_constraint_exclusions(train_data, args.max_span_length),
            "dev": span_length_constraint_exclusions(dev_data, args.max_span_length),
        },
    }
    atomic_write_json(output_dir / "dataset_stats.json", dataset_stats)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=datamodule.collator,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=datamodule.collator,
    )
    device = torch.device(
        "cuda" if args.accelerator == "cuda" and torch.cuda.is_available()
        else "mps" if args.accelerator in {"auto", "mps"} and torch.backends.mps.is_available()
        else "cpu"
    )
    requested = (
        ("baseline", "boundary", "span_pair")
        if args.approach == "all"
        else (args.approach,)
    )
    records_by_approach: dict[str, list[dict[str, Any]]] = {}
    training: dict[str, dict[str, Any]] = {}
    for approach in ("baseline", "boundary", "span_pair"):
        if approach not in requested:
            prediction_path = output_dir / approach / "predictions.json"
            if prediction_path.is_file():
                records_by_approach[approach] = json.loads(
                    prediction_path.read_text(encoding="utf-8")
                )
            metrics_path = output_dir / approach / "metrics.json"
            if metrics_path.is_file():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                if approach == "baseline":
                    config["baseline_expansion"] = saved["parser_expansion"]
                    checkpoint_sha256 = saved.get("checkpoint_sha256")
                    if checkpoint_sha256 is None and saved.get("checkpoint"):
                        checkpoint_sha256 = file_sha256(Path(saved["checkpoint"]))
                    if checkpoint_sha256 is None:
                        raise ValueError(
                            f"Baseline metrics at {metrics_path} do not identify "
                            "a checkpoint to hash."
                        )
                    config["approach_configurations"]["baseline"][
                        "checkpoint_sha256"
                    ] = checkpoint_sha256
                elif "training" in saved:
                    training[approach] = saved["training"]
                    training_info = training[approach]
                    checkpoint_sha256 = training_info.get("checkpoint_sha256")
                    if checkpoint_sha256 is None and training_info.get("checkpoint"):
                        checkpoint_sha256 = file_sha256(
                            Path(training_info["checkpoint"])
                        )
                    if checkpoint_sha256 is None:
                        raise ValueError(
                            f"Saved {approach} training metrics at {metrics_path} "
                            "do not identify a checkpoint to hash."
                        )
                    training_info["checkpoint_sha256"] = checkpoint_sha256
                    config["approach_configurations"][approach].update({
                        "checkpoint_sha256": checkpoint_sha256,
                        "best_epoch": training_info["best_epoch"],
                        "training_runtime_seconds": (
                            training_info["training_runtime_seconds"]
                        ),
                    })
            continue

        approach_dir = output_dir / approach
        approach_dir.mkdir(parents=True, exist_ok=True)
        if approach == "baseline":
            if not args.exp002_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Frozen EXP-002 checkpoint not found: {args.exp002_checkpoint}"
                )
            expander = NounPhraseExpander()
            records = _evaluate_affinity_baseline(
                args.exp002_checkpoint,
                dev_loader,
                device,
                expander,
            )
            records_by_approach[approach] = records
            metrics = {
                **_metrics_with_records(records),
                "checkpoint": str(args.exp002_checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(args.exp002_checkpoint),
                "parser_expansion": expander.mode,
            }
            atomic_write_json(approach_dir / "predictions.json", records)
            atomic_write_json(approach_dir / "metrics.json", metrics)
            config["baseline_expansion"] = expander.mode
            config["approach_configurations"]["baseline"]["checkpoint_sha256"] = (
                metrics["checkpoint_sha256"]
            )
            config["approach"] = (
                "all"
                if all(
                    name in records_by_approach
                    for name in ("baseline", "boundary", "span_pair")
                )
                else args.approach
            )
            config_path.write_text(yaml_compatible_json(config), encoding="utf-8")
            continue

        module, training_info = _train_approach(
            approach,
            args,
            train_loader,
            dev_loader,
            output_dir,
        )
        records = _evaluate_learned_model(module, dev_loader, device, approach)
        records_by_approach[approach] = records
        metrics = {
            **_metrics_with_records(records),
            "training": training_info,
            "loss_excluded_unrepresentable_relations": {
                split: value["relations_with_no_gold_candidate_within_maximum"]
                for split, value in dataset_stats["span_length_constraint_exclusions"].items()
            },
            "checkpoint_selection": {
                "monitor": "val/span_em",
                "mode": "max",
                "tie_breaker": "lower development loss",
                "best_epoch": training_info["best_epoch"],
            },
        }
        training[approach] = training_info
        config["approach_configurations"][approach].update({
            "checkpoint_sha256": training_info["checkpoint_sha256"],
            "best_epoch": training_info["best_epoch"],
            "training_runtime_seconds": training_info["training_runtime_seconds"],
        })
        # Geometric candidates are hints only; semantic error classes require review.
        incorrect = [record for record in records if not record["exact_match"]]
        for record in incorrect:
            record["error_class"] = None
            record["error_reviewed"] = False
            record["error_class_candidate"] = _error_class_candidate(record)
        atomic_write_json(approach_dir / "predictions.json", records)
        atomic_write_json(approach_dir / "error_review.json", incorrect[:20])
        atomic_write_json(approach_dir / "metrics.json", metrics)

    config["approach"] = (
        "all"
        if all(
            name in records_by_approach
            for name in ("baseline", "boundary", "span_pair")
        )
        else args.approach
    )
    config_path.write_text(yaml_compatible_json(config), encoding="utf-8")
    _attach_review_context(records_by_approach, dev_data)
    for approach, records in records_by_approach.items():
        approach_dir = output_dir / approach
        atomic_write_json(approach_dir / "predictions.json", records)
        if approach in {"boundary", "span_pair"}:
            incorrect = [record for record in records if not record["exact_match"]]
            atomic_write_json(approach_dir / "error_review.json", incorrect[:20])

    length_metrics = span_length_metrics(records_by_approach)
    comparison = _comparison(records_by_approach)
    comparison["h4"] = _h4_status(length_metrics)
    atomic_write_json(output_dir / "comparison_metrics.json", comparison)
    atomic_write_json(output_dir / "span_length_metrics.json", length_metrics)
    update_report(
        output_dir,
        comparison,
        length_metrics,
        training,
        records_by_approach,
        config,
        config.get("baseline_expansion", "baseline not run"),
    )
    update_experiment_log(output_dir)
    return {
        "status": "completed" if all(
            name in records_by_approach for name in ("baseline", "boundary", "span_pair")
        ) else "partial",
        "comparison": comparison,
        "output_dir": str(output_dir.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
