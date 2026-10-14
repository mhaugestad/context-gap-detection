"""Run EXP-002: full-data training and development baseline comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.datamodule import CoreferenceDataModule
from src.models.losses import coreference_loss
from src.training.metrics import coreference_hit_counts
from src.training.module import CoreferenceLightningModule


DISTANCE_BANDS = (
    ("0-16", 0, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    (">128", 129, None),
)

DEFAULTS = {
    "dataset_name": "mhaugestad/contextualise-context",
    "dataset_revision": None,
    "encoder_name": "distilbert-base-uncased",
    "projection_dim": 128,
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
    "num_workers": 0,
    "accelerator": "auto",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULTS["dataset_name"])
    parser.add_argument("--dataset-revision", default=DEFAULTS["dataset_revision"])
    parser.add_argument("--encoder-name", default=DEFAULTS["encoder_name"])
    parser.add_argument("--projection-dim", type=int, default=DEFAULTS["projection_dim"])
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
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--accelerator", default=DEFAULTS["accelerator"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "exp002_full_training",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the dataset, tokenizer, and model weights to be cached locally.",
    )
    parser.add_argument(
        "--skip-untrained-reference",
        action="store_true",
        help="Skip optional pre-training model evaluation.",
    )
    args = parser.parse_args(argv)
    for name in ("projection_dim", "batch_size", "max_epochs", "patience", "max_length"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0 or args.gradient_clip_val < 0:
        parser.error("--weight-decay and --gradient-clip-val cannot be negative")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0, 1)")
    return args


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def git_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        metadata["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["dirty"] = bool(
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
    return metadata


def dataset_statistics(dataset: Any) -> dict[str, int]:
    """Count preprocessed supervision; keep empty examples in example totals."""
    examples_with_relations = 0
    relation_count = 0
    anaphor_token_count = 0
    antecedent_span_count = 0
    dropped_relation_count = 0
    dropped_annotation_count = 0
    for example in dataset.examples:
        relations = example["relations"]
        examples_with_relations += bool(relations)
        relation_count += len(relations)
        dropped_relation_count += int(example.get("dropped_relations", 0))
        dropped_annotation_count += int(example.get("dropped_annotations", 0))
        for relation in relations:
            anaphor_token_count += (
                relation["anaphor_end"] - relation["anaphor_start"]
            )
            antecedent_span_count += len(relation["antecedents"])
    return {
        "examples": len(dataset),
        "examples_with_valid_relations": examples_with_relations,
        "examples_without_valid_relations": len(dataset) - examples_with_relations,
        "valid_relations": relation_count,
        "anaphor_tokens": anaphor_token_count,
        "valid_antecedent_spans": antecedent_span_count,
        "preprocessor_reported_dropped_relations": dropped_relation_count,
        "preprocessor_reported_dropped_annotations": dropped_annotation_count,
    }


def _valid_context(batch: dict[str, Any], batch_index: int, sequence_length: int) -> torch.Tensor:
    """Context-only candidates come from the established input mask."""
    return batch["context_mask"][batch_index, :sequence_length].to(dtype=torch.bool)


def evaluate_baselines(dataloader: DataLoader) -> dict[str, Any]:
    """Evaluate exact random expectation and last-context-token recency."""
    expected_random_correct = 0.0
    recency_correct = 0
    relation_count = 0
    for batch in dataloader:
        sequence_length = batch["input_ids"].shape[1]
        for batch_index in range(batch["input_ids"].shape[0]):
            valid_context = _valid_context(batch, batch_index, sequence_length)
            for relation_index in batch["relation_mask"][batch_index].nonzero(
                as_tuple=False
            ).flatten().tolist():
                gold_context = torch.zeros(sequence_length, dtype=torch.bool)
                valid_antecedents = batch["antecedent_mask"][
                    batch_index, relation_index
                ]
                for antecedent_index in valid_antecedents.nonzero(
                    as_tuple=False
                ).flatten().tolist():
                    start, end = batch["antecedent_spans"][
                        batch_index, relation_index, antecedent_index
                    ].tolist()
                    gold_context[start:end] = True
                eligible_gold_count = int((gold_context & valid_context).sum())
                context_count = int(valid_context.sum())
                if context_count == 0 or eligible_gold_count == 0:
                    raise ValueError(
                        "Development relation has no valid gold/context tokens"
                    )
                # This is the exact expectation for a uniform draw, so the
                # reported baseline is deterministic and has no Monte Carlo
                # sampling error.
                expected_random_correct += eligible_gold_count / context_count
                last_context_token = int(
                    valid_context.nonzero(as_tuple=False)[-1].item()
                )
                recency_correct += int(gold_context[last_context_token].item())
                relation_count += 1

    return {
        "denominator_relations": relation_count,
        "random_expected_relation_hit_at_1": {
            "expected_correct": expected_random_correct,
            "total": relation_count,
            "value": expected_random_correct / relation_count if relation_count else 0.0,
            "interpretation": "Exact expected accuracy under uniform sampling of a valid context token.",
        },
        "recency_relation_hit_at_1": {
            "correct": recency_correct,
            "total": relation_count,
            "value": recency_correct / relation_count if relation_count else 0.0,
        },
    }


def distance_in_tokens(
    anaphor_span: Sequence[int],
    antecedent_spans: Sequence[Sequence[int]],
) -> int:
    """Return nearest token-to-token absolute encoder-index distance.

    Spans are half-open; the distance is the minimum absolute difference
    between token indices in the anaphor and any antecedent mention.
    """
    anaphor_start, anaphor_end = map(int, anaphor_span)
    if anaphor_end <= anaphor_start:
        raise ValueError("Anaphor span must be a non-empty half-open interval")
    distances = []
    for antecedent_start, antecedent_end in antecedent_spans:
        antecedent_start, antecedent_end = int(antecedent_start), int(antecedent_end)
        if antecedent_end <= antecedent_start:
            raise ValueError("Antecedent spans must be non-empty half-open intervals")
        nearest_anaphor_token = min(
            max(antecedent_start, anaphor_start),
            anaphor_end - 1,
        )
        nearest_antecedent_token = min(
            max(nearest_anaphor_token, antecedent_start),
            antecedent_end - 1,
        )
        distances.append(abs(nearest_anaphor_token - nearest_antecedent_token))
    if not distances:
        raise ValueError("At least one antecedent span is required for distance")
    return min(distances)


def distance_band(distance: int) -> str:
    if distance < 0:
        raise ValueError("Distance cannot be negative")
    for label, lower, upper in DISTANCE_BANDS:
        if distance >= lower and (upper is None or distance <= upper):
            return label
    raise AssertionError(f"Distance {distance} does not fall into any distance band")


def build_distance_metrics(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        groups[distance_band(int(prediction["antecedent_distance"]))].append(
            prediction
        )
    result = {}
    for label, _, _ in DISTANCE_BANDS:
        group = groups[label]
        correct = sum(bool(item["correct"]) for item in group)
        result[label] = {
            "relations": len(group),
            "correct": correct,
            "relation_hit_at_1": correct / len(group) if group else None,
        }
    return result


def tokenizer_offsets(tokenizer: Any, example: dict[str, Any], max_length: int):
    encoding = tokenizer(
        example["context"],
        example["chunk"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    if len(encoding["offset_mapping"]) != len(example["input_ids"]):
        raise ValueError(f"Offset mapping length mismatch in example {example['id']}")
    return [tuple(offset) for offset in encoding["offset_mapping"]]


def span_text(text: str, offsets: Sequence[Sequence[int]], start: int, end: int) -> str:
    nonempty = [offset for offset in offsets[start:end] if offset[1] > offset[0]]
    if not nonempty:
        return ""
    return text[int(nonempty[0][0]) : int(nonempty[-1][1])]


@torch.no_grad()
def evaluate_model(
    model: CoreferenceLightningModule,
    dataloader: DataLoader,
    examples: Any,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate all supervised relations with shared Task 3 Hit@1 logic."""
    was_training = model.training
    model.eval()
    model.to(device)
    relation_weighted_loss = 0.0
    relation_count = token_correct = token_count = relation_correct = 0
    predictions = []
    examples_by_id = {str(example["id"]): example for example in examples.examples}

    for batch in dataloader:
        device_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(device_batch)
        loss = coreference_loss(
            logits=output["logits"],
            pair_mask=output["pair_mask"],
            relations=device_batch["relations"],
            antecedent_spans=device_batch["antecedent_spans"],
            relation_mask=device_batch["relation_mask"],
            antecedent_mask=device_batch["antecedent_mask"],
        )
        batch_counts = coreference_hit_counts(
            output["logits"],
            output["pair_mask"],
            device_batch["relations"],
            device_batch["antecedent_spans"],
            device_batch["relation_mask"],
            device_batch["antecedent_mask"],
        )
        batch_token_correct, batch_token_count, batch_relation_correct, batch_relation_count = (
            batch_counts
        )
        token_correct += batch_token_correct
        token_count += batch_token_count
        relation_correct += batch_relation_correct
        relation_count += batch_relation_count
        relation_weighted_loss += float(loss.detach().cpu()) * batch_relation_count

        for batch_index, example_id in enumerate(batch["id"]):
            example = examples_by_id[str(example_id)]
            offsets = tokenizer_offsets(tokenizer, example, max_length)
            relation_indices = batch["relation_mask"][batch_index].nonzero(
                as_tuple=False
            ).flatten().tolist()
            for relation_index in relation_indices:
                anaphor_start, anaphor_end = map(
                    int,
                    batch["relations"][batch_index, relation_index].tolist(),
                )
                query_positions = torch.arange(
                    anaphor_start,
                    anaphor_end,
                    device=device,
                )
                logits = output["logits"][batch_index, query_positions]
                valid_by_query = output["pair_mask"][batch_index, query_positions]
                # Average each context candidate only over query tokens for
                # which that candidate is valid, matching coreference_hit_counts.
                valid_counts = valid_by_query.sum(dim=0)
                valid_context = valid_counts > 0
                mean_scores = logits.masked_fill(~valid_by_query, 0.0).sum(dim=0)
                mean_scores = mean_scores / valid_counts.clamp_min(1)
                mean_scores = mean_scores.masked_fill(~valid_context, -torch.inf)
                predicted_position = int(mean_scores.argmax().item())

                gold_positions: set[int] = set()
                gold_spans = []
                gold_texts = []
                for antecedent_index in batch["antecedent_mask"][
                    batch_index, relation_index
                ].nonzero(as_tuple=False).flatten().tolist():
                    start, end = map(
                        int,
                        batch["antecedent_spans"][
                            batch_index, relation_index, antecedent_index
                        ].tolist(),
                    )
                    gold_positions.update(range(start, end))
                    gold_spans.append((start, end))
                    gold_texts.append(
                        span_text(example["context"], offsets, start, end)
                    )

                valid_gold_positions = sorted(
                    gold_positions
                    & set(valid_context.nonzero(as_tuple=False).flatten().tolist())
                )
                if not valid_gold_positions:
                    raise ValueError(
                        f"Gold antecedent for {example_id}/{relation_index} "
                        "is absent from valid context positions"
                    )
                best_gold_score = mean_scores[valid_gold_positions].max()
                gold_rank = 1 + int(
                    (mean_scores[valid_context] > best_gold_score).sum().item()
                )
                recency_position = int(
                    valid_context.nonzero(as_tuple=False)[-1].item()
                )
                distance = distance_in_tokens(
                    (anaphor_start, anaphor_end),
                    gold_spans,
                )
                predictions.append(
                    {
                        "example_id": str(example_id),
                        "anaphor_text": span_text(
                            example["chunk"],
                            offsets,
                            anaphor_start,
                            anaphor_end,
                        ),
                        "gold_antecedent_texts": gold_texts,
                        "predicted_context_token": span_text(
                            example["context"],
                            offsets,
                            predicted_position,
                            predicted_position + 1,
                        ),
                        "predicted_context_token_index": predicted_position,
                        "correct": predicted_position in gold_positions,
                        "highest_scoring_gold_rank": gold_rank,
                        "antecedent_distance": distance,
                        "recency_baseline_correct": recency_position in gold_positions,
                    }
                )

    metrics = {
        "loss": relation_weighted_loss / relation_count if relation_count else 0.0,
        "token_hit_at_1": {
            "correct": token_correct,
            "total": token_count,
            "value": token_correct / token_count if token_count else 0.0,
        },
        "relation_hit_at_1": {
            "correct": relation_correct,
            "total": relation_count,
            "value": relation_correct / relation_count if relation_count else 0.0,
        },
    }
    model.train(was_training)
    return metrics, predictions


def baseline_prediction_counts(predictions: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Provide model-comparable relation denominator and recency successes."""
    return {
        "recency_correct": sum(
            bool(prediction["recency_baseline_correct"])
            for prediction in predictions
        ),
        "relations": len(predictions),
    }


def _yaml_subset(config: dict[str, Any]) -> str:
    """Serialize nested experiment configuration as YAML without a dependency."""
    def render_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        return json.dumps(str(value), ensure_ascii=False)

    def render_mapping(values: dict[str, Any], indent: int = 0) -> list[str]:
        lines = []
        for key, value in values.items():
            prefix = " " * indent + f"{key}:"
            if isinstance(value, dict):
                lines.append(prefix)
                lines.extend(render_mapping(value, indent + 2))
            else:
                lines.append(f"{prefix} {render_value(value)}")
        return lines

    return "\n".join(render_mapping(config)) + "\n"


def resolve_device(accelerator: str) -> torch.device:
    """Resolve the evaluation device consistently with Lightning's auto order."""
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"Accelerator {accelerator!r} requested but CUDA is unavailable.")
        return torch.device("cuda")
    if accelerator == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Accelerator 'mps' requested but MPS is unavailable.")
        return torch.device("mps")
    if accelerator != "auto":
        raise ValueError(f"Cannot determine evaluation device for {accelerator!r}.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lightning_scalar_metrics(trainer: L.Trainer, prefix: str) -> dict[str, float]:
    """Extract final-epoch values Lightning logged, excluding unrelated metrics."""
    return {
        name: float(value.detach().cpu())
        for name, value in trainer.callback_metrics.items()
        if name.startswith(f"{prefix}/") and isinstance(value, torch.Tensor)
    }


class BestValidationEpoch(Callback):
    """Track the epoch associated with the best validation loss."""

    def __init__(self) -> None:
        super().__init__()
        self.best_epoch = -1
        self.best_score = math.inf

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        score = trainer.callback_metrics.get("val/loss")
        if score is None:
            raise RuntimeError("Lightning did not log required checkpoint metric val/loss.")
        score_value = float(score.detach().cpu())
        if not math.isfinite(score_value):
            raise RuntimeError(f"Validation loss is not finite: {score_value}")
        if score_value < self.best_score:
            self.best_score = score_value
            self.best_epoch = trainer.current_epoch


def _run(args: argparse.Namespace, captured_warnings: list[str]) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    git = git_metadata()
    config = {
        **{key: getattr(args, key) for key in DEFAULTS},
        "dataset_revision": args.dataset_revision,
        "early_stopping_monitor": "val/loss",
        "early_stopping_patience": args.patience,
        "checkpoint_monitor": "val/loss",
        "checkpoint_mode": "min",
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git["commit"],
        "git_worktree_dirty": git["dirty"],
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
        "device": {
            "accelerator_requested": args.accelerator,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        },
    }
    (output_dir / "config.yaml").write_text(
        _yaml_subset(config),
        encoding="utf-8",
    )
    started = time.monotonic()
    result: dict[str, Any] = {"status": "running", "configuration": config}

    try:
        L.seed_everything(args.seed, workers=True)
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
        # The DataModule's fit stage loads train and dev only. In particular,
        # do not call setup(None), test, or predict: test is reserved for EXP-003.
        datamodule.setup("fit")
        train_data = datamodule.splits["train"]
        dev_data = datamodule.splits["dev"]
        train_stats = dataset_statistics(train_data)
        dev_stats = dataset_statistics(dev_data)
        dataset_stats = {
            "dataset_name": args.dataset_name,
            "dataset_revision": args.dataset_revision,
            "train": train_stats,
            "dev": dev_stats,
            "dropped_count_note": (
                "Counts are reported by CoreferencePreprocessor. "
                "CoreferenceDataModule filters invalid/unmatched source annotations "
                "before preprocessing and does not expose those filter counts."
            ),
            "test_split_used": False,
            "test_split_note": "Reserved for EXP-003; not preprocessed or evaluated by this run.",
        }
        atomic_write_json(output_dir / "dataset_stats.json", dataset_stats)
        if train_stats["valid_relations"] == 0 or dev_stats["valid_relations"] == 0:
            raise ValueError("Training and development splits must contain valid relations.")

        collator = datamodule.collator
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        train_eval_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        dev_loader = DataLoader(
            dev_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

        model = CoreferenceLightningModule(
            encoder_name=args.encoder_name,
            projection_dim=args.projection_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )
        device = resolve_device(args.accelerator)
        untrained_metrics = None
        if not args.skip_untrained_reference:
            untrained_metrics, _ = evaluate_model(
                model,
                dev_loader,
                dev_data,
                datamodule.tokenizer,
                args.max_length,
                device,
            )

        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            enable_version_counter=False,
        )
        early_stopping = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=args.patience,
            check_finite=True,
        )
        best_epoch_tracker = BestValidationEpoch()
        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=1,
            max_epochs=args.max_epochs,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm="norm",
            precision=args.precision,
            deterministic=True,
            logger=False,
            callbacks=[checkpoint, early_stopping, best_epoch_tracker],
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=False,
            log_every_n_steps=10,
            num_sanity_val_steps=0,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=dev_loader,
        )
        if not checkpoint.best_model_path:
            raise RuntimeError("No best validation-loss checkpoint was saved.")
        if best_epoch_tracker.best_epoch < 0:
            raise RuntimeError("No finite development loss was observed.")
        if not math.isclose(
            best_epoch_tracker.best_score,
            float(checkpoint.best_model_score),
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "Best-epoch tracker and checkpoint disagree on the selected "
                "development loss."
            )

        # Keep the last completed epoch's Lightning logs separately from the
        # full-split re-evaluation of the best validation-loss checkpoint.
        final_epoch_metrics = {
            **lightning_scalar_metrics(trainer, "train"),
            **lightning_scalar_metrics(trainer, "val"),
        }
        best_model = CoreferenceLightningModule.load_from_checkpoint(
            checkpoint.best_model_path,
            map_location="cpu",
        )
        best_model.to(device)
        train_metrics, _ = evaluate_model(
            best_model,
            train_eval_loader,
            train_data,
            datamodule.tokenizer,
            args.max_length,
            device,
        )
        dev_metrics, predictions = evaluate_model(
            best_model,
            dev_loader,
            dev_data,
            datamodule.tokenizer,
            args.max_length,
            device,
        )
        baseline_metrics = evaluate_baselines(dev_loader)
        recency = baseline_metrics["recency_relation_hit_at_1"]["value"]
        random_expected = baseline_metrics["random_expected_relation_hit_at_1"]["value"]
        stronger_baseline = max(recency, random_expected)
        model_hit = dev_metrics["relation_hit_at_1"]["value"]
        baseline_metrics.update(
            {
                "same_relation_denominator_as_model": (
                    baseline_metrics["denominator_relations"]
                    == dev_metrics["relation_hit_at_1"]["total"]
                ),
                "stronger_baseline_value": stronger_baseline,
                "stronger_baseline_name": (
                    "recency" if recency >= random_expected else "random_expected"
                ),
                "model_absolute_improvement_percentage_points": (
                    model_hit - stronger_baseline
                )
                * 100.0,
            }
        )
        distance_metrics = build_distance_metrics(predictions)

        # Re-evaluate the untrained reference already measured above if the
        # optional run was requested; it shares the identical development rows.
        metrics = {
            "status": "completed",
            "configuration": config,
            "epochs_completed": trainer.current_epoch + 1,
            "stopped_early": trainer.current_epoch + 1 < args.max_epochs,
            "early_stopping_best_epoch": best_epoch_tracker.best_epoch + 1,
            "checkpoint_best_epoch": best_epoch_tracker.best_epoch + 1,
            "best_checkpoint_path": str(
                Path(checkpoint.best_model_path).resolve()
            ),
            "best_development_loss": float(checkpoint.best_model_score),
            "training_full_split": train_metrics,
            "final_epoch_metrics": final_epoch_metrics,
            "development_best_checkpoint": dev_metrics,
            "untrained_development_reference": untrained_metrics,
            "train_dev_loss_gap": train_metrics["loss"] - dev_metrics["loss"],
            "train_dev_relation_hit_gap": (
                train_metrics["relation_hit_at_1"]["value"]
                - dev_metrics["relation_hit_at_1"]["value"]
            ),
            "runtime_seconds": time.monotonic() - started,
            "device_used": str(device),
            "warnings": captured_warnings,
            "incorrect_predictions_inspected": min(
                10,
                sum(not item["correct"] for item in predictions),
            ),
            "incorrect_predictions": [
                item for item in predictions if not item["correct"]
            ][:10],
        }
        atomic_write_json(output_dir / "baseline_metrics.json", baseline_metrics)
        atomic_write_json(output_dir / "distance_metrics.json", distance_metrics)
        atomic_write_json(output_dir / "predictions.json", predictions)
        result.update(metrics)
        atomic_write_json(output_dir / "metrics.json", result)
        return result
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "runtime_seconds": time.monotonic() - started,
                "warnings": captured_warnings,
            }
        )
        atomic_write_json(output_dir / "metrics.json", result)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        try:
            result = _run(args, captured)
        finally:
            captured.extend(str(item.message) for item in warning_records)
            metrics_path = args.output_dir.resolve() / "metrics.json"
            if metrics_path.exists():
                saved = json.loads(metrics_path.read_text(encoding="utf-8"))
                saved["warnings"] = captured
                atomic_write_json(metrics_path, saved)
    result["warnings"] = captured
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
