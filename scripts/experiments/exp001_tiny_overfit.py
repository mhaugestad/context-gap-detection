"""Run EXP-001: deterministic tiny-set overfitting for coreference affinity."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.datamodule import CoreferenceDataModule
from src.models.losses import coreference_loss
from src.training.metrics import coreference_hit_counts
from src.training.module import CoreferenceLightningModule


DEFAULTS = {
    "dataset_name": "mhaugestad/contextualise-context",
    "dataset_revision": None,
    "encoder_name": "distilbert-base-uncased",
    "projection_dim": 128,
    "seed": 42,
    "num_examples": 16,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "dropout": 0.0,
    "max_epochs": 100,
    "gradient_clip_val": 1.0,
    "precision": "32-true",
    "max_length": 512,
    "num_workers": 0,
    "accelerator": "auto",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Create a CLI while preserving the experiment's documented defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULTS["dataset_name"])
    parser.add_argument("--dataset-revision", default=DEFAULTS["dataset_revision"])
    parser.add_argument("--encoder-name", default=DEFAULTS["encoder_name"])
    parser.add_argument("--projection-dim", type=int, default=DEFAULTS["projection_dim"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--num-examples", type=int, default=DEFAULTS["num_examples"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=DEFAULTS["gradient_clip_val"],
    )
    parser.add_argument("--precision", default=DEFAULTS["precision"])
    parser.add_argument("--max-length", type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--accelerator", default=DEFAULTS["accelerator"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "exp001_tiny_overfit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the dataset, tokenizer, and model weights to be cached locally.",
    )
    args = parser.parse_args(argv)

    for name in ("projection_dim", "num_examples", "batch_size", "max_epochs", "max_length"):
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


def _atomic_write_json(path: Path, value: Any) -> None:
    """Avoid leaving truncated result files if the process is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _is_eligible(example: dict[str, Any]) -> bool:
    """Require every retained relation to map to real chunk/context tokens."""
    if not example["relations"]:
        return False

    sequence_length = len(example["input_ids"])
    context_mask = example["context_mask"]
    chunk_mask = example["chunk_mask"]
    for relation in example["relations"]:
        anaphor_start = relation["anaphor_start"]
        anaphor_end = relation["anaphor_end"]
        if not 0 <= anaphor_start < anaphor_end <= sequence_length:
            return False
        if not all(chunk_mask[anaphor_start:anaphor_end]):
            return False

        valid_antecedent = False
        for antecedent in relation["antecedents"]:
            start, end = antecedent["start"], antecedent["end"]
            if not 0 <= start < end <= sequence_length:
                continue
            if all(context_mask[start:end]):
                valid_antecedent = True
                break
        if not valid_antecedent:
            return False
    return True


def select_examples(
    dataset: Any,
    num_examples: int,
    seed: int,
) -> tuple[list[int], list[str]]:
    """Sample eligible preprocessed train rows without using test data."""
    eligible_indices = [
        index
        for index, example in enumerate(dataset.examples)
        if _is_eligible(example)
    ]
    if len(eligible_indices) < num_examples:
        raise ValueError(
            f"Requested {num_examples} eligible training examples, but only "
            f"{len(eligible_indices)} of {len(dataset)} are available after "
            "annotation filtering and token alignment."
        )

    selected_indices = random.Random(seed).sample(eligible_indices, num_examples)
    selected_ids = [str(dataset[index]["id"]) for index in selected_indices]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(
            "Selected example IDs are not unique; ID-only subset reconstruction "
            "would be ambiguous."
        )
    return selected_indices, selected_ids


class EpochHistoryCallback(Callback):
    """Capture the full-epoch metrics produced by the LightningModule."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, Any]] = []

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        metrics = trainer.callback_metrics
        row: dict[str, Any] = {"epoch": int(trainer.current_epoch) + 1}
        for name in (
            "train/loss",
            "train/token_hit_at_1",
            "train/relation_hit_at_1",
            "val/loss",
            "val/token_hit_at_1",
            "val/relation_hit_at_1",
        ):
            value = metrics.get(name)
            if value is not None:
                row[name] = float(value.detach().cpu().item())
        if len(row) > 1:
            self.history.append(row)


def _offsets_for_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_length: int,
) -> list[tuple[int, int]]:
    encoding = tokenizer(
        example["context"],
        example["chunk"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    if len(offsets) != len(example["input_ids"]):
        raise ValueError(
            f"Tokenizer offsets for example {example['id']} do not match "
            "the preprocessed sequence length."
        )
    return [tuple(offset) for offset in offsets]


def _surface_text(text: str, offsets: list[tuple[int, int]], start: int, end: int) -> str:
    valid_offsets = [offset for offset in offsets[start:end] if offset[1] > offset[0]]
    if not valid_offsets:
        return ""
    return text[valid_offsets[0][0] : valid_offsets[-1][1]]


@torch.no_grad()
def evaluate_subset(
    model: CoreferenceLightningModule,
    dataloader: DataLoader,
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure a fixed checkpoint on the complete subset without gradients."""
    was_training = model.training
    model.eval()
    model.to(device)

    weighted_loss = 0.0
    token_correct = token_count = relation_correct = relation_count = 0
    predictions: list[dict[str, Any]] = []

    for batch in dataloader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(batch)
        loss = coreference_loss(
            logits=output["logits"],
            pair_mask=output["pair_mask"],
            relations=batch["relations"],
            antecedent_spans=batch["antecedent_spans"],
            relation_mask=batch["relation_mask"],
            antecedent_mask=batch["antecedent_mask"],
        )
        batch_counts = coreference_hit_counts(
            output["logits"],
            output["pair_mask"],
            batch["relations"],
            batch["antecedent_spans"],
            batch["relation_mask"],
            batch["antecedent_mask"],
        )
        batch_token_correct, batch_token_count, batch_relation_correct, batch_relation_count = (
            batch_counts
        )
        token_correct += batch_token_correct
        token_count += batch_token_count
        relation_correct += batch_relation_correct
        relation_count += batch_relation_count
        weighted_loss += float(loss.detach().cpu()) * batch_relation_count

        for batch_index, example_id in enumerate(batch["id"]):
            example = next(item for item in examples if str(item["id"]) == str(example_id))
            offsets = _offsets_for_example(tokenizer, example, max_length)
            valid_relations = batch["relation_mask"][batch_index].nonzero(
                as_tuple=False
            ).flatten()
            for relation_index_tensor in valid_relations:
                relation_index = int(relation_index_tensor.item())
                start, end = (
                    int(value)
                    for value in batch["relations"][batch_index, relation_index].tolist()
                )
                query_positions = torch.arange(start, end, device=device)
                valid_by_query = output["pair_mask"][
                    batch_index, query_positions
                ]
                rows = output["logits"][batch_index, query_positions]
                counts = valid_by_query.sum(dim=0)
                valid_context = counts > 0
                mean_scores = rows.masked_fill(~valid_by_query, 0.0).sum(dim=0)
                mean_scores = mean_scores / counts.clamp_min(1)
                mean_scores = mean_scores.masked_fill(~valid_context, -torch.inf)
                predicted_position = int(mean_scores.argmax().item())

                gold_positions = set()
                gold_texts = []
                for antecedent_index_tensor in batch["antecedent_mask"][
                    batch_index, relation_index
                ].nonzero(as_tuple=False).flatten():
                    antecedent_index = int(antecedent_index_tensor.item())
                    antecedent_start, antecedent_end = (
                        int(value)
                        for value in batch["antecedent_spans"][
                            batch_index, relation_index, antecedent_index
                        ].tolist()
                    )
                    gold_positions.update(range(antecedent_start, antecedent_end))
                    gold_texts.append(
                        _surface_text(
                            example["context"],
                            offsets,
                            antecedent_start,
                            antecedent_end,
                        )
                    )

                gold_scores = mean_scores[
                    torch.tensor(sorted(gold_positions), device=device)
                ]
                best_gold_score = gold_scores.max()
                gold_rank = 1 + int((mean_scores[valid_context] > best_gold_score).sum())
                predicted_text = _surface_text(
                    example["context"],
                    offsets,
                    predicted_position,
                    predicted_position + 1,
                )
                predictions.append(
                    {
                        "example_id": str(example_id),
                        "anaphor_text": _surface_text(
                            example["chunk"],
                            offsets,
                            start,
                            end,
                        ),
                        "gold_antecedent_texts": gold_texts,
                        "predicted_context_token": predicted_text,
                        "predicted_context_token_index": predicted_position,
                        "highest_scoring_gold_rank": gold_rank,
                        "correct": predicted_position in gold_positions,
                    }
                )

    if was_training:
        model.train()
    result = {
        "loss": weighted_loss / relation_count if relation_count else 0.0,
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
    return result, predictions


def _run(args: argparse.Namespace, captured_warnings: list[str]) -> dict[str, Any]:
    output_dir: Path = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    previous_attempts: list[dict[str, Any]] = []
    prior_metrics_path = output_dir / "metrics.json"
    if prior_metrics_path.exists():
        prior_metrics = json.loads(prior_metrics_path.read_text(encoding="utf-8"))
        previous_attempts = list(prior_metrics.get("previous_attempts", []))
        if prior_metrics.get("status") == "failed":
            previous_attempts.append(
                {
                    "status": "failed",
                    "failure": prior_metrics.get("failure"),
                    "runtime_seconds": prior_metrics.get("runtime_seconds"),
                    "warnings": prior_metrics.get("warnings", []),
                }
            )

    config = {
        **DEFAULTS,
        **{
            key: getattr(args, key)
            for key in DEFAULTS
        },
        "output_dir": str(output_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_worktree_dirty": _git_worktree_dirty(),
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
    # JSON is a YAML 1.2 subset; keeping this file dependency-free avoids
    # requiring a YAML package solely to record scalar experiment settings.
    (output_dir / "config.yaml").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "running",
        "configuration": config,
    }
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
        datamodule.setup("fit")
        config["source_split_sizes"] = {
            split: len(dataset)
            for split, dataset in datamodule.splits.items()
        }
        (output_dir / "config.yaml").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        train_dataset = datamodule.splits["train"]
        selected_indices, selected_ids = select_examples(
            train_dataset,
            args.num_examples,
            args.seed,
        )
        selected_dataset = Subset(train_dataset, selected_indices)
        selected_examples = [train_dataset[index] for index in selected_indices]

        _atomic_write_json(output_dir / "selected_ids.json", selected_ids)
        collator = datamodule.collator
        train_generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            selected_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        evaluation_loader = DataLoader(
            selected_dataset,
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
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            auto_insert_metric_name=False,
            monitor="train/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
            enable_version_counter=False,
        )
        epoch_history = EpochHistoryCallback()
        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=1,
            max_epochs=args.max_epochs,
            num_sanity_val_steps=0,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm="norm",
            precision=args.precision,
            deterministic=True,
            logger=False,
            callbacks=[checkpoint, epoch_history],
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=evaluation_loader)

        if not checkpoint.best_model_path:
            raise RuntimeError("Lightning did not save a best training-loss checkpoint.")

        # Evaluate the selected checkpoint, not whichever weights happened to
        # remain in memory after the final epoch.
        best_model = CoreferenceLightningModule.load_from_checkpoint(
            checkpoint.best_model_path,
            map_location="cpu",
        )
        device = trainer.strategy.root_device
        best_model.to(device)
        train_subset_metrics, train_predictions = evaluate_subset(
            best_model,
            evaluation_loader,
            selected_examples,
            datamodule.tokenizer,
            args.max_length,
            device,
        )
        eval_metrics, eval_predictions = evaluate_subset(
            best_model,
            evaluation_loader,
            selected_examples,
            datamodule.tokenizer,
            args.max_length,
            device,
        )
        if train_predictions != eval_predictions:
            raise RuntimeError(
                "The deterministic train-subset and evaluation passes differed."
            )

        training_history = epoch_history.history
        best_epoch = None
        if training_history:
            best_epoch = min(
                training_history,
                key=lambda row: row.get("train/loss", float("inf")),
            )["epoch"]
        best_metrics = next(
            (row for row in training_history if row["epoch"] == best_epoch),
            None,
        )
        met_targets = all(
            (
                train_subset_metrics["loss"] <= 0.05,
                train_subset_metrics["token_hit_at_1"]["value"] >= 0.99,
                train_subset_metrics["relation_hit_at_1"]["value"] >= 0.99,
                eval_metrics["token_hit_at_1"]["value"] >= 0.99,
                eval_metrics["relation_hit_at_1"]["value"] >= 0.99,
            )
        )
        result.update(
            {
                "status": "completed",
                "selected_example_count": len(selected_ids),
                "selected_ids": selected_ids,
                "epochs_completed": len(training_history),
                "best_epoch_by_training_loss": best_epoch,
                "best_checkpoint_path": str(Path(checkpoint.best_model_path).resolve()),
                "best_checkpoint_training_loss": float(checkpoint.best_model_score),
                "best_epoch_metrics": best_metrics,
                "final_epoch_metrics": training_history[-1] if training_history else None,
                "training_subset": train_subset_metrics,
                "same_subset_evaluation": eval_metrics,
                "success_criteria_met": bool(met_targets),
                "epoch_history": training_history,
                "runtime_seconds": time.monotonic() - started,
                "device_used": str(device),
                "warnings": captured_warnings,
                "previous_attempts": previous_attempts,
            }
        )
        _atomic_write_json(output_dir / "predictions.json", train_predictions)
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "failure": f"{type(error).__name__}: {error}",
                "runtime_seconds": time.monotonic() - started,
                "warnings": captured_warnings,
                "previous_attempts": previous_attempts,
            }
        )
        _atomic_write_json(output_dir / "metrics.json", result)
        raise

    _atomic_write_json(output_dir / "metrics.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        try:
            result = _run(args, caught)
        finally:
            caught.extend(str(record.message) for record in warning_records)
            # Rewrite the final warning list even if the training run failed.
            metrics_path = args.output_dir.resolve() / "metrics.json"
            if metrics_path.exists():
                recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
                recorded["warnings"] = caught
                _atomic_write_json(metrics_path, recorded)

    result["warnings"] = caught
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
