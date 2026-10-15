"""PyTorch Lightning training for frozen-backbone NULL scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch

from src.evaluation.null_metrics import evaluate_null_predictions
from src.evaluation.span_metrics import score_span_prediction
from src.models.null_antecedent import (
    NullAntecedentModel,
    marginal_link_null_loss,
)


class NullAntecedentLightningModule(L.LightningModule):
    def __init__(
        self,
        model: NullAntecedentModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self._validation_records: list[dict[str, Any]] = []
        self.last_validation_metrics: dict[str, Any] = {}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.model(batch)

    def _loss(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> tuple[torch.Tensor, int, int]:
        return marginal_link_null_loss(
            relation_outputs=output["relations"],
            antecedent_spans=batch["antecedent_spans"],
            antecedent_mask=batch["antecedent_mask"],
            is_null=batch["is_null"],
            zero=output["zero"],
        )

    @staticmethod
    def records_from_batch(
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        records = []
        for relation_output in output["relations"]:
            batch_index = int(relation_output["batch_index"])
            relation_index = int(relation_output["relation_index"])
            candidates = relation_output["candidate_spans"]
            span_scores = relation_output["scores"]
            span_index = int(span_scores.argmax().item())
            predicted_start, predicted_end = (
                int(value) for value in candidates[span_index].tolist()
            )
            gold_indices = batch["antecedent_mask"][
                batch_index, relation_index
            ].nonzero(as_tuple=False).flatten()
            gold_spans = [
                tuple(
                    int(value)
                    for value in batch["antecedent_spans"][
                        batch_index, relation_index, index
                    ].tolist()
                )
                for index in gold_indices.tolist()
            ]
            is_null = bool(batch["is_null"][batch_index, relation_index].item())
            predicted_is_null = bool(relation_output["predicted_is_null"])
            null_probability = float(
                relation_output["probabilities"][-1].detach().cpu().item()
            )
            metadata = batch["relation_metadata"][batch_index][relation_index] or {}
            metric = score_span_prediction(
                (predicted_start, predicted_end),
                gold_spans,
            ) if gold_spans else {
                "exact_match": False,
                "token_overlap_f1": 0.0,
                "localization_hit": False,
            }
            span_exact = bool(metric["exact_match"])
            records.append({
                "batch_index": batch_index,
                "relation_index": relation_index,
                "example_id": batch["id"][batch_index],
                "source_example_id": metadata.get("source_example_id"),
                "relation_id": int(
                    batch["relation_ids"][batch_index, relation_index].item()
                ),
                "gold_is_null": is_null,
                "predicted_is_null": predicted_is_null,
                "predicted_span": [predicted_start, predicted_end],
                "gold_spans": [list(span) for span in gold_spans],
                "forced_link_exact": span_exact,
                "forced_link_token_f1": float(metric["token_overlap_f1"]),
                "end_to_end_correct": (
                    predicted_is_null if is_null else (
                        not predicted_is_null and span_exact
                    )
                ),
                "null_score": float(
                    relation_output["null_score"].detach().cpu().item()
                ),
                "highest_span_score": float(span_scores[span_index].detach().cpu().item()),
                "null_probability": null_probability,
                "candidate_count": int(candidates.shape[0]),
                "context_token_count": int(
                    batch["context_mask"][batch_index].sum().item()
                ),
                "negative_strategy": metadata.get("strategy"),
                "pair_id": metadata.get("pair_id"),
                "replacement_source_id": metadata.get("replacement_source_id"),
                "context_text": batch["context"][batch_index],
                "chunk_text": batch["chunk"][batch_index],
            })
        return records

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self(batch)
        loss, supervised_count, excluded = self._loss(batch, output)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=max(supervised_count, 1),
        )
        self.log("train/excluded_relations", float(excluded), on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_records = []

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self(batch)
        loss, _, _ = self._loss(batch, output)
        self._validation_records.extend(self.records_from_batch(output, batch))
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = evaluate_null_predictions(self._validation_records)
        self.last_validation_metrics = metrics
        self.log(
            "val/macro_f1",
            metrics["macro_f1"],
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            "val/e2e_exact",
            metrics["end_to_end_exact_accuracy"],
            prog_bar=False,
            sync_dist=False,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("The selected NULL approach has no trainable parameters")
        return torch.optim.AdamW(
            parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
