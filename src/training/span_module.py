"""Lightning training module for explicitly supervised antecedent spans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import nn

from src.evaluation.span_metrics import aggregate_span_metrics, score_span_prediction
from src.models.span_extraction import SpanExtractionModel, marginal_span_loss


class SpanExtractionLightningModule(L.LightningModule):
    """Train boundary or candidate-span scoring over gold anaphors."""

    def __init__(
        self,
        approach: str = "boundary",
        encoder_name: str = "distilbert-base-uncased",
        head_dim: int = 128,
        max_span_length: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.save_hyperparameters(ignore=["model"])
        self.model = model if model is not None else SpanExtractionModel(
            encoder_name=encoder_name,
            head_dim=head_dim,
            max_span_length=max_span_length,
            dropout=dropout,
            approach=approach,
        )
        self._validation_records: list[dict[str, Any]] = []
        self._validation_loss_sum = 0.0
        self._validation_loss_count = 0
        self._validation_excluded = 0
        self.last_validation_metrics: dict[str, Any] = {}

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            context_mask=batch["context_mask"],
            relations=batch["relations"],
            relation_mask=batch["relation_mask"],
        )

    def _loss(
        self,
        batch: Mapping[str, Any],
        output: Mapping[str, Any],
    ) -> tuple[torch.Tensor, int, int]:
        return marginal_span_loss(
            relation_outputs=output["relations"],
            antecedent_spans=batch["antecedent_spans"],
            antecedent_mask=batch["antecedent_mask"],
            zero=output["zero"],
        )

    @staticmethod
    def _record_predictions(
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        records = []
        for relation_output in output["relations"]:
            batch_index = int(relation_output["batch_index"])
            relation_index = int(relation_output["relation_index"])
            candidate_spans = relation_output["candidate_spans"]
            scores = relation_output["scores"]
            predicted_start, predicted_end = candidate_spans[
                int(scores.argmax().item())
            ].tolist()
            gold_indices = batch["antecedent_mask"][
                batch_index, relation_index
            ].nonzero(as_tuple=False).flatten()
            gold_spans = [
                tuple(int(value) for value in batch["antecedent_spans"][
                    batch_index, relation_index, index
                ].tolist())
                for index in gold_indices.tolist()
            ]
            metric = score_span_prediction(
                (int(predicted_start), int(predicted_end)),
                gold_spans,
            )
            records.append({
                "batch_index": batch_index,
                "relation_index": relation_index,
                "predicted_span": [int(predicted_start), int(predicted_end)],
                "gold_spans": [list(span) for span in gold_spans],
                **metric,
            })
        return records

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self.forward(batch)
        loss, supervised_count, _ = self._loss(batch, output)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=max(supervised_count, 1),
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_records = []
        self._validation_loss_sum = 0.0
        self._validation_loss_count = 0
        self._validation_excluded = 0

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        output = self.forward(batch)
        loss, supervised_count, excluded = self._loss(batch, output)
        self._validation_records.extend(self._record_predictions(output, batch))
        self._validation_loss_sum += float(loss.detach().item()) * supervised_count
        self._validation_loss_count += supervised_count
        self._validation_excluded += excluded
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = aggregate_span_metrics(self._validation_records)
        loss = (
            self._validation_loss_sum / self._validation_loss_count
            if self._validation_loss_count
            else 0.0
        )
        self.last_validation_metrics = {
            **metrics,
            "loss": loss,
            "loss_supervised_relations": self._validation_loss_count,
            "loss_excluded_unrepresentable_relations": self._validation_excluded,
        }
        span_em = metrics["exact_span_match"]["value"]
        self.log(
            "val/span_em",
            0.0 if span_em is None else span_em,
            prog_bar=True,
            sync_dist=False,
        )
        self.log("val/loss", loss, prog_bar=False, sync_dist=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
