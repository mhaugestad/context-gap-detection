"""PyTorch Lightning training wrapper for the affinity coreference model."""

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import nn

from src.models.affinity import CoreferenceModel
from src.models.losses import coreference_loss
from src.training.metrics import CoreferenceMetrics


class CoreferenceLightningModule(L.LightningModule):
    """Train and evaluate token affinity with multi-antecedent supervision."""

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        projection_dim: int = 128,
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
        self.model = model if model is not None else CoreferenceModel(
            encoder_name=encoder_name,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.train_metrics = CoreferenceMetrics()
        self.validation_metrics = CoreferenceMetrics()
        self.test_metrics = CoreferenceMetrics()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the encoder and affinity head on a collated batch."""
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            context_mask=batch["context_mask"],
            chunk_mask=batch["chunk_mask"],
        )

    def _step(
        self,
        batch: Mapping[str, Any],
        metrics: CoreferenceMetrics,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output = self.forward(batch)
        if "logits" not in output or "pair_mask" not in output:
            raise ValueError("CoreferenceModel output must include logits and pair_mask")

        loss = coreference_loss(
            logits=output["logits"],
            pair_mask=output["pair_mask"],
            relations=batch["relations"],
            antecedent_spans=batch["antecedent_spans"],
            relation_mask=batch["relation_mask"],
            antecedent_mask=batch["antecedent_mask"],
        )
        metrics.update(
            logits=output["logits"],
            pair_mask=output["pair_mask"],
            relations=batch["relations"],
            antecedent_spans=batch["antecedent_spans"],
            relation_mask=batch["relation_mask"],
            antecedent_mask=batch["antecedent_mask"],
            loss=loss,
        )
        return output, loss

    def training_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        _, loss = self._step(batch, self.train_metrics)
        return loss

    def validation_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        _, loss = self._step(batch, self.validation_metrics)
        return loss

    def test_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        _, loss = self._step(batch, self.test_metrics)
        return loss

    def _log_and_reset(self, stage: str, metrics: CoreferenceMetrics) -> None:
        # The custom accumulator weights each batch loss by its supervised
        # relation count, and sums metric numerators/denominators across batches.
        for name, value in metrics.compute().items():
            self.log(f"{stage}/{name}", value, sync_dist=False)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._log_and_reset("train", self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._log_and_reset("val", self.validation_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_and_reset("test", self.test_metrics)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use AdamW with the configured learning rate and weight decay."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )