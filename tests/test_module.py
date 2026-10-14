"""Offline tests for the Lightning training wrapper and CPU smoke run."""

from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.models.affinity import CoreferenceModel
from src.training.module import CoreferenceLightningModule


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(12, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> CoreferenceLightningModule:
    encoder = FakeEncoder()
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        staticmethod(lambda _name: encoder),
    )
    return CoreferenceLightningModule(
        encoder_name="offline-test-encoder",
        projection_dim=3,
        dropout=0.0,
    )


def make_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "context_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        "chunk_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "relations": torch.tensor([[[3, 4]]]),
        "antecedent_spans": torch.tensor([[[[0, 1], [1, 2]]]]),
        "relation_mask": torch.tensor([[True]]),
        "antecedent_mask": torch.tensor([[[True, True]]]),
    }


def test_training_validation_and_test_steps_execute(
    module: CoreferenceLightningModule,
) -> None:
    batch = make_batch()
    train_loss = module.training_step(batch, 0)
    val_loss = module.validation_step(batch, 0)
    test_loss = module.test_step(batch, 0)

    assert train_loss.ndim == val_loss.ndim == test_loss.ndim == 0
    train_loss.backward()
    assert module.model.query_projection.weight.grad is not None
    assert module.model.query_projection.weight.grad.abs().sum() > 0


def test_configured_optimizer_is_adamw(
    module: CoreferenceLightningModule,
) -> None:
    module.hparams.learning_rate = 1e-3
    module.hparams.weight_decay = 0.2
    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(1e-3)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.2)


def test_cpu_trainer_smoke_run_completes_train_and_validation(
    module: CoreferenceLightningModule,
) -> None:
    batch = make_batch()
    loader = DataLoader([batch], batch_size=None)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)

    assert "train/loss" in trainer.callback_metrics
    assert "val/loss" in trainer.callback_metrics
    assert "val/token_hit_at_1" in trainer.callback_metrics
    assert "val/relation_hit_at_1" in trainer.callback_metrics
