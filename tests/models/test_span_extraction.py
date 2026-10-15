from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.span_metrics import score_span_prediction
from src.models.span_extraction import (
    SpanExtractionModel,
    enumerate_context_spans,
    marginal_span_loss,
)
from src.training.span_module import SpanExtractionLightningModule


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 6) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(20, hidden_size)
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict=True):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def batch(relations=None, relation_mask=None):
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "context_mask": torch.tensor([[0, 1, 1, 1, 0, 0]]),
        "relations": torch.tensor(relations or [[[4, 5], [5, 6]]]),
        "relation_mask": torch.tensor(
            relation_mask if relation_mask is not None else [[True, True]]
        ),
        "antecedent_spans": torch.tensor([[[[1, 2], [2, 3]], [[2, 3], [-1, -1]]]]),
        "antecedent_mask": torch.tensor([[[True, True], [True, False]]]),
    }


def test_candidate_enumeration_uses_half_open_spans_and_respects_runs_and_limit():
    mask = torch.tensor([False, True, True, False, True, True, True, False])
    spans = enumerate_context_spans(mask, max_span_length=2)
    assert spans.tolist() == [
        [1, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 6], [5, 7], [6, 7],
    ]
    assert all(mask[start:end].all() for start, end in spans.tolist())
    assert all(end - start <= 2 for start, end in spans.tolist())


@pytest.mark.parametrize("approach", ["boundary", "span_pair"])
def test_learned_models_score_multiple_relations_with_one_encoder_pass(approach):
    encoder = FakeEncoder()
    model = SpanExtractionModel(
        head_dim=4,
        max_span_length=2,
        dropout=0,
        approach=approach,
        encoder=encoder,
    )
    output = model(**{
        key: value for key, value in batch().items()
        if key in {"input_ids", "attention_mask", "context_mask", "relations", "relation_mask"}
    })
    assert len(output["relations"]) == 2
    assert encoder.calls == 1
    for relation_output in output["relations"]:
        candidates = relation_output["candidate_spans"]
        assert candidates.numel() > 0
        assert (candidates[:, 0] < candidates[:, 1]).all()
        assert (candidates[:, 0] >= 1).all()
        assert (candidates[:, 1] <= 4).all()
        assert (candidates[:, 1] - candidates[:, 0] <= 2).all()


def test_multiple_alternative_and_overlapping_gold_spans_are_marginalized():
    scores = torch.tensor([1000.0, 999.0, -1000.0], requires_grad=True)
    outputs = [{
        "batch_index": 0,
        "relation_index": 0,
        "candidate_spans": torch.tensor([[1, 2], [1, 3], [2, 3]]),
        "scores": scores,
    }]
    gold = torch.tensor([[[[1, 2], [1, 3], [1, 3]]]])
    mask = torch.tensor([[[True, True, True]]])
    loss, count, excluded = marginal_span_loss(outputs, gold, mask, scores.sum() * 0)
    expected = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[:2], dim=0)
    assert torch.allclose(loss, expected)
    assert count == 1
    assert excluded == 0
    loss.backward()
    assert scores.grad is not None


def test_padded_relations_are_ignored_and_zero_relation_loss_is_differentiable():
    encoder = FakeEncoder()
    model = SpanExtractionModel(
        head_dim=4, max_span_length=2, dropout=0, encoder=encoder
    )
    empty_output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        context_mask=torch.tensor([[0, 0, 0]]),
        relations=torch.tensor([[[-1, -1]]]),
        relation_mask=torch.tensor([[False]]),
    )
    gold = torch.full((1, 1, 1, 2), -1)
    gold_mask = torch.zeros((1, 1, 1), dtype=torch.bool)
    loss, count, excluded = marginal_span_loss(
        empty_output["relations"], gold, gold_mask, empty_output["zero"]
    )
    assert loss.item() == 0
    assert loss.requires_grad
    assert (count, excluded) == (0, 0)
    loss.backward()
    assert encoder.embedding.weight.grad is not None


def test_unrepresentable_long_gold_is_reported_without_altering_annotation():
    scores = torch.tensor([0.0], requires_grad=True)
    output = [{
        "batch_index": 0,
        "relation_index": 0,
        "candidate_spans": torch.tensor([[2, 3]]),
        "scores": scores,
    }]
    gold = torch.tensor([[[[1, 4]]]])
    mask = torch.tensor([[[True]]])
    loss, count, excluded = marginal_span_loss(output, gold, mask, scores.sum() * 0)
    assert loss.item() == 0
    assert (count, excluded) == (0, 1)
    assert gold[0, 0, 0].tolist() == [1, 4]


def test_metrics_cover_exact_overlap_boundaries_and_character_spans():
    result = score_span_prediction(
        (3, 5),
        [(2, 5), (3, 5)],
        (10, 18),
        [(10, 18), (20, 24)],
    )
    assert result == {
        "exact_match": True,
        "token_overlap_f1": 1.0,
        "localization_hit": True,
        "start_boundary_correct": True,
        "end_boundary_correct": True,
        "character_exact_match": True,
    }

    incomplete = score_span_prediction((3, 4), [(2, 5)])
    assert incomplete["exact_match"] is False
    assert incomplete["localization_hit"] is True
    assert incomplete["token_overlap_f1"] == pytest.approx(0.5)
    assert incomplete["start_boundary_correct"] is False
    assert incomplete["end_boundary_correct"] is False


@pytest.mark.parametrize("approach", ["boundary", "span_pair"])
def test_gradients_reach_encoder_and_span_head(approach):
    encoder = FakeEncoder()
    model = SpanExtractionModel(
        head_dim=4, max_span_length=3, dropout=0, approach=approach, encoder=encoder
    )
    current = batch()
    output = model(
        input_ids=current["input_ids"],
        attention_mask=current["attention_mask"],
        context_mask=current["context_mask"],
        relations=current["relations"],
        relation_mask=current["relation_mask"],
    )
    loss, count, _ = marginal_span_loss(
        output["relations"],
        current["antecedent_spans"],
        current["antecedent_mask"],
        output["zero"],
    )
    assert count == 2
    loss.backward()
    assert encoder.embedding.weight.grad is not None
    assert encoder.embedding.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    )


def test_state_dict_reload_reproduces_predictions():
    torch.manual_seed(4)
    original = SpanExtractionModel(
        head_dim=4, max_span_length=3, dropout=0, approach="span_pair", encoder=FakeEncoder()
    ).eval()
    current = batch()
    inputs = {
        key: value for key, value in current.items()
        if key in {"input_ids", "attention_mask", "context_mask", "relations", "relation_mask"}
    }
    with torch.no_grad():
        expected = original(**inputs)["relations"][0]["scores"]
    restored = SpanExtractionModel(
        head_dim=4, max_span_length=3, dropout=0, approach="span_pair", encoder=FakeEncoder()
    ).eval()
    restored.load_state_dict(original.state_dict())
    with torch.no_grad():
        actual = restored(**inputs)["relations"][0]["scores"]
    assert torch.equal(expected, actual)


def test_cpu_lightning_smoke_training():
    encoder = FakeEncoder()
    model = SpanExtractionModel(
        head_dim=4, max_span_length=3, dropout=0, approach="boundary", encoder=encoder
    )
    module = SpanExtractionLightningModule(
        approach="boundary",
        head_dim=4,
        max_span_length=3,
        dropout=0,
        model=model,
    )
    raw = batch()
    raw.update({
        "id": ["example"],
        "context": ["a b c"],
        "chunk": ["d e"],
        "offset_mapping": torch.tensor([[[0, 0], [0, 1], [2, 3], [4, 5], [0, 1], [2, 3]]]),
        "antecedent_char_spans": torch.tensor([[[[0, 1], [2, 3]], [[2, 3], [-1, -1]]]]),
        "relation_char_spans": torch.tensor([[[0, 1], [2, 3]]]),
        "relation_ids": torch.tensor([[1, 2]]),
        "dropped_relations": torch.tensor([0]),
    })
    loader = DataLoader([raw], batch_size=None)
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
    assert "val/span_em" in trainer.callback_metrics
    assert encoder.calls >= 2
