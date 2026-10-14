"""Offline tests for the encoder-based token affinity model."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModel

from src.models.affinity import CoreferenceModel


class FakeEncoder(nn.Module):
    """Small trainable stand-in that avoids loading pretrained weights."""

    def __init__(self, hidden_size: int, vocab_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.forward_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> SimpleNamespace:
        self.forward_calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def make_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hidden_size: int = 4,
    projection_dim: int = 3,
    dropout: float = 0.0,
    fine_tune_encoder: bool = True,
) -> CoreferenceModel:
    """Inject a random, trainable encoder without any network access."""
    encoder = FakeEncoder(hidden_size)
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        staticmethod(lambda _encoder_name: encoder),
    )
    return CoreferenceModel(
        encoder_name="offline-test-encoder",
        projection_dim=projection_dim,
        dropout=dropout,
        fine_tune_encoder=fine_tune_encoder,
    )


@pytest.mark.parametrize(
    ("batch_size", "sequence_length"),
    [(1, 4), (3, 7)],
)
def test_output_shapes_vary_with_batch_and_sequence_lengths(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    sequence_length: int,
) -> None:
    model = make_model(monkeypatch)
    input_ids = torch.randint(0, 16, (batch_size, sequence_length))
    attention_mask = torch.ones_like(input_ids)
    context_mask = torch.zeros_like(input_ids)
    chunk_mask = torch.zeros_like(input_ids)
    context_mask[:, 1] = 1
    chunk_mask[:, -1] = 1

    output = model(input_ids, attention_mask, context_mask, chunk_mask)

    assert output["logits"].shape == (
        batch_size,
        sequence_length,
        sequence_length,
    )
    assert output["pair_mask"].shape == output["logits"].shape
    assert output["pair_mask"].dtype == torch.bool


def test_logits_use_chunk_queries_and_context_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch, hidden_size=2, projection_dim=2)
    encoder = model.encoder

    # Choose token vectors whose directional dot products can be checked
    # explicitly; identity projections make the expected orientation clear.
    with torch.no_grad():
        encoder.embedding.weight.zero_()
        encoder.embedding.weight[1] = torch.tensor([1.0, 0.0])
        encoder.embedding.weight[2] = torch.tensor([0.0, 1.0])
        encoder.embedding.weight[3] = torch.tensor([1.0, 1.0])
        encoder.embedding.weight[4] = torch.tensor([-1.0, 1.0])
        model.query_projection.weight.copy_(
            torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        )
        model.query_projection.bias.zero_()
        model.key_projection.weight.copy_(
            torch.tensor([[2.0, 0.0], [1.0, 1.0]])
        )
        model.key_projection.bias.zero_()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    context_mask = torch.tensor([[1, 1, 0, 0]])
    chunk_mask = torch.tensor([[0, 0, 1, 1]])

    output = model(input_ids, attention_mask, context_mask, chunk_mask)
    logits = output["logits"]

    # Query position 2 ([1, 1]) maps to [3, 1], while context position 0
    # maps to [2, 1]. The asymmetric projections make a transposed orientation
    # produce a different value, so this assertion checks more than indexing.
    assert logits[0, 2, 0].item() == pytest.approx(7 / 2**0.5)
    assert logits[0, 2, 1].item() == pytest.approx(1 / 2**0.5)
    assert logits[0, 3, 0].item() == pytest.approx(3 / 2**0.5)
    assert logits[0, 3, 1].item() == pytest.approx(1 / 2**0.5)
    assert output["pair_mask"][0, 2, 0]
    assert not output["pair_mask"][0, 0, 2]


def test_pair_mask_excludes_special_tokens_and_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    # Positions 0 and 3 stand in for tokenizer-added special tokens. Segment
    # masks are false there, just as in the data preprocessor.
    # Deliberately mark the padded positions as segment members: attention
    # masking must still veto them even if masks supplied by a caller disagree.
    context_mask = torch.tensor([[0, 1, 0, 0, 1, 0]])
    chunk_mask = torch.tensor([[0, 0, 1, 0, 0, 1]])

    output = model(input_ids, attention_mask, context_mask, chunk_mask)
    expected = torch.zeros((1, 6, 6), dtype=torch.bool)
    expected[0, 2, 1] = True

    assert torch.equal(output["pair_mask"], expected)
    assert torch.isfinite(output["logits"]).all()
    assert not torch.isneginf(output["logits"]).any()


def test_logits_and_masks_keep_invalid_pairs_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3]])

    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        context_mask=torch.tensor([[1, 0, 0]]),
        chunk_mask=torch.tensor([[0, 1, 0]]),
    )

    assert output["pair_mask"].tolist() == [
        [[False, False, False], [True, False, False], [False, False, False]]
    ]
    assert torch.isfinite(output["logits"]).all()


def test_gradients_reach_affinity_head_and_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        context_mask=torch.tensor([[1, 1, 0, 0]]),
        chunk_mask=torch.tensor([[0, 0, 1, 1]]),
    )

    output["logits"].sum().backward()

    assert model.encoder.embedding.weight.grad is not None
    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert model.query_projection.weight.grad is not None
    assert model.query_projection.weight.grad.abs().sum() > 0
    assert model.key_projection.weight.grad is not None
    assert model.key_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("hidden_size", "projection_dim"),
    [(4, 3), (7, 5)],
)
def test_model_supports_different_encoder_hidden_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    hidden_size: int,
    projection_dim: int,
) -> None:
    model = make_model(
        monkeypatch,
        hidden_size=hidden_size,
        projection_dim=projection_dim,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    output = model(
        input_ids,
        torch.ones_like(input_ids),
        torch.tensor([[1, 0, 0]]),
        torch.tensor([[0, 1, 0]]),
    )

    assert output["logits"].shape == (1, 3, 3)


def test_no_valid_pairs_returns_finite_logits_and_empty_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3]])
    output = model(
        input_ids,
        torch.ones_like(input_ids),
        torch.zeros_like(input_ids),
        torch.zeros_like(input_ids),
    )

    assert output["pair_mask"].shape == (1, 3, 3)
    assert not output["pair_mask"].any()
    assert torch.isfinite(output["logits"]).all()


def test_encoder_runs_once_per_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch)
    input_ids = torch.tensor([[1, 2, 3]])

    model(
        input_ids,
        torch.ones_like(input_ids),
        torch.tensor([[1, 0, 0]]),
        torch.tensor([[0, 1, 0]]),
    )

    assert model.encoder.forward_calls == 1


def test_encoder_is_fine_tunable_by_default_and_can_be_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fine_tuned_model = make_model(monkeypatch)
    assert all(parameter.requires_grad for parameter in fine_tuned_model.encoder.parameters())

    frozen_model = make_model(monkeypatch, fine_tune_encoder=False)
    assert not any(parameter.requires_grad for parameter in frozen_model.encoder.parameters())
