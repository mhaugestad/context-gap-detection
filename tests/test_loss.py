"""Offline tests for multi-antecedent token-level coreference loss."""

import pytest
import torch

from src.models.losses import coreference_loss


def make_supervision(
    *,
    batch_size: int = 1,
    relation_count: int = 1,
    antecedent_count: int = 2,
    sequence_length: int = 5,
) -> dict[str, torch.Tensor]:
    relations = torch.full((batch_size, relation_count, 2), -1, dtype=torch.long)
    antecedent_spans = torch.full(
        (batch_size, relation_count, antecedent_count, 2),
        -1,
        dtype=torch.long,
    )
    relation_mask = torch.zeros((batch_size, relation_count), dtype=torch.bool)
    antecedent_mask = torch.zeros(
        (batch_size, relation_count, antecedent_count),
        dtype=torch.bool,
    )
    if relation_count and antecedent_count:
        relations[0, 0] = torch.tensor([3, 4])
        antecedent_spans[0, 0, 0] = torch.tensor([0, 1])
        relation_mask[0, 0] = True
        antecedent_mask[0, 0, 0] = True

    pair_mask = torch.zeros(
        (batch_size, sequence_length, sequence_length),
        dtype=torch.bool,
    )
    pair_mask[:, 3:, :3] = True
    return {
        "pair_mask": pair_mask,
        "relations": relations,
        "antecedent_spans": antecedent_spans,
        "relation_mask": relation_mask,
        "antecedent_mask": antecedent_mask,
    }


def loss_for(logits: torch.Tensor, supervision: dict[str, torch.Tensor]) -> torch.Tensor:
    return coreference_loss(logits=logits, **supervision)


def test_increasing_gold_logit_reduces_loss() -> None:
    supervision = make_supervision()
    logits = torch.zeros((1, 5, 5))
    low = loss_for(logits, supervision)
    logits[0, 3, 0] = 5.0
    high = loss_for(logits, supervision)
    assert high < low


def test_alternative_antecedents_form_one_gold_probability_mass() -> None:
    supervision = make_supervision()
    supervision["antecedent_spans"][0, 0, 1] = torch.tensor([1, 2])
    supervision["antecedent_mask"][0, 0, 1] = True
    logits = torch.zeros((1, 5, 5))
    actual = loss_for(logits, supervision)
    assert actual.item() == pytest.approx(torch.log(torch.tensor(3 / 2)).item())


def test_overlapping_gold_spans_form_a_set_union() -> None:
    supervision = make_supervision()
    supervision["antecedent_spans"][0, 0, 0] = torch.tensor([0, 2])
    supervision["antecedent_spans"][0, 0, 1] = torch.tensor([1, 2])
    supervision["antecedent_mask"][0, 0, 1] = True
    logits = torch.zeros((1, 5, 5))

    assert loss_for(logits, supervision).item() == pytest.approx(
        torch.log(torch.tensor(3 / 2)).item()
    )


def test_multitoken_anaphor_averages_per_token_losses() -> None:
    supervision = make_supervision()
    supervision["relations"][0, 0] = torch.tensor([3, 5])
    logits = torch.zeros((1, 5, 5))
    logits[0, 3, 0] = 3.0
    logits[0, 4, 1] = 3.0

    expected = torch.stack([
        torch.logsumexp(logits[0, 3, :3], dim=0)
        - torch.logsumexp(logits[0, 3, :1], dim=0),
        torch.logsumexp(logits[0, 4, :3], dim=0)
        - torch.logsumexp(logits[0, 4, :1], dim=0),
    ]).mean()
    assert loss_for(logits, supervision) == pytest.approx(expected.item())


def test_padded_relations_and_antecedents_are_ignored() -> None:
    supervision = make_supervision(relation_count=3, antecedent_count=3)
    logits = torch.zeros((1, 5, 5))
    baseline = loss_for(logits, supervision)
    supervision["relations"][0, 1] = torch.tensor([100, 101])
    supervision["antecedent_spans"][0, 0, 1] = torch.tensor([100, 101])

    assert loss_for(logits, supervision) == baseline


def test_zero_relation_batch_has_differentiable_zero_loss() -> None:
    supervision = make_supervision(relation_count=0)
    logits = torch.randn((1, 5, 5), requires_grad=True)
    loss = loss_for(logits, supervision)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_unsupervised_example_can_share_a_batch_with_a_relation() -> None:
    supervision = make_supervision(batch_size=2)
    logits = torch.zeros((2, 5, 5), requires_grad=True)
    loss = loss_for(logits, supervision)
    loss.backward()

    assert loss.item() > 0
    assert logits.grad is not None
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))


def test_loss_is_finite_for_extreme_logits_and_backpropagates() -> None:
    supervision = make_supervision()
    logits = torch.full((1, 5, 5), -10000.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 3, 0] = 10000.0
        logits[0, 3, 1:3] = -10000.0
    loss = loss_for(logits, supervision)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_valid_relation_without_antecedent_fails_clearly() -> None:
    supervision = make_supervision()
    supervision["antecedent_mask"][0, 0, 0] = False
    with pytest.raises(ValueError, match="no valid antecedent spans"):
        loss_for(torch.zeros((1, 5, 5)), supervision)


def test_invalid_antecedent_bounds_fail_clearly() -> None:
    supervision = make_supervision()
    supervision["antecedent_spans"][0, 0, 0] = torch.tensor([4, 6])
    with pytest.raises(ValueError, match="invalid bounds"):
        loss_for(torch.zeros((1, 5, 5)), supervision)


def test_gold_antecedent_outside_valid_context_fails_clearly() -> None:
    supervision = make_supervision()
    supervision["pair_mask"][0, 3, 0] = False
    supervision["pair_mask"][0, 3, 1] = False
    supervision["pair_mask"][0, 3, 2] = True
    with pytest.raises(ValueError, match="no gold antecedent tokens available"):
        loss_for(torch.zeros((1, 5, 5)), supervision)


def test_padded_relation_slots_do_not_change_relation_normalization() -> None:
    base = make_supervision(relation_count=1, antecedent_count=1)
    padded = make_supervision(relation_count=8, antecedent_count=4)
    logits = torch.zeros((1, 5, 5))

    assert loss_for(logits, base) == loss_for(logits, padded)
