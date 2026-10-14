"""Tests for token and relation antecedent localization metrics."""

import pytest
import torch

from src.training.metrics import CoreferenceMetrics, coreference_hit_counts


def make_batch(
    anaphor: tuple[int, int] = (3, 4),
    antecedents: tuple[tuple[int, int], ...] = ((0, 1),),
    relation_count: int = 1,
) -> dict[str, torch.Tensor]:
    logits = torch.zeros((1, 5, 5))
    pair_mask = torch.zeros((1, 5, 5), dtype=torch.bool)
    pair_mask[0, 3:, :3] = True
    relations = torch.full((1, relation_count, 2), -1, dtype=torch.long)
    antecedent_spans = torch.full(
        (1, relation_count, max(1, len(antecedents)), 2),
        -1,
        dtype=torch.long,
    )
    relation_mask = torch.zeros((1, relation_count), dtype=torch.bool)
    antecedent_mask = torch.zeros(
        (1, relation_count, max(1, len(antecedents))),
        dtype=torch.bool,
    )
    if relation_count:
        relations[0, 0] = torch.tensor(anaphor)
        relation_mask[0, 0] = True
        for index, span in enumerate(antecedents):
            antecedent_spans[0, 0, index] = torch.tensor(span)
            antecedent_mask[0, 0, index] = True
    return {
        "logits": logits,
        "pair_mask": pair_mask,
        "relations": relations,
        "antecedent_spans": antecedent_spans,
        "relation_mask": relation_mask,
        "antecedent_mask": antecedent_mask,
    }


def count_hits(batch: dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    return coreference_hit_counts(
        batch["logits"],
        batch["pair_mask"],
        batch["relations"],
        batch["antecedent_spans"],
        batch["relation_mask"],
        batch["antecedent_mask"],
    )


def test_token_hit_at_one_is_correct() -> None:
    batch = make_batch(anaphor=(3, 5))
    batch["logits"][0, 3, 0] = 4
    batch["logits"][0, 4, 0] = 4
    assert count_hits(batch) == (2, 2, 1, 1)


def test_relation_hit_uses_mean_anaphor_logits() -> None:
    batch = make_batch(anaphor=(3, 5))
    batch["logits"][0, 3, 0] = 10
    batch["logits"][0, 4, 0] = -10
    batch["logits"][0, 3, 1] = 4
    batch["logits"][0, 4, 1] = 4

    assert count_hits(batch) == (1, 2, 0, 1)


def test_all_gold_antecedents_are_acceptable() -> None:
    batch = make_batch(antecedents=((0, 1), (1, 2)))
    batch["logits"][0, 3, 1] = 2
    assert count_hits(batch) == (1, 1, 1, 1)


def test_invalid_context_positions_cannot_win() -> None:
    batch = make_batch()
    batch["pair_mask"][0, 3, 2] = False
    batch["logits"][0, 3, 2] = 100
    assert count_hits(batch) == (1, 1, 1, 1)


def test_padded_relations_are_ignored() -> None:
    batch = make_batch(relation_count=3)
    batch["relations"][0, 1] = torch.tensor([50, 51])
    assert count_hits(batch) == (1, 1, 1, 1)


def test_empty_supervision_computes_zero_without_division_error() -> None:
    batch = make_batch(relation_count=0)
    metric = CoreferenceMetrics()
    metric.update(**batch, loss=torch.tensor(0.0))

    assert metric.compute() == {
        "loss": torch.tensor(0.0),
        "token_hit_at_1": torch.tensor(0.0),
        "relation_hit_at_1": torch.tensor(0.0),
    }


def test_accumulation_uses_total_denominators_across_batches() -> None:
    metric = CoreferenceMetrics()
    one_relation = make_batch(relation_count=1)
    one_relation["logits"][0, 3, 0] = 1
    metric.update(**one_relation, loss=torch.tensor(2.0))

    three_token_relation = make_batch(anaphor=(3, 5), relation_count=1)
    three_token_relation["relations"][0, 0] = torch.tensor([3, 5])
    three_token_relation["logits"][0, 3, 0] = 1
    three_token_relation["logits"][0, 4, 1] = 5
    metric.update(**three_token_relation, loss=torch.tensor(4.0))

    result = metric.compute()
    assert result["loss"].item() == pytest.approx(3.0)
    assert result["token_hit_at_1"].item() == pytest.approx(2 / 3)
    assert result["relation_hit_at_1"].item() == pytest.approx(0.5)
