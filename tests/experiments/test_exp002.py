"""Offline tests for EXP-002 baselines, distances, and split statistics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.collator import CoreferenceCollator


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "exp002_full_training.py"
)
SPEC = importlib.util.spec_from_file_location("exp002_full_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exp002 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp002)


def example(relations, *, identifier="sample"):
    return {
        "id": identifier,
        "context": "alpha beta gamma",
        "chunk": "delta epsilon",
        "input_ids": [101, 11, 12, 13, 102, 21, 22, 102],
        "attention_mask": [1] * 8,
        "context_mask": [0, 1, 1, 1, 0, 0, 0, 0],
        "chunk_mask": [0, 0, 0, 0, 0, 1, 1, 0],
        "relations": relations,
        "dropped_relations": 0,
        "dropped_annotations": 0,
    }


def collated_loader(examples):
    return DataLoader(
        examples,
        batch_size=2,
        collate_fn=CoreferenceCollator(pad_token_id=0),
    )


def relation(start, end, antecedents):
    return {
        "relation_id": start,
        "anaphor_start": start,
        "anaphor_end": end,
        "antecedents": [
            {"start": antecedent_start, "end": antecedent_end}
            for antecedent_start, antecedent_end in antecedents
        ],
    }


def test_random_baseline_reports_exact_expected_accuracy() -> None:
    batch = example([relation(5, 6, [(1, 2)])])
    result = exp002.evaluate_baselines(collated_loader([batch]))

    # A single gold token among three context candidates has expected
    # accuracy 1/3; this is an expectation, not a sampled prediction.
    assert result["random_expected_relation_hit_at_1"] == {
        "expected_correct": pytest.approx(1 / 3),
        "total": 1,
        "value": pytest.approx(1 / 3),
        "interpretation": (
            "Exact expected accuracy under uniform sampling of a valid context token."
        ),
    }


def test_recency_baseline_checks_last_context_position() -> None:
    batch = example([relation(5, 6, [(3, 4)])])
    result = exp002.evaluate_baselines(collated_loader([batch]))
    assert result["recency_relation_hit_at_1"] == {
        "correct": 1,
        "total": 1,
        "value": 1.0,
    }


def test_multiple_gold_mentions_are_alternatives_for_baselines() -> None:
    batch = example([relation(5, 6, [(1, 2), (3, 4)])])
    result = exp002.evaluate_baselines(collated_loader([batch]))

    assert result["random_expected_relation_hit_at_1"]["expected_correct"] == pytest.approx(
        2 / 3
    )
    assert result["recency_relation_hit_at_1"]["correct"] == 1
    assert result["denominator_relations"] == 1


def test_padding_and_non_context_positions_are_excluded() -> None:
    first = {
        **example([relation(4, 5, [(1, 2)])], identifier="short"),
        "input_ids": [101, 11, 12, 102, 21, 102],
        "attention_mask": [1] * 6,
        "context_mask": [0, 1, 1, 0, 0, 0],
        "chunk_mask": [0, 0, 0, 0, 1, 0],
    }
    second = example([], identifier="empty")
    result = exp002.evaluate_baselines(collated_loader([first, second]))

    assert result["denominator_relations"] == 1
    assert result["random_expected_relation_hit_at_1"]["total"] == 1
    assert result["random_expected_relation_hit_at_1"]["value"] == pytest.approx(
        1 / 2
    )
    # Recency stops at context token 2, rather than including the separator or
    # padding to the second example's length.
    assert result["recency_relation_hit_at_1"]["correct"] == 0


def test_zero_relation_examples_do_not_enter_metric_denominator() -> None:
    supervised = example([relation(5, 6, [(1, 2)])], identifier="positive")
    empty = example([], identifier="negative")
    result = exp002.evaluate_baselines(collated_loader([supervised, empty]))

    assert result["denominator_relations"] == 1
    assert result["recency_relation_hit_at_1"]["total"] == 1


@pytest.mark.parametrize(
    ("distance", "expected_band"),
    [
        (0, "0-16"),
        (16, "0-16"),
        (17, "17-32"),
        (32, "17-32"),
        (33, "33-64"),
        (64, "33-64"),
        (65, "65-128"),
        (128, "65-128"),
        (129, ">128"),
    ],
)
def test_distance_band_boundaries(distance: int, expected_band: str) -> None:
    assert exp002.distance_band(distance) == expected_band


def test_distance_is_nearest_token_across_all_gold_mentions() -> None:
    # The second mention is nearer to the anaphor and must determine the
    # relation's primary exploratory distance.
    assert exp002.distance_in_tokens((20, 22), [(2, 4), (15, 17)]) == 4


def test_distance_metrics_report_counts_and_ignore_empty_bands() -> None:
    result = exp002.build_distance_metrics(
        [
            {"antecedent_distance": 2, "correct": True},
            {"antecedent_distance": 16, "correct": False},
            {"antecedent_distance": 40, "correct": True},
        ]
    )

    assert result["0-16"] == {
        "relations": 2,
        "correct": 1,
        "relation_hit_at_1": 0.5,
    }
    assert result["17-32"]["relations"] == 0
    assert result["17-32"]["relation_hit_at_1"] is None
    assert result["33-64"]["relations"] == 1


def test_invalid_negative_distance_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        exp002.distance_band(-1)


def test_preprocessed_dataset_statistics_separate_retained_and_dropped() -> None:
    class FakeDataset:
        examples = [
            example(
                [
                    relation(5, 6, [(1, 2), (2, 4)]),
                    relation(6, 7, [(1, 2)]),
                ]
            ),
            {
                **example([]),
                "dropped_relations": 1,
                "dropped_annotations": 2,
            },
        ]

        def __len__(self):
            return len(self.examples)

    stats = exp002.dataset_statistics(FakeDataset())
    assert stats == {
        "examples": 2,
        "examples_with_valid_relations": 1,
        "examples_without_valid_relations": 1,
        "valid_relations": 2,
        "anaphor_tokens": 2,
        "valid_antecedent_spans": 3,
        "preprocessor_reported_dropped_relations": 1,
        "preprocessor_reported_dropped_annotations": 2,
    }


def test_best_validation_epoch_tracks_the_minimum_loss() -> None:
    tracker = exp002.BestValidationEpoch()
    module = SimpleNamespace()
    for epoch, loss in enumerate((0.8, 0.4, 0.5)):
        trainer = SimpleNamespace(
            current_epoch=epoch,
            callback_metrics={"val/loss": torch.tensor(loss)},
        )
        tracker.on_validation_end(trainer, module)

    assert tracker.best_epoch == 1
    assert tracker.best_score == pytest.approx(0.4)


def test_best_validation_epoch_fails_when_required_metric_is_missing() -> None:
    tracker = exp002.BestValidationEpoch()
    trainer = SimpleNamespace(current_epoch=0, callback_metrics={})

    with pytest.raises(RuntimeError, match="did not log required"):
        tracker.on_validation_end(trainer, SimpleNamespace())


def test_yaml_config_keeps_nested_device_metadata_as_a_mapping() -> None:
    serialized = exp002._yaml_subset(
        {"seed": 42, "device": {"accelerator": "mps", "cuda_available": False}}
    )

    assert "device:\n  accelerator: \"mps\"\n  cuda_available: false\n" in serialized
