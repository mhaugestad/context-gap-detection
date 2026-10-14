"""Offline tests for the EXP-003 behavioral evaluation helpers."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "experiments"
    / "exp003_behavioral_eval.py"
)
SPEC = importlib.util.spec_from_file_location("exp003_behavioral_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exp003 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp003)


def _records():
    return [
        {
            "example_id": "A01",
            "category": "A",
            "subtest": "A01",
            "status": "EVALUABLE",
            "status_flags": [],
            "gold_relation_count": 2,
            "correct_relations": 1,
            "all_relations_correct": False,
            "token_hit_at_1": {"correct": 2, "total": 3},
        },
        {
            "example_id": "C01",
            "category": "C",
            "subtest": "C01",
            "status": "EVALUABLE",
            "status_flags": [],
            "gold_relation_count": 1,
            "correct_relations": 1,
            "all_relations_correct": True,
            "token_hit_at_1": {"correct": 1, "total": 1},
        },
        {
            "example_id": "B01",
            "category": "B",
            "subtest": "B01",
            "status": "EVALUABLE",
            "status_flags": [],
            "gold_relation_count": 1,
            "correct_relations": 0,
            "all_relations_correct": False,
            "token_hit_at_1": {"correct": 0, "total": 1},
        },
        {
            "example_id": "F01",
            "category": "F",
            "subtest": "F01",
            "status": "EVALUABLE",
            "status_flags": [],
            "gold_relation_count": 1,
            "correct_relations": 1,
            "all_relations_correct": True,
            "token_hit_at_1": {"correct": 1, "total": 1},
        },
        {
            "example_id": "H01",
            "category": "H",
            "subtest": "H01",
            "status": "EVALUABLE",
            "status_flags": [],
            "gold_relation_count": 1,
            "correct_relations": 0,
            "all_relations_correct": False,
            "token_hit_at_1": {"correct": 0, "total": 1},
        },
        {
            "example_id": "G01",
            "category": "G",
            "subtest": "G01",
            "status": "UNSUPPORTED",
            "status_flags": ["NO_GOLD_RELATION", "UNSUPPORTED"],
            "gold_relation_count": 0,
            "correct_relations": 0,
            "all_relations_correct": None,
            "token_hit_at_1": {"correct": 0, "total": 0},
        },
    ]


def _pred(example_id, relation_id, correct, random_expected, recency):
    return {
        "example_id": example_id,
        "relation_id": relation_id,
        "correct": correct,
        "random_expected_accuracy": random_expected,
        "recency_baseline_correct": recency,
    }


def test_taxonomy_mapping_covers_all_36_verified_subtest_ids():
    examples = [{"id": identifier} for identifier in reversed(exp003.SUBTESTS)]

    mapping = exp003.validate_taxonomy_mapping(examples)

    assert len(mapping) == 36
    assert {row["example_id"] for row in mapping} == set(exp003.SUBTESTS)
    assert all(row["category"] == row["subtest"][0] for row in mapping)
    assert any(row["example_id"] == "A08" for row in mapping)


def test_taxonomy_mapping_rejects_unmapped_or_duplicate_examples():
    with pytest.raises(ValueError, match="unknown=.*missing"):
        exp003.validate_taxonomy_mapping([{"id": "unknown"}])
    with pytest.raises(ValueError, match="duplicate"):
        exp003.validate_taxonomy_mapping([{"id": "A01"}, {"id": "A01"}])


def test_stable_ranking_excludes_non_context_and_breaks_ties_by_token_index():
    assert exp003.stable_ranked_context_tokens(
        [4.0, 5.0, 5.0, 100.0, 0.0],
        [True, True, True, False, False],
    ) == [1, 2, 0]


def test_recency_baseline_uses_last_valid_context_position_not_model_rank():
    assert exp003.final_valid_context_token([False, True, True, False, True]) == 4
    with pytest.raises(ValueError, match="No valid context tokens"):
        exp003.final_valid_context_token([False, False])


def test_relation_metrics_use_union_of_alternative_gold_antecedents():
    scores = torch.tensor([[0.1, 0.9, 0.0, 0.0], [0.2, 0.0, 0.8, 0.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, False]])
    gold = torch.tensor([False, True, True, False])

    token_correct, token_total, relation_correct, relation_total, gold_n, candidate_n = (
        exp003.relation_metric_counts(scores, valid, gold)
    )

    assert (token_correct, token_total) == (2, 2)
    assert (relation_correct, relation_total) == (1, 1)
    assert (gold_n, candidate_n) == (2, 3)


def test_category_and_subtest_aggregation_use_relation_denominators():
    records = _records()
    predictions = [
        _pred("A01", 1, True, 1 / 3, False),
        _pred("A01", 2, False, 2 / 3, True),
        _pred("C01", 1, True, 1 / 4, False),
        _pred("B01", 1, False, 1 / 2, True),
        _pred("F01", 1, True, 1 / 3, False),
        _pred("H01", 1, False, 1 / 3, True),
    ]

    categories, subtests = exp003._aggregate_records(records, predictions)
    h1 = exp003.h1_group_comparison(categories)

    assert categories["A"]["valid_relations"] == 2
    assert categories["A"]["correct_relations"] == 1
    assert categories["A"]["relation_hit_at_1"] == 0.5
    assert categories["A"]["token_hit_at_1"] == pytest.approx(2 / 3)
    assert categories["G"]["valid_relations"] == 0
    assert categories["G"]["relation_hit_at_1"] is None
    assert categories["G"]["unsupported_or_non_evaluable_examples"] == 1
    assert subtests["A01"]["correct_relations"] == 1
    assert subtests["A01"]["examples_detail"][0]["all_relations_correct"] is False
    assert subtests["A01"]["examples_detail"][0]["localization_pass"] is False
    assert h1["straightforward"]["relations_correct"] == 2
    assert h1["straightforward"]["relations_total"] == 3
    assert h1["challenging"]["relations_correct"] == 1
    assert h1["challenging"]["relations_total"] == 3
    assert h1["status"] == "supported"
    assert h1["difference_percentage_points"] == pytest.approx(100 / 3)


def test_all_relations_pass_requires_every_relation_and_is_undefined_when_empty():
    assert exp003.all_relations_correct([]) is None
    assert exp003.all_relations_correct([{"correct": True}]) is True
    assert exp003.all_relations_correct(
        [{"correct": True}, {"correct": False}]
    ) is False


def test_unsupported_cases_are_not_converted_to_success():
    records = [_records()[-1]]
    categories, subtests = exp003._aggregate_records(records, [])
    failures = exp003.build_failure_analysis(
        [
            {
                "example_id": "G01",
                "subtest": "G01",
                "relation_id": 1,
                "correct": False,
                "predicted_token": "alpha",
                "gold_antecedents": [],
            }
        ]
    )

    assert categories["G"]["relation_hit_at_1"] is None
    assert subtests["G01"]["examples_detail"][0]["localization_pass"] is None
    assert len(failures) == 1
    assert failures[0]["diagnostic_label"] == "OTHER"


def test_known_failure_patterns_have_explicit_qualitative_labels():
    failures = exp003.build_failure_analysis(
        [
            {
                "example_id": "H04",
                "subtest": "H04",
                "relation_id": 1,
                "correct": False,
                "predicted_token": "Northern",
                "gold_antecedents": ["the Southern Rail Expansion"],
            },
            {
                "example_id": "D03",
                "subtest": "D03",
                "relation_id": 1,
                "correct": False,
                "predicted_token": "Riverside",
                "gold_antecedents": ["the Northbridge railway station"],
            },
        ]
    )

    assert [item["diagnostic_label"] for item in failures] == [
        "DISTRACTOR_ENTITY",
        "DISTRACTOR_ENTITY",
    ]
    assert all(item["implication"] == "model_weakness" for item in failures)


def test_multiple_anaphors_with_one_relation_are_reported_as_invalid():
    context = "John arrived."
    chunk = "John spoke. He left."
    raw = {
        "id": "D02",
        "context": context,
        "chunk": chunk,
        "context_spans": [
            {"label": "ANTECEDENT_1", "start": 0, "end": 4, "text": "John"}
        ],
        "chunk_spans": [
            {"label": "ANAPHOR_1", "start": 0, "end": 4, "text": "John"},
            {"label": "ANAPHOR_1", "start": 12, "end": 14, "text": "He"},
        ],
    }

    def no_relations(_filtered):
        return {"relations": [], "dropped_annotations": 0, "dropped_relations": 0}

    _, audit = exp003.preprocess_with_audit(
        raw,
        no_relations,
        {
            "D02": {
                "example_id": "D02",
                "category": "D",
                "subtest": "D02",
                "subtest_name": exp003.SUBTESTS["D02"][0],
                "intended_behavior": exp003.SUBTESTS["D02"][1],
            }
        },
    )

    assert audit["status"] == "ANNOTATION_INVALID"
    assert audit["preprocessing_status"] == "ANNOTATION_INVALID"
    assert "ANNOTATION_INVALID" in audit["status_flags"]
    assert audit["valid_relations_after_alignment"] == 0
    assert "Multiple anaphors for relation 1" in (
        audit["annotation_filter_and_alignment_warnings"][0]
    )


def test_expected_unmatched_negative_control_remains_unsupported_not_invalid():
    raw = {
        "id": "G01",
        "context": "A salient person appeared.",
        "chunk": "No referring expression appears here.",
        "context_spans": [
            {"label": "ANTECEDENT_1", "start": 2, "end": 16, "text": "salient person"}
        ],
        "chunk_spans": [],
    }

    def no_relations(_filtered):
        return {"relations": [], "dropped_annotations": 0, "dropped_relations": 0}

    _, audit = exp003.preprocess_with_audit(
        raw,
        no_relations,
        {
            "G01": {
                "example_id": "G01",
                "category": "G",
                "subtest": "G01",
                "subtest_name": exp003.SUBTESTS["G01"][0],
                "intended_behavior": exp003.SUBTESTS["G01"][1],
            }
        },
    )

    assert audit["status"] == "UNSUPPORTED"
    assert audit["preprocessing_status"] == "NO_GOLD_RELATION"
    assert "NO_GOLD_RELATION" in audit["status_flags"]
    assert "UNSUPPORTED" in audit["status_flags"]
    assert "ANNOTATION_INVALID" not in audit["status_flags"]


def test_json_result_serialization_is_stable(tmp_path):
    output = tmp_path / "metrics.json"
    value = {"scores": [1, 2], "title": "deterministic"}

    exp003._atomic_write_json(output, value)
    first = output.read_bytes()
    exp003._atomic_write_json(output, value)

    assert output.read_bytes() == first
    assert json.loads(first) == value
