from __future__ import annotations

from scripts.experiments.exp005b_null_validation import (
    _manual_review_summary,
    _sample_manual_review,
    false_null_analysis,
    select_threshold,
    source_grouped_stratified_split,
)


def _record(
    source_id: str,
    is_null: bool,
    strategy: str | None,
    *,
    predicted_null: bool = False,
    margin: float = -1.0,
    exact: bool = True,
) -> dict:
    return {
        "example_id": f"{source_id}::{strategy or 'positive'}",
        "source_example_id": source_id,
        "relation_id": 1,
        "gold_is_null": is_null,
        "predicted_is_null": predicted_null,
        "negative_strategy": strategy,
        "null_score": margin,
        "highest_span_score": 0.0,
        "forced_link_exact": exact,
        "forced_link_token_f1": 1.0 if exact else 0.0,
        "end_to_end_correct": (
            predicted_null if is_null else not predicted_null and exact
        ),
        "context_token_count": 40,
        "candidate_count": 4,
        "second_highest_span_score": -0.5,
        "top_two_span_score_gap": 0.5,
        "gold_character_spans": [[0, 4]],
        "gold_target": ["Alex"],
        "highest_scoring_span_text": "Alex",
        "highest_scoring_character_span": [0, 4],
        "context_text": "Alex spoke.",
        "chunk_text": "He arrived.",
        "anaphor_text": "He",
        "predicted_target": "Alex" if not predicted_null else "NULL",
    }


def test_source_split_is_deterministic_stratified_and_group_disjoint():
    rows = []
    for strategy in ("context_removal", "hard_context_replacement"):
        for index in range(8):
            source = f"{strategy}-{index}"
            rows.append(_record(source, False, None))
            rows.append(_record(source, True, strategy, predicted_null=True))

    first_cal, first_eval, details = source_grouped_stratified_split(rows, 9)
    second_cal, second_eval, _ = source_grouped_stratified_split(rows, 9)

    cal_sources = {row["source_example_id"] for row in first_cal}
    eval_sources = {row["source_example_id"] for row in first_eval}
    assert cal_sources.isdisjoint(eval_sources)
    assert first_cal == second_cal
    assert first_eval == second_eval
    assert details["calibration_relation_count"] == 16
    assert details["evaluation_relation_count"] == 16
    assert {
        row["negative_strategy"] for row in first_cal if row["gold_is_null"]
    } == {"context_removal", "hard_context_replacement"}
    assert {
        row["negative_strategy"] for row in first_eval if row["gold_is_null"]
    } == {"context_removal", "hard_context_replacement"}


def test_threshold_selection_optimizes_calibration_e2e_then_positive_em():
    rows = [
        _record("positive", False, None, margin=0.5, exact=True),
        _record("negative", True, "context_removal", predicted_null=True, margin=1.0),
    ]

    result = select_threshold(rows)

    assert result["selected_threshold"] == 0.5
    assert result["calibration_metrics_at_selected_threshold"][
        "end_to_end_exact_accuracy"
    ] == 1.0
    assert result["calibration_metrics_at_selected_threshold"][
        "actual_positive_span_em_after_null"
    ] == 1.0


def test_manual_review_sample_has_25_per_strategy_and_keeps_all_available_errors():
    rows = []
    for strategy in ("context_removal", "hard_context_replacement"):
        for index in range(30):
            rows.append(_record(
                f"{strategy}-{index}",
                True,
                strategy,
                predicted_null=index >= 4,
            ))

    sample = _sample_manual_review(rows, seed=42)
    summary = _manual_review_summary(sample)

    assert len(sample) == 50
    for strategy in ("context_removal", "hard_context_replacement"):
        selected = [row for row in sample if row["negative_strategy"] == strategy]
        assert len(selected) == 25
        assert sum(not row["predicted_is_null"] for row in selected) == 4
        assert sum(row["predicted_is_null"] for row in selected) == 21
        assert summary["by_strategy"][strategy]["pending_human_review"] == 25
    assert summary["validated_count"] == 0
    assert summary["review_status"] == "PENDING_HUMAN_REVIEW"


def test_false_null_analysis_inspects_each_false_rejection_and_stratifies():
    rows = [
        _record("accepted", False, None, predicted_null=False, exact=True),
        _record("false-null", False, None, predicted_null=True, exact=False),
    ]

    result = false_null_analysis(rows)

    assert result["positive_relation_count"] == 2
    assert result["false_null_count"] == 1
    assert result["false_null_predictions"][0][
        "margin_null_minus_best_span"
    ] == -1.0
    assert result["false_null_predictions"][0][
        "forced_link_prediction_correct"
    ] is False
    assert result["by_forced_link_localization"]["forced_link_incorrect"][
        "false_null_count"
    ] == 1
