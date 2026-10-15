from pathlib import Path
from types import SimpleNamespace

from src.evaluation.span_metrics import aggregate_span_metrics, length_band
from scripts.experiments.exp004_span_extraction import (
    BestSpanCheckpoint,
    _comparison,
    _error_class_candidate,
    _span_to_char,
    _attach_review_context,
    parse_args,
    span_length_metrics,
    update_report,
)
import torch


def prediction(example_id, predicted, gold, *, relation_id=1):
    exact = predicted in gold
    localization = any(
        max(predicted[0], start) < min(predicted[1], end)
        for start, end in gold
    )
    return {
        "example_id": example_id,
        "relation_id": relation_id,
        "predicted_token_start": predicted[0],
        "predicted_token_end": predicted[1],
        "gold_token_spans": [list(span) for span in gold],
        "gold_antecedent_token_lengths": [end - start for start, end in gold],
        "exact_match": exact,
        "localization_hit": localization,
        "start_boundary_correct": any(predicted[0] == start for start, _ in gold),
        "end_boundary_correct": any(predicted[1] == end for _, end in gold),
        "token_overlap_f1": 1.0 if exact else 0.0,
        "character_exact_match": exact,
    }


def test_length_bands_and_metrics_report_counts_and_intervals():
    assert [length_band(value) for value in (1, 2, 3, 5, 9, 32)] == [
        "1", "2", "3-4", "5-8", "9+", "9+"
    ]
    metrics = aggregate_span_metrics([
        {
            "exact_match": True,
            "localization_hit": True,
            "start_boundary_correct": True,
            "end_boundary_correct": True,
            "character_exact_match": True,
            "token_overlap_f1": 1.0,
        },
        {
            "exact_match": False,
            "localization_hit": False,
            "start_boundary_correct": False,
            "end_boundary_correct": False,
            "character_exact_match": False,
            "token_overlap_f1": 0.0,
        },
    ])
    assert metrics["total_relations"] == 2
    assert metrics["exact_span_match"]["correct"] == 1
    assert metrics["exact_span_match"]["ci95"] is not None
    assert metrics["token_overlap_f1"]["mean"] == 0.5


def test_paired_comparison_uses_intersection_and_reports_paired_counts():
    baseline = [
        prediction("a", (1, 2), [(1, 2)]),
        prediction("b", (2, 3), [(1, 2)]),
    ]
    boundary = [
        prediction("a", (1, 3), [(1, 2)]),
        prediction("b", (1, 2), [(1, 2)]),
        prediction("extra", (3, 4), [(3, 4)]),
    ]
    result = _comparison({"baseline": baseline, "boundary": boundary})
    assert result["common_relation_count"] == 2
    assert result["paired_exact_match_comparisons"]["boundary_vs_baseline"][
        "first_only_correct"
    ] == 1
    assert result["paired_exact_match_comparisons"]["boundary_vs_baseline"][
        "second_only_correct"
    ] == 1


def test_character_mapping_uses_first_and_last_included_tokens():
    offsets = torch.tensor([[0, 0], [3, 5], [6, 9], [0, 0]])
    assert _span_to_char(offsets, (1, 3)) == (3, 9)
    assert _span_to_char(offsets, (0, 1)) is None


def test_error_class_suggestions_do_not_claim_semantic_classification():
    incomplete = prediction("a", (2, 3), [(1, 4)])
    assert "incomplete" in _error_class_candidate(incomplete)
    oversized = prediction("b", (1, 5), [(2, 4)])
    assert "oversized" in _error_class_candidate(oversized)
    wrong = prediction("c", (8, 9), [(1, 3)])
    assert "manual review" in _error_class_candidate(wrong)


def test_cli_can_run_an_approach_independently():
    args = parse_args(["--approach", "span_pair", "--max-span-length", "12"])
    assert args.approach == "span_pair"
    assert args.max_span_length == 12


def test_checkpoint_selection_uses_em_then_loss(tmp_path: Path):
    callback = BestSpanCheckpoint(tmp_path / "best.ckpt")
    saved = []

    class Trainer:
        sanity_checking = False
        callback_metrics = {
            "val/span_em": torch.tensor(0.5),
            "val/loss": torch.tensor(1.0),
        }
        current_epoch = 0

        def save_checkpoint(self, path, weights_only=True):
            saved.append((path, weights_only))

    trainer = Trainer()
    callback.on_validation_end(trainer, SimpleNamespace())
    assert callback.best_epoch == 1
    trainer.current_epoch = 1
    trainer.callback_metrics = {
        "val/span_em": torch.tensor(0.5),
        "val/loss": torch.tensor(0.9),
    }
    callback.on_validation_end(trainer, SimpleNamespace())
    assert callback.best_epoch == 2
    assert len(saved) == 2
    trainer.current_epoch = 2
    trainer.callback_metrics = {
        "val/span_em": torch.tensor(0.4),
        "val/loss": torch.tensor(0.1),
    }
    callback.on_validation_end(trainer, SimpleNamespace())
    assert callback.best_epoch == 2
    assert len(saved) == 2


def test_report_contains_table_hypothesis_status_and_length_counts(tmp_path: Path):
    records = {
        "baseline": [prediction("a", (1, 2), [(1, 2)])],
        "boundary": [prediction("a", (1, 3), [(1, 2)])],
        "span_pair": [prediction("a", (1, 2), [(1, 2)])],
    }
    comparison = _comparison(records)
    lengths = span_length_metrics(records)
    update_report(
        tmp_path,
        comparison,
        lengths,
        {"boundary": {
            "best_epoch": 1,
            "best_development_span_em": 0.0,
            "best_development_loss": 1.0,
            "training_runtime_seconds": 1.0,
            "epochs_completed": 1,
        }},
        records,
        {"seed": 42},
        "test parser",
    )
    report = (tmp_path / "comparison_report.md").read_text(encoding="utf-8")
    assert "| Approach | Dev Span EM | Token F1 | Localization Hit | Best Epoch |" in report
    assert "- H1:" in report
    assert "- H4:" in report
    assert "| 1 | 1 / 100.0% |" in report


def test_review_context_is_attached_to_cached_predictions():
    records = {"boundary": [prediction("a", (1, 2), [(1, 2)])]}
    _attach_review_context(
        records,
        [{"id": "a", "context": "source text", "chunk": "mention"}],
    )
    assert records["boundary"][0]["context_text"] == "source text"
    assert records["boundary"][0]["chunk_text"] == "mention"
