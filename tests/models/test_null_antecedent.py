from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import lightning as L
from torch import nn
from torch.utils.data import DataLoader

from scripts.experiments.exp005_null_antecedent import BestNullCheckpoint
from scripts.experiments.exp005_null_antecedent import (
    _adjust_for_candidate_count,
    _span_to_char,
)
from src.data.null_antecedent import (
    NullAntecedentCollator,
    audit_source_split,
    construct_split_negatives,
)
from src.data.preprocessor import CoreferencePreprocessor
from src.evaluation.null_metrics import (
    calibration_metrics,
    evaluate_null_predictions,
    paired_contrast_metrics,
    stratified_performance,
)
from src.models.null_antecedent import (
    NullAntecedentModel,
    marginal_link_null_loss,
)
from src.models.span_extraction import SpanExtractionModel
from src.training.null_module import NullAntecedentLightningModule


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(20, hidden_size)
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict=True):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class StubSpanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(1, 6)
        self.encoder.config = SimpleNamespace(hidden_size=6)
        self.score_bias = nn.Parameter(torch.tensor([2.0, 0.0]))
        self.max_span_length = 32

    def forward(self, **_kwargs):
        features = self.encoder(torch.tensor(0))
        return {
            "zero": self.score_bias.sum() * 0.0,
            "relations": [{
                "batch_index": 0,
                "relation_index": 0,
                "candidate_spans": torch.tensor([[1, 2], [2, 3]]),
                "scores": self.score_bias,
                "anaphor_repr": features,
                "context_repr": features * 0.5,
            }],
        }


class FakeEncoding(dict):
    def __init__(self, input_ids, offset_mapping, sequence_ids):
        super().__init__(
            input_ids=input_ids,
            attention_mask=[1] * len(input_ids),
            offset_mapping=offset_mapping,
        )
        self._sequence_ids = sequence_ids

    def sequence_ids(self):
        return self._sequence_ids


class FakeTokenizer:
    is_fast = True
    pad_token_id = 0

    @staticmethod
    def _tokens(text):
        return [
            (match.group(), match.start(), match.end())
            for match in __import__("re").finditer(r"\S+", text)
        ]

    def __call__(
        self,
        context,
        chunk=None,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    ):
        context_tokens = self._tokens(context)
        chunk_tokens = self._tokens(chunk or "")
        ids = [1]
        offsets = [(0, 0)]
        sequence_ids = [None]
        for index, (_, start, end) in enumerate(context_tokens):
            ids.append(index + 2)
            offsets.append((start, end))
            sequence_ids.append(0)
        ids.append(2)
        offsets.append((0, 0))
        sequence_ids.append(None)
        for index, (_, start, end) in enumerate(chunk_tokens):
            ids.append(index + 3)
            offsets.append((start, end))
            sequence_ids.append(1)
        ids.append(2)
        offsets.append((0, 0))
        sequence_ids.append(None)
        return FakeEncoding(ids[:max_length], offsets[:max_length], sequence_ids[:max_length])


def _span(text, label, start):
    return {"text": text, "start": start, "end": start + len(text), "label": label}


def _source_examples():
    return [
        {
            "id": "source",
            "context": "Alex spoke.",
            "chunk": "He arrived.",
            "context_spans": [_span("Alex", "ANTECEDENT_1", 0)],
            "chunk_spans": [_span("He", "ANAPHOR_1", 0)],
        },
        {
            "id": "unannotated",
            "context": "A new programme began.",
            "chunk": "It was successful.",
            "context_spans": [],
            "chunk_spans": [],
        },
        {
            "id": "hard-donor",
            "context": "Jordan visited the lab.",
            "chunk": "They left.",
            "context_spans": [_span("Jordan", "ANTECEDENT_2", 0)],
            "chunk_spans": [_span("They", "ANAPHOR_2", 0)],
        },
    ]


def _raw_example(is_null=False):
    return {
        "id": "x",
        "source_example_id": "x",
        "context": "a b c",
        "chunk": "he",
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "offset_mapping": [(0, 0), (0, 1), (2, 3), (0, 0)],
        "context_mask": [0, 1, 1, 0],
        "chunk_mask": [0, 0, 0, 1],
        "dropped_relations": 0,
        "relations": [{
            "relation_id": 1,
            "anaphor_start": 3,
            "anaphor_end": 4,
            "anaphor_char_start": 0,
            "anaphor_char_end": 2,
            "antecedents": [] if is_null else [{"start": 1, "end": 2}],
            "is_null": is_null,
            "strategy": "context_removal" if is_null else None,
            "source_example_id": "x",
            "replacement_source_id": "donor" if is_null else None,
            "pair_id": "x:1",
        }],
    }


def _collator_batch(is_null=False):
    return NullAntecedentCollator(0)([_raw_example(is_null)])


def test_explicit_null_relation_has_zero_gold_spans_and_positive_is_preserved():
    positive, audit = audit_source_split(
        _source_examples(), CoreferencePreprocessor(FakeTokenizer(), annotation_policy="drop")
    )
    assert audit["valid_aligned_positive_relations"] == 2
    source = next(example for example in positive if example["id"] == "source")
    assert source["relations"][0]["is_null"] is False
    assert len(source["relations"][0]["antecedents"]) == 1

    negatives, _, _ = construct_split_negatives(
        _source_examples(), positive, FakeTokenizer(), 512, "train", 42
    )
    null_relation = negatives[0]["relations"][0]
    assert null_relation["is_null"] is True
    assert null_relation["antecedents"] == []


def test_zero_relation_examples_are_audited_but_never_null_supervision():
    positives, audit = audit_source_split(
        _source_examples(), CoreferencePreprocessor(FakeTokenizer(), annotation_policy="drop")
    )
    assert audit["examples_without_any_annotations"] == 1
    assert audit["explicit_null_annotation"]["supported_by_existing_relation_format"] is False
    assert all(
        not relation["is_null"]
        for example in positives
        for relation in example["relations"]
    )


def test_negative_construction_is_split_local_and_rejects_reserved_test_split():
    rows = _source_examples()
    positives, _ = audit_source_split(
        rows, CoreferencePreprocessor(FakeTokenizer(), annotation_policy="drop")
    )
    negatives, report, manifest = construct_split_negatives(
        rows, positives, FakeTokenizer(), 512, "train", 42
    )
    available = {str(row["id"]) for row in rows}
    assert negatives
    assert report["reserved_test_split_loaded"] is False
    assert all(row["relations"][0]["replacement_source_id"] in available for row in negatives)
    assert all(item["source_split"] == "train" for item in manifest)
    with pytest.raises(ValueError, match="train or dev"):
        construct_split_negatives(
            rows, positives, FakeTokenizer(), 512, "test", 42
        )


def test_train_and_dev_negative_pools_do_not_cross_leak():
    rows = _source_examples()
    train_rows = [{**row, "id": f"train-{row['id']}"} for row in rows]
    dev_rows = [{**row, "id": f"dev-{row['id']}"} for row in rows]
    processor = CoreferencePreprocessor(FakeTokenizer(), annotation_policy="drop")
    train_positive, _ = audit_source_split(train_rows, processor)
    dev_positive, _ = audit_source_split(dev_rows, processor)
    train_negative, _, _ = construct_split_negatives(
        train_rows, train_positive, FakeTokenizer(), 512, "train", 42
    )
    dev_negative, _, _ = construct_split_negatives(
        dev_rows, dev_positive, FakeTokenizer(), 512, "dev", 43
    )
    train_pool = {str(row["id"]) for row in train_rows}
    dev_pool = {str(row["id"]) for row in dev_rows}
    assert all(
        row["relations"][0]["replacement_source_id"] in train_pool
        for row in train_negative
    )
    assert all(
        row["relations"][0]["replacement_source_id"] in dev_pool
        for row in dev_negative
    )
    assert not train_pool & dev_pool


def test_collator_preserves_explicit_link_and_null_labels_and_metadata():
    positive = _collator_batch()
    negative = _collator_batch(is_null=True)
    assert positive["is_null"].tolist() == [[False]]
    assert negative["is_null"].tolist() == [[True]]
    assert positive["antecedent_mask"].tolist() == [[[True]]]
    assert negative["antecedent_mask"].tolist() == [[[False]]]
    assert negative["relation_metadata"][0][0]["pair_id"] == "x:1"


@pytest.mark.parametrize(
    ("approach", "expected_null"),
    [("constant", False), ("learned_bias", True), ("conditioned", False)],
)
def test_null_candidate_is_present_and_approaches_score_it(approach, expected_null):
    model = NullAntecedentModel(StubSpanModel(), approach, dropout=0)
    if approach == "learned_bias":
        with torch.no_grad():
            model.null_bias.fill_(3.0)
    elif approach == "conditioned":
        with torch.no_grad():
            model.null_head[-1].bias.fill_(-10.0)
    output = model(_collator_batch())
    relation = output["relations"][0]
    assert relation["all_scores"].shape == (3,)
    assert relation["candidate_spans"].tolist() == [[1, 2], [2, 3]]
    assert relation["predicted_is_null"] is expected_null


def test_constant_null_score_is_fixed_and_has_no_trainable_parameters():
    model = NullAntecedentModel(StubSpanModel(), "constant")
    assert model.null_constant.item() == 0.0
    assert all(not param.requires_grad for param in model.parameters())
    assert model.null_bias is None


def test_link_and_null_losses_are_stable_differentiable_and_marginal():
    scores = torch.tensor([1000.0, 999.0, -1000.0], requires_grad=True)
    null = torch.tensor(998.0, requires_grad=True)
    outputs = [
        {
            "batch_index": 0,
            "relation_index": 0,
            "candidate_spans": torch.tensor([[1, 2], [2, 3], [3, 4]]),
            "scores": scores,
            "null_score": null,
        },
        {
            "batch_index": 0,
            "relation_index": 1,
            "candidate_spans": torch.tensor([[1, 2], [2, 3]]),
            "scores": scores[:2],
            "null_score": null,
        },
    ]
    gold = torch.tensor([[[[1, 2], [2, 3]], [[-1, -1], [-1, -1]]]])
    mask = torch.tensor([[[True, True], [False, False]]])
    labels = torch.tensor([[False, True]])
    loss, count, excluded = marginal_link_null_loss(
        outputs, gold, mask, labels, scores.sum() * 0
    )
    assert torch.isfinite(loss)
    assert count == 2 and excluded == 0
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert null.grad is not None and torch.isfinite(null.grad)


def test_link_loss_requires_gold_but_null_loss_rejects_gold_spans():
    scores = torch.tensor([0.0], requires_grad=True)
    output = [{
        "batch_index": 0,
        "relation_index": 0,
        "candidate_spans": torch.tensor([[1, 2]]),
        "scores": scores,
        "null_score": scores.new_zeros(()),
    }]
    with pytest.raises(ValueError, match="requires at least one"):
        marginal_link_null_loss(
            output,
            torch.full((1, 1, 1, 2), -1),
            torch.zeros((1, 1, 1), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.bool),
            scores.sum() * 0,
        )


def test_frozen_encoder_and_span_scorer_receive_no_gradients_null_head_does():
    model = NullAntecedentModel(StubSpanModel(), "conditioned", dropout=0)
    model.train()
    output = model(_collator_batch(is_null=True))
    batch = _collator_batch(is_null=True)
    loss, _, _ = marginal_link_null_loss(
        output["relations"],
        batch["antecedent_spans"],
        batch["antecedent_mask"],
        batch["is_null"],
        output["zero"],
    )
    loss.backward()
    assert all(parameter.grad is None for parameter in model.span_model.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.null_head.parameters()
    )
    assert not model.span_model.training


def test_real_span_model_encodes_once_for_multiple_relations_with_null_head():
    encoder = FakeEncoder()
    span_model = SpanExtractionModel(
        head_dim=4,
        max_span_length=2,
        dropout=0,
        approach="span_pair",
        encoder=encoder,
    )
    model = NullAntecedentModel(span_model, "conditioned", dropout=0)
    current = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
        "context_mask": torch.tensor([[0, 1, 1, 1, 0, 0]]),
        "relations": torch.tensor([[[4, 5], [5, 6]]]),
        "relation_mask": torch.tensor([[True, True]]),
    }
    output = model(current)
    assert len(output["relations"]) == 2
    assert encoder.calls == 1
    for relation in output["relations"]:
        candidates = relation["candidate_spans"]
        assert (candidates[:, 0] >= 1).all()
        assert (candidates[:, 1] <= 4).all()
        assert relation["all_scores"].shape[0] == candidates.shape[0] + 1


def test_metrics_report_confusion_end_to_end_and_paired_contrast():
    records = [
        {
            "gold_is_null": False,
            "predicted_is_null": False,
            "forced_link_exact": True,
            "forced_link_token_f1": 1.0,
            "end_to_end_correct": True,
            "null_probability": 0.1,
            "negative_strategy": None,
            "pair_id": "p1",
            "context_token_count": 5,
            "candidate_count": 6,
        },
        {
            "gold_is_null": True,
            "predicted_is_null": True,
            "forced_link_exact": False,
            "forced_link_token_f1": 0.0,
            "end_to_end_correct": True,
            "null_probability": 0.9,
            "negative_strategy": "context_removal",
            "pair_id": "p1",
            "context_token_count": 3,
            "candidate_count": 3,
        },
    ]
    metrics = evaluate_null_predictions(records)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["end_to_end_exact_accuracy"] == pytest.approx(1.0)
    assert metrics["forced_link_span_em"] == pytest.approx(1.0)
    assert metrics["actual_positive_span_em_after_null"] == pytest.approx(1.0)
    assert paired_contrast_metrics(records)["both_decisions_correct"] == 1
    assert calibration_metrics(records)["brier_score"] == pytest.approx(0.01)


def test_span_offsets_reconstruct_characters():
    offsets = torch.tensor([[0, 0], [0, 4], [5, 10], [0, 0]])
    assert _span_to_char(offsets, (1, 3)) == (0, 10)
    assert _span_to_char(offsets, (0, 1)) is None


def test_multiple_relations_and_alternative_gold_mentions_are_preserved():
    row = {
        "id": "multi",
        "context": "Alex Jones spoke.",
        "chunk": "He arrived. It changed.",
        "context_spans": [
            _span("Alex", "ANTECEDENT_1", 0),
            _span("Jones", "ANTECEDENT_1", 5),
            _span("spoke", "ANTECEDENT_2", 11),
        ],
        "chunk_spans": [
            _span("He", "ANAPHOR_1", 0),
            _span("It", "ANAPHOR_2", 12),
        ],
    }
    positives, _ = audit_source_split(
        [row], CoreferencePreprocessor(FakeTokenizer(), annotation_policy="drop")
    )
    relations = positives[0]["relations"]
    assert len(relations) == 2
    assert [len(relation["antecedents"]) for relation in relations] == [2, 1]


def test_state_dict_reload_reproduces_null_predictions():
    torch.manual_seed(7)
    original = NullAntecedentModel(StubSpanModel(), "conditioned", dropout=0).eval()
    inputs = _collator_batch()
    expected = original(inputs)["relations"][0]["all_scores"].detach()
    restored = NullAntecedentModel(StubSpanModel(), "conditioned", dropout=0).eval()
    restored.load_state_dict(deepcopy(original.state_dict()))
    actual = restored(inputs)["relations"][0]["all_scores"].detach()
    assert torch.equal(expected, actual)


def test_candidate_count_adjustment_uses_calibration_then_held_out_pairs():
    records = []
    for pair_index in range(10):
        for is_null in (False, True):
            records.append({
                "example_id": f"pair-{pair_index}-{is_null}",
                "relation_id": 1,
                "source_example_id": f"pair-{pair_index}",
                "pair_id": f"pair-{pair_index}",
                "gold_is_null": is_null,
                "predicted_is_null": False,
                "forced_link_exact": not is_null,
                "forced_link_token_f1": 1.0 if not is_null else 0.0,
                "end_to_end_correct": not is_null,
                "null_probability": 0.4,
                "null_score": 0.0,
                "highest_span_score": 0.5,
                "candidate_count": pair_index + 1,
                "context_token_count": 10,
                "negative_strategy": "context_removal" if is_null else None,
            })
    result = _adjust_for_candidate_count(records, seed=42)
    assert result["status"] == "completed exploratory diagnostic"
    assert result["calibration_pair_count"] == 2
    assert result["held_out_pair_count"] == 8
    assert "held_out_adjusted_metrics" in result


def test_candidate_count_stratification_works_with_small_balanced_sample():
    rows = []
    for index in range(4):
        rows.append({
            "gold_is_null": index % 2 == 0,
            "predicted_is_null": index % 2 == 0,
            "forced_link_exact": index % 2 == 1,
            "forced_link_token_f1": 1.0 if index % 2 else 0.0,
            "end_to_end_correct": True,
            "null_probability": 0.5,
            "candidate_count": index + 1,
            "context_token_count": index * 40 + 1,
            "negative_strategy": "context_removal",
        })
    metrics = evaluate_null_predictions(rows)
    stratified = stratified_performance(rows)
    assert metrics["null"]["recall"] == 1.0
    assert stratified["by_candidate_count_quantile"]["q1"]["count"] == 1
    assert stratified["by_context_length"]["1-32"]["count"] == 1


def test_lightning_training_saves_best_checkpoint_without_default_checkpointing(tmp_path):
    model = NullAntecedentModel(StubSpanModel(), "learned_bias")
    module = NullAntecedentLightningModule(model, learning_rate=0.1)
    loader = DataLoader(
        [_raw_example(is_null=True)],
        batch_size=1,
        collate_fn=NullAntecedentCollator(0),
    )
    checkpoint = tmp_path / "best.ckpt"
    selector = BestNullCheckpoint(checkpoint)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[selector],
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    assert checkpoint.is_file()
    assert selector.best_epoch == 1

    restored = NullAntecedentLightningModule(
        NullAntecedentModel(StubSpanModel(), "learned_bias"),
        learning_rate=0.1,
    )
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    batch = _collator_batch(is_null=True)
    assert torch.equal(
        module(batch)["relations"][0]["all_scores"],
        restored(batch)["relations"][0]["all_scores"],
    )
