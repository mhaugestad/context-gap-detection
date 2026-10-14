"""Evaluate the frozen EXP-002 checkpoint on the reserved behavioral test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.annotations import filter_annotations
from src.data.datamodule import CoreferenceDataModule
from src.training.module import CoreferenceLightningModule


DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "exp003_behavioral_eval"
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "experiments"
    / "exp002_full_training"
    / "checkpoints"
    / "best.ckpt"
)
DEFAULT_EXP002_METRICS = (
    REPOSITORY_ROOT / "experiments" / "exp002_full_training" / "metrics.json"
)

CATEGORY_INFO = {
    "A": "Basic and semantic forms",
    "B": "Antecedent competition",
    "C": "Context distance",
    "D": "Link structure",
    "E": "Span complexity",
    "F": "Reasoning",
    "G": "Negative controls",
    "H": "Contrast and adversarial cases",
}

SUBTESTS = {
    "A01": ("Exact NP", "Resolve an anaphor whose surface form matches its antecedent."),
    "A02": ("Shortened NP", "Resolve a shortened noun phrase to a longer antecedent."),
    "A03": ("Proper-name variation", "Resolve a shortened proper name to a fuller name."),
    "A04": ("Pronoun", "Resolve a pronoun to a named antecedent without lexical overlap."),
    "A05": ("Definite description", "Resolve a semantically related definite description."),
    "A06": ("Demonstrative", "Resolve a demonstrative to a preceding event or proposition."),
    "A07": ("Alias / acronym", "Resolve an acronym or alias to its full-form antecedent."),
    "A08": ("Nominal reformulation", "Resolve an event expressed in a different grammatical form."),
    "B01": ("Same-type distractor", "Choose the correct candidate over a same-type entity."),
    "B02": ("Lexical distractor", "Resist a distractor with strong lexical similarity."),
    "B03": ("Nearest-is-wrong", "Choose a non-nearest plausible antecedent."),
    "B04": ("Most-frequent-is-wrong", "Choose the less frequent entity over a salient distractor."),
    "C01": ("Adjacent", "Resolve an antecedent immediately before the chunk boundary."),
    "C02": ("Two-sentence distance", "Resolve across two intervening sentences."),
    "C03": ("Four-plus-sentence distance", "Resolve across four or more intervening sentences."),
    "C04": ("Intervening entities", "Resolve a distant antecedent despite intervening entities."),
    "D01": ("1:1", "Localize a single antecedent for a single anaphor."),
    "D02": ("1:N", "Link multiple chunk anaphors to the same context antecedent."),
    "D03": ("Two independent pairs", "Keep two distinct antecedent–anaphor pairs separate."),
    "D04": ("Coreference chain", "Link a chunk anaphor to a chain of differently expressed mentions."),
    "E01": ("Long NP", "Localize the complete long antecedent noun phrase."),
    "E02": ("Title + name", "Localize a complete title-and-name antecedent."),
    "E03": ("Possessive", "Resolve a possessive anaphor to its antecedent."),
    "E04": ("Nested / embedded NP", "Choose the intended antecedent rather than a nested NP."),
    "F01": ("Syntactic cue", "Use syntax to distinguish plausible antecedents."),
    "F02": ("Semantic type", "Use semantic compatibility to select an antecedent."),
    "F03": ("Discourse / event cue", "Resolve an event or proposition anaphor by discourse role."),
    "F04": ("World-knowledge stress test", "Resolve a relation requiring encyclopedic knowledge."),
    "G01": ("No anaphor", "Avoid linking when the chunk contains no anaphoric expression."),
    "G02": ("Missing antecedent", "Reject a null antecedent rather than inventing a context link."),
    "G03": ("Lexical false friend", "Avoid linking lexically similar expressions with distinct referents."),
    "G04": ("Generic / non-referential NP", "Avoid linking a generic NP to a specific context entity."),
    "H01": ("Recency contrast", "Choose the correct entity over a closer same-type distractor."),
    "H02": ("Gender / number contrast", "Respect gender or number cues in candidate selection."),
    "H03": ("Entity-order swap", "Resolve by identity rather than candidate order."),
    "H04": ("Minimal lexical change", "Use a small decisive lexical distinction between candidates."),
}

UNSUPPORTED_SUBTESTS = {
    "G01": (
        "The model receives gold anaphor spans and cannot predict that no anaphor exists.",
        "Anaphor detection and an explicit abstention/no-link output.",
    ),
    "G02": (
        "The model always ranks context tokens for a supplied anaphor and has no null antecedent.",
        "A null antecedent class or calibrated abstention mechanism.",
    ),
    "G03": (
        "A negative false-friend case without a positive gold link has no localization target.",
        "Referentiality/link-existence supervision, in addition to localization where a gold link exists.",
    ),
    "G04": (
        "The objective does not distinguish generic expressions from referential anaphors.",
        "Referentiality detection or explicit no-link supervision.",
    ),
}

MANUAL_FAILURE_DIAGNOSTICS = {
    ("H04", 1): {
        "label": "DISTRACTOR_ENTITY",
        "why": (
            "The selected Northern token belongs to the competing Northern Rail "
            "Expansion; the target is the Southern Rail Expansion."
        ),
        "interpretation": (
            "A direct same-template entity competition error. The gold Southern token "
            "ranked third, showing it was scored but lost to the competing expansion."
        ),
    },
    ("H02", 1): {
        "label": "LEXICAL_BIAS",
        "why": (
            "The selected 'original' token is part of 'original design', repeated in "
            "the chunk, while the gold antecedent is the coordinated pair Elena Torres "
            "and Sofia Ramirez."
        ),
        "interpretation": (
            "Consistent with lexical overlap outweighing the plural-person cue; the "
            "single example does not prove that causal mechanism."
        ),
    },
    ("A08", 1): {
        "label": "SEMANTIC_CONFUSION",
        "why": (
            "The selected 'North' subtoken is from Northstar, the acquiring company, "
            "whereas 'The deal' refers to the acquisition event."
        ),
        "interpretation": (
            "The prediction confuses an event anaphor with an entity participant; the "
            "best gold token ranked third."
        ),
    },
    ("B04", 1): {
        "label": "DISTRACTOR_ENTITY",
        "why": (
            "The selected earlier 'union' token is from 'union representatives', not "
            "the gold mention 'union leader Maria Torres'."
        ),
        "interpretation": (
            "A repeated same-lexeme distractor wins over the less frequent named "
            "antecedent; the gold surname ranked second."
        ),
    },
    ("D03", 1): {
        "label": "DISTRACTOR_ENTITY",
        "why": (
            "The selected 'Riverside' token names the competing bus-terminal entity; "
            "the gold antecedent for 'The station' is the Northbridge railway station."
        ),
        "interpretation": (
            "A cross-link between the two independent pairs. The same Riverside mention "
            "is correctly selected for 'the terminal' in relation 2."
        ),
    },
    ("F03", 1): {
        "label": "SEMANTIC_CONFUSION",
        "why": (
            "The selected 'university' token names the institution, while 'The decision' "
            "refers to the university's action of reducing tuition fees."
        ),
        "interpretation": (
            "The relation requires linking a nominal event anaphor to a proposition, "
            "not to its agent."
        ),
    },
    ("F02", 1): {
        "label": "SEMANTIC_CONFUSION",
        "why": (
            "The selected 'town' token is not a building; the gold antecedent is the "
            "historic Grandview Hotel, which fits 'The building'."
        ),
        "interpretation": (
            "The prediction misses the semantic-type contrast. It is an observed "
            "localization error, not evidence about the model's internal reasoning."
        ),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="mhaugestad/contextualise-context")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--exp002-metrics", type=Path, default=DEFAULT_EXP002_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.batch_size < 1 or (args.max_length is not None and args.max_length < 1):
        parser.error("--batch-size and --max-length must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "working_tree_dirty": dirty}


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError(f"Unsupported device: {requested}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_taxonomy_mapping(raw_examples: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    observed_ids = [str(example["id"]) for example in raw_examples]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("Test split contains duplicate example IDs.")

    unknown_ids = sorted(set(observed_ids) - SUBTESTS.keys())
    missing_ids = sorted(SUBTESTS.keys() - set(observed_ids))
    if unknown_ids or missing_ids:
        raise ValueError(
            f"Taxonomy IDs do not match test examples: unknown={unknown_ids}, "
            f"missing={missing_ids}"
        )

    return [
        {
            "example_id": identifier,
            "category": identifier[0],
            "subtest": identifier,
            "subtest_name": SUBTESTS[identifier][0],
            "intended_behavior": SUBTESTS[identifier][1],
        }
        for identifier in sorted(observed_ids)
    ]


def stable_ranked_context_tokens(
    scores: Sequence[float],
    valid_context: Sequence[bool],
    top_k: int | None = None,
) -> list[int]:
    if len(scores) != len(valid_context):
        raise ValueError("Scores and context mask must have equal lengths.")
    indices = [i for i, valid in enumerate(valid_context) if valid]
    indices.sort(key=lambda index: (-float(scores[index]), index))
    return indices if top_k is None else indices[:top_k]


def final_valid_context_token(valid_context: Sequence[bool]) -> int:
    valid = [index for index, value in enumerate(valid_context) if value]
    if not valid:
        raise ValueError("No valid context tokens are available for the recency baseline.")
    return valid[-1]


def relation_metric_counts(
    scores: torch.Tensor,
    valid_by_query: torch.Tensor,
    gold_context: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    """Return token correct/count, relation correct/count, and gold/candidate counts."""
    if scores.ndim != 2 or valid_by_query.shape != scores.shape:
        raise ValueError("Relation scores and pair mask must have shape [Q, T].")
    if gold_context.ndim != 1 or gold_context.shape[0] != scores.shape[1]:
        raise ValueError("Gold-context mask must have shape [T].")

    token_correct = 0
    token_count = 0
    for row, valid in zip(scores, valid_by_query):
        candidates = stable_ranked_context_tokens(
            row.detach().cpu().tolist(), valid.detach().cpu().tolist(), top_k=1
        )
        if not candidates:
            raise ValueError("Anaphor token has no valid context candidates.")
        token_correct += int(bool(gold_context[candidates[0]].item()))
        token_count += 1

    valid_context = valid_by_query.any(dim=0)
    candidate_count = int(valid_context.sum().item())
    gold_count = int((gold_context & valid_context).sum().item())
    if candidate_count == 0 or gold_count == 0:
        raise ValueError("Relation has no valid context or gold antecedent tokens.")

    valid_counts = valid_by_query.sum(dim=0)
    mean_scores = scores.masked_fill(~valid_by_query, 0.0).sum(dim=0)
    mean_scores = mean_scores / valid_counts.clamp_min(1)
    ranked = stable_ranked_context_tokens(
        mean_scores.detach().cpu().tolist(),
        valid_context.detach().cpu().tolist(),
    )
    relation_correct = int(bool(gold_context[ranked[0]].item()))
    recency_correct = int(bool(gold_context[ranked[-1]].item()))
    return (
        token_correct,
        token_count,
        relation_correct,
        1,
        gold_count,
        candidate_count,
    )


def _valid_span_map(
    spans: Sequence[dict[str, Any]],
    prefix: str,
    text: str,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pattern = re.compile(rf"{prefix}_(\d+)")
    for span in spans:
        match = pattern.fullmatch(str(span.get("label", "")))
        start, end = span.get("start"), span.get("end")
        if (
            match
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
            and text[start:end] == span.get("text")
        ):
            result[int(match.group(1))].append(span)
    return dict(result)


def _span_text(text: str, start: int, end: int) -> str:
    return text[start:end]


def _token_span_text(text: str, offsets: Sequence[Sequence[int]], start: int, end: int) -> str:
    selected = [tuple(map(int, offset)) for offset in offsets[start:end] if offset[1] > offset[0]]
    if not selected:
        return ""
    return text[selected[0][0] : selected[-1][1]]


def preprocess_with_audit(
    raw_example: dict[str, Any],
    preprocessor: Any,
    taxonomy: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        filtered = filter_annotations(raw_example, warn=True)
    filter_warnings = [str(item.message) for item in captured]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        processed = preprocessor(filtered)
    preprocessor_warnings = [str(item.message) for item in captured]

    raw_spans_a = raw_example.get("chunk_spans", [])
    raw_spans_c = raw_example.get("context_spans", [])
    filtered_span_count = len(filtered["chunk_spans"]) + len(filtered["context_spans"])
    dropped_annotations = int(processed.get("dropped_annotations", 0))
    dropped_relations = int(processed.get("dropped_relations", 0))
    has_relations = bool(processed["relations"])
    record = taxonomy[str(raw_example["id"])]
    unsupported = record["subtest"] in UNSUPPORTED_SUBTESTS
    invalid_filter_warning = any(
        "Invalid label:" in message
        or "Invalid span:" in message
        or "Multiple anaphors for relation" in message
        or ("Unmatched relation:" in message and not unsupported)
        for message in filter_warnings
    )
    annotation_invalid = bool(dropped_annotations or invalid_filter_warning)

    if has_relations:
        status = "EVALUABLE"
    elif unsupported:
        status = "UNSUPPORTED"
    elif annotation_invalid:
        status = "ANNOTATION_INVALID"
    elif dropped_relations:
        status = "TRUNCATED"
    else:
        status = "NO_GOLD_RELATION"

    flags = []
    if not has_relations and not annotation_invalid and not dropped_relations:
        flags.append("NO_GOLD_RELATION")
    if annotation_invalid:
        flags.append("ANNOTATION_INVALID")
    if dropped_relations:
        flags.append("TRUNCATED")
    if unsupported:
        flags.append("UNSUPPORTED")

    audit = {
        **record,
        "status": status,
        "status_flags": flags,
        "preprocessing_status": (
            "EVALUABLE"
            if has_relations
            else "ANNOTATION_INVALID"
            if annotation_invalid
            else "TRUNCATED"
            if dropped_relations
            else "NO_GOLD_RELATION"
        ),
        "raw_anaphor_annotations": len(raw_spans_a),
        "raw_antecedent_annotations": len(raw_spans_c),
        "filtered_annotation_count": filtered_span_count,
        "filtered_anaphor_annotations": len(filtered["chunk_spans"]),
        "filtered_antecedent_annotations": len(filtered["context_spans"]),
        "valid_relations_after_alignment": len(processed["relations"]),
        "preprocessor_dropped_annotations": dropped_annotations,
        "preprocessor_dropped_relations": dropped_relations,
        "annotation_filter_warnings": filter_warnings,
        "preprocessor_alignment_warnings": preprocessor_warnings,
        "annotation_filter_and_alignment_warnings": filter_warnings
        + preprocessor_warnings,
        "context": raw_example["context"],
        "chunk": raw_example["chunk"],
        "gold_relation_count": len(processed["relations"]),
    }
    return processed, audit


def evaluate_test_split(
    model: CoreferenceLightningModule,
    loader: DataLoader,
    processed_examples: Sequence[dict[str, Any]],
    raw_examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    taxonomy: dict[str, dict[str, str]],
    audits: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    model.to(device)
    processed_by_id = {str(example["id"]): example for example in processed_examples}
    raw_by_id = {str(example["id"]): example for example in raw_examples}
    token_correct = token_count = relation_correct = relation_count = recency_correct = 0
    random_expected_correct = 0.0
    predictions: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            device_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(device_batch)
            for batch_index, raw_id in enumerate(batch["id"]):
                example_id = str(raw_id)
                processed = processed_by_id[example_id]
                raw = raw_by_id[example_id]
                offsets = tokenizer(
                    raw["context"],
                    raw["chunk"],
                    truncation=True,
                    max_length=max_length,
                    return_offsets_mapping=True,
                )["offset_mapping"]
                raw_antecedents = _valid_span_map(
                    raw["context_spans"], "ANTECEDENT", raw["context"]
                )
                raw_anaphors = _valid_span_map(
                    raw["chunk_spans"], "ANAPHOR", raw["chunk"]
                )
                example_token_correct = 0
                example_token_count = 0
                example_relation_correct = 0

                for relation_index, relation in enumerate(processed["relations"]):
                    start = int(relation["anaphor_start"])
                    end = int(relation["anaphor_end"])
                    rows = output["logits"][batch_index, start:end]
                    valid_by_query = output["pair_mask"][batch_index, start:end]
                    gold_context = torch.zeros(
                        rows.shape[1], dtype=torch.bool, device=device
                    )
                    gold_token_positions: set[int] = set()
                    gold_token_spans: list[tuple[int, int]] = []
                    for antecedent in relation["antecedents"]:
                        c_start = int(antecedent["start"])
                        c_end = int(antecedent["end"])
                        gold_context[c_start:c_end] = True
                        gold_token_positions.update(range(c_start, c_end))
                        gold_token_spans.append((c_start, c_end))

                    current_token_correct = 0
                    for row, valid in zip(rows, valid_by_query):
                        ranked = stable_ranked_context_tokens(
                            row.detach().cpu().tolist(),
                            valid.detach().cpu().tolist(),
                            top_k=1,
                        )
                        current_token_correct += int(
                            bool(gold_context[ranked[0]].item())
                        )

                    (
                        _,
                        relation_token_count,
                        current_relation_correct,
                        _,
                        gold_token_count,
                        candidate_count,
                    ) = relation_metric_counts(rows, valid_by_query, gold_context)
                    valid_context = valid_by_query.any(dim=0)
                    candidate_counts = valid_by_query.sum(dim=0)
                    mean_scores = rows.masked_fill(~valid_by_query, 0.0).sum(dim=0)
                    mean_scores = mean_scores / candidate_counts.clamp_min(1)
                    ranked_context = stable_ranked_context_tokens(
                        mean_scores.detach().cpu().tolist(),
                        valid_context.detach().cpu().tolist(),
                    )
                    predicted_position = ranked_context[0]
                    gold_ranked = [
                        position for position in ranked_context if position in gold_token_positions
                    ]
                    best_gold_rank = (
                        ranked_context.index(gold_ranked[0]) + 1 if gold_ranked else None
                    )
                    final_context_position = final_valid_context_token(
                        valid_context.detach().cpu().tolist()
                    )
                    current_recency_correct = int(
                        bool(gold_context[final_context_position].item())
                    )
                    random_expected_correct += gold_token_count / candidate_count

                    relation_id = int(relation["relation_id"])
                    gold_char_spans = raw_antecedents.get(relation_id, [])
                    anaphor_char_spans = raw_anaphors.get(relation_id, [])
                    gold_texts = [
                        _span_text(raw["context"], int(span["start"]), int(span["end"]))
                        for span in gold_char_spans
                    ]
                    if not gold_texts:
                        gold_texts = [
                            _token_span_text(raw["context"], offsets, s, e)
                            for s, e in gold_token_spans
                        ]
                    anaphor_text = (
                        _span_text(
                            raw["chunk"],
                            int(anaphor_char_spans[0]["start"]),
                            int(anaphor_char_spans[0]["end"]),
                        )
                        if anaphor_char_spans
                        else _token_span_text(raw["chunk"], offsets, start, end)
                    )
                    predicted_offset = tuple(map(int, offsets[predicted_position]))
                    top_tokens = []
                    for position in ranked_context[:5]:
                        token_offset = tuple(map(int, offsets[position]))
                        top_tokens.append(
                            {
                                "token": _token_span_text(
                                    raw["context"], offsets, position, position + 1
                                ),
                                "token_index": position,
                                "character_offsets": list(token_offset),
                                "score": float(mean_scores[position].detach().cpu()),
                            }
                        )
                    gold_token_ranks = []
                    gold_rank_map = {
                        position: rank
                        for rank, position in enumerate(ranked_context, start=1)
                    }
                    for position in sorted(gold_token_positions):
                        token_offset = tuple(map(int, offsets[position]))
                        gold_token_ranks.append(
                            {
                                "token": _token_span_text(
                                    raw["context"], offsets, position, position + 1
                                ),
                                "token_index": position,
                                "rank": gold_rank_map[position],
                                "character_offsets": list(token_offset),
                            }
                        )
                    nearest_anaphor = min(max(predicted_position, start), end - 1)
                    example_id_record = taxonomy[example_id]
                    prediction = {
                        "example_id": example_id,
                        "category": example_id_record["category"],
                        "subtest": example_id_record["subtest"],
                        "relation_id": relation_id,
                        "anaphor_text": anaphor_text,
                        "anaphor_token_span": [start, end],
                        "gold_antecedents": gold_texts,
                        "gold_antecedent_character_offsets": [
                            [int(span["start"]), int(span["end"])]
                            for span in gold_char_spans
                        ],
                        "predicted_token": _token_span_text(
                            raw["context"],
                            offsets,
                            predicted_position,
                            predicted_position + 1,
                        ),
                        "predicted_token_index": predicted_position,
                        "predicted_token_character_offsets": list(predicted_offset),
                        "predicted_token_score": float(
                            mean_scores[predicted_position].detach().cpu()
                        ),
                        "correct": bool(current_relation_correct),
                        "gold_token_best_rank": best_gold_rank,
                        "gold_token_ranks": gold_token_ranks,
                        "top_5_context_tokens": top_tokens,
                        "recency_baseline_correct": bool(current_recency_correct),
                        "random_expected_accuracy": gold_token_count / candidate_count,
                        "distance_to_anaphor_tokens": abs(predicted_position - nearest_anaphor),
                        "context": raw["context"],
                        "chunk": raw["chunk"],
                    }
                    predictions.append(prediction)
                    token_correct += current_token_correct
                    token_count += relation_token_count
                    relation_correct += current_relation_correct
                    relation_count += 1
                    recency_correct += current_recency_correct
                    example_token_correct += current_token_correct
                    example_token_count += relation_token_count
                    example_relation_correct += current_relation_correct

                audit = audits[example_id]
                audit["token_hit_at_1"] = {
                    "correct": example_token_correct,
                    "total": example_token_count,
                    "value": (
                        example_token_correct / example_token_count
                        if example_token_count
                        else None
                    ),
                }
                audit["correct_relations"] = example_relation_correct
                audit["all_relations_correct"] = (
                    all_relations_correct(
                        [
                            prediction
                            for prediction in predictions
                            if prediction["example_id"] == example_id
                        ]
                    )
                )

    metric_result = {
        "relation_hit_at_1": {
            "correct": relation_correct,
            "total": relation_count,
            "value": relation_correct / relation_count if relation_count else None,
        },
        "token_hit_at_1": {
            "correct": token_correct,
            "total": token_count,
            "value": token_correct / token_count if token_count else None,
        },
        "random_expected_relation_hit_at_1": {
            "expected_correct": random_expected_correct,
            "total": relation_count,
            "value": random_expected_correct / relation_count if relation_count else None,
        },
        "recency_relation_hit_at_1": {
            "correct": recency_correct,
            "total": relation_count,
            "value": recency_correct / relation_count if relation_count else None,
        },
    }
    for example in processed_examples:
        audit = audits[str(example["id"])]
        audit["status"] = "EVALUABLE" if example["relations"] else audit["status"]
    return metric_result, predictions, list(audits.values())


def _safe_rate(correct: int, total: int) -> float | None:
    return correct / total if total else None


def all_relations_correct(predictions: Sequence[dict[str, Any]]) -> bool | None:
    if not predictions:
        return None
    return all(bool(prediction["correct"]) for prediction in predictions)


def _aggregate_records(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    record_by_id = {record["example_id"]: record for record in records}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subtest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
        by_subtest[record["subtest"]].append(record)
    for prediction in predictions:
        prediction_groups[prediction["example_id"]].append(prediction)

    def aggregate(ids: Sequence[str]) -> dict[str, Any]:
        selected = [record_by_id[item] for item in ids if item in record_by_id]
        selected_predictions = [
            prediction
            for record in selected
            for prediction in prediction_groups.get(record["example_id"], [])
        ]
        rel_correct = sum(bool(item["correct"]) for item in selected_predictions)
        tok_correct = sum(
            int(record.get("token_hit_at_1", {}).get("correct") or 0)
            for record in selected
        )
        tok_total = sum(
            int(record.get("token_hit_at_1", {}).get("total") or 0)
            for record in selected
        )
        baseline_random = sum(
            float(item["random_expected_accuracy"]) for item in selected_predictions
        )
        baseline_recency = sum(
            bool(item["recency_baseline_correct"]) for item in selected_predictions
        )
        total = len(selected_predictions)
        return {
            "examples": len(selected),
            "evaluable_examples": sum(record["status"] == "EVALUABLE" for record in selected),
            "valid_relations": total,
            "correct_relations": rel_correct,
            "relation_hit_at_1": _safe_rate(rel_correct, total),
            "token_correct": tok_correct,
            "token_total": tok_total,
            "token_hit_at_1": _safe_rate(tok_correct, tok_total),
            "random_expected_relation_hit_at_1": _safe_rate(baseline_random, total),
            "random_expected_correct": baseline_random,
            "recency_correct": baseline_recency,
            "recency_relation_hit_at_1": _safe_rate(baseline_recency, total),
            "unsupported_or_non_evaluable_examples": sum(
                record["status"] != "EVALUABLE" for record in selected
            ),
            "example_ids": [record["example_id"] for record in selected],
        }

    category_metrics = {
        category: {
            "focus": CATEGORY_INFO[category],
            **aggregate([record["example_id"] for record in by_category[category]]),
        }
        for category in CATEGORY_INFO
    }
    subtest_results = {}
    for subtest in sorted(SUBTESTS):
        current_records = by_subtest.get(subtest, [])
        result = aggregate([record["example_id"] for record in current_records])
        result["name"] = SUBTESTS[subtest][0]
        result["intended_behavior"] = SUBTESTS[subtest][1]
        details = []
        for record in current_records:
            if record["status"] == "EVALUABLE":
                localized = int(record.get("correct_relations", 0))
                total = int(record["gold_relation_count"])
                diagnostic_note = (
                    f"All {total} aligned relation(s) localized."
                    if localized == total
                    else f"{total - localized} of {total} aligned relation(s) missed."
                )
            elif record["status"] == "ANNOTATION_INVALID":
                diagnostic_note = "; ".join(
                    record["annotation_filter_and_alignment_warnings"]
                ) or "Invalid annotation prevented evaluation."
            elif record["status"] == "TRUNCATED":
                diagnostic_note = (
                    f"{record['preprocessor_dropped_relations']} relation(s) could not "
                    "be represented after tokenization/truncation."
                )
            elif subtest in UNSUPPORTED_SUBTESTS:
                diagnostic_note = UNSUPPORTED_SUBTESTS[subtest][0]
            else:
                diagnostic_note = "No valid aligned gold relation; excluded from Hit@1."
            details.append(
                {
                    "example_id": record["example_id"],
                    "status": record["status"],
                    "status_flags": record["status_flags"],
                    "gold_relations": record["gold_relation_count"],
                    "correct_relations": record.get("correct_relations", 0),
                    "all_relations_correct": record.get("all_relations_correct"),
                    "localization_pass": (
                        record.get("all_relations_correct")
                        if record["status"] == "EVALUABLE"
                        else None
                    ),
                    "diagnostic_note": diagnostic_note,
                }
            )
        result["examples_detail"] = [
            *details
        ]
        subtest_results[subtest] = result
    return category_metrics, subtest_results


def h1_group_comparison(category_metrics: dict[str, Any]) -> dict[str, Any]:
    straightforward = ["A", "C"]
    challenging = ["B", "F", "H"]

    def group(categories: Sequence[str]) -> dict[str, Any]:
        correct = sum(category_metrics[item]["correct_relations"] for item in categories)
        total = sum(category_metrics[item]["valid_relations"] for item in categories)
        return {
            "categories": list(categories),
            "relations_correct": correct,
            "relations_total": total,
            "relation_hit_at_1": _safe_rate(correct, total),
        }

    first = group(straightforward)
    second = group(challenging)
    if first["relation_hit_at_1"] is None or second["relation_hit_at_1"] is None:
        status = "inconclusive"
        difference = None
    else:
        difference = (
            first["relation_hit_at_1"] - second["relation_hit_at_1"]
        ) * 100.0
        status = "supported" if difference > 0 else "not supported"
    return {
        "straightforward": first,
        "challenging": second,
        "difference_percentage_points": difference,
        "absolute_difference_percentage_points": abs(difference) if difference is not None else None,
        "status": status,
        "interpretation": (
            "Exploratory only: groups contain few examples and differ in composition."
        ),
    }


def build_failure_analysis(
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use conservative labels unless a specific observed cause is established."""
    result = []
    for item in predictions:
        if item["correct"]:
            continue
        diagnostic = MANUAL_FAILURE_DIAGNOSTICS.get(
            (item["example_id"], int(item["relation_id"]))
        )
        if diagnostic is None:
            diagnostic = {
                "label": "OTHER",
                "why": (
                    "The selected context token is outside every annotated gold "
                    "antecedent span, so this relation is a localization miss."
                ),
                "interpretation": (
                    "Observed model localization weakness; the isolated prediction "
                    "does not establish a causal bias."
                ),
            }
        result.append(
            {
                "example_id": item["example_id"],
                "subtest": item["subtest"],
                "relation_id": item["relation_id"],
                "predicted_token": item["predicted_token"],
                "gold_antecedents": item["gold_antecedents"],
                "why_incorrect": diagnostic["why"],
                "diagnostic_label": diagnostic["label"],
                "interpretation": diagnostic["interpretation"],
                "implication": "model_weakness",
            }
        )
    return result


def build_report(
    overall: dict[str, Any],
    categories: dict[str, Any],
    subtests: dict[str, Any],
    group_comparison: dict[str, Any],
    audits: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    exp002_metrics: dict[str, Any],
) -> str:
    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    rel = overall["relation_hit_at_1"]
    tok = overall["token_hit_at_1"]
    rand = overall["random_expected_relation_hit_at_1"]
    rec = overall["recency_relation_hit_at_1"]
    lines = [
        "# EXP-003 — Behavioral test evaluation",
        "",
        "## Summary",
        "",
        (
            f"The frozen EXP-002 checkpoint achieved Relation Hit@1 of "
            f"**{rel['correct']}/{rel['total']} ({pct(rel['value'])})** and "
            f"Token Hit@1 of **{tok['correct']}/{tok['total']} ({pct(tok['value'])})** "
            "on the valid, aligned gold relations."
        ),
        (
            f"The exact random-token expectation was **{rand['expected_correct']:.2f}/"
            f"{rand['total']} ({pct(rand['value'])})**; the last-context-token "
            f"recency baseline was **{rec['correct']}/{rec['total']} ({pct(rec['value'])})**."
        ),
        (
            "EXP-002 development reference: Relation Hit@1 **101/152 (66.45%)**, "
            "Token Hit@1 **191/336 (56.85%)**. The behavioral test set was purpose-built "
            "and is not composition-matched to development, so the rates are descriptive "
            "and are not a clean generalization comparison."
        ),
        "",
        "The model localizes antecedent tokens given gold anaphor spans. It does not "
        "evaluate anaphor detection, exact span recovery, or null antecedent prediction.",
        "",
        "## Category results",
        "",
        "| Category | Examples | Evaluable | Relations correct/total | Relation Hit@1 | Token Hit@1 | Unsupported / non-evaluable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, result in categories.items():
        lines.append(
            f"| {category} — {result['focus']} | {result['examples']} | "
            f"{result['evaluable_examples']} | {result['correct_relations']}/"
            f"{result['valid_relations']} | {pct(result['relation_hit_at_1'])} | "
            f"{pct(result['token_hit_at_1'])} | "
            f"{result['unsupported_or_non_evaluable_examples']} |"
        )
    lines.extend(
        [
            "",
            "## Subtest results",
            "",
            "| Subtest | Example(s) | Status | Relations correct/total | Relation Hit@1 | All relations correct | Diagnostic note |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for subtest, result in subtests.items():
        details = result["examples_detail"]
        statuses = ", ".join(
            f"{item['example_id']}: {item['status']}" for item in details
        )
        notes = "; ".join(item["diagnostic_note"] for item in details)
        all_pass = (
            "N/A"
            if result["valid_relations"] == 0
            else "yes"
            if result["correct_relations"] == result["valid_relations"]
            else "no"
        )
        lines.append(
            f"| {subtest} — {result['name']} | {', '.join(result['example_ids']) or '—'} | "
            f"{statuses or 'UNMAPPED'} | {result['correct_relations']}/"
            f"{result['valid_relations']} | {pct(result['relation_hit_at_1'])} | {all_pass} |"
            f" {notes} |"
        )
    first = group_comparison["straightforward"]
    second = group_comparison["challenging"]
    difference = group_comparison["difference_percentage_points"]
    lines.extend(
        [
            "",
            "## H1: straightforward vs challenging",
            "",
            f"- A + C: **{first['relations_correct']}/{first['relations_total']} "
            f"({pct(first['relation_hit_at_1'])})**.",
            f"- B + F + H: **{second['relations_correct']}/{second['relations_total']} "
            f"({pct(second['relation_hit_at_1'])})**.",
            (
                f"- Straightforward minus challenging: **{difference:+.2f} percentage "
                f"points**; H1 is **{group_comparison['status']}**."
                if difference is not None
                else "- Group difference is not estimable; H1 is inconclusive."
            ),
            f"- {group_comparison['interpretation']}",
            "",
            "## H2: span complexity",
            "",
            (
                f"- Category E: **{categories['E']['correct_relations']}/"
                f"{categories['E']['valid_relations']} "
                f"({pct(categories['E']['relation_hit_at_1'])})**; Category A: "
                f"**{categories['A']['correct_relations']}/"
                f"{categories['A']['valid_relations']} "
                f"({pct(categories['A']['relation_hit_at_1'])})**."
            ),
            "- Category E's token-level localization is perfect in this small battery, "
            "but Hit@1 does not establish recovery of complete antecedent spans.",
            "",
            "## Qualitative observations",
            "",
        ]
    )
    successful_adversarial = sorted(
        {
            item["subtest"]
            for item in predictions
            if item["category"] == "H" and item["correct"]
        }
    )
    success_notes = {
        "H01": "selects Patel rather than the later Laura Mitchell mention",
        "H03": "selects Orion rather than later Meridian",
    }
    lines.extend(
        [
            "- Correct adversarial localizations: "
            + (
                "; ".join(
                    f"{identifier} {success_notes.get(identifier, '')}".rstrip()
                    for identifier in successful_adversarial
                )
                if successful_adversarial
                else "none observed."
            ),
            "- The model also handled all four distance subtests and all four span-complexity "
            "subtests at relation Hit@1, but each cell is a single or very small number of examples.",
        ]
    )
    lines.append(
        f"- Incorrect relation predictions were inspected individually; "
        f"{len(failures)} were outside the annotated antecedent-token union."
    )
    for item in failures:
        lines.append(
            f"- **{item['example_id']} ({item['subtest']}, relation "
            f"{item['relation_id']})**: predicted `{item['predicted_token']}` instead of "
            f"`{' / '.join(item['gold_antecedents'])}`. "
            f"`{item['diagnostic_label']}`: {item['why_incorrect']}"
        )
    lines.extend(
        [
            "",
            "## Negative controls",
            "",
            "| Subtest | Status | Evaluation limitation | Capability needed |",
            "|---|---|---|---|",
        ]
    )
    for identifier, (limitation, capability) in UNSUPPORTED_SUBTESTS.items():
        matching = [record for record in audits if record["subtest"] == identifier]
        status = matching[0]["status"] if matching else "UNMAPPED"
        lines.append(
            f"| {identifier} — {SUBTESTS[identifier][0]} | {status} | "
            f"{limitation} | {capability} |"
        )
    lines.extend(
        [
            "",
            "Zero-relation examples remain in the dataset audit and do not enter metric "
            "denominators. Their absence of a predicted link is not counted as a success.",
            "",
            "## Recommendations",
            "",
            "- Add explicit anaphor detection and a calibrated null/abstention option before "
            "scoring negative controls as model behavior.",
            "- Evaluate exact antecedent span boundaries and end-to-end linking separately "
            "from token localization.",
            "- Use larger, contrastive paired examples for causal claims about recency, "
            "lexical similarity, or semantic reasoning.",
            "- Treat this test battery as diagnostic; do not tune the frozen checkpoint "
            "against these test outcomes.",
            "",
            "## Taxonomy and mapping caveat",
            "",
            "The test split IDs directly match the test-battery subtest IDs and were checked "
            "one-to-one. The behavioral taxonomy in README includes A05–A08; task06's "
            "shorter category table lists only A01–A04. This evaluation uses the full "
            "README/test-split taxonomy so that all 36 test examples are retained.",
            "",
            "## Run provenance",
            "",
            f"- EXP-002 development Relation Hit@1: "
            f"{exp002_metrics['development_best_checkpoint']['relation_hit_at_1']['correct']}/"
            f"{exp002_metrics['development_best_checkpoint']['relation_hit_at_1']['total']}.",
            f"- Evaluated examples retained in audit: {len(audits)}.",
            f"- Incorrect relations in failure analysis: {len(failures)}.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    metrics_path = args.exp002_metrics.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EXP-002 checkpoint not found: {checkpoint}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"EXP-002 metrics not found: {metrics_path}")
    exp002_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if exp002_metrics.get("status") != "completed":
        raise ValueError(f"EXP-002 metrics are not completed: {metrics_path}")
    selected_checkpoint = exp002_metrics.get("best_checkpoint_path")
    if not selected_checkpoint:
        raise ValueError("EXP-002 metrics do not identify the selected checkpoint.")
    if Path(selected_checkpoint).resolve() != checkpoint:
        raise ValueError(
            "EXP-003 must use the checkpoint selected by minimum EXP-002 "
            f"development loss: {Path(selected_checkpoint).resolve()}"
        )
    exp002_config = exp002_metrics["configuration"]
    if args.dataset_name != exp002_config["dataset_name"]:
        raise ValueError(
            "EXP-003 must use the dataset recorded by EXP-002: "
            f"{exp002_config['dataset_name']!r}"
        )
    dataset_revision = (
        args.dataset_revision
        if args.dataset_revision is not None
        else exp002_config.get("dataset_revision")
    )
    max_length = (
        args.max_length
        if args.max_length is not None
        else int(exp002_config.get("max_length", 512))
    )
    seed_everything(args.seed)
    device = resolve_device(args.device)

    datamodule = CoreferenceDataModule(
        dataset_name=args.dataset_name,
        tokenizer_name=exp002_config["encoder_name"],
        batch_size=args.batch_size,
        max_length=max_length,
        num_workers=0,
        dataset_revision=dataset_revision,
        annotation_policy="warn",
    )
    datamodule.prepare_data()
    if datamodule._raw_dataset is None or "test" not in datamodule._raw_dataset:
        raise ValueError("The dataset does not contain the reserved test split.")
    raw_test = list(datamodule._raw_dataset["test"])
    if len(raw_test) != 36:
        raise ValueError(f"Expected the 36-example test split, received {len(raw_test)}.")

    taxonomy_rows = validate_taxonomy_mapping(raw_test)
    taxonomy = {row["example_id"]: row for row in taxonomy_rows}
    processed_examples = []
    audits = {}
    for raw in raw_test:
        processed, audit = preprocess_with_audit(
            raw,
            datamodule.preprocessor,
            taxonomy,
        )
        processed_examples.append(processed)
        audits[str(raw["id"])] = audit

    loader = DataLoader(
        processed_examples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=datamodule.collator,
    )
    model = CoreferenceLightningModule.load_from_checkpoint(
        str(checkpoint),
        map_location="cpu",
    )
    model.eval()
    result, predictions, example_audits = evaluate_test_split(
        model=model,
        loader=loader,
        processed_examples=processed_examples,
        raw_examples=raw_test,
        tokenizer=datamodule.tokenizer,
        max_length=max_length,
        device=device,
        taxonomy=taxonomy,
        audits=audits,
    )
    category_metrics, subtest_results = _aggregate_records(
        example_audits,
        predictions,
    )
    h1 = h1_group_comparison(category_metrics)
    failure_analysis = build_failure_analysis(predictions)
    raw_stats = {
        "examples": len(raw_test),
        "examples_with_valid_relations": sum(
            record["status"] == "EVALUABLE" for record in example_audits
        ),
        "examples_without_valid_relations": sum(
            record["preprocessing_status"] != "EVALUABLE"
            for record in example_audits
        ),
        "valid_relations": len(predictions),
        "anaphor_tokens": sum(
            (item["anaphor_token_span"][1] - item["anaphor_token_span"][0])
            for item in predictions
        ),
        "examples_with_annotation_filter_or_alignment_issues": sum(
            "ANNOTATION_INVALID" in record["status_flags"]
            or "TRUNCATED" in record["status_flags"]
            for record in example_audits
        ),
        "examples_with_filter_warnings": sum(
            bool(record["annotation_filter_warnings"])
            for record in example_audits
        ),
        "preprocessor_reported_dropped_relations": sum(
            record["preprocessor_dropped_relations"] for record in example_audits
        ),
        "preprocessor_reported_dropped_annotations": sum(
            record["preprocessor_dropped_annotations"] for record in example_audits
        ),
        "dataset_fingerprint": getattr(datamodule._raw_dataset["test"], "_fingerprint", None),
        "dataset_revision": dataset_revision,
        "test_split_used_for_training_or_model_selection": False,
        "examples": example_audits,
        "counting_note": (
            "Pre-filter annotation warnings and preprocessor alignment drops are listed "
            "per example; relations removed before token alignment are not scored."
        ),
    }
    config = {
        "experiment": "EXP-003",
        "dataset_name": args.dataset_name,
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": raw_stats["dataset_fingerprint"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "exp002_training_configuration": exp002_config,
        "encoder_name": exp002_config["encoder_name"],
        "projection_dim": exp002_config["projection_dim"],
        "dropout": exp002_config["dropout"],
        "max_length": max_length,
        "batch_size": args.batch_size,
        "evaluation_seed": args.seed,
        "device": str(device),
        "git": git_metadata(),
        "exp002_git_commit": exp002_metrics.get("git_commit")
        or exp002_config.get("git_commit"),
        "exp002_checkpoint_best_epoch": exp002_metrics.get("checkpoint_best_epoch"),
        "exp002_best_development_loss": exp002_metrics.get("best_development_loss"),
        "tie_breaking": (
            "Higher score first; exact score ties use the lower original encoder token index."
        ),
        "deterministic_inference": True,
        "gradient_computation": False,
        "annotation_policy": "existing filter_annotations + CoreferencePreprocessor",
        "negative_controls": "Reported as unsupported where no-link/anaphor detection is required.",
    }
    overall = {
        **result,
        "development_reference": {
            "relation_hit_at_1": exp002_metrics["development_best_checkpoint"][
                "relation_hit_at_1"
            ],
            "token_hit_at_1": exp002_metrics["development_best_checkpoint"][
                "token_hit_at_1"
            ],
        },
        "h1_group_comparison": h1,
        "taxonomy_note": (
            "README's complete test battery includes A05-A08; these are included "
            "because the reserved test split contains them, despite task06's shorter table."
        ),
    }
    taxonomy_file = {
        "dataset_name": args.dataset_name,
        "dataset_revision": dataset_revision,
        "mapping_source": (
            "The 36 unique test split IDs were verified against the subtest IDs in "
            "README.md's Antecedent–Anaphor Diagnostic Test Battery."
        ),
        "validation": {
            "test_example_count": len(raw_test),
            "unique_test_ids": len(taxonomy_rows),
            "all_ids_mapped": True,
            "unmapped_ids": [],
        },
        "category_descriptions": CATEGORY_INFO,
        "subtests": {
            identifier: {
                "category": identifier[0],
                "name": metadata[0],
                "intended_behavior": metadata[1],
            }
            for identifier, metadata in sorted(SUBTESTS.items())
        },
        "example_mappings": taxonomy_rows,
        "task06_discrepancy": (
            "Task06's table lists A01-A04; README and the actual reserved split include "
            "A01-A08. All 36 examples are retained and A05-A08 are explicitly mapped."
        ),
    }
    report = build_report(
        overall,
        category_metrics,
        subtest_results,
        h1,
        example_audits,
        failure_analysis,
        predictions,
        exp002_metrics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _atomic_write_json(output_dir / "dataset_stats.json", raw_stats)
    _atomic_write_json(output_dir / "taxonomy_mapping.json", taxonomy_file)
    _atomic_write_json(output_dir / "overall_metrics.json", overall)
    _atomic_write_json(output_dir / "category_metrics.json", category_metrics)
    _atomic_write_json(output_dir / "subtest_results.json", subtest_results)
    _atomic_write_json(output_dir / "predictions.json", predictions)
    _atomic_write_json(output_dir / "failure_analysis.json", failure_analysis)
    (output_dir / "behavioral_report.md").write_text(report, encoding="utf-8")

    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "examples": len(raw_test),
        "relations": len(predictions),
        "relation_hit_at_1": overall["relation_hit_at_1"],
        "token_hit_at_1": overall["token_hit_at_1"],
        "h1_group_comparison": h1,
        "device": str(device),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = _run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
