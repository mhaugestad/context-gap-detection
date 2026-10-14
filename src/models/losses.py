"""Numerically stable objectives for cross-boundary coreference."""

from collections.abc import Iterator
from typing import NamedTuple

import torch


class RelationTargets(NamedTuple):
    """Validated token targets for one annotated anaphor relation."""

    batch_index: int
    relation_index: int
    query_positions: torch.Tensor
    gold_context: torch.Tensor
    valid_context_by_query: torch.Tensor


def _validate_shapes(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    relations: torch.Tensor,
    antecedent_spans: torch.Tensor,
    relation_mask: torch.Tensor,
    antecedent_mask: torch.Tensor,
) -> None:
    if logits.ndim != 3 or logits.shape[1] != logits.shape[2]:
        raise ValueError("logits must have shape [batch_size, sequence_length, sequence_length]")
    batch_size, sequence_length, _ = logits.shape
    if pair_mask.shape != logits.shape:
        raise ValueError("pair_mask must have the same shape as logits")
    if relations.ndim != 3 or relations.shape[0] != batch_size or relations.shape[2] != 2:
        raise ValueError("relations must have shape [batch_size, relation_count, 2]")
    if relation_mask.shape != relations.shape[:2]:
        raise ValueError("relation_mask must have shape [batch_size, relation_count]")
    if (
        antecedent_spans.ndim != 4
        or antecedent_spans.shape[:2] != relations.shape[:2]
        or antecedent_spans.shape[3] != 2
    ):
        raise ValueError(
            "antecedent_spans must have shape "
            "[batch_size, relation_count, antecedent_count, 2]"
        )
    if antecedent_mask.shape != antecedent_spans.shape[:3]:
        raise ValueError(
            "antecedent_mask must have shape "
            "[batch_size, relation_count, antecedent_count]"
        )
    if sequence_length == 0:
        raise ValueError("logits must contain at least one sequence position")


def _iter_relation_targets(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    relations: torch.Tensor,
    antecedent_spans: torch.Tensor,
    relation_mask: torch.Tensor,
    antecedent_mask: torch.Tensor,
) -> Iterator[RelationTargets]:
    """Yield validated relation targets shared by loss and evaluation."""
    _validate_shapes(
        logits,
        pair_mask,
        relations,
        antecedent_spans,
        relation_mask,
        antecedent_mask,
    )

    sequence_length = logits.shape[1]
    valid_pairs = pair_mask.to(device=logits.device, dtype=torch.bool)
    valid_relations = relation_mask.to(device=logits.device, dtype=torch.bool)
    valid_antecedents = antecedent_mask.to(device=logits.device, dtype=torch.bool)

    for batch_index, relation_index in valid_relations.nonzero(as_tuple=False).tolist():
        anaphor_start, anaphor_end = (
            int(value)
            for value in relations[batch_index, relation_index].tolist()
        )
        if not 0 <= anaphor_start < anaphor_end <= sequence_length:
            raise ValueError(
                f"Valid relation ({batch_index}, {relation_index}) has invalid "
                f"anaphor span [{anaphor_start}, {anaphor_end}) for sequence "
                f"length {sequence_length}"
            )

        relation_antecedents = valid_antecedents[batch_index, relation_index]
        antecedent_indices = relation_antecedents.nonzero(as_tuple=False).flatten().tolist()
        if not antecedent_indices:
            raise ValueError(
                f"Valid relation ({batch_index}, {relation_index}) has no valid "
                "antecedent spans"
            )

        gold_context = torch.zeros(
            sequence_length,
            dtype=torch.bool,
            device=logits.device,
        )
        for antecedent_index in antecedent_indices:
            antecedent_start, antecedent_end = (
                int(value)
                for value in antecedent_spans[
                    batch_index, relation_index, antecedent_index
                ].tolist()
            )
            if not 0 <= antecedent_start < antecedent_end <= sequence_length:
                raise ValueError(
                    f"Valid antecedent span ({batch_index}, {relation_index}, "
                    f"{antecedent_index}) has invalid bounds "
                    f"[{antecedent_start}, {antecedent_end}) for sequence "
                    f"length {sequence_length}"
                )
            # Boolean assignment forms a set union, so overlapping or repeated
            # alternative mentions never count a token more than once.
            gold_context[antecedent_start:antecedent_end] = True

        query_positions = torch.arange(
            anaphor_start,
            anaphor_end,
            device=logits.device,
        )
        valid_context_by_query = valid_pairs[batch_index, query_positions]
        for offset, valid_context in enumerate(valid_context_by_query):
            query_position = anaphor_start + offset
            if not valid_context.any():
                raise ValueError(
                    f"Anaphor token {query_position} in relation "
                    f"({batch_index}, {relation_index}) has no valid context "
                    "tokens according to pair_mask"
                )
            if not (gold_context & valid_context).any():
                raise ValueError(
                    f"Anaphor token {query_position} in relation "
                    f"({batch_index}, {relation_index}) has no gold antecedent "
                    "tokens available in pair_mask"
                )

        yield RelationTargets(
            batch_index,
            relation_index,
            query_positions,
            gold_context,
            valid_context_by_query,
        )


def coreference_loss(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    relations: torch.Tensor,
    antecedent_spans: torch.Tensor,
    relation_mask: torch.Tensor,
    antecedent_mask: torch.Tensor,
) -> torch.Tensor:
    """Return mean multi-antecedent loss over valid relations.

    For each anaphor token, the objective is the difference between a
    log-sum-exp over valid context positions and one over the union of all
    gold antecedent tokens. Token losses are averaged within each relation,
    then relation losses are averaged so padding and relation count per
    example cannot skew the objective.
    """
    _validate_shapes(
        logits,
        pair_mask,
        relations,
        antecedent_spans,
        relation_mask,
        antecedent_mask,
    )
    targets = list(
        _iter_relation_targets(
            logits,
            pair_mask,
            relations,
            antecedent_spans,
            relation_mask,
            antecedent_mask,
        )
    )
    if not targets:
        # Retain a gradient path for batches that contain only unannotated data.
        return logits.sum() * 0.0

    relation_losses = []
    for target in targets:
        token_losses = []
        for query_position, valid_context in zip(
            target.query_positions,
            target.valid_context_by_query,
        ):
            row_logits = logits[target.batch_index, query_position]
            gold_context = target.gold_context & valid_context
            denominator = torch.logsumexp(
                row_logits.masked_fill(~valid_context, -torch.inf),
                dim=0,
            )
            numerator = torch.logsumexp(
                row_logits.masked_fill(~gold_context, -torch.inf),
                dim=0,
            )
            token_losses.append(denominator - numerator)
        relation_losses.append(torch.stack(token_losses).mean())

    return torch.stack(relation_losses).mean()


class CoreferenceLoss(torch.nn.Module):
    """Module wrapper for :func:`coreference_loss`."""

    def forward(
        self,
        logits: torch.Tensor,
        pair_mask: torch.Tensor,
        relations: torch.Tensor,
        antecedent_spans: torch.Tensor,
        relation_mask: torch.Tensor,
        antecedent_mask: torch.Tensor,
    ) -> torch.Tensor:
        return coreference_loss(
            logits,
            pair_mask,
            relations,
            antecedent_spans,
            relation_mask,
            antecedent_mask,
        )