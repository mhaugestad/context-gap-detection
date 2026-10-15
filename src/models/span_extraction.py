"""Encoder-only models and objectives for antecedent span extraction."""

from collections.abc import Mapping
from typing import Any

import math

import torch
from torch import nn
from transformers import AutoModel


def enumerate_context_spans(
    context_mask: torch.Tensor,
    max_span_length: int = 32,
) -> torch.Tensor:
    """Enumerate half-open spans contained in contiguous context-token runs."""
    if context_mask.ndim != 1:
        raise ValueError("context_mask must have shape [sequence_length]")
    if max_span_length < 1:
        raise ValueError("max_span_length must be positive")

    positions = context_mask.to(dtype=torch.bool).nonzero(
        as_tuple=False
    ).flatten().tolist()
    runs: list[list[int]] = []
    for position in positions:
        if not runs or position != runs[-1][-1] + 1:
            runs.append([position])
        else:
            runs[-1].append(position)

    spans: list[tuple[int, int]] = []
    for run in runs:
        for start_offset, start in enumerate(run):
            last_offset = min(len(run), start_offset + max_span_length)
            spans.extend(
                (start, run[end_offset] + 1)
                for end_offset in range(start_offset, last_offset)
            )
    if not spans:
        return torch.empty((0, 2), dtype=torch.long, device=context_mask.device)
    return torch.tensor(spans, dtype=torch.long, device=context_mask.device)


class SpanExtractionModel(nn.Module):
    """Score context spans conditioned on each supplied gold anaphor.

    The encoder runs once for the entire batch. Context projections used by
    span-pair scoring have ``head_dim`` width, so candidate memory is
    O(number_of_candidates * head_dim), rather than O(candidates * hidden_size).
    At 16,000 candidates, 128 dimensions, and float32, one candidate-feature
    matrix is about 8 MiB; scoring work is linear in candidates times head_dim.
    """

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        head_dim: int = 128,
        max_span_length: int = 32,
        dropout: float = 0.1,
        approach: str = "boundary",
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if max_span_length < 1:
            raise ValueError("max_span_length must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")
        if approach not in {"boundary", "span_pair"}:
            raise ValueError("approach must be 'boundary' or 'span_pair'")

        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(
            encoder_name
        )
        self.encoder.train()
        config = getattr(self.encoder, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Encoder config must define a positive hidden_size or d_model")

        self.approach = approach
        self.head_dim = head_dim
        self.max_span_length = max_span_length
        self.dropout = nn.Dropout(dropout)

        if approach == "boundary":
            self.context_projection = nn.Linear(hidden_size, head_dim)
            self.anaphor_projection = nn.Linear(hidden_size, head_dim)
            self.boundary_hidden = nn.Linear(head_dim * 3, head_dim)
            self.start_scorer = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )
            self.end_scorer = nn.Sequential(
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, 1),
            )
        else:
            self.start_projection = nn.Linear(hidden_size, head_dim, bias=False)
            self.end_projection = nn.Linear(hidden_size, head_dim, bias=False)
            self.mean_projection = nn.Linear(hidden_size, head_dim, bias=False)
            self.anaphor_projection = nn.Linear(hidden_size, head_dim)
            self.span_scorer = nn.Linear(head_dim, 1)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        encoded: Any = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.dropout(encoded.last_hidden_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        relations: torch.Tensor,
        relation_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Return ragged candidate-span scores for each valid relation."""
        if context_mask.shape != input_ids.shape:
            raise ValueError("context_mask must match input_ids")
        if relations.ndim != 3 or relations.shape[0] != input_ids.shape[0] \
                or relations.shape[2] != 2:
            raise ValueError("relations must have shape [batch, relation_count, 2]")
        if relation_mask.shape != relations.shape[:2]:
            raise ValueError("relation_mask must have shape [batch, relation_count]")

        hidden = self._encode(input_ids, attention_mask)
        valid_context = context_mask.to(dtype=torch.bool) & attention_mask.to(
            dtype=torch.bool
        )
        relation_outputs: list[dict[str, torch.Tensor | int]] = []

        for batch_index, relation_index in relation_mask.to(
            dtype=torch.bool
        ).nonzero(as_tuple=False).tolist():
            anaphor_start, anaphor_end = (
                int(index)
                for index in relations[batch_index, relation_index].tolist()
            )
            sequence_length = input_ids.shape[1]
            if not 0 <= anaphor_start < anaphor_end <= sequence_length:
                raise ValueError(
                    f"Invalid anaphor span [{anaphor_start}, {anaphor_end})"
                )
            if not attention_mask[batch_index, anaphor_start:anaphor_end].bool().all():
                raise ValueError("Anaphor span includes padding")

            context = valid_context[batch_index]
            candidates = enumerate_context_spans(context, self.max_span_length)
            if candidates.numel() == 0:
                raise ValueError("A supervised relation has no valid context tokens")

            sequence_hidden = hidden[batch_index]
            anaphor = sequence_hidden[anaphor_start:anaphor_end].mean(dim=0)
            context_repr = sequence_hidden[context].mean(dim=0)
            if self.approach == "boundary":
                scores = self._boundary_scores(
                    sequence_hidden,
                    anaphor,
                    candidates,
                    context,
                )
            else:
                scores = self._span_pair_scores(
                    sequence_hidden,
                    anaphor,
                    candidates,
                )
            relation_outputs.append({
                "batch_index": batch_index,
                "relation_index": relation_index,
                "candidate_spans": candidates,
                "scores": scores,
                "anaphor_repr": anaphor,
                "context_repr": context_repr,
            })

        return {
            "relations": relation_outputs,
            "zero": hidden.sum() * 0.0,
        }

    def _boundary_scores(
        self,
        hidden: torch.Tensor,
        anaphor: torch.Tensor,
        candidates: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        context_repr = self.context_projection(hidden)
        anaphor_repr = self.anaphor_projection(anaphor)
        conditioned = torch.cat(
            (
                context_repr,
                anaphor_repr.expand_as(context_repr),
                context_repr * anaphor_repr,
            ),
            dim=-1,
        )
        conditioned = torch.tanh(self.boundary_hidden(conditioned))
        start_logits = self.start_scorer(conditioned).squeeze(-1)
        end_logits = self.end_scorer(conditioned).squeeze(-1)
        start_logits = start_logits.masked_fill(~context_mask, -torch.inf)
        end_logits = end_logits.masked_fill(~context_mask, -torch.inf)
        return start_logits[candidates[:, 0]] + end_logits[candidates[:, 1] - 1]

    def _span_pair_scores(
        self,
        hidden: torch.Tensor,
        anaphor: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        start_projection = self.start_projection(hidden)
        end_projection = self.end_projection(hidden)
        mean_projection = self.mean_projection(hidden)
        prefix = torch.cat(
            (
                mean_projection.new_zeros((1, self.head_dim)),
                mean_projection.cumsum(dim=0),
            ),
            dim=0,
        )
        starts, ends = candidates.unbind(dim=-1)
        mean_repr = (prefix[ends] - prefix[starts]) / (ends - starts).unsqueeze(-1)
        span_repr = (
            start_projection[starts]
            + end_projection[ends - 1]
            + mean_repr
        )
        query = self.anaphor_projection(anaphor)
        scores = (span_repr * query).sum(dim=-1) / math.sqrt(self.head_dim)
        return scores + self.span_scorer(span_repr).squeeze(-1)


def marginal_span_loss(
    relation_outputs: list[Mapping[str, torch.Tensor | int]],
    antecedent_spans: torch.Tensor,
    antecedent_mask: torch.Tensor,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Marginal negative log likelihood over alternative complete gold spans.

    Relations whose every gold span exceeds the candidate maximum are excluded
    from the loss and reported in the returned excluded count; their gold
    annotations remain available to evaluation and are never shortened.
    """
    if antecedent_spans.ndim != 4 or antecedent_spans.shape[-1] != 2:
        raise ValueError(
            "antecedent_spans must have shape [batch, relations, alternatives, 2]"
        )
    if antecedent_mask.shape != antecedent_spans.shape[:3]:
        raise ValueError("antecedent_mask must match antecedent_spans")

    losses: list[torch.Tensor] = []
    excluded = 0
    for output in relation_outputs:
        batch_index = int(output["batch_index"])
        relation_index = int(output["relation_index"])
        candidates = output["candidate_spans"]
        scores = output["scores"]
        if not isinstance(candidates, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("candidate_spans and scores must be tensors")

        alternative_indices = antecedent_mask[
            batch_index, relation_index
        ].to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
        if alternative_indices.numel() == 0:
            raise ValueError("A valid relation must have at least one gold antecedent")
        gold = antecedent_spans[batch_index, relation_index, alternative_indices]
        matches = (candidates[:, None, :] == gold[None, :, :]).all(dim=-1).any(dim=-1)
        if not matches.any():
            excluded += 1
            continue
        losses.append(
            torch.logsumexp(scores, dim=0)
            - torch.logsumexp(scores.masked_fill(~matches, -torch.inf), dim=0)
        )

    if not losses:
        return zero, 0, excluded
    return torch.stack(losses).mean(), len(losses), excluded
