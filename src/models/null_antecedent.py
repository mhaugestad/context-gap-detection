"""NULL-augmented scoring for supplied gold anaphor spans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .span_extraction import SpanExtractionModel


class NullAntecedentModel(nn.Module):
    """Add a NULL candidate to the existing span-pair candidate scores."""

    def __init__(
        self,
        span_model: SpanExtractionModel,
        approach: str,
        dropout: float = 0.1,
        freeze_span_model: bool = True,
    ) -> None:
        super().__init__()
        if approach not in {"constant", "learned_bias", "conditioned"}:
            raise ValueError(
                "approach must be constant, learned_bias, or conditioned"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        hidden_size = getattr(
            getattr(span_model.encoder, "config", None),
            "hidden_size",
            None,
        )
        if hidden_size is None:
            hidden_size = getattr(
                getattr(span_model.encoder, "config", None),
                "d_model",
                None,
            )
        if not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("The span model encoder must define hidden_size")

        self.span_model = span_model
        self.approach = approach
        self.freeze_span_model = freeze_span_model
        if approach == "constant":
            self.register_buffer("null_constant", torch.tensor(0.0))
            self.register_parameter("null_bias", None)
            self.null_head = None
        elif approach == "learned_bias":
            self.register_buffer("null_constant", None)
            self.null_bias = nn.Parameter(torch.tensor(0.0))
            self.null_head = None
        else:
            self.register_buffer("null_constant", None)
            self.register_parameter("null_bias", None)
            self.null_head = nn.Sequential(
                nn.Linear(hidden_size * 3, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        if freeze_span_model:
            self.span_model.requires_grad_(False)
            self.span_model.eval()

    def train(self, mode: bool = True) -> NullAntecedentModel:
        super().train(mode)
        if self.freeze_span_model:
            self.span_model.eval()
        return self

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        output = self.span_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            context_mask=batch["context_mask"],
            relations=batch["relations"],
            relation_mask=batch["relation_mask"],
        )
        relation_outputs = []
        for relation in output["relations"]:
            if self.approach == "constant":
                null_score = self.null_constant.to(relation["scores"].device)
            elif self.approach == "learned_bias":
                if self.null_bias is None:
                    raise RuntimeError("learned_bias is missing its scalar parameter")
                null_score = self.null_bias
            else:
                if self.null_head is None:
                    raise RuntimeError("conditioned approach is missing its NULL head")
                anaphor = relation["anaphor_repr"]
                context = relation["context_repr"]
                features = torch.cat((anaphor, context, anaphor * context), dim=-1)
                null_score = self.null_head(features).squeeze(-1)

            candidate_scores = relation["scores"]
            all_scores = torch.cat((candidate_scores, null_score.reshape(1)))
            probabilities = torch.softmax(all_scores, dim=0)
            relation_outputs.append({
                **relation,
                "null_score": null_score,
                "all_scores": all_scores,
                "probabilities": probabilities,
                "predicted_is_null": bool(
                    all_scores.argmax().item() == candidate_scores.numel()
                ),
            })

        return {**output, "relations": relation_outputs}


def marginal_link_null_loss(
    relation_outputs: list[Mapping[str, Any]],
    antecedent_spans: torch.Tensor,
    antecedent_mask: torch.Tensor,
    is_null: torch.Tensor,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Marginal negative log likelihood over LINK alternatives plus NULL."""
    if antecedent_spans.ndim != 4 or antecedent_spans.shape[-1] != 2:
        raise ValueError(
            "antecedent_spans must have shape [batch, relations, alternatives, 2]"
        )
    if antecedent_mask.shape != antecedent_spans.shape[:3]:
        raise ValueError("antecedent_mask must match antecedent_spans")
    if is_null.shape != antecedent_spans.shape[:2]:
        raise ValueError("is_null must have shape [batch, relations]")

    losses: list[torch.Tensor] = []
    excluded = 0
    for output in relation_outputs:
        batch_index = int(output["batch_index"])
        relation_index = int(output["relation_index"])
        candidates = output["candidate_spans"]
        scores = output["scores"]
        null_score = output["null_score"]
        if not all(isinstance(value, torch.Tensor) for value in (candidates, scores, null_score)):
            raise TypeError("candidate spans and scores must be tensors")
        if scores.ndim != 1 or scores.numel() != candidates.shape[0]:
            raise ValueError("Each candidate span must have exactly one score")

        logits = torch.cat((scores, null_score.reshape(1)))
        denominator = torch.logsumexp(logits, dim=0)
        if bool(is_null[batch_index, relation_index].item()):
            gold_indices = antecedent_mask[
                batch_index, relation_index
            ].to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
            if gold_indices.numel():
                raise ValueError("NULL supervision cannot include gold antecedent spans")
            losses.append(denominator - null_score)
            continue

        alternative_indices = antecedent_mask[
            batch_index, relation_index
        ].to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
        if alternative_indices.numel() == 0:
            raise ValueError("LINK supervision requires at least one gold antecedent")
        gold = antecedent_spans[batch_index, relation_index, alternative_indices]
        matches = (candidates[:, None, :] == gold[None, :, :]).all(dim=-1).any(dim=-1)
        if not matches.any():
            excluded += 1
            continue
        losses.append(
            denominator
            - torch.logsumexp(scores.masked_fill(~matches, -torch.inf), dim=0)
        )

    if not losses:
        return zero, 0, excluded
    return torch.stack(losses).mean(), len(losses), excluded
