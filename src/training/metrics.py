"""Relation-weighted evaluation metrics for antecedent localization.

Hit@1 measures whether the best context token belongs to any annotated
antecedent. It does not measure exact span extraction, anaphor detection, or
end-to-end coreference resolution.
"""

import torch
from torchmetrics import Metric

from src.models.losses import _iter_relation_targets


def coreference_hit_counts(
    logits: torch.Tensor,
    pair_mask: torch.Tensor,
    relations: torch.Tensor,
    antecedent_spans: torch.Tensor,
    relation_mask: torch.Tensor,
    antecedent_mask: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Return token-correct, token-count, relation-correct, relation-count."""
    token_correct = 0
    token_count = 0
    relation_correct = 0
    relation_count = 0

    for target in _iter_relation_targets(
        logits,
        pair_mask,
        relations,
        antecedent_spans,
        relation_mask,
        antecedent_mask,
    ):
        rows = logits[target.batch_index, target.query_positions]
        valid_by_query = target.valid_context_by_query

        for row, valid_context in zip(rows, valid_by_query):
            best_context = row.masked_fill(~valid_context, -torch.inf).argmax()
            token_correct += int(target.gold_context[best_context].item())
            token_count += 1

        # Mean each candidate's score over the anaphor tokens for which that
        # candidate is valid; context validity is normally identical per row.
        valid_counts = valid_by_query.sum(dim=0)
        valid_context = valid_counts > 0
        mean_scores = (
            rows.masked_fill(~valid_by_query, 0.0).sum(dim=0)
            / valid_counts.clamp_min(1)
        )
        best_context = mean_scores.masked_fill(
            ~valid_context,
            -torch.inf,
        ).argmax()
        relation_correct += int(target.gold_context[best_context].item())
        relation_count += 1

    return token_correct, token_count, relation_correct, relation_count


class CoreferenceMetrics(Metric):
    """Accumulate Hit@1 and relation-weighted loss across any number of batches."""

    full_state_update = False

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "token_correct",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "token_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "relation_correct",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "relation_count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "weighted_loss",
            # Float32 remains supported by MPS and is sufficient for the
            # relation-count-weighted epoch loss accumulator.
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        logits: torch.Tensor,
        pair_mask: torch.Tensor,
        relations: torch.Tensor,
        antecedent_spans: torch.Tensor,
        relation_mask: torch.Tensor,
        antecedent_mask: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        token_correct, token_count, relation_correct, relation_count = (
            coreference_hit_counts(
                logits,
                pair_mask,
                relations,
                antecedent_spans,
                relation_mask,
                antecedent_mask,
            )
        )
        self.token_correct += token_correct
        self.token_count += token_count
        self.relation_correct += relation_correct
        self.relation_count += relation_count
        if relation_count:
            self.weighted_loss += (
                loss.detach().to(dtype=self.weighted_loss.dtype) * relation_count
            )

    def compute(self) -> dict[str, torch.Tensor]:
        """Return zero for empty splits and exact totals otherwise."""
        zero = self.weighted_loss.new_zeros(())
        return {
            "loss": torch.where(
                self.relation_count > 0,
                self.weighted_loss / self.relation_count.clamp_min(1),
                zero,
            ).to(dtype=torch.float32),
            "token_hit_at_1": torch.where(
                self.token_count > 0,
                self.token_correct / self.token_count.clamp_min(1),
                zero,
            ).to(dtype=torch.float32),
            "relation_hit_at_1": torch.where(
                self.relation_count > 0,
                self.relation_correct / self.relation_count.clamp_min(1),
                zero,
            ).to(dtype=torch.float32),
        }