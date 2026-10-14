"""Encoder-based token-to-token affinity for cross-boundary coreference."""

import math
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class CoreferenceModel(nn.Module):
    """Score chunk tokens against context tokens with one encoder pass.

    The input is the concatenated sequence produced by the data preprocessor.
    ``context_mask`` and ``chunk_mask`` identify the two token segments and
    are expected to be false for tokenizer-added special tokens. The
    ``attention_mask`` independently excludes padding.

    The returned matrix uses chunk-to-context orientation:
    ``logits[b, i, j]`` scores chunk token ``i`` against context token ``j``.
    Logits are deliberately left unnormalized and unmasked so that a
    downstream objective or inference procedure can choose its own masking
    and normalization strategy.
    """

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        projection_dim: int = 128,
        dropout: float = 0.1,
        fine_tune_encoder: bool = True,
    ) -> None:
        super().__init__()

        if projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        # Keep the backbone configurable through Hugging Face's common
        # AutoModel interface instead of coupling the head to a specific model.
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            # Some encoder configurations call this dimension ``d_model``.
            hidden_size = getattr(self.encoder.config, "d_model", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                "The encoder configuration must define a positive "
                "'hidden_size' or 'd_model'"
            )

        self.projection_dim = projection_dim
        self.dropout = nn.Dropout(dropout)

        # Separate projections let the model learn distinct representations
        # for chunk queries and context candidate keys.
        self.query_projection = nn.Linear(hidden_size, projection_dim)
        self.key_projection = nn.Linear(hidden_size, projection_dim)

        # Fine-tuning is the default, but inference or head-only experiments
        # can freeze the backbone without changing the forward interface.
        if not fine_tune_encoder:
            self.encoder.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return raw pairwise affinity logits and their validity mask.

        Each input mask has shape ``[B, T]``. The output logits and pair mask
        have shape ``[B, T, T]``; the first token axis is the chunk/query axis,
        and the second is the context/key axis.
        """
        expected_shape = input_ids.shape
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")
        for name, mask in (
            ("attention_mask", attention_mask),
            ("context_mask", context_mask),
            ("chunk_mask", chunk_mask),
        ):
            if mask.ndim != 2 or mask.shape != expected_shape:
                raise ValueError(
                    f"{name} must have the same [batch_size, sequence_length] "
                    "shape as input_ids"
                )

        # The same encoded sequence supplies both query and key vectors. This
        # is intentionally the model's only encoder invocation per batch.
        encoder_outputs: Any = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = self.dropout(encoder_outputs.last_hidden_state)

        # Rows are chunk queries and columns are context keys. Scaling by the
        # square root of the projection width controls dot-product magnitude.
        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        logits = torch.matmul(queries, keys.transpose(-1, -2))
        logits = logits / math.sqrt(self.projection_dim)

        # The preprocessing segment masks exclude special tokens; combining
        # them with attention_mask also prevents padding from becoming either
        # a query or a candidate antecedent. Keep this as a separate boolean
        # tensor rather than overwriting invalid logits with -infinity.
        attended = attention_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        )
        valid_context = context_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        ) & attended
        valid_chunk = chunk_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        ) & attended
        pair_mask = valid_chunk.unsqueeze(2) & valid_context.unsqueeze(1)

        return {"logits": logits, "pair_mask": pair_mask}