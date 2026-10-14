# Task 02: Implement Encoder-Based Token Affinity Model

## Objective

Implement a lightweight, encoder-only neural network for cross-boundary coreference resolution.

Given a preceding `context` and subsequent `chunk`, the model must produce pairwise affinity logits indicating how strongly each chunk token refers to each context token.

The model must use a single transformer encoder forward pass and remain independent of PyTorch Lightning.

**Scope:** Model architecture, forward pass, masking, and unit tests. Do not implement the loss function or training loop.

## Background

This project investigates resolving missing context when documents are divided into chunks.

Example:

**Context:** "Barack Obama was president. The former president spoke."

**Chunk:** "He travelled to Chicago."

The model should learn that the chunk token "He" refers to the antecedent mentions "Barack Obama" and "The former president."

The existing pipeline provides:

- `src/data/annotations.py` — annotation filtering
- `src/data/preprocessor.py` — character-to-token alignment
- `src/data/collator.py` — padded batches
- `src/data/datamodule.py` — Lightning DataModule

All data-pipeline tests currently pass (29 tests).

## Architecture

### 1. Transformer encoder

Use Hugging Face `AutoModel` with a configurable pretrained encoder.

Default backbone: `distilbert-base-uncased`.

Input tensors:

- `input_ids`: `[B, T]`
- `attention_mask`: `[B, T]`
- `context_mask`: `[B, T]`
- `chunk_mask`: `[B, T]`

The encoder produces hidden states:

`H = Encoder(input_ids, attention_mask)`

Shape: `[B, T, D]`

Where `D` is the encoder hidden dimension.

### 2. Token affinity head

Implement a trainable pairwise affinity head.

Suggested baseline: projected dot-product attention.

```
Q = Wq(H)
K = Wk(H)

S = Q @ K.transpose(-1, -2) / sqrt(d)
```

Where:

- `Q` represents chunk-token queries.
- `K` represents context-token keys.
- `d` is the projection dimension.
- `S` contains raw token-to-token affinity logits.

Output shape: `[B, T, T]`.

Define the orientation explicitly:

`logits[b, i, j]` is the affinity between chunk token `i` and context token `j`.

Make the projection dimension configurable (default 128).

Use separate learnable linear projections for queries and keys.

### 3. Pairwise masking

Only chunk-to-context token pairs are valid.

Construct:

```
pair_mask[b, i, j] =
    chunk_mask[b, i] AND context_mask[b, j]
```

Shape: `[B, T, T]`.

Padding and special tokens must not be considered valid candidates.

Return the raw logits and the boolean pair mask separately. Do not replace invalid logits with negative infinity inside the model.

This allows the loss and inference code to control masking and normalization.

### 4. Forward interface

Implement a class named `CoreferenceModel` in `src/model.py`, or follow the repository's existing model organization.

Suggested interface:

```python
model = CoreferenceModel(
    encoder_name="distilbert-base-uncased",
    projection_dim=128,
)

output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    context_mask=batch["context_mask"],
    chunk_mask=batch["chunk_mask"],
)

logits = output["logits"]
pair_mask = output["pair_mask"]
```

Required output:

- `logits`: float tensor `[B, T, T]`
- `pair_mask`: bool tensor `[B, T, T]`

Do not apply sigmoid or softmax to the logits.

### 5. Configurability

Support:

- Configurable Hugging Face encoder name.
- Configurable projection dimension.
- Optional dropout.
- Encoder fine-tuning enabled by default.

Do not hardcode DistilBERT-specific hidden dimensions.

## Tests

Create `tests/test_model.py`.

Tests should run offline using a small randomly initialized transformer configuration or a mocked encoder. Do not download pretrained weights during unit tests.

Test:

1. Correct output shapes for varying batch sizes and sequence lengths.
2. Correct orientation of chunk-to-context pairs.
3. Pair masking excludes padding and special tokens.
4. Invalid pairs are marked false.
5. Raw logits remain finite.
6. Gradients propagate through the affinity head and encoder.
7. Model works with different encoder hidden dimensions.
8. No valid pairs produces a valid output without errors.

## Acceptance Criteria

- [ ] Model subclasses `torch.nn.Module`.
- [ ] Uses `transformers.AutoModel` or an equivalent configurable encoder.
- [ ] Performs one encoder forward pass.
- [ ] Implements separate query and key projections.
- [ ] Produces logits shaped `[B, T, T]`.
- [ ] Produces a boolean pair mask shaped `[B, T, T]`.
- [ ] Scores are oriented chunk-token → context-token.
- [ ] Padding and special tokens are excluded by the mask.
- [ ] No softmax or sigmoid is applied inside the model.
- [ ] No PyTorch Lightning dependency in the model implementation.
- [ ] Unit tests require no network access.
- [ ] Existing data-pipeline tests continue passing.

## Deliverables

1. Model implementation.
2. Offline model unit tests.
3. Short usage example.
4. Brief documentation of tensor shapes and masking semantics.

## Out of Scope

- Loss function
- Training loop
- Optimizer or scheduler
- Coreference metrics
- Anaphor detection
- Span-level classification
- Generative models

## Definition of Done

The model can consume a batch produced by `CoreferenceDataModule` and return correctly shaped affinity logits and masks.

A backward pass through the affinity scores must produce gradients in both the affinity head and the encoder.

The model is ready to be used by a separate Lightning training module.