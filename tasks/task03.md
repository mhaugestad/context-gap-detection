# Task 03: Implement Coreference Loss and PyTorch Lightning Training Module

## Objective

Implement the training and evaluation logic for the encoder-based cross-boundary coreference model developed in Task 1.

The model produces chunk-token × context-token affinity logits.

The training objective is to maximize the probability assigned to valid antecedent tokens for each gold anaphor, supporting multiple alternative antecedent mentions.

**Scope:** Loss function, LightningModule, optimizer, training/validation/test steps, metrics, and tests.

## Dependencies

Task 1 must provide:

```python
output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    context_mask=batch["context_mask"],
    chunk_mask=batch["chunk_mask"],
)
```

With:

- `output["logits"]`: `[B, T, T]`
- `output["pair_mask"]`: `[B, T, T]`

The existing collator provides:

| Tensor | Shape |
|---|---|
| `relations` | `[B, R, 2]` |
| `antecedent_spans` | `[B, R, A, 2]` |
| `relation_mask` | `[B, R]` |
| `antecedent_mask` | `[B, R, A]` |

All span boundaries are absolute encoder-token indices, with exclusive end offsets.

`relations[b, r]` contains the anaphor span `[start, end]`.

`antecedent_spans[b, r, a]` contains an alternative antecedent span.

Padded spans use `-1` and must be ignored using the corresponding masks.

## 1. Loss Function

Implement a standalone PyTorch loss function, separate from the LightningModule.

Suggested location: `src/loss.py`.

### Training objective

For each gold anaphor relation:

1. Identify all chunk tokens belonging to the gold anaphor span.
2. Identify all context tokens belonging to any valid gold antecedent span.
3. Combine the antecedent spans into a union of acceptable gold context tokens.
4. Normalize affinity logits across valid context tokens.
5. Maximize the total probability mass assigned to the gold antecedent-token union.

For an anaphor token `i`, define:

```
P(j | i) = softmax_j(S[i, j])
```

The softmax denominator must include only valid context tokens.

Let `G` be the union of gold antecedent-token indices.

Then:

```
L_i = -log(sum(P(j | i) for j in G))
```

Compute this using a numerically stable log-sum-exp formulation:

```
L_i =
    logsumexp(S[i, valid_context])
    - logsumexp(S[i, gold_context])
```

For a multi-token anaphor, compute the loss for each anaphor token and average across those tokens.

Then average across valid relations in the batch.

### Important semantics

Multiple antecedent mentions with the same relation ID are **alternative correct mentions**, not separate independent classification targets.

Do not sum a separate loss for each antecedent mention.

Instead, construct a union of all acceptable antecedent tokens and compute one gold probability mass.

Overlapping antecedent spans must not double-count tokens.

### Empty and invalid supervision

Handle:

- Batches with zero valid relations.
- Examples with zero relations mixed with supervised examples.
- Padded relations.
- Padded antecedent spans.
- Multiple antecedents.
- Multi-token anaphors.
- Multi-token antecedents.
- Overlapping antecedent spans.

A zero-relation batch must produce a finite, differentiable zero loss connected to the model computation graph, for example `logits.sum() * 0.0`.

Never divide by zero.

If a supposedly valid relation has no gold antecedent tokens after masking, raise a descriptive error rather than silently producing NaN or infinity.

## 2. PyTorch Lightning Module

Implement `CoreferenceLightningModule(pl.LightningModule)`.

Suggested location: `src/module.py`.

The module should:

- Own a `CoreferenceModel` instance.
- Call the model forward pass.
- Compute the standalone coreference loss.
- Implement `training_step`.
- Implement `validation_step`.
- Implement `test_step`.
- Implement `configure_optimizers`.

Use AdamW by default.

Configurable hyperparameters:

- Encoder name
- Projection dimension
- Learning rate (default `2e-5`)
- Weight decay (default `0.01`)
- Optional dropout

Log:

- `train/loss`
- `val/loss`
- `test/loss`

Use epoch-level validation and test logging.

Ensure losses are weighted appropriately by the number of supervised relations rather than padded or zero-relation examples when aggregating epoch metrics.

## 3. Evaluation Metrics

Implement simple, interpretable metrics for the initial proof of concept.

Suggested location: `src/metrics.py`.

### Primary metric: Antecedent Token Hit@1

For each gold anaphor token:

1. Find the highest-scoring valid context token.
2. Check whether that token belongs to the union of gold antecedent tokens.

Report the proportion of correctly resolved anaphor tokens.

### Secondary metric: Relation Hit@1

For each gold relation:

1. Aggregate affinity scores across the anaphor span (mean logits).
2. Select the highest-scoring valid context token.
3. Check whether that token belongs to any gold antecedent span.

Report the proportion of correctly resolved relations.

### Metric limitations

These metrics evaluate token-level antecedent localization.

They do not evaluate exact antecedent span extraction, anaphor detection, or end-to-end coreference resolution.

Document this distinction explicitly.

### Metric aggregation

Aggregate using total correct predictions and total evaluated units.

Do not average batch percentages without accounting for different numbers of valid relations.

Log:

- `val/token_hit_at_1`
- `val/relation_hit_at_1`
- `test/token_hit_at_1`
- `test/relation_hit_at_1`

## 4. Tests

Create:

- `tests/test_loss.py`
- `tests/test_module.py`
- `tests/test_metrics.py`

Use small synthetic tensors and mocked model outputs. Tests must run without downloading model weights or accessing Hugging Face Hub.

### Loss tests

Verify:

1. Lower loss when gold antecedent logits increase relative to distractors.
2. Multiple valid antecedents are treated as alternatives.
3. Gold token unions do not double-count overlapping spans.
4. Multi-token anaphors are supported.
5. Padded relations and antecedents are ignored.
6. Zero-relation batches produce differentiable zero loss.
7. Loss remains finite for large positive and negative logits.
8. Gradients propagate to affinity logits.
9. Invalid gold supervision produces a clear exception.
10. Loss normalization is not biased by examples containing more padded relations.

### Metrics tests

Verify:

1. Correct token Hit@1.
2. Correct relation Hit@1.
3. Multiple gold antecedents count as correct alternatives.
4. Invalid context positions cannot be selected.
5. Padded relations are ignored.
6. Empty-supervision batches do not cause division by zero.
7. Aggregate metrics use correct denominators across batches.

### Lightning tests

Verify:

1. `training_step` returns a scalar loss.
2. `validation_step` and `test_step` execute.
3. Optimizer configuration works.
4. A backward pass produces gradients.
5. A minimal CPU training run completes using synthetic batches.

## 5. Integration Smoke Test

Run a small CPU-only training experiment using the existing DataModule or synthetic batches.

The smoke test should verify that:

- A batch can pass through the encoder.
- The affinity head produces logits.
- The loss accepts the batch's relation tensors.
- Backpropagation completes.
- Validation metrics are calculated.
- Lightning can complete at least one training step and one validation step.

No GPU should be required.

## Acceptance Criteria

- [ ] Standalone coreference loss implemented.
- [ ] Loss uses stable log-sum-exp calculations.
- [ ] Multiple antecedent mentions are treated as alternative gold targets.
- [ ] Loss normalizes across valid context tokens only.
- [ ] Multi-token anaphors are supported.
- [ ] Padded relations and antecedents are ignored.
- [ ] Empty-supervision batches are safe and differentiable.
- [ ] Invalid supervision fails with a descriptive error.
- [ ] LightningModule supports training, validation, and testing.
- [ ] AdamW optimizer is configurable.
- [ ] Training and evaluation losses are logged.
- [ ] Token Hit@1 and Relation Hit@1 are implemented.
- [ ] Metrics aggregate correctly across batches.
- [ ] Offline unit tests pass.
- [ ] CPU smoke test completes successfully.
- [ ] Existing model and data-pipeline tests continue passing.

## Deliverables

1. `src/loss.py`
2. `src/module.py`
3. `src/metrics.py`
4. Corresponding test files
5. Minimal CPU training smoke test
6. Brief usage example documenting how to connect the LightningModule and DataModule

## Out of Scope

- Automatic anaphor detection
- Exact antecedent span prediction
- Coreference clustering
- Event or temporal contextualization
- Hyperparameter optimization
- Distributed training
- Model publishing
- Hugging Face Spaces deployment

## Definition of Done

A Lightning Trainer can consume batches from `CoreferenceDataModule`, run the encoder affinity model, calculate a numerically stable multi-antecedent loss, backpropagate, and report validation metrics.

The system must work with zero-relation examples and variable numbers of antecedents without special-case handling in the training loop.