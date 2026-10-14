# Experiment 001 — Tiny-Set Overfitting

## Objective

Verify that the encoder-based cross-boundary coreference model can successfully memorize a small set of annotated examples.

This experiment is intended to validate the complete training pipeline before running experiments on the full dataset.

**Primary hypothesis:** Given 16 training examples containing valid anaphor–antecedent relations, the model should be capable of achieving near-zero training loss and approximately 100% training Relation Hit@1.

This is a training sanity check, **not a generalization experiment**.

## Background

The project implements an encoder-only coreference resolution architecture consisting of:

- Hugging Face transformer encoder (default: `distilbert-base-uncased`).
- Projected token-to-token affinity head.
- Supervised loss maximizing probability mass over gold antecedent tokens.
- Support for multiple valid antecedent mentions per anaphor.
- PyTorch Lightning training and evaluation.
- Hugging Face dataset integration through `CoreferenceDataModule`.

The available dataset contains 1,000 training examples, 196 development examples, and 36 manually constructed test examples.

The model receives gold anaphor spans during training and evaluation. Anaphor detection is not part of this experiment.

## Scope

Implement and run a reproducible tiny-set overfitting experiment using the existing training infrastructure.

Avoid modifying the model architecture, loss function, or annotation rules unless a demonstrable implementation bug is discovered.

## Task Requirements

### 1. Create a reproducible experiment script

Create:

`scripts/experiments/exp001_tiny_overfit.py`

The script must:

1. Load the existing dataset and tokenizer.
2. Apply the established annotation filtering and preprocessing pipeline.
3. Select exactly 16 training examples containing at least one valid, token-aligned relation.
4. Use a fixed random seed (`42`) for example selection and training.
5. Use the same 16 examples for training and evaluation.
6. Initialize the existing LightningModule and model.
7. Train using PyTorch Lightning.
8. Save experiment results and configuration.

The 16 selected example IDs must be saved to the experiment output directory so the same subset can be reconstructed later.

Do not select the first 16 rows blindly. Sample deterministically from eligible training examples.

Ensure the experiment does not accidentally include examples with zero valid relations.

### 2. Training configuration

Use the following initial configuration:

| Parameter | Value |
|---|---|
| Encoder | `distilbert-base-uncased` |
| Projection dimension | 128 |
| Random seed | 42 |
| Training examples | 16 |
| Batch size | 4 |
| Learning rate | `1e-4` |
| Weight decay | 0 |
| Dropout | 0, where configurable |
| Maximum epochs | 100 |
| Gradient clipping | 1.0 |
| Precision | 32-bit |
| Validation dataset | Same 16 training examples |
| Checkpointing | Best training-loss checkpoint |

Use the existing optimizer and training module wherever possible.

The higher learning rate and disabled regularization are intentional: the objective is memorization rather than generalization.

Make the experiment configurable through CLI arguments or a configuration file, but preserve these defaults.

### 3. Training monitoring

Track the following metrics:

- Training loss
- Training Token Hit@1
- Training Relation Hit@1
- Evaluation loss on the same 16 examples
- Evaluation Token Hit@1
- Evaluation Relation Hit@1

If the LightningModule currently logs only training loss, extend it minimally to support the training metrics.

**Important:** Training metrics must be aggregated over the full 16-example subset, not inferred from the final training batch.

Use deterministic evaluation with `model.eval()` and disabled gradient computation.

### 4. Success criteria

The primary success criteria are:

| Metric | Target |
|---|---|
| Training Relation Hit@1 | ≥ 99% |
| Training Token Hit@1 | ≥ 99% |
| Training loss | ≤ 0.05 |
| Evaluation Relation Hit@1 on same subset | ≥ 99% |
| Evaluation Token Hit@1 on same subset | ≥ 99% |

These are experimental targets, not guaranteed outcomes.

Because the dataset is small, aim for perfect antecedent localization. Report exact numerator/denominator counts alongside percentages.

If the targets are not reached after 100 epochs, record the outcome rather than manipulating results.

### 5. Diagnostics if overfitting fails

If the model fails to memorize the 16 examples, investigate systematically:

1. Confirm that the gold antecedent spans map to valid context-token positions.
2. Confirm that gold anaphor tokens map to valid chunk-token positions.
3. Verify that the pair mask includes every gold token pair.
4. Confirm that the loss decreases when gold antecedent logits are increased.
5. Verify nonzero, finite gradients for both encoder and affinity head.
6. Check for NaN/Inf losses, gradients, or logits.
7. Inspect predictions for a few failed relations.
8. Check whether the model is repeatedly predicting the same context positions.

For failed relations, report:

- Example ID
- Anaphor text
- Gold antecedent text(s)
- Predicted context token
- Rank of the highest-scoring gold antecedent token

Use tokenizer offsets and original text to reconstruct readable predictions where possible.

Do not introduce architectural changes during this experiment without documenting the reason.

### 6. Experiment outputs

Store outputs under:

`experiments/exp001_tiny_overfit/`

Suggested contents:

```text
experiments/exp001_tiny_overfit/
├── config.yaml
├── selected_ids.json
├── metrics.json
├── predictions.json
└── checkpoints/
    └── best.ckpt
```

Use the project's existing DVCLive/DVC infrastructure if configured. Do not introduce a second experiment-tracking system.

Save:

- Exact configuration and random seed
- Selected example IDs
- Best and final epoch metrics
- Number of epochs completed
- Best checkpoint path
- Git commit hash, if available
- Runtime and device information
- Any warnings or failures

Do not commit large model checkpoints to Git.

### 7. Update `EXPERIMENTLOG.md`

**This is a mandatory deliverable.**

Append a concise entry to the existing `EXPERIMENTLOG.md` file. Preserve its existing format and previous entries.

The entry must contain:

**Experiment:** EXP-001 — Tiny-Set Overfitting

**Description:** A short explanation (2–4 sentences) of the experiment's objective, architecture, selected subset, and training setup.

**Hypothesis:** The model can memorize 16 annotated examples and achieve near-perfect antecedent localization.

**Configuration:** Record the most important hyperparameters, seed, dataset size, and relevant Git commit.

**Results:** Report actual measured training loss, Token Hit@1, Relation Hit@1, best epoch, and whether success criteria were achieved.

**Observations:** Write 3–5 concise, evidence-based observations about training behavior, convergence, masking, predictions, or failure modes. Do not invent observations or describe unmeasured outcomes as facts.

**Conclusion:** State whether the overfitting sanity check passed, partially passed, or failed.

**Next steps:** Recommend whether to proceed to full-dataset training or investigate specific issues first.

If the experiment cannot be completed, still append an entry describing what ran, the blocker, and the available observations.

Example structure (replace placeholders with actual results):

```markdown
## EXP-001 — Tiny-Set Overfitting

**Description:** Tested whether the encoder affinity model can
memorize 16 deterministically selected training examples using
gold anaphor supervision.

**Hypothesis:** The model should achieve near-zero loss and
approximately 100% antecedent localization accuracy.

**Configuration:** DistilBERT, projection_dim=128, seed=42,
batch_size=4, lr=1e-4, max_epochs=100.

**Results:**
- Best epoch: ...
- Training loss: ...
- Training Token Hit@1: ...
- Training Relation Hit@1: ...
- Success criteria: PASS / PARTIAL / FAIL

**Observations:**
- ...
- ...
- ...

**Conclusion:** ...

**Next steps:** ...
```

Keep the entry short and useful for comparing future experiments.

## Acceptance Criteria

- [ ] Experiment script created and runnable from the repository root.
- [ ] Exactly 16 eligible training examples selected deterministically.
- [ ] Selected example IDs saved.
- [ ] Same examples used for training and evaluation.
- [ ] Fixed seed and configuration recorded.
- [ ] Existing model, loss, and DataModule components reused.
- [ ] Training and evaluation metrics collected over the full subset.
- [ ] Multiple antecedents handled correctly.
- [ ] Training completes for up to 100 epochs without NaN/Inf.
- [ ] Best checkpoint saved.
- [ ] Configuration and measured results saved.
- [ ] Failed predictions inspected if success targets are missed.
- [ ] `EXPERIMENTLOG.md` updated with description, results, observations, and conclusion.
- [ ] Existing unit tests continue passing.
- [ ] No test-set examples used for training or hyperparameter tuning.

## Deliverables

1. `scripts/experiments/exp001_tiny_overfit.py`
2. Experiment configuration and selected example IDs
3. Training metrics and checkpoint
4. Prediction diagnostics
5. Updated `EXPERIMENTLOG.md`
6. Brief execution summary stating whether the sanity check passed

## Definition of Done

The experiment has been executed, its results are reproducible, and its outcome is documented in `EXPERIMENTLOG.md`.

**The task is complete even if the model fails to overfit**, provided the failure is measured, investigated, and documented.

Do not proceed to full-dataset training as part of this task.