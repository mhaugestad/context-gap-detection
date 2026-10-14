# EXP-002 — Full-Dataset Training and Baseline Evaluation

## 1. Objective

Determine whether the encoder-based token-affinity model learns transferable cross-boundary coreference resolution rather than merely memorizing training examples.

Train the existing model on the full training split and evaluate it against independent development data and simple non-learned baselines.

This experiment builds directly on EXP-001, which demonstrated successful memorization of 16 examples, achieving 100% Token Hit@1 and Relation Hit@1 with near-zero training loss.

**This experiment measures generalization, not memorization.**

## 2. Hypothesis

### Primary hypothesis — H1

**A fine-tuned DistilBERT token-affinity model will achieve higher Relation Hit@1 on the held-out development set than both a random valid-context-token baseline and a recency baseline that always selects the last valid context token.**

This would provide initial evidence that the model learns transferable antecedent-localization behavior beyond simple positional heuristics.

### Null hypothesis — H0

The trained model does not outperform the stronger of the two baselines on held-out development examples.

### Secondary hypotheses

**H2 — Generalization:** Development loss will decrease during training, and development Relation Hit@1 will improve relative to the initial, untrained affinity model.

**H3 — Distance sensitivity:** Antecedent-localization accuracy will vary with the distance between the gold antecedent and the anaphor, with lower accuracy expected for more distant antecedents.

H3 is exploratory. Its purpose is to characterize positional bias, not establish a required performance threshold.

### Decision criteria

H1 is supported if the trained model achieves a higher development Relation Hit@1 than both baselines on the same eligible relations.

Also report the absolute percentage-point improvement over the stronger baseline.

Because the development set is small, treat the result as preliminary evidence rather than a statistically definitive conclusion. Report numerator/denominator counts and uncertainty intervals where practical.

## 3. Dataset

Hugging Face dataset: `mhaugestad/contextualise-context`

| Split | Original examples | Usage |
|---|---:|---|
| Train | 1,000 | Model training |
| Dev | 196 | Validation and model selection |
| Test | 36 | Reserved for EXP-003 |

Use the existing annotation filtering, preprocessor, collator, and Lightning DataModule.

### Dataset requirements

- Preserve examples with zero valid relations.
- Report the number of examples containing at least one valid relation after preprocessing.
- Report the number of valid relations and anaphor tokens in each split.
- Report dropped annotations, truncated relations, and their reasons where available.
- Support multiple antecedent mentions per relation.
- Do not use the test split for training, checkpoint selection, or hyperparameter tuning.

Zero-relation examples must not contribute to supervised loss or Hit@1 denominators.

## 4. Model and Training Configuration

Reuse the existing `CoreferenceModel` and `CoreferenceLightningModule`.

Do not change the model architecture or loss function during this experiment unless an implementation defect is identified.

### Initial configuration

| Parameter | Value |
|---|---|
| Encoder | `distilbert-base-uncased` |
| Projection dimension | 128 |
| Random seed | 42 |
| Training examples | Full train split |
| Validation examples | Full dev split |
| Batch size | 16 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Maximum epochs | 20 |
| Early stopping patience | 4 epochs |
| Early stopping monitor | `val/loss` |
| Checkpoint monitor | `val/loss` |
| Checkpoint mode | `min` |
| Gradient clipping | 1.0 |
| Precision | 32-bit |
| Dropout | Existing model default |
| Optimizer | AdamW |

Use the existing Lightning infrastructure.

Save both the best checkpoint and the final training metrics.

Record the actual number of epochs completed and the epoch selected by early stopping.

Do not automatically increase the epoch count or adjust hyperparameters based on the development results. If a configuration change becomes necessary, record it as a separate run.

## 5. Baseline Implementations

Implement two non-learned baselines and one optional initialization reference.

All methods must be evaluated on exactly the same valid development relations.

### Baseline A — Random valid-context-token selection

For each gold relation:

1. Identify all valid context tokens using `context_mask`.
2. Uniformly sample one context token.
3. Count the prediction as correct if the sampled token belongs to any gold antecedent span.

Use a fixed random seed.

To reduce sampling noise, evaluate over 1,000 independent random draws per relation, or calculate the exact expected accuracy from the fraction of valid context tokens belonging to the gold antecedent union.

Prefer the exact expectation for the primary comparison.

Report this as expected Relation Hit@1, not as an observed count of successful predictions.

### Baseline B — Recency heuristic

For each gold relation:

1. Identify all valid context tokens.
2. Select the final valid context token.
3. Count the prediction as correct if that token belongs to any gold antecedent span.

Report Relation Hit@1 using the same denominator as the trained model.

This baseline tests whether a trivial positional strategy can solve the task.

### Optional reference — Untrained model

Evaluate the randomly initialized affinity head with the pretrained encoder before fine-tuning.

Use the same development set and metrics.

This provides a reference for the contribution of supervised fine-tuning, although it is not required for testing H1.

## 6. Evaluation Metrics

### Primary metric — Relation Hit@1

For each gold relation:

1. Average affinity logits across the gold anaphor tokens.
2. Mask invalid context positions.
3. Select the highest-scoring context token.
4. Count the prediction as correct if it belongs to any gold antecedent span.

Aggregate over all valid relations in the development set.

### Secondary metrics

**Token Hit@1:** For each gold anaphor token, select the highest-scoring valid context token and check membership in the gold antecedent-token union.

**Development loss:** Compute the existing multi-antecedent probability-mass loss.

**Train–dev gap:** Compare training and development losses and Hit@1 values to identify overfitting.

**Distance analysis:** Measure the distance from the gold antecedent to the anaphor, using absolute encoder-token positions.

For multiple valid antecedents, use the nearest valid gold antecedent for the primary distance analysis.

Use the following distance bands as an initial exploratory grouping:

- 0–16 tokens
- 17–32 tokens
- 33–64 tokens
- 65–128 tokens
- More than 128 tokens

Report sample counts for each band. Do not draw conclusions from very small groups.

### Important limitations

These metrics evaluate antecedent-token localization given gold anaphor spans.

They do not measure:

- Anaphor detection.
- Exact antecedent-span extraction.
- End-to-end coreference resolution.
- Whether the model correctly identifies that no antecedent exists.

Do not describe Relation Hit@1 as end-to-end coreference accuracy.

## 7. Experiment Implementation

Create:

`scripts/experiments/exp002_full_training.py`

The script must:

1. Set all relevant random seeds.
2. Initialize the existing DataModule.
3. Load and preprocess train/dev splits.
4. Calculate dataset eligibility and relation statistics.
5. Initialize the existing LightningModule.
6. Optionally evaluate the initial model.
7. Train on the full training split.
8. Apply early stopping based on development loss.
9. Save the best checkpoint.
10. Reload the best checkpoint.
11. Evaluate the best checkpoint on the full development split.
12. Evaluate both non-learned baselines.
13. Calculate distance-stratified metrics.
14. Save configuration, metrics, and prediction diagnostics.
15. Append the experiment results to `EXPERIMENTLOG.md`.

The experiment should run from the repository root.

Use CLI arguments or configuration files for configurable parameters, while retaining the defaults above.

## 8. Experiment Tracking and Outputs

Store outputs under:

```text
experiments/exp002_full_training/
├── config.yaml
├── dataset_stats.json
├── metrics.json
├── baseline_metrics.json
├── distance_metrics.json
├── predictions.json
└── checkpoints/
    └── best.ckpt
```

Use the project's existing DVCLive/DVC setup if available.

Record:

- Dataset identifier and revision, if available.
- Git commit hash and dirty-working-tree status.
- Random seed.
- Training configuration.
- Device and runtime.
- Best checkpoint and epoch.
- Final training and development metrics.
- Baseline metrics.
- Distance-stratified results.
- Dropped-annotation and relation counts.

Do not commit large checkpoints directly to Git.

## 9. Prediction Diagnostics

Save development-set predictions at relation level.

For each relation, include:

- Example ID.
- Anaphor text.
- Gold antecedent text(s).
- Predicted context token.
- Whether the prediction was correct.
- Highest-scoring gold token rank.
- Antecedent distance.
- Whether the recency baseline was correct.

Use original text and tokenizer offset mappings to reconstruct readable predictions where possible.

Inspect at least 10 incorrect predictions, or all incorrect predictions if fewer than 10 exist.

Look for recurring errors involving:

- Recency bias.
- Multiple antecedent mentions.
- Long-distance references.
- Pronouns versus nominal anaphors.
- Semantic confusion between candidate entities.
- Tokenization or annotation alignment.

Do not infer causal explanations from individual examples without supporting evidence.

## 10. Tests

Create offline tests for the experiment-specific functionality.

Suggested file:

`tests/experiments/test_exp002.py`

Test:

- Random baseline expected accuracy.
- Recency baseline selection.
- Multiple gold antecedents counted as alternatives.
- Correct masking of padding and special tokens.
- Correct relation-level metric denominators.
- Distance calculation and bin assignment.
- Zero-relation examples excluded from metric denominators.
- Correct checkpoint reload and evaluation behavior where practical.

Reuse existing metric implementations rather than duplicating them.

All existing tests must continue passing.

## 11. Mandatory Experiment Log

Append an entry to the existing `EXPERIMENTLOG.md`.

Preserve previous entries, including EXP-001.

### Required structure

#### EXP-002 — Full-Dataset Training and Baseline Evaluation

**Description:** 2–4 sentences describing the experiment and why it was performed.

**Hypotheses:** Clearly state H1, H0, H2, and H3.

**Configuration:** Record dataset, architecture, seed, hyperparameters, epochs, checkpoint selection, and Git commit.

**Dataset statistics:** Report valid examples, relations, anaphor tokens, and dropped relations.

**Results:** Include a comparison table:

| Method | Dev Relation Hit@1 | Dev Token Hit@1 | Dev Loss |
|---|---|---|---|
| Random baseline | ... | N/A | N/A |
| Recency baseline | ... | N/A | N/A |
| Untrained model (if evaluated) | ... | ... | ... |
| Fine-tuned model | ... | ... | ... |

Use `N/A` where a metric is not applicable or was not computed.

Include best epoch, best development loss, and the improvement over the strongest baseline.

**Observations:** Write 3–6 concise, evidence-based observations about convergence, generalization, positional bias, and failure patterns.

**Hypothesis assessment:** State whether each hypothesis was supported, not supported, or inconclusive.

**Conclusion:** Summarize whether the model demonstrates meaningful held-out antecedent localization beyond the baselines.

**Next steps:** Recommend whether to proceed to EXP-003 behavioral evaluation or investigate identified weaknesses.

Do not fabricate metrics, observations, or conclusions. If execution fails, document the blocker and available evidence.

## 12. Acceptance Criteria

- [ ] Experiment script created and executable from the repository root.
- [ ] Full 1,000-example training split used, subject only to existing preprocessing rules.
- [ ] Independent 196-example development split used for validation.
- [ ] Test split remains untouched.
- [ ] Fixed seed and complete configuration recorded.
- [ ] Existing model, loss, metrics, and DataModule reused.
- [ ] Early stopping and best-checkpoint selection implemented.
- [ ] Best checkpoint reloaded for final evaluation.
- [ ] Development Relation Hit@1 and Token Hit@1 reported with counts.
- [ ] Random and recency baselines evaluated on the same eligible relations.
- [ ] Absolute improvement over the stronger baseline reported.
- [ ] Distance-stratified results produced with sample counts.
- [ ] At least 10 incorrect predictions inspected, where available.
- [ ] Experiment outputs saved reproducibly.
- [ ] `EXPERIMENTLOG.md` updated with hypotheses, results, observations, and conclusions.
- [ ] Existing tests and new offline experiment tests pass.
- [ ] No model architecture changes introduced without explicit documentation.

## 13. Definition of Done

EXP-002 is complete when the full training experiment has been executed, the best checkpoint has been evaluated on the independent development set, and the trained model has been compared with both baselines.

The experiment must produce enough evidence to assess H1, H2, and H3, with limitations documented.

**The task is complete even if the model does not outperform the baselines.** A negative result is scientifically useful if the experiment is correctly executed, reproducible, and documented.

Do not run EXP-003 or evaluate the reserved test split as part of this task.