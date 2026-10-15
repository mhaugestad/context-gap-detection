# EXP-005 — Null Antecedent Prediction

## 1. Objective

Extend the existing encoder-only, anaphor-conditioned span-pair coreference model to support **explicit NULL antecedent prediction**.

The current model assumes that every supplied gold anaphor has at least one valid antecedent in the provided context. It therefore always predicts a context span.

EXP-005 will investigate whether the model can learn to distinguish between:

1. **LINK:** The supplied anaphor has a valid antecedent span in the provided context.
2. **NULL:** The supplied anaphor has no valid antecedent anywhere in the provided context.

The experiment must compare three NULL-scoring mechanisms while preserving the existing span-pair architecture.

**Primary research question:**

Can a lightweight encoder-only model learn to abstain from linking an anaphor when its antecedent is absent, without substantially degrading antecedent span extraction on positive examples?

This is an architectural proof of concept. Do not scale the training dataset or perform extensive data grooming as part of this experiment.

---

## 2. Scope and Definitions

### LINK

An anaphor is LINK when at least one valid antecedent span exists within the supplied context.

Example:

**Context:** Dr. Marcus Holloway was appointed director of the National Archaeological Institute.

**Chunk:** He will take up the position in September.

**Gold anaphor:** `He`

**Gold output:** `Dr. Marcus Holloway`

### NULL

An anaphor is NULL when it is a valid anaphoric expression but **no valid antecedent exists anywhere in the supplied context**.

Example:

**Context:** The National Archaeological Institute announced a new research programme.

**Chunk:** He will take up the position in September.

**Gold anaphor:** `He`

**Gold output:** `NULL`

NULL does not mean that the expression is non-referential, generic, or not an anaphor.

It means that the antecedent is unavailable within the provided context.

### Out of scope

Do not implement:

- Automatic anaphor detection.
- Referentiality classification.
- Generic-expression detection.
- End-to-end coreference resolution.
- Generative span extraction.
- New encoder architectures.
- Large-scale data collection.

All examples must continue to use supplied gold anaphor spans.

---

## 3. Background

### EXP-002 — Antecedent token localization

The best DistilBERT affinity checkpoint achieved:

- Dev Relation Hit@1: 101/152 (66.45%).
- Dev Token Hit@1: 191/336 (56.85%).

### EXP-003 — Behavioral evaluation

The frozen EXP-002 checkpoint achieved:

- Behavioral Relation Hit@1: 25/32 (78.12%).
- Behavioral Token Hit@1: 46/60 (76.67%).

Negative-control cases were unsupported because the model had no mechanism for predicting that an antecedent was absent.

### EXP-004 — Antecedent span extraction

Three approaches were evaluated on 152 development relations:

| Approach | Span EM | Token F1 | Localization Hit |
|---|---:|---:|---:|
| Frozen affinity expansion | 11.18% | 0.3642 | 66.45% |
| Boundary head | 50.00% | 0.5860 | 69.08% |
| Span-pair scoring | 53.29% | 0.6012 | 65.13% |

The span-pair model achieved the highest development Exact Span Match.

Use this architecture as the foundation for EXP-005.

Reference checkpoint:

`experiments/exp004_span_extraction/span_pair/checkpoints/best.ckpt`

SHA-256:

`2ee1b7046bbddc298e6b394d6470f093a13380fc812b7a1c1a92f06f021b1be5`

---

## 4. Hypotheses

### H1 — NULL prediction is learnable

Adding an explicit NULL candidate allows the model to distinguish LINK from NULL substantially better than an always-LINK baseline.

**Assessment:**

Compare LINK/NULL classification performance against the always-LINK baseline on a balanced, held-out development evaluation.

H1 is supported if the best NULL-enabled model improves both balanced accuracy and macro F1 over the always-LINK baseline.

Report the actual differences and uncertainty estimates.

### H2 — Context-conditioned NULL scoring is beneficial

A NULL score conditioned on the anaphor and context will outperform a constant or globally learned NULL score.

**Assessment:**

Compare all three approaches on identical development examples.

H2 is supported if the conditioned NULL model achieves the highest LINK/NULL macro F1 and end-to-end exact correctness.

If the metrics disagree, report H2 as mixed or inconclusive.

### H3 — Span extraction can be preserved

Adding NULL prediction does not substantially degrade span extraction on examples where an antecedent exists.

**Assessment:**

Compare positive-only Span EM and Token F1 against the frozen EXP-004 span-pair checkpoint.

Use a provisional engineering tolerance of **5 percentage points** for Span EM degradation.

This threshold is an experiment-specific criterion, not a statistically established equivalence margin.

Report both:

- Conditional span quality when forced to select a span.
- Actual span correctness after NULL decisions.

### H4 — The model uses antecedent availability

The model can distinguish LINK and NULL for the same anaphor and chunk when only the supplied context changes.

**Assessment:**

Evaluate controlled positive/negative context pairs.

H4 is supported descriptively if the model correctly changes its decision from LINK to NULL more frequently than the baselines.

Report paired consistency and failure examples.

---

## 5. Dataset Audit

Before implementing new training behavior, audit the existing dataset.

Dataset:

`mhaugestad/contextualise-context`

Existing splits:

- Train: 1,000 examples.
- Dev: 196 examples.
- Test: 36 examples.

EXP-004 used:

- Train: 699 valid positive relations.
- Dev: 152 valid positive relations.

The reserved test split must not be loaded during EXP-005.

### Audit requirements

Determine:

1. How many examples contain valid positive relations.
2. How many examples contain no annotated relations.
3. How many relations are removed by annotation filtering.
4. How many are removed by token alignment.
5. Whether any existing examples explicitly annotate an anaphor with no antecedent.
6. Whether the existing annotation format can represent a NULL relation.

**Important:** Do not automatically interpret zero-relation examples as NULL.

An example without a valid relation may contain no anaphor, missing annotations, or invalid annotations.

Only examples with an explicitly identified anaphor and a verified absent antecedent may be used as NULL supervision.

Save the audit results before training.

---

## 6. Controlled Negative Data Construction

Because the existing dataset primarily contains positive coreference relations, construct a small, controlled negative dataset from existing positive examples.

Do not generate new annotations using an LLM.

### 6.1 Positive examples

Retain the original context/chunk pairs and gold antecedent spans.

Each valid gold relation becomes a positive training instance.

For examples containing multiple anaphors, retain the relation-specific labels.

### 6.2 Negative construction methods

Implement two negative-generation strategies.

#### Strategy A — Context removal

Construct a negative by replacing the context with a short, unrelated or non-identifying context while keeping the chunk and anaphor unchanged.

Example:

Original:

**Context:** Dr. Marcus Holloway was appointed director.

**Chunk:** He will begin in September.

**Label:** LINK

Modified:

**Context:** The institute announced a new research programme.

**Chunk:** He will begin in September.

**Label:** NULL

The replacement context must not contain an antecedent for the target anaphor.

#### Strategy B — Hard context replacement

Replace the original context with another context from the same dataset containing superficially plausible but incorrect antecedent candidates.

Example:

**Context:** Professor Elena Torres was appointed director of the institute.

**Chunk:** He will begin in September.

**Label:** NULL

This creates a more difficult negative than unrelated-context replacement.

However, the replacement must be checked carefully: a plausible but incorrect antecedent must not accidentally become a valid referent.

### 6.3 Negative validity

Synthetic negatives are only useful if their NULL labels are credible.

Implement a conservative validation policy:

- Preserve the original anaphor and chunk text.
- Reject replacements containing the original gold antecedent mention.
- Reject replacements containing known alternative gold antecedent mentions.
- Prefer replacements with no obvious valid referent for the target anaphor.
- Exclude ambiguous cases.
- Record the negative-generation strategy and source example IDs.

Lexical exclusion alone is not sufficient to prove that an antecedent is absent.

For hard negatives, create a small manually reviewable audit set and explicitly record any uncertain labels.

### 6.4 Leakage prevention

Split source examples before generating negatives.

All synthetic variants of a source example must remain in the same split.

Never use a development source example to construct a training negative.

Do not use the reserved test split as a negative-context pool.

When sampling replacement contexts, use only contexts from the corresponding split.

Record the replacement source ID.

### 6.5 Class balance

For the initial architecture comparison, target:

- One positive instance per valid gold relation.
- One synthetic negative instance per positive instance.

Use an approximately balanced LINK/NULL training dataset.

Create a balanced development evaluation using the same source-split isolation rules.

Also report performance separately by negative-generation strategy.

Do not assume the balanced class distribution represents real-world deployment conditions.

---

## 7. Model Architecture

Reuse the EXP-004 span-pair scorer.

The model should continue to:

1. Encode the context and chunk jointly.
2. Extract the gold anaphor representation.
3. Enumerate candidate antecedent spans in the context.
4. Score each candidate span.
5. Select the best candidate.

EXP-005 adds one additional candidate:

`NULL`

### Shared scoring formulation

Let:

- `a` be the anaphor representation.
- `c` be the encoded context.
- `S(c)` be the set of valid context spans.
- `f(a, s, c)` be the existing span-pair score.
- `g(a, c)` be the NULL score.

The candidate scores are:

```python
scores = [
    span_score_1,
    span_score_2,
    ...,
    span_score_n,
    null_score,
]
```

Prediction:

```python
prediction = argmax(scores)
```

If the winning candidate is NULL, return no antecedent span.

Otherwise, return the selected span and its character offsets.

Use a single encoder forward pass per example, including examples containing multiple anaphor relations.

---

## 8. Architectural Variants

Compare three approaches.

### Approach A — Constant NULL score

Use a fixed NULL score competing against the existing span scores.

```python
null_score = 0.0
```

The score is not trainable.

This tests whether the existing span-score distribution already supports abstention.

Use the frozen EXP-004 span-pair checkpoint.

Do not retrain the encoder or span scorer.

The default fixed score must be recorded before evaluation.

If exploring additional fixed thresholds, do so on a separate calibration subset and report the default separately.

### Approach B — Learned NULL bias

Introduce a single trainable scalar:

```python
null_score = nn.Parameter(torch.tensor(0.0))
```

The NULL score is global and does not depend directly on the individual anaphor or context.

Initialize the span scorer from the EXP-004 checkpoint.

Freeze the encoder and span-pair scoring head for the primary comparison.

Train only the NULL bias on the synthetic LINK/NULL training set.

This isolates the contribution of learning a global NULL score.

### Approach C — Context-conditioned NULL head

Introduce a trainable NULL head conditioned on the anaphor and context representations.

Suggested architecture:

```python
anaphor_rep = mean_pool(
    hidden_states[anaphor_start:anaphor_end]
)

context_rep = masked_mean_pool(
    hidden_states,
    context_mask
)

null_features = torch.cat(
    [
        anaphor_rep,
        context_rep,
        anaphor_rep * context_rep,
    ],
    dim=-1,
)

null_score = null_head(null_features)
```

The NULL head may be a small MLP.

Suggested structure:

```python
Linear(3 * hidden_size, 128)
GELU()
Dropout(0.1)
Linear(128, 1)
```

Initialize the encoder and span scorer from the EXP-004 checkpoint.

For the primary architecture comparison, freeze the encoder and span-pair scorer and train only the NULL head.

This allows a controlled comparison between:

- Fixed NULL score.
- Learned global NULL score.
- Learned context-conditioned NULL score.

### Optional fine-tuning ablation

After completing the frozen-backbone comparison, optionally unfreeze the full model for Approach C.

Use a lower learning rate for the pretrained encoder.

Report this separately as:

`conditioned_null_finetuned`

Do not mix fine-tuned and frozen-backbone results in the primary architecture comparison.

---

## 9. Training Objective

For each supplied gold anaphor, the model predicts a distribution over:

- All valid candidate antecedent spans.
- One NULL candidate.

### Positive relations

For LINK examples, the gold target consists of all valid gold antecedent spans.

Use marginal log likelihood over the valid gold alternatives:

\[
\mathcal{L}_{LINK}
=
-\log
\frac{\sum_{s\in G}\exp f(a,s,c)}
{\exp g(a,c)+\sum_{s'\in S(c)}\exp f(a,s',c)}
\]

where `G` contains the valid gold antecedent spans.

### Negative relations

For NULL examples:

\[
\mathcal{L}_{NULL}
=
-\log
\frac{\exp g(a,c)}
{\exp g(a,c)+\sum_{s'\in S(c)}\exp f(a,s',c)}
\]

Implement both objectives using numerically stable log-sum-exp operations.

### Training requirements

- Preserve multiple valid gold antecedent alternatives.
- Mask invalid candidate spans.
- Support variable numbers of relations per example.
- Support NULL-labeled relations with no gold antecedent spans.
- Avoid treating padded relations as training targets.
- Do not infer NULL labels from empty annotation arrays without explicit supervision.
- Ensure gradients reach the NULL head.
- In frozen variants, verify that the encoder and span scorer receive no parameter updates.

---

## 10. Candidate-Count Calibration

The number of candidate spans depends on context length.

A NULL score competes against all candidate span scores, so the number and distribution of candidates may influence NULL predictions.

Investigate this explicitly.

For each evaluation relation, record:

- Context token count.
- Number of candidate spans.
- Highest non-NULL span score.
- NULL score.
- NULL probability.
- Predicted class.

Report LINK/NULL performance by context-length band.

Suggested bands:

- 1–32 context tokens.
- 33–64.
- 65–128.
- 129+.

Also compare performance across candidate-count quantiles where sample sizes permit.

Do not interpret the softmax NULL probability as calibrated without checking calibration.

As an exploratory diagnostic, evaluate whether a candidate-count-adjusted NULL score improves performance on a held-out calibration subset.

Do not add the adjustment to the main model unless the experiment provides evidence that it is useful.

---

## 11. Evaluation Metrics

Evaluate all approaches on identical development examples.

### 11.1 LINK/NULL classification

Report:

- LINK precision.
- LINK recall.
- LINK F1.
- NULL precision.
- NULL recall.
- NULL F1.
- Macro F1.
- Balanced accuracy.
- Confusion matrix.

Include an always-LINK baseline.

### 11.2 End-to-end exact correctness

For each example:

- LINK is correct only if the predicted antecedent span exactly matches one of the gold spans.
- NULL is correct only if the model predicts NULL.

Report:

`End-to-end exact accuracy = correct outcomes / all evaluated relations`

This is the main combined metric.

### 11.3 Positive-only span extraction

On original positive development relations, report:

- Forced-LINK Exact Span Match.
- Forced-LINK Token F1.
- Actual Exact Span Match after NULL decisions.
- Actual Token F1 after NULL decisions.
- False-NULL rate.

Forced-LINK evaluation ignores the NULL candidate and selects the highest-scoring valid context span.

This distinguishes changes in span quality from changes in LINK/NULL decisions.

### 11.4 Negative-only performance

Report:

- NULL recall.
- False-LINK rate.
- Performance by negative-generation strategy.
- Performance by context-length band.

### 11.5 Paired contrast evaluation

For each original positive example and its synthetic negative counterpart, evaluate whether the model produces:

- LINK on the original.
- NULL on the modified context.

Report the proportion of pairs where both decisions are correct.

Also report pairs where:

- Both are predicted LINK.
- Both are predicted NULL.
- The positive is NULL and the negative is LINK.

### 11.6 Calibration

Report:

- Brier score for NULL probability.
- Reliability diagram or tabular calibration bins.
- Predicted NULL rate by candidate-count band.

Treat calibration analysis as exploratory.

---

## 12. Experimental Protocol

### Stage 1 — Dataset audit

Run the annotation audit and document which examples can support LINK/NULL supervision.

Do not proceed by automatically labeling all zero-relation examples as negative.

### Stage 2 — Synthetic negative construction

Generate controlled negatives from training and development source examples separately.

Create a small manual review set covering both easy and hard negatives.

Record ambiguous or rejected cases.

### Stage 3 — Baseline evaluation

Evaluate:

- Always-LINK.
- Frozen EXP-004 span-pair model with constant NULL score.

Do not train either baseline.

### Stage 4 — Learned NULL bias

Freeze the EXP-004 model.

Train the global NULL scalar.

Evaluate on the development set.

### Stage 5 — Conditioned NULL head

Freeze the EXP-004 model.

Train the conditioned NULL head.

Evaluate on the same development examples.

### Stage 6 — Optional fine-tuning

If the frozen conditioned NULL head shows promise, optionally unfreeze the model and fine-tune jointly.

Report this as an additional experiment variant.

### Stage 7 — Comparison and error analysis

Compare all approaches using:

- LINK/NULL classification.
- End-to-end exact correctness.
- Positive-only Span EM.
- Negative-only false-LINK rate.
- Paired contrast accuracy.

### Stage 8 — Experiment logging

Write results, diagnostics, limitations, and conclusions to `EXPERIMENTLOG.md`.

---

## 13. Training Configuration

Suggested initial configuration:

| Parameter | Value |
|---|---|
| Encoder | `distilbert-base-uncased` |
| Base checkpoint | EXP-004 span-pair best |
| Seed | 42 |
| Batch size | 16 |
| NULL-head learning rate | `1e-3` |
| Weight decay | `0.01` |
| Dropout | `0.1` |
| Max epochs | 20 |
| Early stopping patience | 4 |
| Gradient clipping | 1.0 |
| Precision | 32-bit |
| Maximum span length | 32 tokens |
| Positive:negative ratio | Approximately 1:1 |

Select checkpoints using development LINK/NULL macro F1.

Use development end-to-end exact correctness as the tie-breaker.

Record best epochs and training runtime.

Do not select checkpoints using the reserved test split.

For optional full-model fine-tuning, use separately configurable learning rates for the encoder, span scorer, and NULL head.

---

## 14. Implementation

Create:

`scripts/experiments/exp005_null_antecedent.py`

Support:

```bash
python scripts/experiments/exp005_null_antecedent.py \
    --approach constant
```

```bash
python scripts/experiments/exp005_null_antecedent.py \
    --approach learned_bias
```

```bash
python scripts/experiments/exp005_null_antecedent.py \
    --approach conditioned
```

Also support:

```bash
python scripts/experiments/exp005_null_antecedent.py \
    --approach all
```

The script should:

1. Audit the source dataset.
2. Construct or load deterministic synthetic negatives.
3. Validate negative examples.
4. Load the EXP-004 checkpoint.
5. Initialize the selected NULL architecture.
6. Train where applicable.
7. Select and reload the best checkpoint.
8. Evaluate on the common development dataset.
9. Export relation-level predictions.
10. Produce comparison metrics.
11. Generate failure diagnostics.
12. Update `EXPERIMENTLOG.md`.

Use the existing PyTorch Lightning infrastructure.

Preserve existing model interfaces where practical.

---

## 15. Required Tests

Add offline unit tests covering:

1. Explicit NULL relation representation.
2. Positive relation preservation.
3. Negative example construction.
4. Source-split isolation.
5. No development-to-training leakage.
6. No use of the reserved test split.
7. Multiple anaphor relations per example.
8. Multiple alternative gold antecedents.
9. Correct candidate-span masking.
10. NULL candidate always present.
11. Numerically stable LINK loss.
12. Numerically stable NULL loss.
13. Differentiable loss.
14. Frozen encoder receives no gradients.
15. Frozen span scorer receives no gradients.
16. NULL head receives gradients.
17. Constant NULL score remains fixed.
18. Correct LINK/NULL predictions.
19. Correct confusion-matrix metrics.
20. Correct end-to-end exact correctness.
21. Correct forced-LINK evaluation.
22. Correct paired contrast evaluation.
23. Correct character-offset reconstruction.
24. Checkpoint reload reproduces predictions.
25. Existing tests continue passing.

Tests must not require downloading pretrained weights.

---

## 16. Output Artifacts

Store results under:

```text
experiments/exp005_null_antecedent/
├── config.yaml
├── dataset_audit.json
├── negative_generation_report.json
├── synthetic_negative_manifest.json
├── comparison_metrics.json
├── calibration_metrics.json
├── contrast_metrics.json
├── constant/
│   ├── metrics.json
│   └── predictions.json
├── learned_bias/
│   ├── metrics.json
│   ├── predictions.json
│   └── checkpoints/
│       └── best.ckpt
├── conditioned/
│   ├── metrics.json
│   ├── predictions.json
│   └── checkpoints/
│       └── best.ckpt
└── comparison_report.md
```

Record:

- Dataset identifier and revision.
- Git commit and dirty-working-tree status.
- EXP-004 checkpoint SHA-256.
- Random seed.
- Negative-generation strategy.
- Number of positive and negative relations.
- Rejected or ambiguous synthetic negatives.
- Model configuration.
- Best epoch.
- Runtime.
- Evaluation metrics.
- Failure examples.

Do not commit large checkpoints directly to Git.

---

## 17. Error Analysis

Inspect at least 20 incorrect predictions per learned approach, or all errors if fewer than 20 exist.

Categorize failures as:

- False LINK: unrelated antecedent selected.
- False LINK: plausible but incorrect distractor.
- False NULL: valid antecedent rejected.
- Correct LINK decision, incorrect antecedent span.
- Synthetic negative annotation ambiguity.
- Candidate-count sensitivity.
- Other.

For every incorrect prediction, preserve:

- Original context.
- Chunk.
- Anaphor.
- Gold target.
- Predicted target.
- Highest-scoring span.
- NULL score.
- Candidate count.
- Negative-generation strategy, if applicable.

Pay particular attention to cases where the conditioned NULL head succeeds but the global NULL bias fails.

These examples may reveal whether contextual conditioning provides genuine value.

---

## 18. Mandatory EXPERIMENTLOG.md Update

Append an EXP-005 entry.

Preserve all previous experiments.

### Required structure

**EXP-005 — Null Antecedent Prediction**

**Description:** Summarize the architectural extension and experimental setup.

**Hypotheses:** Record H1–H4.

**Dataset:** Record positive and negative counts, construction methods, and exclusions.

**Configuration:** Record encoder, checkpoint, training parameters, and architecture variants.

**Results:**

| Approach | Macro F1 | Balanced Accuracy | E2E Exact | Positive Span EM | NULL Recall |
|---|---|---|---|---|---|
| Always-LINK | ... | ... | ... | ... | ... |
| Constant NULL | ... | ... | ... | ... | ... |
| Learned NULL bias | ... | ... | ... | ... | ... |
| Conditioned NULL | ... | ... | ... | ... | ... |

Clearly distinguish forced-LINK Span EM from actual post-NULL Span EM.

**Observations:** Include 4–8 evidence-based findings.

**Hypothesis assessment:** Supported, not supported, mixed, or inconclusive.

**Limitations:** Discuss synthetic-negative validity, class balance, candidate-count effects, and remaining lack of anaphor detection.

**Conclusion:** State whether explicit NULL scoring improves conditional antecedent resolution.

**Next steps:** Recommend the architecture to carry forward.

Do not fabricate results.

---

## 19. Acceptance Criteria

- [ ] Existing dataset audited.
- [ ] NULL semantics explicitly defined.
- [ ] No zero-relation examples automatically labeled NULL.
- [ ] Synthetic negatives generated reproducibly.
- [ ] Negative-label validity reviewed.
- [ ] Source-split leakage prevented.
- [ ] EXP-003 test split untouched.
- [ ] EXP-004 checkpoint reused.
- [ ] Always-LINK baseline evaluated.
- [ ] Constant NULL baseline evaluated.
- [ ] Learned NULL bias implemented.
- [ ] Conditioned NULL head implemented.
- [ ] Frozen-backbone comparison completed.
- [ ] Multiple gold antecedent alternatives preserved.
- [ ] LINK and NULL losses tested.
- [ ] LINK/NULL classification metrics reported.
- [ ] End-to-end exact correctness reported.
- [ ] Positive-only Span EM reported.
- [ ] False-LINK and false-NULL rates reported.
- [ ] Controlled contrast pairs evaluated.
- [ ] Candidate-count sensitivity analyzed.
- [ ] Prediction diagnostics exported.
- [ ] Experiment artifacts saved.
- [ ] EXPERIMENTLOG.md updated.
- [ ] Offline tests pass.
- [ ] Existing tests continue passing.

---

## 20. Definition of Done

EXP-005 is complete when the constant, learned-bias, and context-conditioned NULL approaches have been evaluated on identical development examples and compared against the always-LINK baseline.

The experiment must establish:

1. Whether explicit NULL scoring improves LINK/NULL decisions.
2. Whether context conditioning adds value beyond a global NULL score.
3. Whether existing antecedent-span extraction performance is preserved.
4. Whether the model can distinguish LINK and NULL for controlled context changes.
5. Whether candidate-count effects materially influence abstention.
6. Which NULL-scoring architecture should be retained for subsequent anaphor-detection experiments.

The experiment is complete even if the NULL-enabled models do not outperform the baselines, provided the comparison is reproducible and the findings are documented.

**Do not expand the dataset or implement anaphor detection during EXP-005.**