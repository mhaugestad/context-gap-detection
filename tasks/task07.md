# EXP-004 — Antecedent Span Extraction

## 1. Objective

Extend the existing encoder-based cross-boundary coreference system from **antecedent-token localization** to **complete antecedent-span extraction**.

The current model predicts which context token is most strongly associated with a supplied gold anaphor. EXP-004 will investigate whether the model can identify the complete antecedent expression, including its start and end boundaries.

For example:

**Context:** "Dr. Marcus Holloway was appointed director of the National Archaeological Institute."

**Chunk:** "He will take up the position in September."

**Current output:** `Marcus`

**Desired output:** `Dr. Marcus Holloway`

The experiment must compare a simple token-affinity-based baseline against two learned span-extraction architectures.

**Primary goal:** Determine whether explicit span-level supervision improves complete antecedent extraction while retaining the lightweight, encoder-only design.

This experiment is not intended to solve automatic anaphor detection, null antecedent prediction, or end-to-end coreference resolution.

---

## 2. Background

### Existing architecture

The project currently implements:

- DistilBERT encoder.
- Projected token-to-token affinity head.
- Single transformer encoder forward pass.
- Gold anaphor supervision.
- Multi-antecedent token-mass loss.
- PyTorch Lightning training infrastructure.
- Hugging Face dataset preprocessing and evaluation.

### Previous experiments

**EXP-001 — Tiny-set overfitting**

The model successfully memorized 16 examples, reaching 100% Relation Hit@1.

**EXP-002 — Full-dataset training**

The best checkpoint achieved:

- Dev Relation Hit@1: 101/152 (66.45%).
- Dev Token Hit@1: 191/336 (56.85%).
- Dev loss: 1.054341.

**EXP-003 — Behavioral evaluation**

The frozen checkpoint achieved:

- Test Relation Hit@1: 25/32 (78.12%).
- Test Token Hit@1: 46/60 (76.67%).

The behavioral evaluation identified weaknesses involving semantic/event references and competing entities.

However, these metrics only evaluate whether a predicted context token belongs to an annotated antecedent span.

They do not evaluate complete span extraction.

### Motivation

The existing token-mass objective rewards the model for identifying any token belonging to a valid antecedent.

For a gold antecedent such as:

`the historic Grandview Hotel`

the model can achieve a correct prediction by selecting only `Hotel`.

For decontextualisation, we ultimately need the complete antecedent expression.

EXP-004 will investigate whether explicit boundary or span supervision addresses this limitation.

---

## 3. Hypotheses

### Primary hypothesis — H1

**A model trained with explicit antecedent-span supervision will achieve higher development-set Exact Span Match than a span-expansion baseline built from the frozen EXP-002 token-affinity model.**

This hypothesis tests whether learning antecedent boundaries provides measurable value beyond postprocessing token-affinity predictions.

H1 is supported if at least one learned approach outperforms the frozen baseline in development Exact Span Match.

Report the absolute percentage-point improvement and the number of correctly extracted spans.

### Null hypothesis — H0

Neither learned span-extraction approach improves development Exact Span Match over the frozen EXP-002 baseline.

### Secondary hypothesis — H2

**Direct span-pair scoring will outperform independent start/end boundary prediction on development Exact Span Match.**

This tests whether jointly scoring candidate spans provides an advantage over predicting their boundaries separately.

H2 is supported if the span-pair approach achieves higher Exact Span Match than the boundary-head approach.

### Secondary hypothesis — H3

**Explicit span supervision will improve antecedent extraction without substantially degrading antecedent localization.**

Evaluate whether predicted spans still overlap the correct antecedent expressions.

Compare localization using token overlap and whether the predicted span intersects any gold antecedent span.

### Exploratory hypothesis — H4

**Span extraction errors will become more frequent as gold antecedent spans increase in length or structural complexity.**

Investigate performance by antecedent token length.

H4 is exploratory; do not impose a required performance threshold.

### Interpretation

This is a model-selection experiment using the development set.

Do not claim statistical significance solely from observed differences.

Report sample sizes, paired comparisons, and uncertainty intervals where practical.

---

## 4. Dataset

Use:

`mhaugestad/contextualise-context`

| Split | Examples | Usage |
|---|---:|---|
| Train | 1,000 | Training |
| Dev | 196 | Evaluation and model selection |
| Test | 36 | Not used in EXP-004 |

The EXP-002 preprocessing produced:

- Train: 630 examples with valid relations, 699 relations, 865 antecedent spans.
- Dev: 137 examples with valid relations, 152 relations, 183 antecedent spans.

Use the existing annotation filtering, tokenization, alignment, and collator infrastructure.

### Annotation semantics

Each relation contains:

- Exactly one gold anaphor span.
- One or more alternative gold antecedent spans.

Multiple antecedents for the same relation are alternative valid answers.

A prediction is correct if it matches **any one complete gold antecedent span**.

Do not require the model to output all alternative mentions.

Do not collapse multiple antecedent spans into one continuous region.

### Important data requirements

- Preserve original character offsets.
- Preserve token-aligned start/end offsets.
- Use half-open span boundaries: `[start, end)`.
- Preserve all valid alternative antecedents.
- Do not silently modify gold annotations.
- Record filtering and alignment exclusions.
- Exclude zero-relation examples from supervised span metrics while preserving them in dataset statistics.

Do not use the EXP-003 test set during this experiment.

---

## 5. Experimental Approaches

Implement and compare three approaches.

All approaches must use the same gold anaphor spans, tokenization, valid context-token masks, and evaluation rules.

### Approach A — Frozen token-affinity span-expansion baseline

**Purpose:** Establish how much complete-span extraction can be achieved without training a new model.

Load the frozen EXP-002 checkpoint.

For each gold anaphor:

1. Compute the mean token-affinity logits across the anaphor span.
2. Select the highest-scoring valid context token.
3. Expand that token into a candidate antecedent span using a deterministic, annotation-independent rule.
4. Return the predicted start/end boundaries.

Use a simple baseline that requires no learned parameters.

Suggested default:

- Decode the selected token to its original character span.
- Expand to the containing noun phrase using a fixed, documented noun-chunking rule where available.
- If no noun phrase contains the token, return the token itself.

For event references or other non-nominal antecedents, the fallback may produce incomplete spans. Preserve these failures rather than adding test-specific rules.

The expansion mechanism must not access gold antecedent spans at inference.

If a parser is used, document its model/version and distinguish this hybrid baseline from the strictly encoder-only learned approaches.

Keep the EXP-002 model frozen.

### Approach B — Learned start/end boundary head

**Purpose:** Determine whether explicit boundary supervision improves span extraction.

Use a configurable transformer encoder and a trainable boundary-prediction head.

For each gold anaphor relation:

1. Encode the context/chunk pair once.
2. Pool the encoder hidden states belonging to the anaphor span.
3. Construct an anaphor-conditioned representation for every valid context token.
4. Predict start logits and end logits over context tokens.
5. Select the highest-scoring valid `(start, end)` pair.

Suggested architecture:

```python
H = encoder(input_ids, attention_mask)

a = mean_pool(H[anaphor_start:anaphor_end])

start_logits = StartHead(H_context, a)
end_logits = EndHead(H_context, a)
```

A suitable head may concatenate the context-token embedding, anaphor embedding, and their elementwise interaction before projecting to a scalar score.

The architecture must support multiple gold anaphor relations per example.

#### Boundary conventions

Gold antecedent spans use half-open offsets `[start, end)`.

The predicted end index must follow the same convention.

When scoring an end boundary, use the final included token `end - 1`, then convert the prediction back to an exclusive end offset.

#### Decoding constraints

Enforce:

- Start and end belong to valid context positions.
- Start is not after end.
- Span contains at least one token.
- Span does not cross special tokens or padding.
- Maximum predicted span length is configurable.

Default maximum span length: 32 tokens.

#### Loss

Use a joint span probability derived from start and end scores:

`score(s, e) = start_logit(s) + end_logit(e)`

Normalize over valid `(s, e)` pairs.

For multiple gold antecedents, maximize the combined probability of all valid gold spans.

Use a numerically stable log-sum-exp implementation.

Do not arbitrarily choose only the first gold antecedent.

### Approach C — Learned span-pair scoring

**Purpose:** Determine whether directly scoring candidate antecedent spans improves extraction.

Use an encoder with an anaphor-conditioned candidate-span scoring head.

For each gold anaphor:

1. Encode the context/chunk pair once.
2. Construct an anaphor representation from its gold token span.
3. Enumerate valid context spans up to the configured maximum length.
4. Construct a representation for each candidate span.
5. Score each candidate span against the anaphor representation.
6. Select the highest-scoring candidate.

Suggested span representation:

```python
span_representation = concat(
    H[start],
    H[end - 1],
    mean_pool(H[start:end]),
)
```

Combine this with the anaphor representation using a trainable scoring head.

Alternative efficient span representations are acceptable if documented.

#### Loss

Use a softmax over valid candidate spans.

For each relation, maximize the combined probability assigned to all gold antecedent spans.

Use the same marginal-likelihood principle as Approach B.

#### Efficiency

Candidate enumeration must be bounded by maximum span length.

Avoid materializing unnecessarily large tensors.

Document approximate memory and computational cost.

The model must support multiple relations per example without requiring multiple transformer encoder passes.

---

## 6. Shared Implementation Requirements

### Single encoder pass

For learned approaches B and C, the encoder must process each context/chunk example once per forward pass.

If an example contains multiple anaphor relations, reuse the encoder hidden states.

Relation-specific heads may operate independently on those shared representations.

### Encoder-only constraint

Do not introduce:

- Generative language models.
- Autoregressive decoding.
- LLM-based span expansion.
- External generative inference.

### Configurability

Support:

- Encoder name.
- Projection/head hidden dimension.
- Maximum span length.
- Dropout.
- Learning rate.
- Weight decay.
- Batch size.
- Random seed.

### Reuse

Reuse the existing tokenizer, preprocessing, masks, Lightning infrastructure, and experiment logging wherever possible.

Avoid duplicating data processing or metric logic.

---

## 7. Evaluation Metrics

### Primary metric — Exact Span Match (EM)

For each valid relation, compare the predicted antecedent span against every gold antecedent span.

A prediction is correct if its start and end token offsets exactly match any one valid gold antecedent span.

Report:

- Correct relations.
- Total evaluated relations.
- Exact Span Match percentage.

Do not use gold spans to adjust predicted boundaries.

### Secondary metric — Token Overlap F1

Compute precision and recall between predicted antecedent tokens and each gold antecedent span.

For predicted token set P and gold token set G:

`Precision = |P ∩ G| / |P|`

`Recall = |P ∩ G| / |G|`

`F1 = 2PR / (P + R)`

For multiple valid gold antecedents, take the maximum F1 over alternatives.

Average relation-level F1 across the evaluation split.

### Secondary metric — Localization Hit

A prediction is a localization hit if the predicted span overlaps at least one valid gold antecedent span.

This is weaker than Exact Span Match.

Report it separately.

### Secondary metric — Boundary accuracy

Report:

- Start-boundary accuracy.
- End-boundary accuracy.

A boundary is correct if it matches the corresponding boundary of at least one valid gold antecedent.

Also report Exact Span Match, which requires both boundaries to match the same gold span.

### Additional analysis

Break down performance by gold antecedent span length:

- 1 token.
- 2 tokens.
- 3–4 tokens.
- 5–8 tokens.
- 9+ tokens.

For relations with multiple alternative gold antecedents, assign the relation to the shortest gold span's length band for this descriptive analysis, and document that choice.

Report sample counts for every group.

### Character-level evaluation

Also reconstruct the predicted span using tokenizer offset mappings and compare it against the original annotated character span.

Report character-level exact match as a supplementary metric.

Token-level EM remains the primary metric because tokenization can prevent perfect character-boundary reconstruction in some cases.

---

## 8. Training Configuration

Train Approaches B and C independently.

Use the same initial training configuration.

| Parameter | Value |
|---|---|
| Encoder | `distilbert-base-uncased` |
| Seed | 42 |
| Batch size | 16 |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Dropout | `0.1` |
| Max epochs | 20 |
| Early stopping patience | 4 |
| Checkpoint monitor | `val/span_em` |
| Checkpoint mode | `max` |
| Gradient clipping | 1.0 |
| Precision | 32-bit |
| Maximum span length | 32 tokens |
| Optimizer | AdamW |

Initialize the learned models from the same pretrained encoder weights, not from each other's trained checkpoints.

The frozen EXP-002 model is used only for Approach A.

This makes the comparison between learned span heads easier to interpret.

### Checkpoint selection

Save the best checkpoint for each learned approach based on development Exact Span Match.

If multiple epochs achieve the same Exact Span Match, use development loss as the tie-breaker.

Record the selected epoch and both metrics.

Do not use the behavioral test split for checkpoint selection.

---

## 9. Experiment Implementation

Create:

`scripts/experiments/exp004_span_extraction.py`

The script must:

1. Load and preprocess the train/dev splits.
2. Audit valid gold spans and token alignment.
3. Load the frozen EXP-002 checkpoint for Approach A.
4. Evaluate the deterministic span-expansion baseline.
5. Train Approach B.
6. Select and reload its best checkpoint.
7. Evaluate Approach B on the full dev split.
8. Train Approach C independently.
9. Select and reload its best checkpoint.
10. Evaluate Approach C on the full dev split.
11. Calculate all shared span metrics.
12. Calculate span-length breakdowns.
13. Export prediction diagnostics.
14. Produce a comparison report.
15. Append results to `EXPERIMENTLOG.md`.

Allow each approach to be run independently through CLI arguments.

Example:

```bash
python scripts/experiments/exp004_span_extraction.py \
    --approach boundary
```

Also support:

```bash
python scripts/experiments/exp004_span_extraction.py \
    --approach span_pair
```

And:

```bash
python scripts/experiments/exp004_span_extraction.py \
    --approach baseline
```

The script should support running all approaches sequentially.

---

## 10. Prediction Diagnostics

Save relation-level predictions for every development example with valid supervision.

For each relation, include:

- Example ID.
- Anaphor text.
- Gold antecedent text(s).
- Predicted antecedent text.
- Predicted token start/end.
- Predicted character start/end.
- Exact-match result.
- Token overlap F1.
- Localization hit.
- Gold antecedent span lengths.
- Model approach.

Inspect at least 20 incorrect predictions per learned approach, or all incorrect predictions if fewer than 20 exist.

Classify errors into:

- Correct entity, incomplete span.
- Correct entity, oversized span.
- Wrong entity.
- Wrong semantic type.
- Event/proposition boundary failure.
- Tokenization/alignment issue.
- Other.

These classifications are diagnostic, not claims about the model's internal reasoning.

Pay particular attention to cases where the predicted span overlaps the gold antecedent but fails exact match.

---

## 11. Output Artifacts

Store results under:

```text
experiments/exp004_span_extraction/
├── config.yaml
├── dataset_stats.json
├── comparison_metrics.json
├── span_length_metrics.json
├── baseline/
│   └── predictions.json
├── boundary/
│   ├── metrics.json
│   ├── predictions.json
│   └── checkpoints/
│       └── best.ckpt
├── span_pair/
│   ├── metrics.json
│   ├── predictions.json
│   └── checkpoints/
│       └── best.ckpt
└── comparison_report.md
```

Record:

- Dataset identifier and revision, if available.
- Git commit and dirty-working-tree status.
- Configuration for each approach.
- Random seed.
- Checkpoint identifiers.
- Best epochs.
- Training runtime.
- Evaluation metrics.
- Memory usage where practical.
- Preprocessing exclusions.

Do not commit large checkpoints directly to Git.

---

## 12. Testing

Create offline unit tests for new components.

Suggested files:

- `tests/model/test_span_boundary.py`
- `tests/model/test_span_pair.py`
- `tests/model/test_span_loss.py`
- `tests/experiments/test_exp004.py`

Tests must not require downloading pretrained weights.

### Required test cases

1. Correct candidate-span enumeration.
2. No candidate spans outside the context.
3. Maximum span length respected.
4. Half-open start/end conventions.
5. Multiple gold antecedents handled as alternatives.
6. Overlapping gold spans handled correctly.
7. Padded relations ignored.
8. Zero-relation batches produce differentiable zero loss.
9. Correct start/end masking.
10. No invalid or reversed predicted spans.
11. Stable log-sum-exp loss.
12. Correct Exact Span Match.
13. Correct token overlap F1.
14. Correct boundary accuracy.
15. Correct character-offset reconstruction.
16. Gradients propagate through the encoder and span heads.
17. Multiple relations reuse a single encoder pass.
18. Checkpoint reload reproduces predictions.
19. CPU smoke training completes successfully.
20. Existing tests continue passing.

---

## 13. Mandatory `EXPERIMENTLOG.md` Update

Append an EXP-004 entry to the existing experiment log.

Preserve EXP-001, EXP-002, and EXP-003.

### Required structure

#### EXP-004 — Antecedent Span Extraction

**Description:** Explain the motivation, three approaches, and evaluation setup in 2–4 sentences.

**Hypotheses:** State H1–H4 and their observed status.

**Configuration:** Record model architecture, dataset, seed, training hyperparameters, maximum span length, checkpoint selection, and Git commit.

**Results:** Include:

| Approach | Dev Span EM | Token F1 | Localization Hit | Best Epoch |
|---|---|---|---|---|
| Frozen affinity expansion | ... | ... | ... | N/A |
| Boundary head | ... | ... | ... | ... |
| Span-pair scoring | ... | ... | ... | ... |

Report counts alongside percentages where applicable.

**Observations:** Write 4–8 concise, evidence-based observations about span boundaries, model convergence, comparison between approaches, and failure patterns.

**Hypothesis assessment:** State whether H1–H4 were supported, not supported, or inconclusive.

**Conclusion:** Explain whether explicit span supervision improves complete antecedent extraction.

**Next steps:** Recommend the most promising approach and the next experiment.

Do not fabricate metrics or conclusions.

If the experiment cannot be completed, document the blocker and partial results.

---

## 14. Acceptance Criteria

- [ ] Frozen EXP-002 checkpoint reused for the baseline.
- [ ] No EXP-003 test examples used.
- [ ] All three approaches implemented.
- [ ] Learned approaches use one encoder forward pass per example.
- [ ] Multiple alternative gold antecedents supported.
- [ ] Exact Span Match implemented correctly.
- [ ] Token overlap F1 implemented correctly.
- [ ] Localization Hit implemented correctly.
- [ ] Boundary metrics implemented.
- [ ] Span-length breakdown produced.
- [ ] Character offsets reconstructed.
- [ ] Both learned approaches trained independently.
- [ ] Best checkpoints selected on development data.
- [ ] Model comparisons use identical eligible relations.
- [ ] Prediction diagnostics saved.
- [ ] Error analysis completed.
- [ ] Experiment artifacts saved.
- [ ] `EXPERIMENTLOG.md` updated with results and observations.
- [ ] Offline tests pass.
- [ ] Existing tests continue passing.
- [ ] No architecture tuning performed against the reserved test set.

---

## 15. Definition of Done

EXP-004 is complete when all three span-extraction approaches have been evaluated on the same development relations and their performance has been compared using Exact Span Match, Token F1, and Localization Hit.

The experiment must establish:

1. Whether learned span supervision improves exact antecedent extraction over frozen token-affinity expansion.
2. Whether independent boundary prediction or direct span-pair scoring performs better.
3. Which span lengths and reference types remain difficult.
4. Whether predicted spans can be reliably mapped back to original context text.
5. Which approach should become the foundation for subsequent null-antecedent and anaphor-detection experiments.

**The experiment is complete even if neither learned approach outperforms the baseline**, provided the comparison is reproducible and the outcome is documented.

Do not implement null antecedent prediction or automatic anaphor detection as part of this experiment.