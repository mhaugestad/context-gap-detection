# EXP-003 — Behavioral and Adversarial Coreference Evaluation

## 1. Objective

Evaluate the cross-boundary coreference model from EXP-002 on the reserved 36-example behavioral test set.

The goal is to understand which coreference phenomena the current encoder-based token-affinity architecture can resolve, which it struggles with, and which cannot be evaluated under its current gold-anaphor localization setup.

**This is a diagnostic experiment, not a model optimization experiment.**

Do not train, fine-tune, change hyperparameters, or modify the model architecture based on test results.

## 2. Background

The project uses an encoder-only architecture with:

- DistilBERT (`distilbert-base-uncased`).
- A projected token-affinity head (projection dimension 128).
- A single encoder forward pass over context and chunk.
- A supervised loss that supports multiple alternative antecedent mentions.
- Gold anaphor spans supplied during evaluation.

### Previous results

**EXP-001 — Tiny-Set Overfitting**

- 16 examples, 17 relations.
- Relation Hit@1: 100%.
- Token Hit@1: 100%.
- Result: PASS.

**EXP-002 — Full-Dataset Training**

- 1,000 training examples, 196 development examples.
- 699 valid training relations.
- 152 valid development relations.
- Best checkpoint selected by development loss at epoch 2.
- Development Relation Hit@1: 101/152 (66.45%).
- Development Token Hit@1: 191/336 (56.85%).
- Development loss: 1.054341.
- Random baseline: 11.06% expected Relation Hit@1.
- Recency baseline: 0.66% Relation Hit@1.

The model demonstrates meaningful held-out antecedent localization, but performance across specific coreference phenomena remains unknown.

## 3. Hypotheses

### Primary hypothesis — H1

**The model will exhibit uneven performance across coreference phenomena, achieving higher Relation Hit@1 on straightforward reference cases than on cases requiring competition resolution, semantic reasoning, or adversarial disambiguation.**

For this experiment, define:

- Straightforward cases: Categories A and C.
- Challenging cases: Categories B, F, and H.

Calculate Relation Hit@1 separately for both groups, using only examples with valid gold anaphor–antecedent relations.

**H1 is supported if the straightforward group achieves higher Relation Hit@1 than the challenging group.**

Report the difference in percentage points and the number of evaluated relations in each group.

This is an exploratory comparison because the test set is small and categories may differ in several ways simultaneously.

### Null hypothesis — H0

There is no observed accuracy advantage for the straightforward group over the challenging group.

### Secondary hypothesis — H2: Span complexity

The model will show reduced accuracy on complex or multi-token antecedent expressions compared with simpler references.

Investigate Category E and compare it descriptively with Category A.

Do not assume that a token-level correct prediction implies the model has recovered the complete antecedent span.

### Secondary hypothesis — H3: Competing entities

The model will make errors when multiple plausible antecedents are present, particularly when the nearest or most lexically similar candidate is incorrect.

Investigate Categories B and H, focusing on whether predictions select annotated distractor entities.

### Secondary hypothesis — H4: Evaluation limitations

Negative-control cases will reveal limitations of the current evaluation protocol because the model cannot explicitly predict that no antecedent exists.

Investigate Category G separately from ordinary localization accuracy.

H4 is an architectural/evaluation capability assessment, not a conventional accuracy hypothesis.

## 4. Dataset

Use the reserved `test` split from:

`mhaugestad/contextualise-context`

Expected original size: 36 examples.

Fields:

- `id`
- `context`
- `chunk`
- `chunk_spans`
- `context_spans`

Use the existing annotation filtering and token preprocessing pipeline.

### Important requirements

- Do not modify test annotations.
- Do not remove difficult examples simply because the model performs poorly.
- Preserve examples with zero valid relations.
- Record all annotation filtering and token alignment outcomes.
- Distinguish examples that are not evaluable under the current metric from examples where the model makes an incorrect prediction.

The test split must remain isolated from training and model selection.

## 5. Behavioral Taxonomy

The test set was designed around the following behavioral categories.

| Category | Focus | Subtests |
|---|---|---|
| A | Basic reference forms | A01 exact NP; A02 shortened NP; A03 proper-name variation; A04 pronoun |
| B | Competing antecedents | B01 same-type distractor; B02 lexical distractor; B03 nearest-is-wrong; B04 most-frequent-is-wrong |
| C | Context distance | C01 adjacent; C02 two sentences; C03 four or more sentences; C04 intervening entities |
| D | Link structure | D01 one-to-one; D02 one-to-many; D03 two independent pairs; D04 coreference chain |
| E | Span complexity | E01 long NP; E02 title and name; E03 possessive; E04 nested/embedded NP |
| F | Reasoning | F01 syntactic cue; F02 semantic type; F03 discourse/event cue; F04 world-knowledge stress test |
| G | Negative controls | G01 no anaphor; G02 missing antecedent; G03 lexical false friend; G04 generic/non-referential NP |
| H | Contrast/adversarial | H01 recency contrast; H02 gender/number contrast; H03 entity-order swap; H04 minimal lexical change |

**Do not assume that the dataset's `id` field contains these taxonomy identifiers.**

If the taxonomy is stored in an existing manifest, use that manifest.

If it is not available, create an explicit mapping from dataset example IDs to taxonomy labels by inspecting the test examples and their original behavioral definitions.

Do not infer categories from row order without verifying the mapping.

If an example's intended category cannot be established reliably, label it `UNMAPPED` and document the issue.

## 6. Freeze the Model and Evaluation Rules

Load the best checkpoint from EXP-002.

Use the exact checkpoint selected by minimum development loss.

Record:

- Checkpoint path.
- Checkpoint hash (SHA-256).
- Model configuration.
- Dataset revision, if available.
- Git commit.
- Device.
- Evaluation seed.

Set the model to evaluation mode and disable gradients.

Do not alter:

- Model weights.
- Tokenizer.
- Projection dimension.
- Pairwise masking.
- Loss definition.
- Prediction aggregation.
- Hit@1 definitions.

If an implementation bug prevents evaluation, document it and apply the smallest necessary correction, with tests.

Do not choose another checkpoint based on test performance.

## 7. Evaluation Protocol

### 7.1 Relation Hit@1 — Primary metric

For each valid gold relation:

1. Obtain the gold anaphor token span.
2. Average model affinity logits over the gold anaphor tokens.
3. Mask all invalid context positions.
4. Select the highest-scoring valid context token.
5. Mark the relation correct if that token belongs to any annotated gold antecedent span.

Multiple gold antecedent mentions are alternative correct answers.

Report:

- Correct relations.
- Total eligible relations.
- Relation Hit@1 percentage.

### 7.2 Token Hit@1 — Secondary metric

For every gold anaphor token:

1. Select its highest-scoring valid context token.
2. Check whether that token belongs to the union of gold antecedent tokens.

Report correct/total counts and percentages.

### 7.3 Development comparison

Include the EXP-002 development performance as a reference:

- Dev Relation Hit@1: 66.45%.
- Dev Token Hit@1: 56.85%.

Do not interpret differences between the development and behavioral test sets as generalization gains or losses without accounting for their different composition.

### 7.4 Baselines

Evaluate the same two baselines from EXP-002:

**Random baseline:** Exact expected accuracy, calculated as the fraction of valid context tokens belonging to the gold antecedent-token union.

**Recency baseline:** Select the final valid context token.

Report both baselines on the eligible test relations.

Do not change baseline definitions.

### 7.5 Zero-relation examples

Examples without valid gold relations must not contribute to Hit@1 denominators.

However, they must remain in the evaluation report.

Classify their evaluation status explicitly:

- `EVALUABLE`: At least one valid gold relation.
- `NO_GOLD_RELATION`: No valid annotated relation.
- `ANNOTATION_INVALID`: Intended relation cannot be evaluated because of invalid annotation.
- `TRUNCATED`: Intended relation was lost during tokenization/truncation.
- `UNSUPPORTED`: The intended behavior requires a capability absent from the model, such as predicting no antecedent.

Use these statuses consistently. Record both preprocessing outcomes and capability limitations where more than one applies.

Do not treat missing predictions for unsupported cases as successful rejections.

## 8. Category-Level Evaluation

For each category A–H, report:

- Number of examples.
- Number of evaluable examples.
- Number of valid relations.
- Correct relations.
- Relation Hit@1.
- Token Hit@1, where applicable.
- Number of unsupported or non-evaluable examples.

Use `N/A` when a metric has no valid denominator.

### Individual subtests

Report results for each identified subtest, such as A01, B03, or H04.

At minimum, include:

- Subtest ID.
- Example ID.
- Evaluation status.
- Number of gold relations.
- Number of correctly localized relations.
- Pass/fail for localization, when evaluable.
- Short diagnostic note.

For examples with multiple relations, report both relation-level correctness and whether all relations were correct.

**Do not count an example as a complete success merely because one of its several relations was correctly localized.**

### Group comparison for H1

Aggregate:

- Group 1: A + C.
- Group 2: B + F + H.

Compute:

- Relation Hit@1 for each group.
- Absolute difference in percentage points.
- Sample counts.

State whether H1 is supported, not supported, or inconclusive.

Do not apply formal significance claims unless a suitable statistical test is actually performed.

## 9. Detailed Prediction Diagnostics

Save relation-level predictions for every evaluable test example.

Each record should contain:

```json
{
  "example_id": "...",
  "category": "B",
  "subtest": "B03",
  "anaphor_text": "...",
  "gold_antecedents": ["..."],
  "predicted_token": "...",
  "predicted_token_index": 0,
  "correct": false,
  "gold_token_best_rank": 0,
  "recency_baseline_correct": false
}
```

Use actual values rather than placeholders.

Where feasible, include:

- Top 5 predicted context tokens and scores.
- Gold antecedent token ranks.
- Predicted token character offsets.
- Gold antecedent character offsets.
- Distance between prediction and anaphor.
- Full context and chunk for manual inspection.

Ranks must be computed only over valid context tokens.

For the highest-ranked gold token, rank 1 means it was the top-scoring valid context token.

Handle ties deterministically and document the tie-breaking rule.

## 10. Failure Analysis

Inspect **every incorrect relation** in the behavioral test set.

Assign one primary diagnostic label:

| Label | Description |
|---|---|
| `DISTRACTOR_ENTITY` | Prediction selects another plausible entity |
| `LEXICAL_BIAS` | Prediction appears driven by lexical similarity |
| `RECENCY_BIAS` | Prediction favors a nearby incorrect mention |
| `SEMANTIC_CONFUSION` | Prediction fails a semantic compatibility distinction |
| `SPAN_BOUNDARY` | Prediction localizes a related token but not an annotated gold token |
| `ANNOTATION_ISSUE` | Gold annotation is questionable or ambiguous |
| `OTHER` | Failure not adequately described by the above |

These labels describe observed patterns, not proven causal mechanisms.

Where evidence is insufficient, use `OTHER` and explain the uncertainty.

For each failed relation, record:

- What the model predicted.
- What the gold annotation expected.
- Why the prediction is incorrect under the current metric.
- Diagnostic label.
- Whether the result suggests a model weakness, annotation problem, or evaluation limitation.

For successful adversarial examples, briefly identify what the model distinguished correctly.

## 11. Negative-Control Analysis

Evaluate Category G separately.

For each negative-control example, report:

1. What behavior the test was designed to check.
2. Whether a valid gold anaphor–antecedent relation exists.
3. Whether the current model and evaluation protocol can test the intended behavior.
4. What additional model capability would be required if it cannot.

Examples:

**G01 — No anaphor:** The current model does not explicitly detect whether an anaphor exists.

**G02 — Missing antecedent:** The current model cannot explicitly predict a null antecedent.

**G03 — Lexical false friend:** May be evaluable as localization if a valid gold relation exists; otherwise assess the capability limitation.

**G04 — Generic/non-referential NP:** Requires distinguishing referential expressions from non-referential expressions, which is not directly supervised by the current objective.

Do not force a numerical Hit@1 score onto cases where the metric is undefined.

Do not claim that the model successfully handles negative controls solely because no valid relation was annotated.

## 12. Experiment Implementation

Create:

`scripts/experiments/exp003_behavioral_eval.py`

The script must:

1. Load the frozen EXP-002 checkpoint.
2. Load the reserved test split.
3. Apply the existing preprocessing pipeline.
4. Load and validate taxonomy mappings.
5. Run deterministic inference.
6. Calculate overall Relation Hit@1 and Token Hit@1.
7. Evaluate random and recency baselines.
8. Calculate category-level metrics.
9. Calculate subtest-level results.
10. Calculate the H1 group comparison.
11. Preserve and report zero-relation examples.
12. Export prediction diagnostics.
13. Produce a human-readable behavioral report.
14. Update `EXPERIMENTLOG.md` with the measured findings.

The script should be runnable from the repository root.

Do not introduce a new training loop or modify the EXP-002 training script.

## 13. Output Files

Store outputs under:

```text
experiments/exp003_behavioral_eval/
├── config.yaml
├── dataset_stats.json
├── taxonomy_mapping.json
├── overall_metrics.json
├── category_metrics.json
├── subtest_results.json
├── predictions.json
├── failure_analysis.json
└── behavioral_report.md
```

The taxonomy mapping should be stable and reusable in future experiments.

Do not overwrite or modify EXP-002 artifacts.

### Human-readable report

`behavioral_report.md` should include:

- Overall performance.
- Comparison with test-set baselines.
- Category-level performance.
- Individual subtest results.
- H1 group comparison.
- Notable successes.
- Failure patterns.
- Negative-control limitations.
- Recommendations for future experiments.

Keep the report evidence-based and concise enough to review manually.

## 14. Tests

Create:

`tests/experiments/test_exp003.py`

Use synthetic fixtures for offline tests.

Verify:

- Taxonomy mapping and category aggregation.
- Correct relation-level denominators.
- Correct example-level all-relations-pass logic.
- Multiple alternative antecedents.
- Zero-relation handling.
- Correct baseline calculations.
- Valid context-token ranking.
- Stable tie-breaking.
- Correct H1 group aggregation.
- Unsupported cases are not counted as correct predictions.
- Deterministic inference and result serialization.

Existing tests must continue passing.

## 15. Mandatory `EXPERIMENTLOG.md` Update

Append a new entry to `EXPERIMENTLOG.md`.

Preserve EXP-001 and EXP-002 unchanged.

Use the following structure:

### EXP-003 — Behavioral and Adversarial Coreference Evaluation

**Description:** Briefly explain that the frozen EXP-002 checkpoint was evaluated against the reserved behavioral test set, with no additional training or tuning.

**Hypotheses:** State H1, H0, H2, H3, and H4, including whether each was supported, not supported, or inconclusive.

**Configuration:** Record checkpoint identifier/hash, dataset, tokenizer, evaluation seed, device, and Git commit.

**Dataset statistics:** Report original examples, evaluable examples, valid relations, anaphor tokens, zero-relation examples, and preprocessing exclusions.

**Overall results:**

| Method | Relation Hit@1 | Token Hit@1 |
|---|---|---|
| Random baseline | ... | N/A |
| Recency baseline | ... | N/A |
| EXP-002 checkpoint | ... | ... |

**Category results:**

| Category | Evaluable relations | Correct | Relation Hit@1 |
|---|---:|---:|---:|
| A | ... | ... | ... |
| B | ... | ... | ... |
| C | ... | ... | ... |
| D | ... | ... | ... |
| E | ... | ... | ... |
| F | ... | ... | ... |
| G | ... | ... | N/A where appropriate |
| H | ... | ... | ... |

**Observations:** Write 4–8 concise observations grounded in actual results, including successes, errors, and evaluation limitations.

**Hypothesis assessment:** Explain which hypotheses were supported and why, using measured counts.

**Conclusion:** State what the experiment reveals about the model's current capabilities.

**Next steps:** Recommend the highest-priority improvement based on the observed failure patterns.

Do not invent observations or performance numbers.

If the experiment cannot be completed, document the blocker and any partial results.

## 16. Acceptance Criteria

- [ ] EXP-002 checkpoint loaded without retraining.
- [ ] Checkpoint identity and hash recorded.
- [ ] All 36 reserved test examples accounted for.
- [ ] Taxonomy mapping verified rather than assumed from row order.
- [ ] Overall Relation Hit@1 and Token Hit@1 calculated.
- [ ] Random and recency baselines calculated.
- [ ] Category A–H results reported.
- [ ] Individual subtest results reported.
- [ ] H1 group comparison calculated.
- [ ] All incorrect relations inspected and classified.
- [ ] Zero-relation and unsupported cases preserved.
- [ ] Negative controls analyzed separately.
- [ ] No unsupported cases counted as successful localization.
- [ ] Human-readable behavioral report generated.
- [ ] Prediction diagnostics saved.
- [ ] `EXPERIMENTLOG.md` updated with results and observations.
- [ ] New offline tests pass.
- [ ] Existing tests continue passing.
- [ ] No model weights, hyperparameters, or annotations modified based on test outcomes.

## 17. Definition of Done

EXP-003 is complete when the frozen EXP-002 checkpoint has been evaluated on all 36 reserved test examples and the results have been analyzed by behavioral category and individual subtest.

The experiment must establish:

1. Which tested reference phenomena the model resolves successfully.
2. Which phenomena produce localization errors.
3. Which negative-control behaviors cannot be evaluated with the current architecture.
4. Whether the primary behavioral hypothesis is supported.
5. Which model or data improvement should be prioritized next.

**A low behavioral-test accuracy does not mean the experiment failed.** The purpose is to expose limitations, not maximize a benchmark score.

Do not modify the model or run further training as part of this task.