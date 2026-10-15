# EXP-005B — NULL Prediction Validation

## Objective

Validate whether the EXP-005 context-conditioned NULL head genuinely detects missing antecedents, rather than exploiting synthetic-negative artifacts.

**This is an evaluation-only experiment. Do not retrain or modify the model.**

## 1. Model

Use the frozen EXP-005 context-conditioned NULL checkpoint.

Preserve the existing encoder, span scorer, NULL head, and inference configuration.

Do not use the reserved 36-example test split.

## 2. Manual Negative Validation

Sample **50 synthetic development negatives**, stratified by:

- 25 context-removal examples.
- 25 hard-replacement examples.
- Include correct NULL predictions and false-LINK predictions from both groups.

For each example, manually classify the label as:

- `VALID_NULL`: No valid antecedent exists in the provided context.
- `INVALID_NULL`: A valid antecedent exists.
- `AMBIGUOUS`: Insufficient evidence to determine whether a valid antecedent exists.

Record the context, chunk, gold anaphor, predicted output, negative-generation strategy, and justification.

**Important:** Do not use the model's prediction as evidence that a synthetic label is correct. If manual validation cannot be performed reliably, export the examples for human review rather than automatically assigning labels.

## 3. False-NULL Analysis

Inspect **all false-NULL predictions** on the 152 original positive development relations.

For each, record:

- Gold antecedent span(s).
- Highest-scoring predicted span.
- NULL score.
- Best-span score.
- Score margin: `NULL − best_span`.
- Whether the forced-LINK prediction was correct.

Identify whether false rejection is associated with weak span localization, context length, or competing antecedents.

## 4. Calibration Analysis

Evaluate whether adjusting the NULL decision threshold reduces false rejections.

Define:

`margin = NULL score − highest span score`

Predict NULL when:

`margin > threshold`

Use a deterministic, stratified **50% calibration / 50% evaluation split**, grouped by original source example so positive/negative pairs stay together.

- Select the threshold using only the calibration subset.
- Optimize end-to-end exact accuracy, with positive Span EM as a tie-breaker.
- Evaluate the selected threshold once on the held-out subset.
- Compare against the original threshold of `0`.
- Report LINK/NULL Macro F1, NULL recall, false-NULL rate, positive Span EM, and end-to-end exact accuracy.

Report both thresholds' performance on identical held-out examples. Do not present calibration-set performance as an independent evaluation.

## 5. Required Outputs

Create:

```text
experiments/exp005b_null_validation/
├── config.yaml
├── manual_review.csv
├── manual_review_summary.json
├── false_null_analysis.json
├── calibration_results.json
├── comparison_report.md
└── predictions.json
```

Append an **EXP-005B** entry to `EXPERIMENTLOG.md` containing:

- Objective and methodology.
- Number of validated, invalid, and ambiguous synthetic negatives.
- False-NULL analysis and recurring error patterns.
- Original versus calibrated threshold metrics.
- Limitations and conclusions.

## 6. Acceptance Criteria

- [ ] Frozen EXP-005 checkpoint used without retraining.
- [ ] 50 negatives sampled and exported for review.
- [ ] Human-reviewed labels recorded, or explicitly marked pending.
- [ ] All false-NULL predictions inspected.
- [ ] Calibration and evaluation subsets are source-disjoint.
- [ ] Threshold selected without evaluation-set leakage.
- [ ] Original and calibrated thresholds compared.
- [ ] Existing tests continue passing.
- [ ] Results and observations appended to `EXPERIMENTLOG.md`.

## 7. Decision Criteria

**Proceed to EXP-006 (Anaphor Detection)** if:

1. Human review confirms that the majority of inspected synthetic negatives are valid.
2. The conditioned NULL model maintains meaningful discrimination on the reviewed valid negatives.
3. Calibration demonstrates an acceptable trade-off between false-NULL and false-LINK errors, or clearly establishes the remaining limitation.

Otherwise, investigate negative-generation quality or NULL-score calibration before extending the architecture.

**Definition of done:** We have evidence about whether the NULL head is learning antecedent availability, how trustworthy the synthetic negatives are, and whether threshold calibration can preserve more valid antecedent links.