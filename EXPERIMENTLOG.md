## EXP-001 — Tiny-Set Overfitting

**Description:** Tested whether the encoder affinity model can memorize 16 deterministically sampled, token-aligned training examples using gold anaphor spans. The same examples were used for validation and a final deterministic evaluation of the best training-loss checkpoint.

**Hypothesis:** The model can memorize 16 annotated examples and achieve near-perfect antecedent localization.

**Configuration:** `mhaugestad/contextualise-context` (1,000 train / 196 dev); DistilBERT, projection_dim=128, seed=42, batch_size=4, lr=1e-4, weight_decay=0, dropout=0, max_epochs=100, gradient clip=1.0, precision=32-bit. Base Git commit: `9a19af658ac447856b398cd3c10aa4551dfc243b` (working tree was dirty).

**Results:**
- Best epoch: 100 of 100.
- Best epoch train loss: 0.000004615; validation loss: 0.000004596.
- Best checkpoint full-subset loss: 0.000004756.
- Training-subset and same-subset evaluation Token Hit@1: 58/58 (100%).
- Training-subset and same-subset evaluation Relation Hit@1: 17/17 (100%).
- Success criteria: PASS.

**Observations:**
- The 16 selected examples contained 17 gold relations and 58 anaphor tokens after annotation filtering and token alignment.
- Epoch-100 training and validation Hit@1 were both 100%; the best checkpoint also achieved 100% on the complete subset.
- All 17 relation predictions selected an annotated antecedent token; no failed-relation diagnostics were needed.
- An initial attempt stopped before training because the epoch loss accumulator used float64, unsupported on MPS. Changing the accumulator to float32 allowed the full run to complete on `mps:0`.
- The experiment completed 100 epochs in about 333 seconds; losses remained finite and decreased to the values above.

**Conclusion:** The tiny-set overfitting sanity check passed. The model memorized the selected examples and met all measured success targets.

## EXP-002 — Full-Dataset Training and Baseline Evaluation

**Description:** Trained the existing DistilBERT token-affinity model on the full preprocessed training split and selected a checkpoint using development loss. Evaluated the selected checkpoint on the independent development split and compared it with exact random-token and last-context-token baselines; the test split was reserved for EXP-003.

**Hypotheses:**
- **H1 — Supported:** The selected model exceeded both non-learned baselines in development Relation Hit@1.
- **H0 — Not supported:** The model did outperform the stronger baseline, so the null claim that it would not outperform either baseline was not observed.
- **H2 — Supported:** Development loss fell from 2.715955 for the untrained model to 1.054341 for the selected checkpoint, and Relation Hit@1 rose from 7/152 to 101/152.
- **H3 — Inconclusive:** Accuracy varied by distance band, but it did not decrease monotonically with distance; the two most distant groups were also small.

**Configuration:** Dataset `mhaugestad/contextualise-context` (revision unspecified); DistilBERT base uncased encoder, projection dimension 128, seed 42, batch size 16, AdamW learning rate `2e-5`, weight decay `0.01`, dropout `0.1`, maximum 20 epochs, early stopping patience 4 on `val/loss`, gradient clipping 1.0, 32-bit precision. The best checkpoint was selected by minimum development loss at epoch 2; early stopping ended training after 7 completed epochs. Execution used MPS and took 700.70 seconds. Base Git commit: `9a19af658ac447856b398cd3c10aa4551dfc243b`; the worktree was dirty.

**Dataset statistics:** Train: 1,000 examples (630 with valid relations), 699 valid relations, 1,558 anaphor tokens, and 865 antecedent spans. Dev: 196 examples (137 with valid relations), 152 valid relations, 336 anaphor tokens, and 183 antecedent spans. The preprocessor reported zero dropped relations and annotations in both splits. **Counting limitation:** `CoreferenceDataModule` filters invalid or unmatched source annotations before the preprocessor and does not expose those filter counts, so these zeros do not establish that no source annotations were filtered. The test split was not preprocessed or evaluated.

**Results:**

| Method | Dev Relation Hit@1 | Dev Token Hit@1 | Dev Loss |
|---|---:|---:|---:|
| Random baseline (exact expected accuracy) | 16.814/152 (11.06%) | N/A | N/A |
| Recency baseline (last valid context token) | 1/152 (0.66%) | N/A | N/A |
| Untrained model | 7/152 (4.61%) | 14/336 (4.17%) | 2.715955 |
| Fine-tuned model (best checkpoint) | 101/152 (66.45%) | 191/336 (56.85%) | 1.054341 |

Best development-loss checkpoint: epoch 2; 7 epochs completed. The selected model improved by **55.39 percentage points** over the stronger baseline, the exact random expectation (11.06%). Re-evaluation of the selected checkpoint on the full train split yielded loss 0.768695, Relation Hit@1 499/699 (71.39%), and Token Hit@1 1,065/1,558 (68.36%). The final completed epoch's logged validation loss was 1.635237 and Relation Hit@1 was 99/152 (65.13%), compared with train Relation Hit@1 89.27%; checkpoint selection therefore retained the earlier epoch with the lower dev loss.

**Observations:**
- Training stopped after epoch 7 under the configured patience; epoch 2 remained the best `val/loss` checkpoint.
- The selected model generalized well beyond the exact-random and recency baselines on this development set, and substantially improved over its untrained initialization.
- Distance-stratified Relation Hit@1 was 13/25 (52.00%) at 0–16 tokens, 30/44 (68.18%) at 17–32, 34/51 (66.67%) at 33–64, 19/24 (79.17%) at 65–128, and 5/8 (62.50%) above 128. The uneven pattern and small outer groups do not support a simple distance-decay claim.
- Inspected the first 10 incorrect development predictions saved by the runner. None of those 10 was correct under the recency heuristic; examples include selecting `Martin` rather than the gold antecedent for `He`, `Maternity` rather than `expectant or new mothers`, and `movement` for both stage anaphors in one example. These examples show competing-token selections but do not establish their causes or represent a randomized error sample.
- Relation Hit@1 is antecedent-token localization given gold anaphor spans; it is not end-to-end coreference accuracy. This is one run on a small dev set, without a statistical significance test.
- The first execution reached training but failed while collecting the reported best epoch because Lightning's `EarlyStopping` callback does not expose a `best_epoch` attribute. A validation-loss tracker was added and the configured experiment was rerun successfully; the completed metrics above are from that rerun.

**Conclusion:** EXP-002 provides preliminary evidence that fine-tuning learns held-out antecedent-token localization beyond the random and recency baselines. Development performance remains imperfect and distance results are exploratory; proceed to EXP-003 behavioral evaluation without using these results to tune against the reserved test split.

**Next steps:** Proceed to EXP-003 with the test split kept isolated, then investigate the recorded competing-entity/semantic confusions and the annotation-filter counting limitation.

## EXP-003 — Behavioral Test Evaluation

**Description:** Evaluated the frozen EXP-002 best-development-loss checkpoint on the reserved 36-example behavioral test split. No training, fine-tuning, or model selection was performed.

**Configuration:** Dataset `mhaugestad/contextualise-context` (revision unspecified; test fingerprint `0996008708007a45`); DistilBERT base uncased, projection dimension 128, dropout 0.1, max length 512, evaluation seed 42, batch size 16, MPS inference. The checkpoint was `experiments/exp002_full_training/checkpoints/best.ckpt`, selected at epoch 2 by minimum `val/loss` 1.054341; SHA-256 `e91a9b2f0e4dbb30d94d8bf2af9af157d5f3e26ec7fd1d380ed1198192b1cf90`. The EXP-002 base/current Git commit was `9a19af658ac447856b398cd3c10aa4551dfc243b` (working tree dirty). The evaluation retains the EXP-002 model, tokenizer, masking, aggregation, and Hit@1 definitions.

**Dataset and annotation outcomes:** All 36 examples and all A01–H04 IDs were mapped one-to-one to the full taxonomy in `README.md`. There were 31 examples with at least one valid relation and 32 valid relations (60 gold anaphor tokens); 5 examples had no usable relation. The preprocessor dropped zero additional relations and annotations. Four G-category negatives are `UNSUPPORTED` for this model objective and were not counted as correct or included in Hit@1. D02 is `ANNOTATION_INVALID`: its two distinct anaphors share `ANAPHOR_1`, which the existing annotation filter rejects. Its one intended 1:N example therefore contributes no metric denominator. No test annotations were edited.

**Results:**

| Metric | Result |
|---|---:|
| Model Relation Hit@1 | 25/32 (78.12%) |
| Model Token Hit@1 | 46/60 (76.67%) |
| Exact random-token expected Relation Hit@1 | 5.120/32 (16.00%) |
| Last-valid-context-token recency Relation Hit@1 | 0/32 (0.00%) |
| EXP-002 development reference Relation Hit@1 | 101/152 (66.45%) |
| EXP-002 development reference Token Hit@1 | 191/336 (56.85%) |

The test result is descriptive, not a direct comparison of generalization: the small behavioral battery is deliberately composed differently from development.

**Category results (Relation Hit@1):** A 7/8 (87.50%); B 3/4 (75.00%); C 4/4 (100%); D 3/4 scored relations (75.00%, with D02 invalid); E 4/4 (100%); F 2/4 (50.00%); G N/A (4/4 unsupported); H 2/4 (50.00%). H1 is supported descriptively: A+C = 11/12 (91.67%) versus B+F+H = 7/12 (58.33%), a 33.33 percentage-point difference. The sample is too small and groups differ in composition, so no formal significance claim is made. Category E achieved 4/4 token localization versus 7/8 in A, but Hit@1 does not establish full-span extraction.

**Qualitative observations:** The model resolved all four distance subtests and all four span-complexity subtests; it also handled H01 (Patel rather than the later Mitchell) and H03 (Orion rather than Meridian). The seven incorrect relations show distinct observed patterns: H04 selected the competing Northern Rail Expansion; H02 selected the repeated phrase token `original` rather than the plural entity pair; A08 selected the acquirer token `North` rather than the acquisition event; B04 selected an earlier `union` mention over `union leader Maria Torres`; D03 crossed the station/terminal links by selecting `Riverside` for “The station” (while the same token was correct for “the terminal”); F03 selected the university agent rather than its tuition-reduction decision; and F02 selected `town` rather than the hotel for “The building”. These are annotated in `failure_analysis.json` using the closest supported diagnostic category; labels describe observed errors and do not prove internal causes.

**Conclusion:** The checkpoint is clearly above both test baselines on the evaluable relations. The strongest evidence is for basic forms, distance, and token localization in the span-complexity examples; errors cluster around competing entities, semantic/event distinctions, and contrastive cases. The negative controls do not test the current model because it is supplied gold anaphor spans and must always select a context token. Future work should add explicit anaphor detection and null/abstention supervision, repair D02's annotation encoding, and use larger controlled pairs. Do not tune this checkpoint on the reserved test results.

**Artifacts:** `experiments/exp003_behavioral_eval/` contains the run configuration and checkpoint hash, per-example preprocessing audit, stable taxonomy mapping, overall/category/subtest metrics, relation-level predictions, failure analysis, and human-readable report. Re-run with `python scripts/experiments/exp003_behavioral_eval.py`.


## EXP-004 — Antecedent Span Extraction

**Description:** Compared deterministic span expansion of the frozen EXP-002 affinity checkpoint with an anaphor-conditioned start/end boundary head and direct span-pair scorer. All approaches use the gold anaphor spans and the same filtered, token-aligned development relations; the reserved EXP-003 test split is not loaded.

**Hypotheses:**

- H1: supported.
- H2: supported.
- H3: inconclusive regarding 'substantial' degradation; no threshold was prespecified; see localization-hit and overlap-F1 comparisons.
- H4: descriptively supported for both learned approaches; exploratory; see the length-band counts and rates.

**Configuration:**

```yaml
{
  "dataset_name": "mhaugestad/contextualise-context",
  "dataset_revision": null,
  "encoder_name": "distilbert-base-uncased",
  "head_dim": 128,
  "seed": 42,
  "batch_size": 16,
  "learning_rate": 2e-05,
  "weight_decay": 0.01,
  "max_epochs": 20,
  "patience": 4,
  "gradient_clip_val": 1.0,
  "precision": "32-true",
  "dropout": 0.1,
  "max_length": 512,
  "max_span_length": 32,
  "num_workers": 0,
  "accelerator": "auto",
  "approach": "all",
  "approach_configurations": {
    "baseline": {
      "checkpoint": "/Users/mathiashaugestad/Documents/Projects/context-gap-detection/experiments/exp002_full_training/checkpoints/best.ckpt",
      "checkpoint_sha256": "e91a9b2f0e4dbb30d94d8bf2af9af157d5f3e26ec7fd1d380ed1198192b1cf90",
      "expansion": "containing spaCy noun chunk if a local parser is available; otherwise selected token"
    },
    "boundary": {
      "encoder_name": "distilbert-base-uncased",
      "head_dim": 128,
      "max_span_length": 32,
      "dropout": 0.1,
      "learning_rate": 2e-05,
      "weight_decay": 0.01,
      "batch_size": 16,
      "seed": 42,
      "checkpoint_sha256": "7de7d3ac4891df0a5c7abb1fe051235e2f4e5a7622d7892e6f9acd7c0f1bbc04",
      "best_epoch": 5,
      "training_runtime_seconds": 964.8674072909635
    },
    "span_pair": {
      "encoder_name": "distilbert-base-uncased",
      "head_dim": 128,
      "max_span_length": 32,
      "dropout": 0.1,
      "learning_rate": 2e-05,
      "weight_decay": 0.01,
      "batch_size": 16,
      "seed": 42,
      "checkpoint_sha256": "2ee1b7046bbddc298e6b394d6470f093a13380fc812b7a1c1a92f06f021b1be5",
      "best_epoch": 5,
      "training_runtime_seconds": 932.8733451669104
    }
  },
  "exp002_checkpoint": "/Users/mathiashaugestad/Documents/Projects/context-gap-detection/experiments/exp002_full_training/checkpoints/best.ckpt",
  "output_dir": "/Users/mathiashaugestad/Documents/Projects/context-gap-detection/experiments/exp004_span_extraction",
  "git": {
    "commit": "0e8104e262d07615dbb1d2d38a30e2c4fad7e3f0",
    "dirty_working_tree": true
  },
  "python_version": "3.11.10 (main, Sep  7 2024, 01:03:31) [Clang 15.0.0 (clang-1500.3.9.4)]",
  "platform": "macOS-26.6.2-arm64-arm-64bit",
  "torch_version": "2.14.1",
  "lightning_version": "2.6.6",
  "kernel_determinism": "seeded RNG; deterministic algorithms disabled because MPS indexed gradient accumulation has no deterministic implementation",
  "created_at_utc": "2026-10-14T14:30:16.825823+00:00",
  "test_split_used": false,
  "baseline_expansion": "selected-token fallback (no local en_core_web_sm parser)"
}
```

**Results:**

| Approach | Dev Span EM | Token F1 | Localization Hit | Best Epoch |
|---|---:|---:|---:|---:|
| Frozen affinity expansion | 17/152 (11.18%) | 0.3642 | 101/152 (66.45%) | N/A |
| Boundary head | 76/152 (50.00%) | 0.5860 | 105/152 (69.08%) | 5 |
| Span-pair scoring | 81/152 (53.29%) | 0.6012 | 99/152 (65.13%) | 5 |

Common relation count: 152.

Paired comparisons and percentile bootstrap intervals are in `comparison_metrics.json`; intervals are descriptive and are not treated as significance tests.

**Observations:**

- The frozen expansion baseline extracted 17/152 relations exactly; its localization hit was 101/152.
- The strongest learned approach was span_pair at 81/152 exact matches, a +42.11 percentage-point difference from baseline.
- Direct span-pair scoring returned 81/152 exact matches versus 76/152 for the independent boundary head.
- boundary selected epoch 5 after 10 completed epochs; runtime was 964.9 seconds.
- span_pair selected epoch 5 after 10 completed epochs; runtime was 932.9 seconds.
- Localized but non-exact predictions (the boundary-completion gap) numbered baseline: 84, boundary: 29, span_pair: 18.
- Observed span-length EM ranged from 0.00% in the 2-token band (34 relations, baseline) to 73.53% in the 2-token band (34 relations, span_pair); these descriptive groups can be small.
- Manual review of the first 20 incorrect predictions per learned approach found boundary: Wrong semantic type 7, Wrong entity 4, Event/proposition boundary failure 4, Correct entity, incomplete span 2, Correct entity, oversized span 2, Tokenization/alignment issue 1; span_pair: Wrong semantic type 7, Correct entity, oversized span 5, Event/proposition boundary failure 3, Wrong entity 2, Correct entity, incomplete span 2, Tokenization/alignment issue 1. These are diagnostic labels, not causal claims.

**Hypothesis assessment:**

- H1: supported.
- H2: supported.
- H3: inconclusive regarding 'substantial' degradation; no threshold was prespecified; see localization-hit and overlap-F1 comparisons.
- H4: descriptively supported for both learned approaches; exploratory; see the length-band counts and rates.

**Conclusion:** span_pair achieved the highest learned development Exact Span Match at 81/152 (53.29%), 42.11 percentage points above the frozen baseline. This supports H1 descriptively, not as a statistical-significance claim.

**Next steps:** Use span_pair as the leading learned approach for a controlled follow-up, review the remaining incorrect predictions, and evaluate boundary improvements on development data only. Preserve the reserved test split for a later locked evaluation.

# EXP-005 — Null Antecedent Prediction

**Description:** Extended the frozen EXP-004 span-pair scorer with a fixed NULL candidate, a learned global NULL bias, and a context-conditioned NULL head. Gold anaphor spans remain supplied; automatic anaphor detection is out of scope.

**Hypotheses:**
- H1 — NULL prediction is learnable: Supported. Confidence intervals are in `comparison_metrics.json`.
- H2 — Context-conditioned NULL scoring is beneficial: Supported.
- H3 — Span extraction can be preserved: Mixed. Forced-LINK span quality is unchanged because the scorer is frozen, but conditioned NULL decisions reduce actual positive Span EM by 5.92 percentage points, beyond the provisional 5-point tolerance.
- H4 — The model uses antecedent availability: Descriptively supported. The conditioned model gets both decisions right for 102/152 pairs (67.11%), versus 0/152 for Always-LINK.

**Dataset:** `mhaugestad/contextualise-context` (revision unspecified); 699 aligned train positive relations and 699 synthetic train negatives; 152 aligned development positive relations and 152 synthetic development negatives. Each split has 50% context-removal and 50% hard-replacement negatives. Annotation filtering removed 86 train and 12 dev matched relations; token alignment removed none. There were 36 train and 4 dev examples with unmatched anaphor labels, but the current relation format does not explicitly encode NULL. No zero-relation examples were labeled NULL. The reserved test split was not loaded. Synthetic labels are heuristic, not human-verified; the review subsets are recorded in `negative_generation_report.json`. Rejected negative candidates: 0 train and 0 dev.

**Configuration:** `distilbert-base-uncased` span-pair model; EXP-004 checkpoint SHA-256 `2ee1b7046bbddc298e6b394d6470f093a13380fc812b7a1c1a92f06f021b1be5`; seed 42; batch size 16; NULL head learning rate 0.001; weight decay 0.01; dropout 0.1; up to 20 epochs; early-stopping patience 4; max span length 32; frozen encoder/span scorer in the primary comparison. Git commit `0e8104e262d07615dbb1d2d38a30e2c4fad7e3f0`, dirty working tree `True`. Training ran on MPS.
- learned_bias: best epoch 2; 7 epochs; 292.37s.
- conditioned: best epoch 4; 9 epochs; 419.02s.

**Results:**

| Approach | Macro F1 | Balanced Accuracy | E2E Exact | Forced-LINK Span EM | Actual positive Span EM | NULL Recall |
|---|---:|---:|---:|---:|---:|---:|
| Always-LINK | 0.3333 | 0.5000 | 0.2664 | 0.5329 | 0.5329 | 0.0000 |
| Constant NULL | 0.3478 | 0.5066 | 0.2730 | 0.5329 | 0.5329 | 0.0132 |
| Learned NULL bias | 0.3549 | 0.5099 | 0.2763 | 0.5329 | 0.5329 | 0.0197 |
| Conditioned NULL | 0.8123 | 0.8125 | 0.6579 | 0.5329 | 0.4737 | 0.8421 |

Forced-LINK Span EM measures span selection with NULL ignored. Actual positive Token F1 after NULL decisions was 0.5204 for conditioned NULL, compared with the forced-LINK/frozen baseline 0.6012.

**Observations:**
- The always-LINK baseline has balanced accuracy 0.5000 on this balanced evaluation.
- The best NULL-enabled result has Macro F1 0.8123, balanced accuracy 0.8125, and E2E exact 0.6579.
- Paired differences versus Always-LINK, including 95% bootstrap intervals:
  - learned_bias vs Always-LINK macro_f1: +0.0216 (paired bootstrap 95% CI [+0.0000, +0.0456]).
  - learned_bias vs Always-LINK balanced_accuracy: +0.0099 (paired bootstrap 95% CI [+0.0000, +0.0212]).
  - learned_bias vs Always-LINK end_to_end_exact_accuracy: +0.0099 (paired bootstrap 95% CI [+0.0000, +0.0230]).
  - conditioned vs Always-LINK macro_f1: +0.4790 (paired bootstrap 95% CI [+0.4231, +0.5326]).
  - conditioned vs Always-LINK balanced_accuracy: +0.3125 (paired bootstrap 95% CI [+0.2650, +0.3556]).
  - conditioned vs Always-LINK end_to_end_exact_accuracy: +0.3914 (paired bootstrap 95% CI [+0.3257, +0.4539]).
- Learned-bias results: Macro F1 0.3549, balanced accuracy 0.5099, E2E exact 0.2763; bootstrap intervals are recorded for Macro F1 and E2E exact.
- Conditioned-head results: Macro F1 0.8123, balanced accuracy 0.8125, E2E exact 0.6579.
- Paired contrast accuracy: 0.6711 for conditioned NULL versus 0.0000 for Always-LINK; the conditioned model predicts both LINK in 17 pairs, both NULL in 26 pairs, and the wrong contrast direction in 7 pairs.
- Forced-LINK Span EM is held constant across NULL approaches because the span scorer is frozen; post-NULL Span EM reflects False-NULL decisions.
- The conditioned NULL head has 0.8421 NULL recall and a 0.2171 False-NULL rate. Context-removal and hard-replacement negative recalls are 0.9079 and 0.7763 respectively.
- A candidate-count adjustment selected gamma=0.1 on 31 calibration pairs; on 121 held-out pairs, Macro F1 increased by 0.0028 and E2E Exact by 0.0207. This diagnostic is exploratory and uses the highest-span score.
- Paired bootstrap differences versus Always-LINK are reported with 95% intervals in `comparison_metrics.json`. For conditioned NULL, Macro F1 difference is +0.4790 [0.4231, 0.5326], balanced accuracy +0.3125 [0.2650, 0.3556], and E2E Exact +0.3914 [0.3257, 0.4539].

**Hypothesis assessment:** H1 Supported; H2 Supported; H3 Mixed; H4 descriptively supported.

**Limitations:** Synthetic negatives may contain implicit valid referents despite lexical exclusion; balanced LINK/NULL proportions do not represent deployment prevalence; candidate-count effects can alter softmax probabilities; calibration is exploratory; supplied gold anaphors mean anaphor detection and referentiality remain unsolved.

**Conclusion:** The context-conditioned NULL head substantially improves LINK/NULL classification and end-to-end exact correctness on this balanced synthetic development set, while a global NULL bias contributes little. Conditional forced-LINK span extraction is preserved, but false-NULL decisions reduce positive post-NULL span EM slightly beyond the provisional tolerance. Treat the gains as preliminary because the NULL labels are heuristic and not human-verified.

**Next steps:** Retain the conditioned NULL head as the leading architecture for follow-up, but first manually review the synthetic audit subset and inspect the saved False-NULL examples; then evaluate on human-reviewed negatives before considering anaphor detection.

# EXP-005B — NULL Prediction Validation

Generated: 2026-10-14T17:04:04.865933+00:00

## Method

- Evaluation-only: reloaded the frozen conditioned EXP-005 checkpoint; no training or fine-tuning was run.
- Loaded only the named `dev` dataset split. The reserved test split was not requested.
- Manual labels are left blank and marked pending; no synthetic label was automatically validated.
- Calibration/evaluation are source-disjoint and stratified by each source group's synthetic-negative strategy mix.

## Manual negative review

- Sampled 50 negatives; reviewed 0.
- Pending: 50; VALID_NULL: 0; INVALID_NULL: 0; AMBIGUOUS: 0.
- Per-strategy sample counts and prediction-outcome composition are in `manual_review_summary.json`.

## False-NULL analysis on original positive development relations

- False NULLs: 33/152 (0.2171).
- Every false-NULL record, its gold span(s), highest-scoring span, scores, margin, forced-LINK correctness, context length, and candidate competition proxies are in `false_null_analysis.json`.
- Localization pattern: false-NULL rate was 0.3380 for forced-LINK span errors (24/71) versus 0.1111 when forced-LINK Span EM was correct (9/81).
- Context-length false-NULL rates: 1-32: 0.1282 (5/39); 129+: 0.2500 (4/16); 33-64: 0.2609 (12/46); 65-128: 0.2353 (12/51).
- Candidate competition proxy: median top-two span-score gap was 1.1536 for false NULLs versus 2.6589 for accepted links; 10 false NULLs had a runner-up within 0.5 score points. Candidate-count bins are weak proxies because they count enumerated spans, not semantic antecedents.
- These are descriptive associations, not causal claims; the comparatively higher false-NULL rate when forced span localization is wrong suggests that span uncertainty and NULL rejection co-occur.

## Held-out threshold comparison

- Calibration selected threshold: `-1.8589554` using only 146 calibration relations.
- Held-out source groups: 70; held-out relations: 158.

| Threshold | Macro F1 | NULL recall | False-NULL rate | Positive Span EM | E2E exact |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7905 | 0.8481 | 0.2658 | 0.4810 | 0.6646 |
| -1.8589554 | 0.7611 | 0.9873 | 0.4430 | 0.4051 | 0.6962 |

Both thresholds were evaluated once on the same held-out source groups. Calibration-subset performance is used for threshold selection only.
On held-out data, calibration changed false-NULL rate by +0.1772, positive Span EM by -0.0759, and E2E exact by +0.0316.

## Conclusion and limitations

Human review is pending, so the validity of synthetic NULL labels and discrimination on human-confirmed valid negatives are not established. The held-out threshold result is preliminary and applies to this balanced synthetic development set only. Do not proceed to EXP-006 on the basis of these pending labels alone.

Frozen conditioned checkpoint SHA-256: `0c962ab5f5217186e5080c90b36fea3d4a3edb5b3a97b974954a1d14712dc888`.
