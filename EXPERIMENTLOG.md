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