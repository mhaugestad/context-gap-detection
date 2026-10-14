# Task01: Implement PyTorch Lightning DataModule for Coreference Resolution

## Objective

Implement a PyTorch Lightning `DataModule` that connects the existing annotation filtering, token preprocessing, and batching components into a reproducible training data pipeline.

The DataModule must load the Hugging Face dataset and expose train, validation, and test DataLoaders suitable for training an encoder-based cross-boundary coreference resolution model.

## Background

The project predicts antecedent spans in a context given an anaphor span in a subsequent chunk.

The dataset contains three splits:

| Split | Examples |
|---|---:|
| Train | 1,000 |
| Dev | 196 |
| Test | 36 |

Each example contains:

- `id`
- `context`
- `chunk`
- `context_spans`
- `chunk_spans`

Multiple antecedent spans may correspond to the same anaphor, identified by matching relation IDs.

The following components already exist and have passing tests:

- `src/data/annotations.py` — `filter_annotations()`
- `src/data/preprocessor.py` — `CoreferencePreprocessor`
- `src/data/collator.py` — `CoreferenceCollator`

Reuse these components rather than reimplementing their functionality.

## Implementation Requirements

### 1. DataModule

Create `src/data/datamodule.py` containing `CoreferenceDataModule(pl.LightningDataModule)`.

Support configurable parameters:

- `dataset_name`: Hugging Face dataset repository identifier
- `tokenizer_name`: Hugging Face tokenizer identifier
- `batch_size`: default 16
- `max_length`: default 512
- `num_workers`: default 0
- `truncation_policy`: `drop` or `error`
- `annotation_policy`: `warn`, `drop`, or `error`

### 2. Dataset loading

Implement `prepare_data()` and `setup(stage=None)` following PyTorch Lightning conventions.

- Load the dataset using Hugging Face `datasets`.
- Map `train`, `dev`, and `test` to training, validation, and testing datasets.
- Do not assume a `validation` split exists.
- Avoid unnecessary repeated downloads or preprocessing.
- Support Lightning's `fit`, `validate`, and `test` stages.

### 3. Preprocessing

For each example:

1. Apply `filter_annotations()`.
2. Apply `CoreferencePreprocessor`.
3. Preserve examples containing zero valid relations unless explicitly configured otherwise.
4. Preserve preprocessing metadata, including dropped-relation counts.

Ensure the pipeline supports multiple antecedents for a single anaphor.

### 4. DataLoaders

Implement:

- `train_dataloader()`
- `val_dataloader()`
- `test_dataloader()`

Use the existing `CoreferenceCollator`.

Training batches should be shuffled. Validation and test batches should not.

### 5. Testing

Create `tests/data/test_datamodule.py`.

Use a small synthetic dataset or mocked Hugging Face loading so unit tests do not depend on network access.

Test stage handling, preprocessing, batching, and zero-relation examples.

## Acceptance Criteria

- [ ] `CoreferenceDataModule` subclasses `pl.LightningDataModule`.
- [ ] Loads the Hugging Face `train`, `dev`, and `test` splits correctly.
- [ ] Annotation filtering runs before token preprocessing.
- [ ] Existing preprocessor and collator are reused without duplicating logic.
- [ ] All three DataLoaders return batches with the expected tensors and masks.
- [ ] Multiple antecedents per anaphor survive the entire pipeline.
- [ ] Examples with zero valid relations are handled without errors.
- [ ] Padding and masks work for variable sequence lengths and relation counts.
- [ ] Train DataLoader shuffles; validation and test DataLoaders do not.
- [ ] `setup(stage)` behaves correctly for `fit`, `validate`, `test`, and `None`.
- [ ] No Hugging Face network access is required for unit tests.
- [ ] All existing tests continue passing.
- [ ] `pytest tests/data/ -v` passes.

## Expected Batch Contract

| Field | Shape |
|---|---|
| `input_ids` | `[B, T]` |
| `attention_mask` | `[B, T]` |
| `context_mask` | `[B, T]` |
| `chunk_mask` | `[B, T]` |
| `relations` | `[B, R, 2]` |
| `antecedent_spans` | `[B, R, A, 2]` |
| `relation_mask` | `[B, R]` |
| `antecedent_mask` | `[B, R, A]` |

Where `B` is batch size, `T` padded token length, `R` maximum relations per example, and `A` maximum antecedents per relation.

Also preserve `id`, `context`, `chunk`, and `dropped_relations`.

## Deliverables

1. `src/data/datamodule.py`
2. `tests/data/test_datamodule.py`
3. A short usage example demonstrating DataModule initialization and retrieval of one training batch.

## Constraints

- Use PyTorch Lightning, Hugging Face Datasets, and Transformers.
- Preserve existing interfaces and naming conventions.
- Keep implementation modular and readable.
- Do not introduce model architecture, loss functions, or training logic.
- Do not modify existing preprocessing or collator behavior unless strictly necessary; explain any required changes.

## Definition of Done

The following should execute successfully:

```python
dm = CoreferenceDataModule(
    dataset_name="YOUR_HF_DATASET",
    tokenizer_name="distilbert-base-uncased",
    batch_size=16,
)

dm.prepare_data()
dm.setup("fit")

batch = next(iter(dm.train_dataloader()))

assert batch["input_ids"].ndim == 2
assert batch["relations"].ndim == 3
assert batch["antecedent_spans"].ndim == 4
```

The DataModule is ready for integration with the encoder affinity model, without further data-pipeline changes.