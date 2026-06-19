# Datasets

## Centralized Dataset Rule

All datasets should stay under:

```text
machine_learning/datasets/
```

No subproject should create local dataset folders. Project folders should contain
code, documentation, requirements, and optional training outputs only.

## Criteo-style CTR Dataset

Used by:

- `01_wide_deep_ctr_prediction`
- `02_deepfm_ctr_prediction`
- `03_dcn_ctr_prediction`
- `04_xdeepfm_ctr_prediction`

Expected location:

```text
machine_learning/datasets/criteo_ctr/
```

Expected processed files:

```text
train.csv
valid.csv
test.csv
```

Expected columns:

```text
label
I1, I2, ..., I13
C1, C2, ..., C26
```

`label` is binary. Numerical features are `I1` through `I13`. Categorical
features are `C1` through `C26`. This dataset is used for CTR feature-interaction
models.

## MovieLens 10M Dataset

Used by:

- `05_din_ctr_prediction`
- `06_dien_ctr_prediction`
- `07_ncf_collaborative_filtering`
- `08_two_tower_retrieval`
- `09_sasrec_sequential_recommendation`
- `10_lightgcn_recommendation`

Expected raw location:

```text
machine_learning/datasets/ml-10M100K/
```

Expected raw files:

```text
ratings.dat
movies.dat
```

MovieLens 10M raw format:

```text
ratings.dat: userId::movieId::rating::timestamp
movies.dat: movieId::title::genres
```

Read these files with pandas using:

```python
sep="::"
engine="python"
```

`movies.dat` should be read with `encoding="latin-1"`.

## Processed Dataset Folders

MovieLens-based projects create compact processed files once:

```text
processed_din
processed_dien
processed_ncf
processed_two_tower
processed_sasrec
processed_lightgcn
```

Processed datasets are used because MovieLens 10M is large. Raw preprocessing
can be expensive, and training should not rebuild raw samples every time.
`scripts/prepare_data.py` creates compact CSV files once, and `scripts/train.py`
reads the processed files directly.

## Debug Mode

Most MovieLens-based projects default to:

```python
debug_mode = True
```

This avoids freezing a laptop during the first run.

Common safety options:

- `debug_mode`
- `max_users`
- `max_interactions`
- `max_samples`
- `max_edges`
- `batch_size`
- `num_workers`
- `n_epochs`
- `eval_max_users`

Do not disable debug mode until the debug run works.
