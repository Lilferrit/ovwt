# ovwt

Trains one XGBoost binary classifier per variant vs. wild-type using CellProfiler morphological features.
For each variant in the dataset, the tool subsets to that variant and the wild-type, trains a model on the training split, selects the best round using a validation split, and reports AUROC and accuracy on all three splits.

## How it works

1. Read a feature file (Parquet or CSV) containing CellProfiler features and a label column.
2. Perform a single stratified 80/10/10 train/val/test split across all cells.
3. For each non-wild-type label found in the dataset:
   - Filter each split to rows belonging to that variant or the wild-type.
   - Optionally compute balanced sample weights to correct for class imbalance.
   - Train an XGBoost classifier with early stopping on the validation set.
   - Evaluate AUROC and accuracy on all three splits.
4. Write `results.csv` and `models.pkl` to the output directory.

Feature columns are identified automatically as any column whose name starts with an uppercase letter and contains an underscore — the standard CellProfiler naming convention (e.g. `Intensity_MeanIntensity_DAPI`, `Texture_Variance_CY5_3_00`).

## Installation

Requires Python 3.13+. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

All XGBoost parameters and optional `app` fields (`log_level`, `seed`) have built-in defaults and do not need to be specified. Only the four required `app` fields must be provided, either on the command line or via a config file.

### Command line only (no config file needed)

Pass the required fields as CLI overrides. All other parameters use their built-in defaults:

```bash
ovwt \
  app.feature_file=/path/to/features.parquet \
  app.out_dir=/path/to/output
```

Any parameter can be overridden the same way:

```bash
ovwt \
  app.feature_file=/path/to/features.parquet \
  app.label_col=aaChanges \
  app.wt_label=WT \
  app.out_dir=/path/to/output \
  xgboost.max_depth=6 \
  xgboost.num_boost_round=200
```

### With a config file

For experiments you want to reproduce or share, write a YAML file with all the settings:

```yaml
# my_experiment.yaml
# @package _global_

app:
  feature_file: /path/to/features.parquet
  label_col: aaChanges
  wt_label: WT
  out_dir: /path/to/output
  log_level: INFO
  seed: 42

xgboost:
  nthread: -1
  max_depth: 3
  colsample_bytree: 0.7
  colsample_bylevel: 0.7
  colsample_bynode: 0.7
  subsample: 0.5
  num_boost_round: 100
  early_stopping_rounds: 5
  weigh_samples: true
```

Then run:

```bash
ovwt --config-path /path/to/dir --config-name my_experiment
```

`--config-path` must be the directory containing the YAML file; `--config-name` is the filename without the `.yaml` extension. CLI overrides can still be appended after the config name.

## Configuration reference

### `app`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `feature_file` | yes | — | Path to the input feature file (`.parquet`, `.pq`, or `.csv`). |
| `label_col` | yes | — | Name of the column containing cell labels. |
| `wt_label` | yes | — | Label value identifying wild-type cells. All other unique values are treated as variants. |
| `out_dir` | yes | — | Directory where `results.csv`, `models.pkl`, and `ovwt.log` are written. Created if it does not exist. |
| `log_level` | no | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Case-insensitive. |
| `seed` | no | `42` | Random seed for the train/val/test split and XGBoost. |

### `xgboost`

| Key | Default | Description |
|-----|---------|-------------|
| `nthread` | `-1` | Number of threads for XGBoost. `-1` uses all available cores. |
| `max_depth` | `3` | Maximum tree depth. |
| `colsample_bytree` | `0.7` | Fraction of features sampled per tree. |
| `colsample_bylevel` | `0.7` | Fraction of features sampled per tree level. |
| `colsample_bynode` | `0.7` | Fraction of features sampled per split node. |
| `subsample` | `0.5` | Fraction of rows sampled per tree. |
| `num_boost_round` | `100` | Maximum number of boosting rounds. |
| `early_stopping_rounds` | `5` | Stop if validation AUC does not improve for this many consecutive rounds. |
| `weigh_samples` | `true` | If `true`, apply balanced class weights to the training set to correct for class imbalance. |

## Outputs

All outputs are written to `out_dir`.

| File | Description |
|------|-------------|
| `results.csv` | One row per variant with columns `variant`, `train_auroc`, `train_accuracy`, `val_auroc`, `val_accuracy`, `test_auroc`, `test_accuracy`. |
| `models.pkl` | Python pickle file containing a `dict` mapping each variant label to its trained `xgboost.Booster`. |
| `ovwt.log` | Full log of the run, mirrored from stdout. |

## Development

Run the test suite:

```bash
uv run pytest
```

Lint and format:

```bash
uv run ruff check
uv run ruff format
```

## Documentation

Full documentation can be found [at the GitHub pages site](https://lilferrit.github.io/ovwt/).