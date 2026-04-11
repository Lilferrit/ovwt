# Configuration

Configuration is handled by [Hydra](https://hydra.cc). Built-in defaults are provided for all optional parameters — only the required `app` fields need to be set.

Settings can be provided via a [YAML config file](usage.md#with-a-config-file), as [CLI overrides](usage.md#command-line-only), or a combination of both.

## `app`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `feature_file` | yes | — | Path to the input feature file (`.parquet`, `.pq`, or `.csv`). |
| `out_dir` | yes | — | Directory where outputs are written. Created if it does not exist. |
| `label_col` | no | `aaChanges` | Name of the column containing cell labels. |
| `wt_label` | no | `WT` | Label value identifying wild-type cells. All other unique values in `label_col` are treated as variants. |
| `log_level` | no | `INFO` | Logging verbosity. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. Case-insensitive. |
| `seed` | no | `42` | Random seed passed to the train/val/test split and XGBoost. |
| `feature_cols` | no | `null` | Explicit list of feature column names to use. If `null`, feature columns are inferred automatically (see below). |
| `min_cells` | no | `250` | If set to an integer, any non-wild-type variant with fewer than this many cells is removed before splitting. Wild-type cells are never filtered by this option. |
| `downsample_wt` | no | `true` | If `true`, wild-type cells are randomly downsampled to match the cell count of the largest remaining variant before splitting. Uses `seed` for reproducibility. |

## `xgboost`

| Key | Default | Description |
|-----|---------|-------------|
| `num_boost_round` | `100` | Maximum number of boosting rounds. |
| `early_stopping_rounds` | `5` | Training stops if validation AUC does not improve for this many consecutive rounds. The best round is retained. |
| `weigh_samples` | `true` | If `true`, apply balanced class weights to the training set. Recommended when classes are imbalanced. |

## `xgboost.params`

Passed directly to `xgb.train`. Any parameter supported by XGBoost can be added here. `objective`, `eval_metric`, and `seed` are set automatically and should not be specified.

| Key | Default | Description |
|-----|---------|-------------|
| `nthread` | `-1` | Number of CPU threads for XGBoost. `-1` uses all available cores. |
| `max_depth` | `3` | Maximum depth of each tree. Shallower trees reduce overfitting. |
| `colsample_bytree` | `0.7` | Fraction of features randomly sampled when building each tree. |
| `colsample_bylevel` | `0.7` | Fraction of features randomly sampled at each tree level. |
| `colsample_bynode` | `0.7` | Fraction of features randomly sampled at each split node. |
| `subsample` | `0.5` | Fraction of training rows randomly sampled per tree. |

## Feature column detection

By default, feature columns are inferred automatically: any column whose name starts with an uppercase letter **and** contains an underscore is treated as a feature. This matches the CellProfiler naming convention (e.g. `Intensity_MeanIntensity_DAPI`, `Texture_Variance_CY5_3_00`). All other columns (including `label_col`) are ignored.

To use a specific set of columns instead, set `app.feature_cols` to an explicit list:

```yaml
app:
  feature_cols:
    - Intensity_MeanIntensity_DAPI
    - Texture_Variance_CY5_3_00
```
