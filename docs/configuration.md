# Configuration

Configuration is handled by [Hydra](https://hydra.cc). Built-in defaults are provided for all optional parameters — only the required `app` fields need to be set.

Settings can be provided via a [YAML config file](usage.md#with-a-config-file), as [CLI overrides](usage.md#command-line-only), or a combination of both.

## `app`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `feature_file` | yes | — | Path to the input feature file (`.parquet`, `.pq`, or `.csv`). |
| `label_col` | yes | — | Name of the column containing cell labels. |
| `wt_label` | yes | — | Label value identifying wild-type cells. All other unique values in `label_col` are treated as variants. |
| `out_dir` | yes | — | Directory where outputs are written. Created if it does not exist. |
| `log_level` | no | `INFO` | Logging verbosity. One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. Case-insensitive. |
| `seed` | no | `42` | Random seed passed to the train/val/test split and XGBoost. |

## `xgboost`

| Key | Default | Description |
|-----|---------|-------------|
| `nthread` | `-1` | Number of CPU threads for XGBoost. `-1` uses all available cores. |
| `max_depth` | `3` | Maximum depth of each tree. Shallower trees reduce overfitting. |
| `colsample_bytree` | `0.7` | Fraction of features randomly sampled when building each tree. |
| `colsample_bylevel` | `0.7` | Fraction of features randomly sampled at each tree level. |
| `colsample_bynode` | `0.7` | Fraction of features randomly sampled at each split node. |
| `subsample` | `0.5` | Fraction of training rows randomly sampled per tree. |
| `num_boost_round` | `100` | Maximum number of boosting rounds. |
| `early_stopping_rounds` | `5` | Training stops if validation AUC does not improve for this many consecutive rounds. The best round is retained. |
| `weigh_samples` | `true` | If `true`, apply balanced class weights to the training set. Recommended when classes are imbalanced. |

## Feature column detection

Feature columns are inferred automatically from the input DataFrame: any column whose name starts with an uppercase letter **and** contains an underscore is treated as a feature. This matches the CellProfiler naming convention (e.g. `Intensity_MeanIntensity_DAPI`, `Texture_Variance_CY5_3_00`). All other columns (including `label_col`) are ignored.
