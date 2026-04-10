# Usage

All XGBoost parameters and optional `app` fields (`log_level`, `seed`) have built-in defaults and do not need to be specified. Only the required `app` fields must be provided, either on the command line or via a config file.

## Command line only

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

## With a config file

For experiments you want to reproduce or share, write a YAML file with your settings:

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

`--config-path` is the directory containing the YAML file. `--config-name` is the filename without the `.yaml` extension.

You can still append CLI overrides after the config name to selectively override values:

```bash
ovwt --config-path /path/to/dir --config-name my_experiment \
  xgboost.max_depth=6
```

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

Serve the documentation locally:

```bash
uv run mkdocs serve
```
