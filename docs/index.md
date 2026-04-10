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

## Outputs

All outputs are written to `out_dir`.

| File | Description |
|------|-------------|
| `results.csv` | One row per variant with columns `variant`, `train_auroc`, `train_accuracy`, `val_auroc`, `val_accuracy`, `test_auroc`, `test_accuracy`. |
| `models.pkl` | Python pickle file containing a `dict` mapping each variant label to its trained `xgboost.Booster`. |
| `ovwt.log` | Full log of the run, mirrored from stdout. |

## Installation

Requires Python 3.13+. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```
