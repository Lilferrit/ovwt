import functools
import logging
import pathlib
import pickle
import sys
from os import PathLike
from typing import Optional

import hydra
import numpy as np
import polars as pl
import sklearn.metrics
import sklearn.model_selection
import sklearn.utils
import xgboost as xgb
from omegaconf import DictConfig


def convert_labels_to_boolean(labels: np.ndarray, wt_label: str) -> np.ndarray:
    """
    Converts an array of labels to boolean values.

    The function identifies the unique classes in the input array and maps the first
    class to False and the second class to True. If there are more than two unique
    classes, a ValueError is raised.

    Args:
        labels (np.ndarray):
            An array of labels to be converted.
        wt_label (str):
            The label corresponding to the positive class (True).
    """
    return labels == wt_label


def get_dmatrix(
    df: pl.DataFrame,
    label_col: str,
    wt_label: str,
    weight: Optional[np.ndarray] = None,
) -> xgb.DMatrix:
    """
    Converts a Polars DataFrame into an XGBoost DMatrix.

    Args:
        df (pl.DataFrame):
            DataFrame containing feature columns and the label column.
        label_col (str):
            The name of the label column.
        wt_label (str):
            The label value corresponding to the wild-type (positive) class.
        weight (Optional[np.ndarray], optional):
            Sample weights. Default is None.

    Returns:
        xgb.DMatrix:
            The XGBoost DMatrix with boolean labels.
    """
    feature_cols = [col for col in df.columns if col != label_col]
    x = df.select(feature_cols).to_numpy()
    y = convert_labels_to_boolean(df.get_column(label_col).to_numpy(), wt_label)
    return xgb.DMatrix(x, label=y, weight=weight)


def train_xgboost(
    train: pl.DataFrame,
    val: pl.DataFrame,
    cfg: DictConfig,
) -> xgb.Booster:
    """
    Trains an XGBoost classifier on the provided training data.

    Args:
        train (pl.DataFrame):
            Training data including feature columns and the label column.
        val (pl.DataFrame):
            Validation data including feature columns and the label column.
        cfg (DictConfig):
            Hydra config. Uses cfg.app.label_col, cfg.app.wt_label, and
            all fields of cfg.xgboost.

    Returns:
        xgb.Booster:
            The trained XGBoost booster.
    """
    label_col = cfg.app.label_col
    wt_label = cfg.app.wt_label

    y_train = convert_labels_to_boolean(
        train.get_column(label_col).to_numpy(), wt_label
    )
    sample_weight = (
        sklearn.utils.compute_sample_weight("balanced", y_train)
        if cfg.xgboost.weigh_samples
        else None
    )

    dtrain = get_dmatrix(train, label_col, wt_label, weight=sample_weight)
    deval = get_dmatrix(val, label_col, wt_label)

    params = {
        "objective": "binary:logistic",
        "max_depth": cfg.xgboost.max_depth,
        "colsample_bytree": cfg.xgboost.colsample_bytree,
        "colsample_bylevel": cfg.xgboost.colsample_bylevel,
        "colsample_bynode": cfg.xgboost.colsample_bynode,
        "subsample": cfg.xgboost.subsample,
        "eval_metric": "auc",
        "seed": cfg.app.seed,
        "nthread": cfg.xgboost.nthread,
    }

    return xgb.train(
        params,
        dtrain,
        num_boost_round=cfg.xgboost.num_boost_round,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=cfg.xgboost.early_stopping_rounds,
        verbose_eval=True,
    )


def evaluate(
    df: pl.DataFrame, model: xgb.Booster, label_col: str, wt_label: str
) -> tuple[float, float]:
    """
    Evaluates an XGBoost model on a dataset, returning AUROC and accuracy.

    Args:
        df (pl.DataFrame):
            Dataset including feature columns and the label column.
        model (xgb.Booster):
            The trained XGBoost booster to evaluate.
        label_col (str):
            The name of the label column.
        wt_label (str):
            The label value corresponding to the wild-type (positive) class.

    Returns:
        tuple[float, float]:
            A tuple of (AUROC, accuracy), where accuracy is computed at a
            decision threshold of 0.5.
    """
    dmatrix = get_dmatrix(df, label_col, wt_label)
    y_true = dmatrix.get_label()
    y_prob = model.predict(dmatrix)
    auroc = sklearn.metrics.roc_auc_score(y_true, y_prob)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_prob >= 0.5)

    return auroc, accuracy


def test_xgboost(
    model: xgb.Booster,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    cfg: DictConfig,
) -> dict:
    """
    Computes the train, validation, and test AUC and accuracy.

    Args:
        model (xgb.Booster):
            The trained XGBoost booster to evaluate.
        train (pl.DataFrame):
            Training DataFrame including feature and label columns.
        val (pl.DataFrame):
            Validation DataFrame including feature and label columns.
        test (pl.DataFrame):
            Test DataFrame including feature and label columns.
        cfg (DictConfig):
            Hydra config. Uses cfg.app.label_col and cfg.app.wt_label.

    Returns:
        dict:
            A dictionary with the keys:
                - "variant": The first non wt_label value in the label column.
                - "train_auroc": The AUC on the training set.
                - "train_accuracy": The accuracy on the training set.
                - "val_auroc": The AUC on the validation set.
                - "val_accuracy": The accuracy on the validation set.
                - "test_auroc": The AUC on the test set.
                - "test_accuracy": The accuracy on the test set.
    """
    label_col = cfg.app.label_col
    wt_label = cfg.app.wt_label

    variant = next(
        v for v in train.get_column(label_col).unique().to_list() if v != wt_label
    )

    evaluate_wrapper = functools.partial(
        evaluate, model=model, label_col=label_col, wt_label=wt_label
    )

    train_auroc, train_accuracy = evaluate_wrapper(train)
    val_auroc, val_accuracy = evaluate_wrapper(val)
    test_auroc, test_accuracy = evaluate_wrapper(test)

    return {
        "variant": variant,
        "train_auroc": train_auroc,
        "train_accuracy": train_accuracy,
        "val_auroc": val_auroc,
        "val_accuracy": val_accuracy,
        "test_auroc": test_auroc,
        "test_accuracy": test_accuracy,
    }


def read_feature_file(file_path: PathLike) -> pl.DataFrame:
    """
    Reads a feature file and returns a Polars DataFrame.

    Args:
        file_path (PathLike):
            Path to the feature file. Supported formats: .parquet, .pq, .csv.

    Returns:
        pl.DataFrame:
            The feature data as a Polars DataFrame.

    Raises:
        ValueError:
            If the file extension is not supported.
    """
    path = pathlib.Path(file_path)
    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pl.read_parquet(path)
    elif suffix == ".csv":
        return pl.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. Expected .parquet or .csv"
        )


def train_test_val_split(
    data_df: pl.DataFrame,
    cfg: DictConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits the data into an 8:1:1 train/test/validation split.

    The split is stratified based on the label column to ensure that the distribution of
    classes is preserved across the train, test, and validation sets.

    Args:
        data_df (pl.DataFrame):
            The input data as a Polars DataFrame.
        cfg (DictConfig):
            Hydra config. Uses cfg.app.label_col and cfg.app.seed. Feature
            columns are inferred as columns that start with an uppercase letter
            and contain an underscore, matching CellProfiler feature naming
            conventions.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
            DataFrames for train, test, and validation sets respectively, each
            containing the feature columns and the label column.
    """
    label_col = cfg.app.label_col
    feature_cols = [col for col in data_df.columns if col[0].isupper() and "_" in col]

    select_cols = feature_cols + [label_col]
    labels = data_df.get_column(label_col).to_numpy()
    data_df = data_df.select(select_cols).with_row_index("__idx__")
    all_idx = data_df.get_column("__idx__").to_numpy()

    train_idx, val_test_idx = sklearn.model_selection.train_test_split(
        all_idx,
        test_size=0.2,
        stratify=labels,
        random_state=cfg.app.seed,
    )

    test_idx, val_idx = sklearn.model_selection.train_test_split(
        val_test_idx,
        test_size=0.5,
        stratify=labels[val_test_idx],
        random_state=cfg.app.seed,
    )

    def select_rows(idx: np.ndarray) -> pl.DataFrame:
        return data_df.filter(pl.col("__idx__").is_in(idx)).select(select_cols)

    return select_rows(train_idx), select_rows(test_idx), select_rows(val_idx)


def configure_logging(
    log_file: Optional[PathLike] = None,
    level: str = "INFO",
) -> None:
    """
    Configures root logger with a stdout handler and optional file handler.

    No-ops if the root logger already has handlers (e.g. called twice).

    Args:
        log_file (Optional[PathLike]):
            If provided, log messages are also written to this file.
        level (str):
            Logging level name (e.g. ``"INFO"``, ``"DEBUG"``). Case-insensitive.
    """
    if logging.root.handlers:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


@hydra.main(config_path="pkg://ovwt.conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Trains and evaluates one XGBoost classifier per variant vs. wild-type.

    Performs a single stratified 8:1:1 train/test/val split on the full
    dataset, then for each unique non-wild-type label trains an XGBoost model
    on the rows belonging to that variant or the wild-type. Results are written
    to ``results.csv`` and trained models are pickled to ``models.pkl`` in
    ``out_dir``.

    Args:
        cfg (DictConfig):
            Hydra config with two groups:
                - cfg.app: feature_file, label_col, wt_label, out_dir
                - cfg.xgboost: XGBoost hyperparameters and training options
    """
    out_dir = pathlib.Path(cfg.app.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir / "ovwt.log", level=cfg.app.log_level)

    logging.info("Reading feature file: %s", cfg.app.feature_file)
    feature_df = read_feature_file(cfg.app.feature_file)

    unique_vars = feature_df.get_column(cfg.app.label_col).unique().to_list()
    variants = [v for v in unique_vars if v != cfg.app.wt_label]
    logging.info("Found %d variant(s) to classify: %s", len(variants), variants)

    train_all, test_all, val_all = train_test_val_split(feature_df, cfg)
    logging.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_all),
        len(val_all),
        len(test_all),
    )

    results = []
    models = {}

    for v in variants:
        logging.info("Training model for variant '%s' vs. '%s'", v, cfg.app.wt_label)
        keep = pl.col(cfg.app.label_col).is_in([v, cfg.app.wt_label])
        
        train, test, val = (
            train_all.filter(keep),
            test_all.filter(keep),
            val_all.filter(keep),
        )
        
        logging.info(
            "Subset sizes — train: %d, val: %d, test: %d",
            len(train),
            len(val),
            len(test),
        )

        model = train_xgboost(train, val, cfg)
        result = test_xgboost(model, train, val, test, cfg)
        
        logging.info(
            "Results for '%s': train_auroc=%.4f, val_auroc=%.4f, test_auroc=%.4f",
            v,
            result["train_auroc"],
            result["val_auroc"],
            result["test_auroc"],
        )
        
        results.append(result)
        models[v] = model

    results_df = pl.DataFrame(results)
    results_path = out_dir / "results.csv"
    results_df.write_csv(results_path)
    logging.info("Results written to %s", results_path)

    models_path = out_dir / "models.pkl"
    logging.info("Writing models to %s", models_path)
    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    logging.info("Done")
