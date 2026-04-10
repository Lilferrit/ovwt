import logging
import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from ovwt import (
    configure_logging,
    convert_labels_to_boolean,
    get_dmatrix,
    read_feature_file,
    test_xgboost as evaluate_splits,
    train_test_val_split,
    train_xgboost,
)


def _make_df(
    n: int = 20,
    label_col: str = "label",
    wt_label: str = "WT",
    variant_label: str = "V1",
) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    labels = [wt_label] * (n // 2) + [variant_label] * (n // 2)
    return pl.DataFrame(
        {
            "Intensity_Mean": rng.random(n).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            label_col: labels,
        }
    )


def _split_cfg(label_col: str = "label") -> OmegaConf:
    return OmegaConf.create({"app": {"label_col": label_col, "seed": 0}})


# ---------------------------------------------------------------------------
# convert_labels_to_boolean
# ---------------------------------------------------------------------------


def test_convert_labels_to_boolean_wt_maps_to_true():
    labels = np.array(["WT", "V1", "WT", "V1"])
    result = convert_labels_to_boolean(labels, "WT")
    np.testing.assert_array_equal(result, [True, False, True, False])


def test_convert_labels_to_boolean_all_wt():
    labels = np.array(["WT", "WT", "WT"])
    assert convert_labels_to_boolean(labels, "WT").all()


def test_convert_labels_to_boolean_none_wt():
    labels = np.array(["V1", "V2", "V3"])
    assert not convert_labels_to_boolean(labels, "WT").any()


# ---------------------------------------------------------------------------
# get_dmatrix
# ---------------------------------------------------------------------------


def test_get_dmatrix_label_values():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    assert set(dm.get_label()) == {0.0, 1.0}


def test_get_dmatrix_wt_label_is_true():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    # 5 WT rows → 5 True (1.0) labels
    assert dm.get_label().sum() == 5.0


def test_get_dmatrix_shape():
    df = _make_df(n=20)
    dm = get_dmatrix(df, "label", "WT")
    assert dm.num_row() == 20
    assert dm.num_col() == 2  # Intensity_Mean, Texture_Var


def test_get_dmatrix_with_weights():
    df = _make_df(n=10)
    weights = np.full(10, 2.0)
    dm = get_dmatrix(df, "label", "WT", weight=weights)
    np.testing.assert_array_equal(dm.get_weight(), weights)


def test_get_dmatrix_no_weights_by_default():
    df = _make_df(n=10)
    dm = get_dmatrix(df, "label", "WT")
    assert len(dm.get_weight()) == 0


# ---------------------------------------------------------------------------
# read_feature_file
# ---------------------------------------------------------------------------


def test_read_feature_file_parquet(tmp_path):
    df = pl.DataFrame({"Intensity_Mean": [1.0, 2.0], "label": ["WT", "V1"]})
    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    assert read_feature_file(path).equals(df)


def test_read_feature_file_pq_extension(tmp_path):
    df = pl.DataFrame({"Intensity_Mean": [1.0, 2.0], "label": ["WT", "V1"]})
    path = tmp_path / "data.pq"
    df.write_parquet(path)
    assert read_feature_file(path).equals(df)


def test_read_feature_file_csv(tmp_path):
    df = pl.DataFrame({"Intensity_Mean": [1.0, 2.0], "label": ["WT", "V1"]})
    path = tmp_path / "data.csv"
    df.write_csv(path)
    assert read_feature_file(path).equals(df)


def test_read_feature_file_unsupported_extension(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_feature_file(path)


# ---------------------------------------------------------------------------
# train_test_val_split
# ---------------------------------------------------------------------------


def test_train_test_val_split_sizes():
    # 100 rows → 80 train, 10 test, 10 val
    df = _make_df(n=100)
    train, test, val = train_test_val_split(df, _split_cfg())
    assert len(train) == 80
    assert len(test) == 10
    assert len(val) == 10


def test_train_test_val_split_no_overlap():
    df = _make_df(n=100)
    # Add a unique id column to track rows across splits
    df = df.with_row_index("__row__")
    train, test, val = train_test_val_split(df, _split_cfg())
    # __row__ is lowercase so not selected as a feature; total rows should sum to 100
    assert len(train) + len(test) + len(val) == 100


def test_train_test_val_split_excludes_non_feature_columns():
    df = _make_df(n=100).with_columns(pl.lit(0).alias("lowercase_extra"))
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        assert "lowercase_extra" not in split.columns
        assert set(split.columns) == {"Intensity_Mean", "Texture_Var", "label"}


def test_train_test_val_split_preserves_class_ratio():
    df = _make_df(n=100)
    train, test, val = train_test_val_split(df, _split_cfg())
    for split in (train, test, val):
        counts = split.get_column("label").value_counts()
        n_wt = counts.filter(pl.col("label") == "WT")["count"][0]
        n_v1 = counts.filter(pl.col("label") == "V1")["count"][0]
        assert n_wt == n_v1


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_default_level():
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = []
        configure_logging()
        mock_basicconfig.assert_called_once()
        assert mock_basicconfig.call_args.kwargs["level"] == logging.INFO


def test_configure_logging_custom_level():
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = []
        configure_logging(level="DEBUG")
        assert mock_basicconfig.call_args.kwargs["level"] == logging.DEBUG


def test_configure_logging_level_case_insensitive():
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = []
        configure_logging(level="warning")
        assert mock_basicconfig.call_args.kwargs["level"] == logging.WARNING


def test_configure_logging_handlers_include_stdout():
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = []
        configure_logging()
        handlers = mock_basicconfig.call_args.kwargs["handlers"]
        assert any(type(h) is logging.StreamHandler for h in handlers)


def test_configure_logging_file_handler(tmp_path):
    log_file = tmp_path / "test.log"
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = []
        configure_logging(log_file=log_file)
        handlers = mock_basicconfig.call_args.kwargs["handlers"]
        assert any(isinstance(h, logging.FileHandler) for h in handlers)


def test_configure_logging_no_file_handler_by_default():
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = []
        configure_logging()
        handlers = mock_basicconfig.call_args.kwargs["handlers"]
        assert not any(isinstance(h, logging.FileHandler) for h in handlers)


def test_configure_logging_noop_if_handlers_exist():
    with (
        patch("logging.root") as mock_root,
        patch("logging.basicConfig") as mock_basicconfig,
    ):
        mock_root.handlers = [logging.NullHandler()]
        configure_logging(level="DEBUG")
        mock_basicconfig.assert_not_called()


# ---------------------------------------------------------------------------
# train_xgboost / test_xgboost
# ---------------------------------------------------------------------------


def _make_xgb_cfg(weigh_samples: bool = True) -> OmegaConf:
    return OmegaConf.create(
        {
            "app": {"label_col": "label", "wt_label": "WT", "seed": 0},
            "xgboost": {
                "nthread": 1,
                "max_depth": 2,
                "colsample_bytree": 1.0,
                "colsample_bylevel": 1.0,
                "colsample_bynode": 1.0,
                "subsample": 1.0,
                "num_boost_round": 5,
                "early_stopping_rounds": 3,
                "weigh_samples": weigh_samples,
            },
        }
    )


def _make_separable_df(n: int = 60) -> pl.DataFrame:
    """Linearly separable dataset: WT has high Intensity_Mean, V1 has low."""
    rng = np.random.default_rng(42)
    half = n // 2
    return pl.DataFrame(
        {
            "Intensity_Mean": np.concatenate(
                [
                    rng.uniform(0.6, 1.0, half),  # WT
                    rng.uniform(0.0, 0.4, half),  # V1
                ]
            ).tolist(),
            "Texture_Var": rng.random(n).tolist(),
            "label": ["WT"] * half + ["V1"] * half,
        }
    )


@pytest.fixture()
def trained_model_and_splits():
    df = _make_separable_df(n=60)
    cfg = _make_xgb_cfg()
    wt = df.filter(pl.col("label") == "WT")
    v1 = df.filter(pl.col("label") == "V1")
    # 20 + 20 train, 5 + 5 val, 5 + 5 test
    train = pl.concat([wt[:20], v1[:20]])
    val = pl.concat([wt[20:25], v1[20:25]])
    test = pl.concat([wt[25:], v1[25:]])
    model = train_xgboost(train, val, cfg)
    return model, train, val, test, cfg


def test_train_xgboost_returns_booster():
    import xgboost as xgb

    df = _make_separable_df(n=60)
    cfg = _make_xgb_cfg()
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], cfg)
    assert isinstance(model, xgb.Booster)


def test_train_xgboost_without_sample_weights():
    import xgboost as xgb

    df = _make_separable_df(n=60)
    cfg = _make_xgb_cfg(weigh_samples=False)
    half = len(df) // 2
    model = train_xgboost(df[:half], df[half:], cfg)
    assert isinstance(model, xgb.Booster)


def test_test_xgboost_result_keys(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    expected_keys = {
        "variant",
        "train_auroc",
        "train_accuracy",
        "val_auroc",
        "val_accuracy",
        "test_auroc",
        "test_accuracy",
    }
    assert set(result.keys()) == expected_keys


def test_test_xgboost_variant_name(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    assert result["variant"] == "V1"


def test_test_xgboost_auroc_in_range(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    for key in ("train_auroc", "val_auroc", "test_auroc"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_test_xgboost_accuracy_in_range(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    for key in ("train_accuracy", "val_accuracy", "test_accuracy"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_test_xgboost_separable_data_high_auroc(trained_model_and_splits):
    model, train, val, test, cfg = trained_model_and_splits
    result = evaluate_splits(model, train, val, test, cfg)
    assert result["train_auroc"] > 0.9
