"""Build aligned feature matrices; callers choose pregame-safe features."""
from __future__ import annotations
from typing import Any
import pandas as pd
import polars as pl

CATEGORICAL_FEATURES = ("player_id", "team", "opponent", "roof", "surface")

RNG_SEED = 0

LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": RNG_SEED,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
}

def _design(
    train: pl.DataFrame,
    test: pl.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    present = [c for c in columns if c in train.columns and c in test.columns]
    train_pdf = train.select(present).to_pandas()
    test_pdf = test.select(present).to_pandas()
    keep: list[str] = []
    cat_cols: list[str] = []
    for col in present:
        if col in CATEGORICAL_FEATURES:
            train_s = train_pdf[col].astype("string")
            test_s = test_pdf[col].astype("string")
            if train_s.notna().sum() == 0:
                continue
            categories = pd.Index(train_s.dropna().unique())
            train_pdf[col] = pd.Categorical(train_s, categories=categories)
            test_pdf[col] = pd.Categorical(test_s, categories=categories)
            cat_cols.append(col)
            keep.append(col)
            continue
        series = pd.to_numeric(train_pdf[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        train_pdf[col] = series
        test_pdf[col] = pd.to_numeric(test_pdf[col], errors="coerce")
        keep.append(col)
    if not keep:
        raise ValueError("no usable features")
    return train_pdf[keep], test_pdf[keep], tuple(cat_cols)
