"""Fit the existing WR chain on prepared historical data and save future draws."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from hierarchical_football import HierarchicalParametric
from hierarchical_football.common import STAT_COLUMNS
from hierarchical_football.scoring import scrimmage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--predict", type=Path, required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.draws <= 0 or args.trees <= 0:
        parser.error("draws and trees must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.output.suffix != ".npz":
        parser.error("output must end in .npz")
    train, predict = pl.read_parquet(args.train), pl.read_parquet(args.predict)
    if train.is_empty() or predict.is_empty():
        parser.error("training and prediction panels must be nonempty")
    for label, frame in (("training", train), ("prediction", predict)):
        required = {"season", "week", *args.features}
        missing = required.difference(frame.columns)
        if missing:
            parser.error(f"{label} panel missing columns: {sorted(missing)}")
        if frame.select(pl.any_horizontal(pl.col("season").is_null(), pl.col("week").is_null()).any()).item():
            parser.error(f"{label} time columns contain nulls")
    missing = set(STAT_COLUMNS).difference(train.columns)
    if missing:
        parser.error(f"training panel missing outcomes: {sorted(missing)}")
    if set(args.features).intersection((*STAT_COLUMNS, "fantasy_points")):
        parser.error("raw outcome columns cannot be selected as features")
    latest_train = max(train.select("season", "week").iter_rows())
    earliest_predict = min(predict.select("season", "week").iter_rows())
    if latest_train >= earliest_predict:
        parser.error("all training weeks must precede every prediction week")
    model = HierarchicalParametric(
        feature_columns=tuple(args.features), n_estimators=args.trees, td_mode="joint"
    ).fit(train)
    draws = model.sample(predict, n_draws=args.draws, rng=np.random.default_rng(args.seed))
    payload = {
        "draws": draws,
        "stat_columns": np.asarray(STAT_COLUMNS),
        "full_ppr": scrimmage(1.0)(draws, STAT_COLUMNS),
        "half_ppr": scrimmage(0.5)(draws, STAT_COLUMNS),
        "standard": scrimmage(0.0)(draws, STAT_COLUMNS),
        "feature_columns": np.asarray(args.features),
        "seed": np.asarray(args.seed),
    }
    for column in ("player_id", "team", "season", "week"):
        if column in predict.columns:
            payload[column] = predict[column].cast(pl.String).fill_null("").to_numpy().astype(str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Saved {len(predict)} rows with {args.draws} draws each to {args.output}")


if __name__ == "__main__":
    main()
