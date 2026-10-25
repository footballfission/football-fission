"""Prepare a single chronological training/prediction bundle without training a model."""
from __future__ import annotations
from pathlib import Path
import json
import polars as pl
from .schema import POSITION_COLUMNS
from .store import atomic_json, write_frame


def split_week(processed: Path, output: Path, position: str, season: int, week: int):
    history = pl.read_parquet(processed / position / "history.parquet")
    future = pl.read_parquet(processed / position / "prediction.parquet")
    features = json.loads((processed / position / "features.json").read_text(encoding="utf-8"))
    panel = pl.concat([history, future], how="diagonal_relaxed")
    target = panel.filter((pl.col("season") == season) & (pl.col("week") == week))
    if target.is_empty():
        raise ValueError("No modeled player/team rows for requested position-week")
    cutoff = target["prediction_timestamp"].min()
    if cutoff is None:
        raise ValueError("Prediction rows have no cutoff")
    train = history.filter((pl.col("kickoff") + pl.duration(hours=4) < cutoff)
        & ((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))))
    if train.is_empty():
        raise ValueError("No completed training outcomes before target cutoff; ingest more history")
    metadata = [c for c in ("player_id", "team", "position", "season", "week", "game_id",
        "kickoff", "prediction_timestamp", "roster_status", "roster_carried_forward")
        if c in target.columns]
    # Keep actual outcomes in a separate evaluation file; they cannot enter the prediction panel.
    prediction = target.select(list(dict.fromkeys([*metadata, *features])))
    observed = target.select(list(dict.fromkeys([*metadata, *POSITION_COLUMNS[position], "fantasy_points"])))
    write_frame(output / "train.parquet", train)
    write_frame(output / "prediction.parquet", prediction)
    write_frame(output / "observed.parquet", observed)
    atomic_json(output / "features.json", features)
    atomic_json(output / "split_report.json", {"position": position, "season": season, "week": week,
        "earliest_cutoff": str(cutoff), "train_rows": train.height, "prediction_rows": prediction.height,
        "completed_history_only": True})
    return {"train_rows": train.height, "prediction_rows": prediction.height}
