"""Leakage-safe lagged player features.

A prior game is eligible only when its kickoff plus ``GAME_KNOWABLE_LAG_HOURS``
is strictly before this row's ``prediction_timestamp``. Season summaries use
completed prior seasons only; in-season form lives in last / EWMA / last-16.
Null timestamps stay null — there is no week-order fallback.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .schema import (
    EWMA_MEDIUM_HALF_LIFE_GAMES, EWMA_SHORT_HALF_LIFE_GAMES,
    LAG_HORIZONS, LAG_SOURCE_COLUMNS, LAST_N_GAMES,
)
GAME_KNOWABLE_LAG_HOURS = 4

_SEASON_WINDOWS = (1, 2, 3)


def _feature_names(sources: tuple[str, ...]) -> tuple[str, ...]:
    lagged = tuple(f"{src}_{horizon}" for src in sources for horizon in LAG_HORIZONS)
    return lagged + ("n_prior_games", "days_since_last_game")


def _ewma(values_oldest_to_newest: np.ndarray, half_life: float) -> float:
    """Newest eligible game has age 0. Half-life is in games, including null slots.

    Per-game decay is ``0.5 ** (1 / half_life)``.
    """
    finite = np.isfinite(values_oldest_to_newest)
    if not finite.any():
        return np.nan
    decay = 0.5 ** (1.0 / half_life)
    ages = np.arange(values_oldest_to_newest.size - 1, -1, -1, dtype=float)
    weights = (decay**ages)[finite]
    return float(np.dot(weights, values_oldest_to_newest[finite]) / weights.sum())


def _lag_one_player(
    kickoff: np.ndarray,
    pred_ts: np.ndarray,
    season: np.ndarray,
    values: dict[str, np.ndarray],
    history_eligible: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n = kickoff.shape[0]
    knowable_at = kickoff + np.timedelta64(GAME_KNOWABLE_LAG_HOURS, "h")
    valid = ~np.isnat(knowable_at)
    if history_eligible is not None:
        valid &= history_eligible
    valid_idx = np.flatnonzero(valid)
    order = np.argsort(knowable_at[valid_idx], kind="mergesort")
    valid_idx = valid_idx[order]
    sorted_know = knowable_at[valid_idx]

    out: dict[str, np.ndarray] = {
        f"{src}_{horizon}": np.full(n, np.nan) for src in values for horizon in LAG_HORIZONS
    }
    n_prior = np.full(n, np.nan)
    days = np.full(n, np.nan)

    for i in range(n):
        ts = pred_ts[i]
        if np.isnat(ts):
            continue
        k = int(np.searchsorted(sorted_know, ts, side="left"))
        eligible = valid_idx[:k]
        eligible = eligible[eligible != i]
        n_prior[i] = float(eligible.size)
        if eligible.size == 0:
            continue

        eligible = eligible[np.argsort(kickoff[eligible], kind="mergesort")]
        latest = int(eligible[-1])
        if not np.isnat(kickoff[i]):
            delta = kickoff[i] - kickoff[latest]
            days[i] = delta / np.timedelta64(1, "h") / 24.0

        row_season = season[i]
        for src, series in values.items():
            ordered = series[eligible]
            finite = np.isfinite(ordered)
            out[f"{src}_last"][i] = ordered[-1]
            out[f"{src}_ewma_short"][i] = _ewma(ordered, EWMA_SHORT_HALF_LIFE_GAMES)
            out[f"{src}_ewma_med"][i] = _ewma(ordered, EWMA_MEDIUM_HALF_LIFE_GAMES)
            tail = ordered[-LAST_N_GAMES:]
            out[f"{src}_last16"][i] = float(np.nanmean(tail)) if np.isfinite(tail).any() else np.nan
            if np.isnan(row_season):
                continue
            for window in _SEASON_WINDOWS:
                lo = row_season - window
                hi = row_season - 1
                mask = (season[eligible] >= lo) & (season[eligible] <= hi) & finite
                if mask.any():
                    out[f"{src}_season_{window}"][i] = float(np.nanmean(ordered[mask]))

    out["n_prior_games"] = n_prior
    out["days_since_last_game"] = days
    return out


def add_lagged_player_features(
    df: pl.DataFrame,
    *,
    sources: tuple[str, ...] = LAG_SOURCE_COLUMNS,
) -> pl.DataFrame:
    """Append lagged columns. Input row order is preserved.

    ``sources`` missing from ``df`` are skipped. Re-calling replaces prior lag columns.
    """
    present = tuple(src for src in sources if src in df.columns)
    names = _feature_names(present)
    if df.height == 0:
        extras = [pl.Series(name, [], dtype=pl.Float64) for name in names if name not in df.columns]
        return df.with_columns(extras) if extras else df

    missing = {"player_id", "season", "kickoff", "prediction_timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    base = df.drop([c for c in names if c in df.columns])
    indexed = base.with_row_index("_row")
    work = indexed.select(
        pl.col("_row"),
        pl.when(pl.col("player_id").is_null())
        .then(pl.concat_str([pl.lit("\0"), pl.col("_row").cast(pl.Utf8)]))
        .otherwise(pl.col("player_id").cast(pl.Utf8))
        .alias("player_id"),
        pl.col("season").cast(pl.Float64, strict=False),
        pl.col("kickoff").cast(pl.Datetime("ns")),
        pl.col("prediction_timestamp").cast(pl.Datetime("ns")),
        *[pl.col(src).cast(pl.Float64, strict=False) for src in present],
    )

    player = work["player_id"].to_numpy()
    row_id = work["_row"].to_numpy()
    kickoff = work["kickoff"].to_numpy().astype("datetime64[ns]")
    pred_ts = work["prediction_timestamp"].to_numpy().astype("datetime64[ns]")
    season = work["season"].to_numpy().astype(float)
    value_cols = {src: work[src].to_numpy().astype(float) for src in present}
    work_eligible = (
        df["history_eligible"].fill_null(False).to_numpy().astype(bool)
        if "history_eligible" in df.columns else np.ones(df.height, dtype=bool)
    )

    blocks: dict[str, np.ndarray] = {name: np.full(df.height, np.nan) for name in names}
    if df.height:
        order = np.argsort(player, kind="mergesort")
        # argsort on the player array; split into contiguous groups after sorting.
        sorted_player = player[order]
        cuts = np.flatnonzero(sorted_player[1:] != sorted_player[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(order)]))
        for start, end in zip(starts, ends, strict=True):
            idx = order[start:end]
            lagged = _lag_one_player(
                kickoff[idx],
                pred_ts[idx],
                season[idx],
                {src: vals[idx] for src, vals in value_cols.items()},
                work_eligible[idx],
            )
            for name, arr in lagged.items():
                blocks[name][idx] = arr

    feature_frame = pl.DataFrame({"_row": row_id, **blocks}).with_columns(
        pl.col("n_prior_games").fill_nan(None).cast(pl.Int32, strict=False),
        *[pl.col(name).fill_nan(None) for name in names if name != "n_prior_games"],
    )
    return (
        indexed.join(feature_frame, on="_row", how="left")
        .sort("_row")
        .drop("_row")
    )


def freeze_history_at_season_start(
    df: pl.DataFrame,
    test_season: int,
    *,
    sources: tuple[str, ...] = LAG_SOURCE_COLUMNS,
) -> pl.DataFrame:
    """Recompute lags so `test_season` rows cannot see that season's games.

    Train rows keep their own timestamps, so their lags still include earlier
    weeks of those seasons. Every test-season row shares the earliest
    prediction timestamp in that season, which sits before its kickoffs, so
    in-season results are not history. That is the frozen protocol. Week-ahead
    scoring must use the unfrozen table instead.
    """
    if "season" not in df.columns or "prediction_timestamp" not in df.columns:
        raise ValueError("season and prediction_timestamp are required")
    test = df.filter(pl.col("season") == test_season)
    if test.height == 0:
        raise ValueError(f"no rows for season {test_season}")
    cutoff = test.select(pl.col("prediction_timestamp").drop_nulls().min()).item()
    if cutoff is None:
        raise ValueError(f"season {test_season} has no prediction_timestamp")
    frozen = df.with_columns(
        pl.when(pl.col("season") == test_season)
        .then(pl.lit(cutoff, dtype=df.schema["prediction_timestamp"]))
        .otherwise(pl.col("prediction_timestamp"))
        .alias("prediction_timestamp")
    )
    return add_lagged_player_features(frozen, sources=sources)


def leakage_safe_columns(
    columns: list[str] | tuple[str, ...],
    *,
    sources: tuple[str, ...] = LAG_SOURCE_COLUMNS,
) -> list[str]:
    """Lag names for ``sources`` only. Raw opportunity columns are post-game and are omitted."""
    allowed = set(_feature_names(sources))
    return [c for c in columns if c in allowed]
