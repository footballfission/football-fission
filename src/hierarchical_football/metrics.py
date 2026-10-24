"""Evaluation summaries for sample-based fantasy-point forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .uncertainty import paired_crps_delta, paired_mae_delta, score_delta_cis

RELIABILITY_EDGES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
TAILS = ("lt_5", "ge_20", "ge_25", "ge_30")


def row_crps(samples: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Fast empirical CRPS in O(n * draws log draws)."""
    samples = np.asarray(samples, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != len(observations):
        raise ValueError("samples must have shape (n_rows, n_draws)")
    ordered = np.sort(samples, axis=1)
    m = ordered.shape[1]
    weights = 2.0 * np.arange(1, m + 1) - m - 1.0
    spread = (ordered * weights).sum(axis=1) / (m * m)
    return np.mean(np.abs(samples - observations[:, None]), axis=1) - spread


def randomized_pit(samples: np.ndarray, y: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Randomized PIT, including mass created by the zero floor."""
    y_col = np.asarray(y, dtype=float)[:, None]
    samples = np.asarray(samples, dtype=float)
    return np.mean(samples < y_col, axis=1) + np.asarray(uniforms) * np.mean(samples == y_col, axis=1)


def prediction_frame(
    keys: pl.DataFrame,
    samples: np.ndarray,
    *,
    model: str,
    pit_uniforms: np.ndarray,
    cross_rate: float | None = None,
) -> pl.DataFrame:
    """Create the stable row-level artifact, including optimizer-ready draws."""
    y = keys["actual"].to_numpy().astype(float)
    levels = (0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95)
    quantiles = np.quantile(samples, levels, axis=1)
    q = {f"q{int(level * 100):02d}": values for level, values in zip(levels, quantiles, strict=True)}
    return keys.with_columns(
        pl.lit(model).alias("model"),
        pl.Series("mean", samples.mean(axis=1)),
        pl.Series("sd", samples.std(axis=1)),
        *[pl.Series(name, values) for name, values in q.items()],
        pl.Series("crps", row_crps(samples, y)),
        pl.Series("pit", randomized_pit(samples, y, pit_uniforms)),
        pl.Series("p_lt_5", np.mean(samples < 5.0, axis=1)),
        pl.Series("p_ge_20", np.mean(samples >= 20.0, axis=1)),
        pl.Series("p_ge_25", np.mean(samples >= 25.0, axis=1)),
        pl.Series("p_ge_30", np.mean(samples >= 30.0, axis=1)),
        pl.Series("draws", samples.tolist()),
        pl.lit(cross_rate).cast(pl.Float64).alias("quantile_cross_rate"),
    )


def _reliability(probability: np.ndarray, occurred: np.ndarray) -> dict[str, Any]:
    bins: list[dict[str, Any]] = []
    edges = np.asarray(RELIABILITY_EDGES)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (probability >= lo) & ((probability <= hi) if i == len(edges) - 2 else (probability < hi))
        block: dict[str, Any] = {"lo": float(lo), "hi": float(hi), "n": int(mask.sum())}
        if mask.any():
            block.update(mean_predicted=float(probability[mask].mean()), empirical_rate=float(occurred[mask].mean()))
        bins.append(block)
    if probability.size == 0:
        return {"mean_predicted": None, "empirical_rate": None, "brier": None, "bins": bins}
    return {
        "mean_predicted": float(probability.mean()),
        "empirical_rate": float(occurred.mean()),
        "brier": float(np.mean((probability - occurred.astype(float)) ** 2)),
        "bins": bins,
    }


def summarize(frame: pl.DataFrame) -> dict[str, Any]:
    y = frame["actual"].to_numpy().astype(float)
    mean = frame["mean"].to_numpy().astype(float)
    pit = frame["pit"].to_numpy().astype(float)
    counts, edges = np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))
    tails = {
        "lt_5": (frame["p_lt_5"].to_numpy(), y < 5.0),
        "ge_20": (frame["p_ge_20"].to_numpy(), y >= 20.0),
        "ge_25": (frame["p_ge_25"].to_numpy(), y >= 25.0),
        "ge_30": (frame["p_ge_30"].to_numpy(), y >= 30.0),
    }
    return {
        "n_rows": frame.height,
        "n_players": frame["player_id"].n_unique(),
        "n_weeks": frame["week"].n_unique(),
        "mae": float(np.mean(np.abs(y - mean))),
        "crps": float(frame["crps"].mean()),
        "coverage_50": float(np.mean((y >= frame["q25"].to_numpy()) & (y <= frame["q75"].to_numpy()))),
        "mean_width_50": float((frame["q75"] - frame["q25"]).mean()),
        "coverage_70": float(np.mean((y >= frame["q15"].to_numpy()) & (y <= frame["q85"].to_numpy()))),
        "mean_width_70": float((frame["q85"] - frame["q15"]).mean()),
        "coverage_80": float(np.mean((y >= frame["q10"].to_numpy()) & (y <= frame["q90"].to_numpy()))),
        "mean_width_80": float((frame["q90"] - frame["q10"]).mean()),
        "coverage_90": float(np.mean((y >= frame["q05"].to_numpy()) & (y <= frame["q95"].to_numpy()))),
        "mean_width_90": float((frame["q95"] - frame["q05"]).mean()),
        "pit": {"counts": counts.tolist(), "edges": edges.tolist()},
        "tails": {name: _reliability(np.asarray(prob), np.asarray(event)) for name, (prob, event) in tails.items()},
    }


def comparison(
    model: pl.DataFrame,
    baseline: pl.DataFrame,
    *,
    n_boot: int,
    rng: np.random.Generator,
    baseline_name: str = "last16_residual",
) -> dict[str, Any]:
    keys = ["player_id", "season", "week"]
    left = model.sort(keys)
    right = baseline.sort(keys)
    if left.select(keys).to_dicts() != right.select(keys).to_dicts():
        raise ValueError("model and baseline rows do not align")
    if not np.allclose(left["actual"].to_numpy(), right["actual"].to_numpy()):
        raise ValueError("model and baseline observations do not align")
    y = left["actual"].to_numpy().astype(float)
    players = left["player_id"].to_numpy()
    weeks = left["week"].to_numpy()
    mae = paired_mae_delta(y, left["mean"].to_numpy(), right["mean"].to_numpy())
    crps = paired_crps_delta(left["crps"].to_numpy(), right["crps"].to_numpy())
    return {
        "note": f"positive delta means the model beats {baseline_name}",
        "mae_delta": score_delta_cis(mae, players, weeks, n_boot=n_boot, rng=rng),
        "crps_delta": score_delta_cis(crps, players, weeks, n_boot=n_boot, rng=rng),
    }
