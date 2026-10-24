"""Shared plumbing for the central comparison.

Direct DRF vs multivariate DRF vs hierarchical parametric model, on identical
week-ahead folds, features, and fantasy scoring rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from .metrics import randomized_pit, row_crps  # noqa: F401  (re-exported)
from .uncertainty import score_delta_cis

# The WR stat line, in the order used everywhere in this experiment.  The plan's
# vector is (REC, Y_rec, TD_rec, N_rush, Y_rush, TD_rush); ``targets`` is added
# because the hierarchical chain needs the opportunity stage.
STAT_COLUMNS: tuple[str, ...] = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
)
STAT_INDEX: dict[str, int] = {name: i for i, name in enumerate(STAT_COLUMNS)}
COMPONENT_LABELS: tuple[str, ...] = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fantasy_points",
)
# Alpha is the per-reception coefficient: 0 = standard, 0.5 = half PPR, 1 = full PPR.
PPR_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0)


def fantasy_points(stats: np.ndarray, *, ppr: float = 1.0) -> np.ndarray:
    """Full scrimmage points from a joint stat sample of shape (..., len(STAT_COLUMNS))."""
    stats = np.asarray(stats, dtype=float)
    receptions = stats[..., STAT_INDEX["receptions"]]
    return (
        ppr * receptions
        + 0.1 * (stats[..., STAT_INDEX["receiving_yards"]] + stats[..., STAT_INDEX["rushing_yards"]])
        + 6.0 * (stats[..., STAT_INDEX["receiving_tds"]] + stats[..., STAT_INDEX["rushing_tds"]])
    )


def receiving_points(stats: np.ndarray, *, ppr: float = 1.0) -> np.ndarray:
    """Receiving-only points, the scoring used by the earlier WR harness."""
    stats = np.asarray(stats, dtype=float)
    return (
        ppr * stats[..., STAT_INDEX["receptions"]]
        + 0.1 * stats[..., STAT_INDEX["receiving_yards"]]
        + 6.0 * stats[..., STAT_INDEX["receiving_tds"]]
    )


def observed_stats(frame: pl.DataFrame) -> np.ndarray:
    """(n, d) observed stat matrix in STAT_COLUMNS order."""
    missing = [name for name in STAT_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing stat columns: {missing}")
    return frame.select(STAT_COLUMNS).to_numpy().astype(float)


def row_energy_score(samples: np.ndarray, observed: np.ndarray, *, chunk: int = 96) -> np.ndarray:
    """Energy score, lower is better. ``samples`` is (n, m, d), ``observed`` is (n, d)."""
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if samples.ndim != 3 or observed.ndim != 2 or samples.shape[0] != observed.shape[0]:
        raise ValueError("expected samples (n, m, d) and observed (n, d)")
    n, m, d = samples.shape
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = samples[start:stop]
        term1 = np.linalg.norm(block - observed[start:stop, None, :], axis=2).mean(axis=1)
        term2 = np.array([pdist(row, metric="euclidean").mean() for row in block])
        out[start:stop] = term1 - 0.5 * term2
    return out


def row_variogram_score(
    samples: np.ndarray, observed: np.ndarray, *, power: float = 0.5, chunk: int = 256
) -> np.ndarray:
    """Variogram score (Scheuerer & Hamill), lower is better. Equal pair weights."""
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if samples.ndim != 3 or observed.ndim != 2:
        raise ValueError("expected samples (n, m, d) and observed (n, d)")
    n, _, d = samples.shape
    out = np.empty(n, dtype=float)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = samples[start:stop]
        obs_gap = np.abs(observed[start:stop, :, None] - observed[start:stop, None, :]) ** power
        pred_gap = np.abs(block[:, :, :, None] - block[:, :, None, :]) ** power
        out[start:stop] = ((obs_gap - pred_gap.mean(axis=1)) ** 2).sum(axis=(1, 2))
    return out


def component_samples(joint: np.ndarray, name: str) -> np.ndarray:
    """Marginal draws for one component of a (n, m, d) joint sample."""
    return np.asarray(joint, dtype=float)[..., STAT_INDEX[name]]


def interval_summary(samples: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Coverage and width at the nominal levels used by the existing harness."""
    out: dict[str, float] = {}
    for nominal in (0.50, 0.70, 0.80, 0.90, 0.95):
        alpha = (1.0 - nominal) / 2.0
        lo, hi = np.quantile(samples, (alpha, 1.0 - alpha), axis=1)
        label = int(round(nominal * 100))
        out[f"coverage_{label}"] = float(np.mean((observed >= lo) & (observed <= hi)))
        out[f"width_{label}"] = float(np.mean(hi - lo))
    return out


def reliability(
    probability: np.ndarray, event: np.ndarray, edges: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
) -> dict[str, Any]:
    """Fixed-bin reliability table and Brier score for one threshold event."""
    probability = np.asarray(probability, dtype=float)
    event = np.asarray(event, dtype=bool)
    bins: list[dict[str, Any]] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        upper_open = i < len(edges) - 2
        mask = (probability >= lo) & ((probability < hi) if upper_open else (probability <= hi))
        row: dict[str, Any] = {"lo": float(lo), "hi": float(hi), "n": int(mask.sum())}
        if mask.any():
            row["predicted"] = float(probability[mask].mean())
            row["empirical"] = float(event[mask].mean())
        bins.append(row)
    return {
        "mean_probability": float(probability.mean()),
        "empirical_rate": float(event.mean()),
        "brier": float(np.mean((probability - event.astype(float)) ** 2)),
        "bins": bins,
    }


def pit_histogram(pit: np.ndarray, bins: int = 10) -> list[int]:
    counts, _ = np.histogram(np.asarray(pit, dtype=float), bins=np.linspace(0.0, 1.0, bins + 1))
    return counts.tolist()


def marginal_summary(samples: np.ndarray, observed: np.ndarray, *, pit_uniforms: np.ndarray) -> dict[str, Any]:
    """Summary for one scalar forecast family (one component or one scoring rule)."""
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    block: dict[str, Any] = {
        "n": int(len(observed)),
        "mean_forecast": float(samples.mean()),
        "mean_observed": float(observed.mean()),
        "mae": float(np.mean(np.abs(observed - samples.mean(axis=1)))),
        "crps": float(row_crps(samples, observed).mean()),
    }
    block.update(interval_summary(samples, observed))
    block["pit_counts"] = pit_histogram(randomized_pit(samples, observed, pit_uniforms))
    return block


def tail_summary(
    samples: np.ndarray,
    observed: np.ndarray,
    *,
    thresholds: tuple[float, ...],
    direction: str = "ge",
) -> dict[str, Any]:
    """Reliability for right-tail (or left-tail) events on one scalar forecast."""
    samples = np.asarray(samples, dtype=float)
    observed = np.asarray(observed, dtype=float)
    out: dict[str, Any] = {}
    for threshold in thresholds:
        probability = (samples >= threshold).mean(axis=1) if direction == "ge" else (samples < threshold).mean(axis=1)
        event = observed >= threshold if direction == "ge" else observed < threshold
        key = f"{direction}_{int(threshold)}" if float(threshold).is_integer() else f"{direction}_{threshold}"
        out[key] = reliability(probability, event)
    return out


def paired_delta(
    delta: np.ndarray,
    players: np.ndarray,
    weeks: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Player-clustered CI plus week-clustered sensitivity for a per-row delta."""
    return score_delta_cis(np.asarray(delta, dtype=float), players, weeks, n_boot=n_boot, rng=rng)


def crps_delta(model_crps: np.ndarray, baseline_crps: np.ndarray) -> np.ndarray:
    """Positive means the model beats the baseline."""
    return np.asarray(baseline_crps, dtype=float) - np.asarray(model_crps, dtype=float)
