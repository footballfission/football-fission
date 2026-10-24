"""Clustered bootstrap for holdout score deltas.

Condition on fitted predictions. Do not refit. Do not iid-resample rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_mae_delta(y: np.ndarray, mu_model: np.ndarray, mu_baseline: np.ndarray) -> np.ndarray:
    """Per-row MAE delta; positive means the model beats the baseline."""
    y = np.asarray(y, dtype=float)
    return np.abs(y - np.asarray(mu_baseline, dtype=float)) - np.abs(y - np.asarray(mu_model, dtype=float))


def paired_crps_delta(crps_model: np.ndarray, crps_baseline: np.ndarray) -> np.ndarray:
    """Per-row CRPS delta; positive means the model beats the baseline."""
    return np.asarray(crps_baseline, dtype=float) - np.asarray(crps_model, dtype=float)


def clustered_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    probs: tuple[float, float] = (0.025, 0.975),
) -> dict[str, Any]:
    """Percentile CI for a mean, resampling *clusters* with replacement.

    Each draw keeps all rows of a sampled cluster (and repeats them if the
    cluster is drawn more than once). Player is the default scientific unit.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.ndim != 1:
        raise ValueError("values must be 1-d")
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must have the same length")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")

    rng = rng or np.random.default_rng(0)
    unique, inverse = np.unique(cluster_ids, return_inverse=True)
    n_clusters = len(unique)
    if n_clusters == 0:
        raise ValueError("no clusters")

    groups = [np.flatnonzero(inverse == k) for k in range(n_clusters)]
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        boot[b] = np.concatenate([values[groups[j]] for j in draw]).mean()

    point = float(values.mean())
    lo, hi = (float(x) for x in np.quantile(boot, probs))
    return {
        "mean": point,
        "ci_low": lo,
        "ci_high": hi,
        "p_positive": float(np.mean(boot > 0.0)),
        "n_boot": int(n_boot),
        "n_clusters": int(n_clusters),
        "n_rows": int(len(values)),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def score_delta_cis(
    delta: np.ndarray,
    player_ids: np.ndarray,
    week_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, Any]]:
    """Player-clustered CI (headline) plus week-clustered sensitivity."""
    rng = rng or np.random.default_rng(0)
    # Independent streams so week sensitivity is not coupled to the player draws.
    seeds = rng.integers(0, 2**31 - 1, size=2, dtype=np.int64)
    player_rng = np.random.default_rng(int(seeds[0]))
    week_rng = np.random.default_rng(int(seeds[1]))
    return {
        "player": clustered_bootstrap_mean(delta, player_ids, n_boot=n_boot, rng=player_rng),
        "week": clustered_bootstrap_mean(delta, week_ids, n_boot=n_boot, rng=week_rng),
    }
