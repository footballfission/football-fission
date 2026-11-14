"""Roster- and defense-conditional pool size models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from ..panels import TEAM_GAME_KEYS
from ..prepare import require_columns
from ..volume import fit_negative_binomial, sample_negative_binomial, volume_scores
from .features import DEFENSE_DIMS

VOLUME_FEATURES = (
    "is_home",
    "week_centered",
    "n_candidates",
    "ol_non_active",
    "vacated_skill_count",
    "vacated_qb_dropback",
    *DEFENSE_DIMS,
    "expected_qb_dropback_share",
)


def _volume_frame(team_feats, player_feats, pool, skills=None):
    teams = team_feats.filter(pl.col("pool") == pool).sort("game_id", "team")
    require_columns(teams, ("total", "is_home", "week", "candidates", "ol_non_active",
                            "vacated_skill_count", "vacated_qb_dropback", *DEFENSE_DIMS),
                    "volume frame")
    # Expected QB: among active QBs, the one with highest lagged dropback share.
    qb = player_feats.filter(
        (pl.col("pool") == pool) & (pl.col("position") == "QB")
    ).select("game_id", "team", "player_id", "lag_dropback_share")
    if pool != "dropbacks":
        # Still want QB signal for target/carry volume; pull from any pool's player feats.
        qb = player_feats.filter(pl.col("position") == "QB").select(
            "game_id", "team", "player_id", "lag_dropback_share"
        ).unique(subset=["game_id", "team", "player_id"], keep="first")
    best_qb = qb.sort("lag_dropback_share", descending=True).group_by("game_id", "team").agg(
        pl.col("lag_dropback_share").first().alias("expected_qb_dropback_share"),
        pl.col("player_id").first().alias("expected_qb_id"),
    )
    frame = teams.join(best_qb, on=["game_id", "team"], how="left").with_columns(
        pl.col("expected_qb_dropback_share").fill_null(0.5),
        pl.col("expected_qb_id").fill_null("UNKNOWN_QB"),
        ((pl.col("week").cast(pl.Float64) - 9.0) / 8.0).alias("week_centered"),
        pl.col("candidates").cast(pl.Float64).alias("n_candidates"),
        pl.col("is_home").cast(pl.Float64),
        pl.col("ol_non_active").cast(pl.Float64),
        pl.col("vacated_skill_count").cast(pl.Float64),
    )
    return frame


def _design(frame, team_levels=None, qb_levels=None):
    x = frame.select(VOLUME_FEATURES).to_numpy().astype(float)
    parts = [x]
    names = list(VOLUME_FEATURES)
    if team_levels is not None:
        teams = frame["team"].to_list()
        index = {t: i for i, t in enumerate(team_levels)}
        mat = np.zeros((frame.height, len(team_levels)))
        for i, team in enumerate(teams):
            if team in index:
                mat[i, index[team]] = 1.0
        parts.append(mat)
        names.extend(f"team_{t}" for t in team_levels)
    if qb_levels is not None:
        qbs = frame["expected_qb_id"].to_list()
        index = {q: i for i, q in enumerate(qb_levels)}
        mat = np.zeros((frame.height, len(qb_levels)))
        for i, qb in enumerate(qbs):
            if qb in index:
                mat[i, index[qb]] = 1.0
        parts.append(mat)
        names.extend(f"qb_{q}" for q in qb_levels)
    return np.concatenate(parts, axis=1), names


def _fit_nb_glm(counts, x, *, ridge, team_slice=None, qb_slice=None,
                identity_ridge=None):
    """Negative-binomial GLM with log link and optional stronger penalty on intercepts."""
    y = np.asarray(counts, dtype=float)
    if y.size < 2 or np.unique(y).size < 2:
        raise ValueError("volume fit needs at least two distinct counts")

    # Start from pooled NB then refine regression.
    pooled = fit_negative_binomial(y)
    n_coef = x.shape[1]
    theta0 = np.zeros(n_coef + 2)
    theta0[0] = float(np.log(pooled["mean"]))
    theta0[-1] = float(np.log(pooled["dispersion"]))

    def unpack(theta):
        intercept = theta[0]
        coef = theta[1:-1]
        if coef.shape[0] != n_coef:
            raise ValueError("volume coefficient length mismatch")
        dispersion = float(np.exp(theta[-1]))
        mu = np.exp(np.clip(intercept + x @ coef, -20, 20))
        return intercept, coef, dispersion, mu

    def objective(theta):
        intercept, coef, dispersion, mu = unpack(theta)
        # NB2 log-likelihood
        from scipy.special import gammaln
        probability = dispersion / (dispersion + mu)
        ll = (gammaln(y + dispersion) - gammaln(dispersion) - gammaln(y + 1.0)
              + dispersion * np.log(probability) + y * np.log1p(-probability))
        nll = -float(ll.sum())
        penalty = 0.5 * ridge * float(np.dot(coef, coef))
        if identity_ridge is not None and team_slice is not None:
            penalty += 0.5 * (identity_ridge - ridge) * float(np.dot(coef[team_slice], coef[team_slice]))
        if identity_ridge is not None and qb_slice is not None:
            penalty += 0.5 * (identity_ridge - ridge) * float(np.dot(coef[qb_slice], coef[qb_slice]))
        return nll + penalty

    fitted = minimize(objective, theta0, method="Nelder-Mead",
                      options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 4000})
    intercept, coef, dispersion, mu = unpack(fitted.x)
    return {
        "intercept": float(intercept),
        "coef": coef,
        "dispersion": float(dispersion),
        "ridge": float(ridge),
        "identity_ridge": None if identity_ridge is None else float(identity_ridge),
        "converged": bool(fitted.success),
        "in_sample_mean": float(mu.mean()),
        "pooled_start": pooled,
    }


@dataclass(frozen=True)
class FittedVolumeModel:
    pool: str
    arm: str
    model: dict
    feature_names: tuple
    team_levels: tuple
    qb_levels: tuple
    train_through: tuple
    training_games: frozenset

    def predict_mean(self, team_feats, player_feats):
        frame = _volume_frame(team_feats, player_feats, self.pool)
        if (any(tuple(row) <= self.train_through
                for row in frame.select("season", "week").iter_rows())
                or set(frame["game_id"]) & self.training_games):
            raise ValueError("volume prediction must follow training weeks without reused games")
        team_levels = self.team_levels or None
        qb_levels = self.qb_levels or None
        x, _ = _design(frame, team_levels=team_levels, qb_levels=qb_levels)
        # Align coefficient length if predict sees fewer/more columns — use stored names order.
        coef = self.model["coef"]
        if x.shape[1] != coef.shape[0]:
            # Rebuild with training level lists already fixed above; mismatch means bug.
            raise ValueError("volume design width does not match fitted coefficients")
        mu = np.exp(np.clip(self.model["intercept"] + x @ coef, -20, 20))
        return {
            "means": mu,
            "dispersion": self.model["dispersion"],
            "team_game_keys": frame.select(*TEAM_GAME_KEYS),
            "frame": frame,
        }

    def sample(self, team_feats, player_feats, *, draws, seed):
        pred = self.predict_mean(team_feats, player_feats)
        rng = np.random.default_rng(seed)
        samples = sample_negative_binomial(pred["means"], pred["dispersion"], draws=draws, rng=rng)
        return {**pred, "samples": samples}

    def report(self):
        return {
            "model": f"{self.arm}_volume",
            "pool": self.pool,
            "arm": self.arm,
            "dispersion": self.model["dispersion"],
            "intercept": self.model["intercept"],
            "ridge": self.model["ridge"],
            "identity_ridge": self.model["identity_ridge"],
            "converged": self.model["converged"],
            "feature_names": list(self.feature_names),
            "coefficients": {
                name: float(value) for name, value in zip(self.feature_names, self.model["coef"])
            },
            "n_team_levels": len(self.team_levels),
            "n_qb_levels": len(self.qb_levels),
            "train_through": {"season": self.train_through[0], "week": self.train_through[1]},
            "interpretation": (
                "Latent roster/defense NB mean without team or QB intercepts"
                if self.arm == "latent" else
                "Same latent volume features plus partially pooled team and QB intercepts"
            ),
        }


def fit_latent_volume(team_feats, player_feats, pool, *, ridge=1.0):
    frame = _volume_frame(team_feats, player_feats, pool)
    x, names = _design(frame)
    # Drop intercept column space: first feature column set has no explicit intercept in x;
    # intercept is separate in the GLM.
    model = _fit_nb_glm(frame["total"].to_numpy(), x, ridge=ridge)
    cutoff = max(tuple(row) for row in frame.select("season", "week").iter_rows())
    return FittedVolumeModel(
        pool, "latent", model, tuple(names), tuple(), tuple(), cutoff, frozenset(frame["game_id"])
    )


def fit_identity_volume(team_feats, player_feats, pool, *, ridge=1.0, identity_ridge=10.0):
    frame = _volume_frame(team_feats, player_feats, pool)
    team_levels = tuple(sorted(frame["team"].unique().to_list()))
    qb_levels = tuple(sorted(set(frame["expected_qb_id"].to_list()) - {"UNKNOWN_QB"}))
    x, names = _design(frame, team_levels=team_levels, qb_levels=qb_levels)
    n_base = len(VOLUME_FEATURES)
    team_slice = slice(n_base, n_base + len(team_levels))
    qb_slice = slice(n_base + len(team_levels), n_base + len(team_levels) + len(qb_levels))
    model = _fit_nb_glm(
        frame["total"].to_numpy(), x, ridge=ridge,
        team_slice=team_slice, qb_slice=qb_slice, identity_ridge=identity_ridge,
    )
    cutoff = max(tuple(row) for row in frame.select("season", "week").iter_rows())
    return FittedVolumeModel(
        pool, "identity", model, tuple(names), team_levels, qb_levels, cutoff,
        frozenset(frame["game_id"]),
    )


def score_volume_predictions(samples, observed_totals):
    return volume_scores(samples, observed_totals)
