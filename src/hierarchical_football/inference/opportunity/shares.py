"""Latent-only and identity-aware opportunity share models."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp

from ..panels import CANDIDATE_KEYS, TEAM_GAME_KEYS
from ..prepare import require_columns
from .features import DEFENSE_DIMS, SKILL_DIMS
from .pools import POOL_POSITIONS, SKILL_POSITIONS

POSITION_LIST = list(SKILL_POSITIONS)
FEATURE_NAMES = (
    *tuple(f"pos_{p}" for p in POSITION_LIST),
    *SKILL_DIMS,
    "vacated_receiver_target",
    "vacated_back_carry",
    "vacated_qb_dropback",
    *DEFENSE_DIMS,
    "is_home",
)
OTHER_FEATURE_NAMES = (
    "intercept",
    *DEFENSE_DIMS,
    "vacated_any_target",
    "vacated_skill_count",
    "n_candidates",
    "is_home",
)


def _design_matrices(player_feats, team_feats, pool):
    """Build per-row feature matrices aligned to candidate and team-game order."""
    players = player_feats.filter(pl.col("pool") == pool).sort(
        "game_id", "team", "candidate_index"
    )
    teams = team_feats.filter(pl.col("pool") == pool).sort("game_id", "team")
    require_columns(
        players,
        (*SKILL_DIMS, "vacated_receiver_target", "vacated_back_carry", "vacated_qb_dropback",
         *DEFENSE_DIMS, "is_home", "count", "player_id", "position"),
        "share design players",
    )
    require_columns(teams, ("other_total", "total", *DEFENSE_DIMS, "vacated_any_target",
                            "vacated_skill_count", "is_home", "candidates"), "share design teams")

    pos = players["position"].to_list()
    pos_mat = np.zeros((players.height, len(POSITION_LIST)), dtype=float)
    for i, value in enumerate(pos):
        if value in POSITION_LIST:
            pos_mat[i, POSITION_LIST.index(value)] = 1.0
    skill = players.select(SKILL_DIMS).to_numpy().astype(float)
    vacated = players.select(
        "vacated_receiver_target", "vacated_back_carry", "vacated_qb_dropback"
    ).to_numpy().astype(float)
    defense = players.select(DEFENSE_DIMS).to_numpy().astype(float)
    home = players["is_home"].cast(pl.Float64).to_numpy().astype(float)[:, None]
    x = np.concatenate([pos_mat, skill, vacated, defense, home], axis=1)

    other = np.column_stack([
        np.ones(teams.height),
        teams.select(DEFENSE_DIMS).to_numpy().astype(float),
        teams["vacated_any_target"].to_numpy().astype(float),
        teams["vacated_skill_count"].cast(pl.Float64).to_numpy().astype(float),
        teams["candidates"].cast(pl.Float64).to_numpy().astype(float),
        teams["is_home"].cast(pl.Float64).to_numpy().astype(float),
    ])
    counts = players["count"].to_numpy().astype(float)
    other_counts = teams["other_total"].to_numpy().astype(float)
    totals = teams["total"].to_numpy().astype(float)
    players_ids = players["player_id"].to_list()
    memberships = []
    pool_map = {tuple(row): i for i, row in enumerate(teams.select(TEAM_GAME_KEYS).iter_rows())}
    for key in players.select(TEAM_GAME_KEYS).iter_rows():
        memberships.append(pool_map[tuple(key)])
    memberships = np.asarray(memberships, dtype=int)
    return {
        "players": players,
        "teams": teams,
        "x": x,
        "other": other,
        "counts": counts,
        "other_counts": other_counts,
        "totals": totals,
        "memberships": memberships,
        "player_ids": players_ids,
    }


def _group_slices(memberships, n_teams):
    order = np.argsort(memberships, kind="mergesort")
    sorted_m = memberships[order]
    starts = np.zeros(n_teams, dtype=int)
    ends = np.zeros(n_teams, dtype=int)
    for g in range(n_teams):
        idx = np.flatnonzero(sorted_m == g)
        if idx.size:
            starts[g] = idx[0]
            ends[g] = idx[-1] + 1
        else:
            starts[g] = 0
            ends[g] = 0
    return order, starts, ends


def _nll_and_grad(theta, design, player_index, n_players, ridge, identity_ridge):
    n_feat = design["x"].shape[1]
    n_other = design["other"].shape[1]
    beta = theta[:n_feat]
    other_beta = theta[n_feat:n_feat + n_other]
    u = theta[n_feat + n_other:] if n_players else np.zeros(0)

    scores = design["x"] @ beta
    if n_players:
        scores = scores + u[player_index]
    other_scores = design["other"] @ other_beta
    memberships = design["memberships"]
    n_teams = design["teams"].height
    order, starts, ends = _group_slices(memberships, n_teams)

    nll = 0.0
    grad_beta = np.zeros_like(beta)
    grad_other = np.zeros_like(other_beta)
    grad_u = np.zeros(n_players) if n_players else None

    # Work in original candidate order via mapped probabilities.
    for g in range(n_teams):
        a, b = starts[g], ends[g]
        if a == b:
            # No candidates: all mass on OTHER.
            p_other = 1.0
            nll -= design["other_counts"][g] * math.log(max(p_other, 1e-15))
            # gradient for other: count * (1 - p) * x = 0 when p=1 and we only have OTHER
            continue
        idx = order[a:b]
        s = scores[idx]
        o = other_scores[g]
        all_scores = np.concatenate([s, np.array([o])])
        log_den = logsumexp(all_scores)
        log_p = s - log_den
        log_p_other = o - log_den
        c = design["counts"][idx]
        c_other = design["other_counts"][g]
        nll -= float(np.dot(c, log_p) + c_other * log_p_other)
        p = np.exp(log_p)
        p_other = math.exp(log_p_other)
        total = design["totals"][g]
        # Multinomial gradient: -sum c_k x_k + total * sum p_k x_k
        residual = total * p - c
        grad_beta += design["x"][idx].T @ residual
        grad_other += (total * p_other - c_other) * design["other"][g]
        if n_players:
            for local, player_i in enumerate(player_index[idx]):
                grad_u[player_i] += residual[local]

    nll += 0.5 * ridge * float(np.dot(beta, beta) + np.dot(other_beta, other_beta))
    grad_beta += ridge * beta
    grad_other += ridge * other_beta
    if n_players:
        nll += 0.5 * identity_ridge * float(np.dot(u, u))
        grad_u += identity_ridge * u
        grad = np.concatenate([grad_beta, grad_other, grad_u])
    else:
        grad = np.concatenate([grad_beta, grad_other])
    return nll, grad


def _fit_share_model(design, *, ridge, identity_ridge, use_identity, max_iter=400):
    player_ids = design["player_ids"]
    if use_identity:
        unique_players = sorted(set(player_ids))
        player_to_index = {p: i for i, p in enumerate(unique_players)}
        player_index = np.array([player_to_index[p] for p in player_ids], dtype=int)
        n_players = len(unique_players)
    else:
        unique_players = []
        player_index = np.zeros(len(player_ids), dtype=int)
        n_players = 0

    n_feat = design["x"].shape[1]
    n_other = design["other"].shape[1]
    theta0 = np.zeros(n_feat + n_other + n_players)

    def objective(theta):
        return _nll_and_grad(theta, design, player_index, n_players, ridge, identity_ridge)

    fitted = minimize(objective, theta0, method="L-BFGS-B", jac=True,
                      options={"maxiter": max_iter, "ftol": 1e-8})
    if not fitted.success and fitted.nit < 5:
        raise ValueError(f"share model failed to converge: {fitted.message}")
    theta = fitted.x
    beta = theta[:n_feat]
    other_beta = theta[n_feat:n_feat + n_other]
    intercepts = {
        unique_players[i]: float(theta[n_feat + n_other + i]) for i in range(n_players)
    }
    return {
        "beta": beta,
        "other_beta": other_beta,
        "player_intercepts": intercepts,
        "feature_names": FEATURE_NAMES,
        "other_feature_names": OTHER_FEATURE_NAMES,
        "ridge": float(ridge),
        "identity_ridge": float(identity_ridge) if use_identity else None,
        "use_identity": use_identity,
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
        "training_nll": float(fitted.fun),
        "n_players": n_players,
    }


def _predict_shares(model, design):
    beta = model["beta"]
    other_beta = model["other_beta"]
    scores = design["x"] @ beta
    if model["use_identity"]:
        intercepts = model["player_intercepts"]
        scores = scores + np.array([intercepts.get(pid, 0.0) for pid in design["player_ids"]])
    other_scores = design["other"] @ other_beta
    memberships = design["memberships"]
    n_teams = design["teams"].height
    order, starts, ends = _group_slices(memberships, n_teams)
    shares = np.zeros(design["x"].shape[0])
    other_shares = np.zeros(n_teams)
    for g in range(n_teams):
        a, b = starts[g], ends[g]
        if a == b:
            other_shares[g] = 1.0
            continue
        idx = order[a:b]
        s = scores[idx]
        o = other_scores[g]
        all_scores = np.concatenate([s, np.array([o])])
        log_den = logsumexp(all_scores)
        shares[idx] = np.exp(s - log_den)
        other_shares[g] = math.exp(o - log_den)
    return shares, other_shares


@dataclass(frozen=True)
class FittedShareModel:
    pool: str
    arm: str
    model: dict
    train_through: tuple
    training_games: frozenset

    def predict(self, team_feats, player_feats):
        design = _design_matrices(player_feats, team_feats, self.pool)
        if (any(tuple(row) <= self.train_through
                for row in design["teams"].select("season", "week").iter_rows())
                or set(design["teams"]["game_id"]) & self.training_games):
            raise ValueError("share prediction must follow training weeks without reused games")
        shares, other_shares = _predict_shares(self.model, design)
        unseen = sum(1 for pid in set(design["player_ids"])
                     if pid not in self.model["player_intercepts"]) if self.model["use_identity"] else 0
        return {
            "shares": shares,
            "other_shares": other_shares,
            "candidate_keys": design["players"].select(*CANDIDATE_KEYS, "player_id"),
            "team_game_keys": design["teams"].select(*TEAM_GAME_KEYS),
            "arm": self.arm,
            "pool": self.pool,
            "unseen_identity_players": unseen,
            "condition": "supplied_roster_with_lagged_skills_and_defense_latents",
        }

    def report(self):
        return {
            "model": f"{self.arm}_share",
            "pool": self.pool,
            "arm": self.arm,
            "ridge": self.model["ridge"],
            "identity_ridge": self.model["identity_ridge"],
            "n_players": self.model["n_players"],
            "converged": self.model["converged"],
            "iterations": self.model["iterations"],
            "training_nll": self.model["training_nll"],
            "feature_names": list(self.model["feature_names"]),
            "other_feature_names": list(self.model["other_feature_names"]),
            "coefficients": {
                name: float(value)
                for name, value in zip(self.model["feature_names"], self.model["beta"])
            },
            "other_coefficients": {
                name: float(value)
                for name, value in zip(self.model["other_feature_names"], self.model["other_beta"])
            },
            "train_through": {"season": self.train_through[0], "week": self.train_through[1]},
            "interpretation": (
                "Identity-free linear scores over lagged skill vectors, vacated-role features, "
                "and opponent defense latents"
                if self.arm == "latent" else
                "Same latent score plus partially pooled player intercepts; no team intercept in shares"
            ),
        }


def fit_latent_shares(team_feats, player_feats, pool, *, ridge=1.0):
    """Fit the identity-free share arm for one pool."""
    if pool not in POOL_POSITIONS:
        raise ValueError(f"unknown pool {pool!r}")
    design = _design_matrices(player_feats, team_feats, pool)
    if not design["teams"].height:
        raise ValueError("latent share fit needs training team games")
    model = _fit_share_model(design, ridge=ridge, identity_ridge=1.0, use_identity=False)
    cutoff = max(tuple(row) for row in design["teams"].select("season", "week").iter_rows())
    return FittedShareModel(pool, "latent", model, cutoff, frozenset(design["teams"]["game_id"]))


def fit_identity_shares(team_feats, player_feats, pool, *, ridge=1.0, identity_ridge=10.0):
    """Fit latents + partially pooled player intercepts for one pool."""
    if pool not in POOL_POSITIONS:
        raise ValueError(f"unknown pool {pool!r}")
    design = _design_matrices(player_feats, team_feats, pool)
    if not design["teams"].height:
        raise ValueError("identity share fit needs training team games")
    model = _fit_share_model(design, ridge=ridge, identity_ridge=identity_ridge, use_identity=True)
    cutoff = max(tuple(row) for row in design["teams"].select("season", "week").iter_rows())
    return FittedShareModel(pool, "identity", model, cutoff, frozenset(design["teams"]["game_id"]))


def select_identity_ridge(team_feats, player_feats, pool, *, train_mask, valid_mask,
                          ridge=1.0, grid=(1.0, 10.0, 100.0, 1000.0)):
    """Choose identity penalty on an earlier validation fold; never on confirmation seasons."""
    train_teams = team_feats.filter(train_mask)
    train_players = player_feats.filter(train_mask)
    valid_teams = team_feats.filter(valid_mask)
    valid_players = player_feats.filter(valid_mask)
    best = None
    scores = []
    for penalty in grid:
        fitted = fit_identity_shares(train_teams, train_players, pool,
                                     ridge=ridge, identity_ridge=penalty)
        pred = fitted.predict(valid_teams, valid_players)
        # Mean cross-entropy without multinomial coefficient for selection.
        design = _design_matrices(valid_players, valid_teams, pool)
        nll = 0.0
        memberships = design["memberships"]
        for g in range(design["teams"].height):
            idx = np.flatnonzero(memberships == g)
            probs = np.concatenate([pred["shares"][idx], [pred["other_shares"][g]]])
            probs = np.clip(probs, 1e-15, 1.0)
            counts = np.concatenate([design["counts"][idx], [design["other_counts"][g]]])
            nll -= float(np.dot(counts, np.log(probs)))
        mean = nll / max(design["teams"].height, 1)
        scores.append({"identity_ridge": penalty, "validation_nll": mean})
        if best is None or mean < best[0] - 1e-12 or (
                abs(mean - best[0]) <= 1e-12 and penalty > best[1]):
            best = (mean, penalty)
    return {"identity_ridge": best[1], "scores": scores}
