"""Partial pooling for identified players: one variance component per block.

A single penalty over every coefficient makes the player prior and the pre-play context prior the
same thing. That is wrong in a specific way: a prior strong enough to control 800 player columns
also flattens target depth and field position, which are the columns the model is supposed to
believe. This module gives the identity block its own penalty, which is the ridge form of a
random-effect prior on players, and leaves the context with its own.

The player penalty is chosen on an inner chronological split, never on the season that is scored: a
prior selected on the holdout would report the selection as evidence. The selection is deliberately
plain - one grid, the primary metric, ties toward more shrinkage - because the point is to state how
the prior was set, not to search for a winner.

A penalty here is a variance component, not a probability of skill. Nothing in this module turns a
shrunk term into credit.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss

from .baselines import (
    DEFAULT_MAX_ITER,
    DEFAULT_PENALTY,
    DEFAULT_TOLERANCE,
    PASSING_CONTEXT_COMPONENTS,
    FittedBaseline,
    baseline_name,
    convergence,
    passing_design,
    penalty_value,
    rushing_design,
    warn_if_not_converged,
)
from .contracts import PASSING_CONTRACT, RUSHING_CONTRACT
from .evaluation import chronological_split, require_task, row_losses

PASSING_IDENTITIES = ("qb_id", "receiver_id")


def fit_block_completion(table, *, identity_penalties, context="linear", context_penalty=1.0,
                         max_iter=DEFAULT_MAX_ITER, tolerance=DEFAULT_TOLERANCE,
                         identities=PASSING_IDENTITIES, opponent_penalty=None):
    """Completion fit with separate, finite positive quarterback and receiver penalties.

    Context, including expanded depth/position/opponent columns when requested, retains its
    own penalty. This is penalized likelihood, not a sampled Bayesian posterior.
    """
    if (len(set(identities)) != len(identities) or
            set(identities) - set(PASSING_IDENTITIES) or set(identity_penalties) != set(identities)):
        raise ValueError("identity penalties must name exactly the requested passing identity blocks")
    penalties = {key: float(value) for key, value in identity_penalties.items()}
    if any(not math.isfinite(value) or value <= 0 for value in
           [*penalties.values(), context_penalty]):
        raise ValueError("all block penalties must be finite and positive")
    design = passing_design(table, identities=identities, context=context)
    if opponent_penalty is not None:
        if (not math.isfinite(opponent_penalty) or opponent_penalty <= 0 or
                "opponent" not in PASSING_CONTEXT_COMPONENTS[context]):
            raise ValueError("opponent penalty must be finite and positive with opponent context")
    target = table["complete_pass"].to_numpy().astype(float)
    if len(np.unique(target)) < 2:
        raise ValueError("completion fitting needs both outcomes")
    scales = np.ones(len(design.columns))
    for identity, penalty in penalties.items():
        scales[identity_mask(design, (identity,))] = math.sqrt(context_penalty / penalty)
    if opponent_penalty is not None:
        scales[identity_mask(design, ("defteam",))] = math.sqrt(context_penalty / opponent_penalty)
    estimator = LogisticRegression(C=1.0 / context_penalty, solver="lbfgs", max_iter=max_iter,
                                   tol=tolerance)
    estimator.fit(design.matrix * scales, target)
    estimator.coef_ *= scales
    probability = estimator.predict_proba(design.matrix)[:, 1]
    state = warn_if_not_converged(convergence(estimator, max_iter), "passing:block_identity")
    block_penalties = {"context": float(context_penalty), "identities": penalties}
    labels = [*(f"{key}={penalties[key]:g}" for key in identities), f"context={context_penalty:g}"]
    if opponent_penalty is not None:
        block_penalties["opponent"] = float(opponent_penalty)
        labels.append(f"opponent={opponent_penalty:g}")
    return FittedBaseline(
        name=baseline_name("passing", identities, design) + "[" + ",".join(labels) + "]",
        task="completion", target="complete_pass", contract=PASSING_CONTRACT.name,
        penalty=float(context_penalty), rows=table.height, estimator=estimator, design=design,
        in_sample={"log_loss": float(log_loss(target, probability, labels=[0, 1])),
                   "brier": float(brier_score_loss(target, probability)), **state},
        block_penalties=block_penalties)


def select_passing_penalties(table, *, train_seasons, holdout_season,
                             penalties=(10.0, 100.0, 1000.0), contexts=("linear", "expanded"),
                             context_penalty=1.0):
    """Select context and two player penalties using an earlier chronological validation season.

    Return all scores and convergence flags. Nonconverged fits are ineligible; exact ties prefer
    stronger total shrinkage, then stronger QB shrinkage, then the earlier declared context.
    Rows outside the declared split never enter either the fit or selection score.
    """
    if not penalties or not contexts:
        raise ValueError("selection needs nonempty penalties and contexts")
    split = chronological_split(table, train_seasons=train_seasons, holdout_season=holdout_season)
    scores = []
    for context, qb, receiver in product(contexts, penalties, penalties):
        fitted = fit_block_completion(split["train"], context=context, context_penalty=context_penalty,
                                      identity_penalties={"qb_id": qb, "receiver_id": receiver})
        scores.append({"context": context, "qb_id": float(qb), "receiver_id": float(receiver),
                       "log_loss": _score(fitted, split["holdout"], "completion"),
                       **{key: fitted.in_sample[key] for key in ("converged", "iterations")}})
    eligible = [row for row in scores if row["converged"] and math.isfinite(row["log_loss"])]
    if not eligible:
        raise ValueError("no converged passing selection candidate")
    best = min(eligible, key=lambda row: (row["log_loss"], -(row["qb_id"] + row["receiver_id"]),
                                         -row["qb_id"], contexts.index(row["context"])))
    return {"selected": {key: best[key] for key in ("context", "qb_id", "receiver_id")},
            "scores": scores, "metric": "log_loss", "train_seasons": list(train_seasons),
            "holdout_season": int(holdout_season), "context_penalty": float(context_penalty),
            "rows": {"train": split["train"].height, "holdout": split["holdout"].height}}

IDENTITY_PENALTIES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def identity_mask(design, identities):
    """True for every design column that belongs to one of the named identity blocks.

    The value is a mask over the design's own columns rather than a list of column names, which is what
    the per-block penalty needs: the columns are scaled before the fit and unscaled after it.
    """
    prefixes = tuple(f"{column}=" for column in identities)
    return np.array([bool(column.startswith(prefixes)) for column in design.columns])


def _scaled_fit(design, target, *, estimator, identity_penalty, context_penalty, identities):
    """Fit with the identity columns rescaled, then fold the scale back into the coefficients.

    Scaling a column by `k` divides its fitted coefficient by `k`, so a block scaled by
    `sqrt(context_penalty / identity_penalty)` carries `identity_penalty` while every other column
    keeps `context_penalty`. The linear predictor is unchanged, so multiplying the fitted identity
    coefficients by `k` afterwards reproduces the same predictions on the ordinary design, which is
    what every other function in this package expects to receive.
    """
    mask = identity_mask(design, identities)
    factor = math.sqrt(context_penalty / identity_penalty)
    matrix = np.array(design.matrix, dtype=float, copy=True)
    matrix[:, mask] *= factor
    estimator.fit(matrix, target)
    fitted = np.array(estimator.coef_, dtype=float, copy=True)
    if fitted.ndim == 1:          # a regressor keeps one coefficient per column
        fitted[mask] *= factor
    else:                         # a classifier keeps one row per outcome
        fitted[:, mask] *= factor
    estimator.coef_ = fitted
    return mask, factor


def _pooled_name(prefix, identities, design, identity_penalty, context_penalty):
    return (f"{baseline_name(prefix, identities, design)}"
            f"[identity_penalty={identity_penalty:g},context_penalty={context_penalty:g}]")


def fit_pooled_completion(table, *, identities=("qb_id", "receiver_id"), composites=None,
                          identity_penalty=1.0, context_penalty=DEFAULT_PENALTY,
                          max_iter=DEFAULT_MAX_ITER, tolerance=DEFAULT_TOLERANCE):
    """Completion baseline with its own penalty on the identified passers and receivers.

    `identity_penalty` governs the one-hot player block and `context_penalty` every other column, so a
    prior strong enough to control hundreds of player columns does not also flatten target depth and
    field position. The result is the ordinary `FittedBaseline`, so prediction, `identity_terms`, and
    the evaluation module work on it unchanged.
    """
    context_penalty = penalty_value(context_penalty)
    identity_penalty = penalty_value(identity_penalty)
    design = passing_design(table, identities=identities, composites=composites)
    target = table["complete_pass"].cast(pl.Float64, strict=False).to_numpy().astype(float)
    if len(np.unique(target)) < 2:
        raise ValueError("passing_targets: complete_pass holds one outcome only, and a logistic "
                         "baseline needs both a completion and an incompletion")
    estimator = LogisticRegression(C=1.0 / context_penalty, solver="lbfgs", max_iter=max_iter,
                                   tol=tolerance)
    _, factor = _scaled_fit(design, target, estimator=estimator, identity_penalty=identity_penalty,
                            context_penalty=context_penalty, identities=identities)
    probability = estimator.predict_proba(design.matrix)[:, 1]
    in_sample = {"log_loss": float(log_loss(target, probability, labels=[0, 1])),
                 "brier": float(brier_score_loss(target, probability)),
                 **warn_if_not_converged(convergence(estimator, max_iter),
                                          "passing_targets:pooled_identity")}
    return FittedBaseline(
        name=_pooled_name("passing", identities, design, identity_penalty, context_penalty),
        task="completion", target="complete_pass", contract=PASSING_CONTRACT.name,
        penalty=context_penalty, rows=table.height, estimator=estimator, design=design,
        in_sample=in_sample, block_penalties={"identity": identity_penalty, "context": context_penalty,
                                             "identity_scale": factor})


def fit_pooled_rushing(table, *, identities=("rusher_id",), composites=None, support=True,
                       identity_penalty=1.0, context_penalty=DEFAULT_PENALTY):
    """Rushing baseline with its own penalty on the identified runners.

    `support=True` is the default here, because the runner block of the plan sits on top of the support
    environment rather than replacing it; pass `support=False` for a runner block scored against
    context alone. The penalties are reported in the fit's `block_penalties`.
    """
    context_penalty = penalty_value(context_penalty)
    identity_penalty = penalty_value(identity_penalty)
    design = rushing_design(table, identities=identities, composites=composites, support=support)
    target = table["rushing_yards"].cast(pl.Float64, strict=False).to_numpy().astype(float)
    estimator = Ridge(alpha=context_penalty)
    _, factor = _scaled_fit(design, target, estimator=estimator, identity_penalty=identity_penalty,
                            context_penalty=context_penalty, identities=identities)
    residual = target - estimator.predict(design.matrix)
    in_sample = {"mae": float(np.mean(np.abs(residual))),
                 "rmse": float(np.sqrt(np.mean(residual ** 2))),
                 **convergence(estimator, None)}
    return FittedBaseline(
        name=_pooled_name("rushing", identities, design, identity_penalty, context_penalty),
        task="yards", target="rushing_yards", contract=RUSHING_CONTRACT.name,
        penalty=context_penalty, rows=table.height, estimator=estimator, design=design,
        in_sample=in_sample, block_penalties={"identity": identity_penalty, "context": context_penalty,
                                             "identity_scale": factor})


def _score(fitted, table, task):
    """Primary metric of one fitted baseline on one table: log loss or mean absolute error."""
    observed = table[fitted.target].cast(pl.Float64, strict=False).to_numpy().astype(float)
    return float(row_losses(observed, fitted.predict(table), task=task).mean())


def select_identity_penalty(table, *, identities, task="completion", train_seasons,
                            holdout_season, penalties=IDENTITY_PENALTIES,
                            composites=None, support=True, context_penalty=DEFAULT_PENALTY,
                            contract=None):
    """Choose the player penalty on an inner chronological split, then report the choice.

    The split fits earlier seasons and scores the next one, and every grid point is scored the same
    way. Ties go to the larger penalty, so an uninformative grid lands on more shrinkage rather than
    on no shrinkage. `scores` keeps every grid value, because a flat grid and a peaked one are
    different evidence and only the table shows which one happened.
    """
    require_task(task)
    contract = contract or (PASSING_CONTRACT if task == "completion" else RUSHING_CONTRACT)
    split = chronological_split(table, train_seasons=train_seasons,
                                holdout_season=holdout_season, contract=contract)
    fitter = fit_pooled_completion if task == "completion" else fit_pooled_rushing
    scores = {}
    for penalty in penalties:
        fitted = fitter(split["train"], identities=identities, composites=composites,
                        identity_penalty=penalty, context_penalty=context_penalty,
                        **(dict(support=support) if task == "yards" else {}))
        scores[float(penalty)] = _score(fitted, split["holdout"], task)
    best = min(scores.values())
    selected = max(float(penalty) for penalty, value in scores.items() if value == best)
    ordered = sorted(scores.items())
    return {
        "task": task,
        "identities": tuple(identities),
        "selected": selected,
        "metric": "log_loss" if task == "completion" else "mae",
        "scores": scores,
        "flat": abs(ordered[-1][1] - ordered[0][1]) < 1e-12,
        "train_seasons": tuple(int(season) for season in train_seasons),
        "holdout_season": int(holdout_season),
        "rows": {"train": split["train"].height, "holdout": split["holdout"].height},
    }
