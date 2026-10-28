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

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss

from .baselines import (
    DEFAULT_PENALTY,
    DEFAULT_TOLERANCE,
    FittedBaseline,
    baseline_name,
    passing_design,
    penalty_value,
    rushing_design,
)
from .contracts import PASSING_CONTRACT, RUSHING_CONTRACT
from .evaluation import TASKS, chronological_split, row_losses

IDENTITY_PENALTIES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def identity_columns(design, identities):
    """Design columns that belong to the named identity blocks."""
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
    mask = identity_columns(design, identities)
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
                          identity_penalty=1.0, context_penalty=DEFAULT_PENALTY, max_iter=1000,
                          tolerance=DEFAULT_TOLERANCE):
    """Completion baseline with its own penalty on the identified passers and receivers."""
    context_penalty = penalty_value(context_penalty)
    identity_penalty = penalty_value(identity_penalty)
    design = passing_design(table, identities=identities, composites=composites)
    target = table["complete_pass"].cast(pl.Float64, strict=False).to_numpy().astype(float)
    if len(np.unique(target)) < 2:
        raise ValueError("passing: complete_pass holds one outcome only, and a logistic baseline "
                         "needs both a completion and an incompletion")
    estimator = LogisticRegression(C=1.0 / context_penalty, solver="lbfgs", max_iter=max_iter,
                                   tol=tolerance)
    _, factor = _scaled_fit(design, target, estimator=estimator, identity_penalty=identity_penalty,
                            context_penalty=context_penalty, identities=identities)
    probability = estimator.predict_proba(design.matrix)[:, 1]
    in_sample = {"log_loss": float(log_loss(target, probability, labels=[0, 1])),
                 "brier": float(brier_score_loss(target, probability))}
    return FittedBaseline(
        name=_pooled_name("passing", identities, design, identity_penalty, context_penalty),
        task="completion", target="complete_pass", contract=PASSING_CONTRACT.name,
        penalty=context_penalty, rows=table.height, estimator=estimator, design=design,
        in_sample=in_sample, block_penalties={"identity": identity_penalty, "context": context_penalty,
                                             "identity_scale": factor})


def fit_pooled_rushing(table, *, identities=("rusher_id",), composites=None, support=True,
                       identity_penalty=1.0, context_penalty=DEFAULT_PENALTY):
    """Rushing baseline with its own penalty on the identified runners."""
    context_penalty = penalty_value(context_penalty)
    identity_penalty = penalty_value(identity_penalty)
    design = rushing_design(table, identities=identities, composites=composites, support=support)
    target = table["rushing_yards"].cast(pl.Float64, strict=False).to_numpy().astype(float)
    estimator = Ridge(alpha=context_penalty)
    _, factor = _scaled_fit(design, target, estimator=estimator, identity_penalty=identity_penalty,
                            context_penalty=context_penalty, identities=identities)
    residual = target - estimator.predict(design.matrix)
    in_sample = {"mae": float(np.mean(np.abs(residual))),
                 "rmse": float(np.sqrt(np.mean(residual ** 2)))}
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
                            validation_season, penalties=IDENTITY_PENALTIES,
                            composites=None, support=True, context_penalty=DEFAULT_PENALTY,
                            contract=None):
    """Choose the player penalty on an inner chronological split, then report the choice.

    The split fits earlier seasons and scores the next one, and every grid point is scored the same
    way. Ties go to the larger penalty, so an uninformative grid lands on more shrinkage rather than
    on no shrinkage. `scores` keeps every grid value, because a flat grid and a peaked one are
    different evidence and only the table shows which one happened.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
    contract = contract or (PASSING_CONTRACT if task == "completion" else RUSHING_CONTRACT)
    split = chronological_split(table, train_seasons=train_seasons,
                                holdout_season=validation_season, contract=contract)
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
        "validation_season": int(validation_season),
        "rows": {"train": split["train"].height, "validation": split["holdout"].height},
    }
