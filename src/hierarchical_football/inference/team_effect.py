"""Team volume with a team effect, shrunk by a penalty chosen on an inner split.

The pooled law in `volume` gives every team game the league's mean. The plan's Model A asks for more: a
partial-pooling model where each team carries a deviation from the league mean, teams with little history
are shrunk toward it, and a team the model has never seen falls back to the league mean without an index
error. This module fits that model by penalized maximum likelihood: a negative binomial whose log mean is
an intercept plus one deviation per team, with a ridge penalty on the deviations. It is the frequentist
cousin of the plan's Bayesian hierarchy, not its posterior, and it is named that way in the record.

The penalty is chosen on an inner chronological split of the training weeks: fit on the earlier part,
score the later part with the count log score, keep the penalty that scored best. The plan asks for exactly
that discipline - no penalty may be chosen on the rows a model is then scored on - so the selector needs a
cutoff inside the training history and refuses to guess one.

The model does not carry an opponent effect, a season trend, or a team-by-season interaction. The plan
introduces those one at a time and only when the simpler model miscalibrates; this module is the simpler
model, and `team_effect_comparison` scores it against the pooled law and the team-history reference on
identical rows.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from .panels import PANEL_SCHEMA
from .prepare import require_columns
from .splits import at_or_before
from .volume import DEFAULT_WINDOW, sample_negative_binomial, volume_comparison, volume_scores

# The grid starts at no shrinkage, so "the data prefers no team effect" is a candidate answer rather than a
# grid edge, and it ends far enough out that a boundary answer means the search needs another order of
# magnitude rather than another grid.
DEFAULT_PENALTIES = (0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DISPERSION_BOUNDS = (np.log(0.5), np.log(1e4))
TEAM_COLUMNS = ("season", "week", "game_id", "team", "targets")


def team_design(panel):
    """One indicator column per team, plus the team order those columns follow.

    A team's column is one where it played. The order is alphabetical so the design is reproducible and
    the mapping back to team names never depends on the row order of the frame.
    """
    require_columns(panel, ("team",), "team_design")
    teams = tuple(sorted(panel["team"].unique().to_list()))
    index = {team: position for position, team in enumerate(teams)}
    design = np.zeros((panel.height, len(teams)), dtype=float)
    for row, team in enumerate(panel["team"].to_list()):
        design[row, index[team]] = 1.0
    return teams, design


def _counts(frame, column="targets"):
    """The counts of a frame as a float array, refused when they are not counts."""
    values = frame[column].to_numpy().astype(float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"team_effect._counts: {column} must be non-negative and finite")
    return values


def _fit_effects(counts, design, penalty, dispersion):
    """The penalized maximum-likelihood intercept and team deviations at a fixed dispersion."""
    def objective(theta):
        intercept, effects = theta[0], theta[1:]
        mu = np.exp(intercept + design @ effects)
        likelihood = np.sum(counts * np.log(mu) - (dispersion + counts) * np.log(dispersion + mu))
        residual = counts - (dispersion + counts) * mu / (dispersion + mu)
        # The penalty's own gradient belongs in the score, or the fit walks uphill and its line search fails.
        gradient = np.concatenate([[residual.sum()], design.T @ residual - 2.0 * penalty * effects])
        return -(likelihood - penalty * float(np.sum(effects ** 2))), -gradient

    start = np.zeros(1 + design.shape[1])
    start[0] = np.log(max(counts.mean(), 1e-6))
    options = {"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-8}
    fitted = minimize(objective, start, jac=True, method="L-BFGS-B", options=options)
    # A line search inside L-BFGS-B occasionally reports ABNORMAL on a smooth problem; restarting from the
    # point it stopped at resolves it. The fit counts as converged when the restart succeeds or when the
    # gradient at the point is already small, so a genuinely stuck fit is still visible to the caller.
    for _ in range(2):
        if fitted.success or np.linalg.norm(fitted.jac, np.inf) < 1e-5:
            break
        fitted = minimize(objective, fitted.x, jac=True, method="L-BFGS-B", options=options)
    return fitted


def _objective(counts, design, penalty, log_dispersion):
    """The penalized negative log likelihood, with the effects solved for at that dispersion."""
    dispersion = float(np.exp(log_dispersion))
    fitted = _fit_effects(counts, design, penalty, dispersion)
    intercept, effects = fitted.x[0], fitted.x[1:]
    mu = np.exp(intercept + design @ effects)
    likelihood = np.sum(gammaln(counts + dispersion) - gammaln(dispersion) - gammaln(counts + 1.0)
                        + dispersion * np.log(dispersion / (dispersion + mu))
                        + counts * np.log(mu / (dispersion + mu)))
    return -(likelihood - penalty * float(np.sum(effects ** 2))), fitted


def fit_team_effect(panel, *, penalty):
    """Fit a team-effect negative binomial at one penalty, returning its parameters and its teams.

    The dispersion is chosen by a bounded search around each penalty's fit, so the reported dispersion and
    the reported effects come from the same objective. Teams absent from the training panel get no effect:
    a team the model has never seen is a fallback case, not an index to guess.
    """
    require_columns(panel, TEAM_COLUMNS, "fit_team_effect")
    if penalty < 0:
        raise ValueError(f"team_effect.fit_team_effect: penalty must be non-negative, found {penalty}")
    counts = _counts(panel)
    if counts.size < 2 or np.unique(counts).size < 2:
        raise ValueError("team_effect.fit_team_effect: the training counts carry no spread to fit")
    teams, design = team_design(panel)
    chosen = minimize_scalar(lambda value: _objective(counts, design, penalty, value)[0],
                            bounds=DISPERSION_BOUNDS, method="bounded")
    objective, fitted = _objective(counts, design, penalty, chosen.x)
    effects = fitted.x[1:]
    return {
        "teams": teams,
        "intercept": float(fitted.x[0]),
        "effects": {team: float(effect) for team, effect in zip(teams, effects, strict=True)},
        "dispersion": float(np.exp(chosen.x)),
        "penalty": float(penalty),
        "objective": float(objective),
        "rows": int(counts.size),
        "converged": bool(fitted.success),
        "iterations": int(fitted.nit),
    }


def team_effect_means(fit, panel):
    """The model's mean targets per row, with a team it has not seen falling back to the intercept."""
    require_columns(panel, ("team",), "team_effect_means")
    effects = np.array([fit["effects"].get(team, 0.0) for team in panel["team"].to_list()], dtype=float)
    return np.exp(fit["intercept"] + effects)


def count_log_score(means, dispersion, observed):
    """The mean negative log predictive mass of the observed counts under the fitted law.

    A proper score that needs no sampling, so a penalty can be chosen without Monte Carlo noise.
    """
    values = np.asarray(means, dtype=float)
    truth = np.asarray(observed, dtype=float)
    if values.size != truth.size:
        raise ValueError(
            f"team_effect.count_log_score: means and observations differ in length: {values.size} against "
            f"{truth.size}"
        )
    mass = (gammaln(truth + dispersion) - gammaln(dispersion) - gammaln(truth + 1.0)
            + dispersion * np.log(dispersion / (dispersion + values))
            + truth * np.log(values / (dispersion + values)))
    return float(-np.mean(mass))


def select_penalty(train_panel, *, penalties=DEFAULT_PENALTIES, inner_through):
    """Choose the team-effect penalty on an inner chronological split of the training weeks.

    `inner_through` is a `(season, week)` cutoff inside the training history. The earlier team games fit the
    model and the later ones score it; the penalty with the lowest count log score wins, and the table of
    every candidate's score is returned with the choice so a reader can see the margin.
    """
    require_columns(train_panel, TEAM_COLUMNS, "select_penalty")
    earlier = train_panel.filter(at_or_before(train_panel, inner_through))
    later = train_panel.filter(at_or_before(train_panel, inner_through).not_())
    if earlier.height < 2 or later.height < 1:
        raise ValueError(
            "team_effect.select_penalty: the inner cutoff leaves nothing to fit or nothing to score: "
            f"{earlier.height} earlier and {later.height} later team games"
        )
    truth = _counts(later)
    scores = []
    for penalty in penalties:
        fit = fit_team_effect(earlier, penalty=penalty)
        scores.append({"penalty": float(penalty),
                       "log_score": count_log_score(team_effect_means(fit, later), fit["dispersion"], truth),
                       "dispersion": fit["dispersion"]})
    best = min(scores, key=lambda row: row["log_score"])
    return {"penalty": best["penalty"], "log_score": best["log_score"], "scores": scores,
            "inner_rows": {"fit": int(earlier.height), "score": int(later.height)},
            "inner_through": {"season": int(inner_through[0]), "week": int(inner_through[1])}}


def team_effect_comparison(train_panel, predict_panel, observed_panel, *, inner_through,
                           penalties=DEFAULT_PENALTIES, draws=2000, seed=0, window=DEFAULT_WINDOW):
    """Score the team-effect model beside the pooled law and the team-history reference.

    The penalty is chosen on the inner split of the training weeks and the model is then refitted on all of
    them. Every model is sampled and scored on the same predicted team games, so the differences are the
    models' and not the rows'.
    """
    require_columns(train_panel, PANEL_SCHEMA, "team_effect_comparison")
    selection = select_penalty(train_panel, penalties=penalties, inner_through=inner_through)
    fit = fit_team_effect(train_panel, penalty=selection["penalty"])
    report = volume_comparison(train_panel, predict_panel, observed_panel, draws=draws, seed=seed,
                               window=window)
    keys = predict_panel.select("game_id", "team")
    truth = _counts(keys.join(observed_panel, on=["game_id", "team"], how="left"))
    means = team_effect_means(fit, predict_panel)
    samples = sample_negative_binomial(means, fit["dispersion"], draws=draws,
                                       rng=np.random.default_rng(seed + 1))
    report["models"]["team_effect"] = volume_scores(samples, truth)
    report["team_effect"] = {**fit, "selection": selection}
    report["predictions"]["team_effect_mean"] = means
    return report
