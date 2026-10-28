"""Team target volume: a league-wide count law and a team-history reference.

The volume model asks how many targets a team's offense generates in a game. Before a hierarchical
model is worth building, the plan asks for the honest comparison underneath it: one pooled count law
that ignores the team, and one reference that uses each team's own recent volume. Both are fitted on the
training weeks of a split and scored on the same predicted team games, so a later Bayesian model has
something to beat rather than a story about how much better it ought to be.

The count law is a negative binomial with mean `mu` and dispersion `k`, so the variance is
`mu + mu**2 / k`. A larger `k` is closer to a Poisson; a smaller `k` is a heavier tail. Both parameters
are fitted by maximum likelihood on the training counts, which needs no new dependency.

The team-history reference is each team's mean over its most recent training team games, with the
training league mean standing in for a team that never appeared. It reads nothing from the predicted
weeks, so it is as-of honest; it is also deliberately simple, because a baseline that needed the same
care as the model it benchmarks would not be a baseline.

Nothing here fits a team effect, an opponent effect, or a season trend. That is the hierarchical model's
job, and this module exists to say what such a model would have to beat.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln

from ..metrics import row_crps
from .panels import PANEL_SCHEMA
from .prepare import require_columns
from .splits import LABEL_COLUMNS, TEAM_GAME_KEYS

DEFAULT_DRAWS = 2000
DEFAULT_WINDOW = 17
LEVELS = (0.05, 0.50, 0.90)


def _counts(frame, column="targets"):
    """The counts of a frame as a float array, refused when they are not counts."""
    values = frame[column].to_numpy().astype(float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"volume._counts: {column} must be non-negative and finite")
    return values


def fit_negative_binomial(counts):
    """Fit the mean and the dispersion of a negative binomial to a sample of counts.

    Returns `{"mean": mu, "dispersion": k}` with `var = mu + mu**2 / k`. The fit is by maximum
    likelihood over both parameters, started from the sample mean and a method-of-moments dispersion.
    A sample with no spread cannot identify a dispersion, so it is refused rather than given a boundary
    value that would look like a fit.
    """
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"volume.fit_negative_binomial: needs at least two counts, found {values.size}")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("volume.fit_negative_binomial: counts must be non-negative and finite")
    spread = float(values.var(ddof=1))
    mean = float(values.mean())
    if spread <= 1e-9 or values.min() == values.max():
        raise ValueError(
            "volume.fit_negative_binomial: the sample needs at least two distinct counts to identify a "
            f"dispersion, found a variance of {spread}"
        )

    def objective(parameters):
        log_mean, log_dispersion = parameters
        mu = np.exp(log_mean)
        dispersion = np.exp(log_dispersion)
        probability = dispersion / (dispersion + mu)
        likelihood = (gammaln(values + dispersion) - gammaln(dispersion) - gammaln(values + 1.0)
                      + dispersion * np.log(probability) + values * np.log1p(-probability))
        return -float(likelihood.sum())

    start = np.array([np.log(mean), np.log(max(mean * mean / max(spread - mean, 1e-6), 1e-3))])
    fitted = minimize(objective, start, method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    if not fitted.success:
        raise ValueError(f"volume.fit_negative_binomial: the fit did not converge: {fitted.message}")
    return {"mean": float(np.exp(fitted.x[0])), "dispersion": float(np.exp(fitted.x[1]))}


def sample_negative_binomial(means, dispersion, *, draws, rng):
    """Draw counts for each mean, sharing one dispersion.

    Row `i` of the result is `draws` independent counts with mean `means[i]` and variance
    `means[i] + means[i]**2 / dispersion`.
    """
    values = np.asarray(means, dtype=float)
    if values.ndim != 1:
        raise ValueError("volume.sample_negative_binomial: means must be one value per row")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("volume.sample_negative_binomial: every mean must be finite and positive")
    if draws < 1:
        raise ValueError(f"volume.sample_negative_binomial: draws must be positive, found {draws}")
    probability = dispersion / (dispersion + values)
    return rng.negative_binomial(dispersion, probability[:, None], size=(values.size, draws)).astype(float)


def team_means(panel, *, window=DEFAULT_WINDOW):
    """Each team's mean targets over its most recent `window` training team games.

    The window counts team games, not weeks, so a bye does not consume a slot. A team that appears fewer
    times than the window keeps the mean of what it has; a team that never appears is absent from the
    result, and the caller decides what stands in for it.
    """
    require_columns(panel, ("team", "targets", "season", "week"), "team_means")
    if window < 1:
        raise ValueError(f"volume.team_means: window must be a positive number of team games, found {window}")
    ordered = panel.sort("season", "week", "game_id", "team")
    recent = ordered.group_by("team", maintain_order=True).tail(window)
    return recent.group_by("team", maintain_order=True).agg(
        pl.col("targets").mean().alias("mean"),
        pl.len().cast(pl.Int32).alias("team_games"),
    )


def volume_scores(samples, observed):
    """MAE, CRPS, and central-interval coverage of a predictive sample against observed counts.

    The sample is one row per team game and one column per draw, which is the shape
    `sample_negative_binomial` returns and the shape `metrics.row_crps` expects.
    """
    drawn = np.asarray(samples, dtype=float)
    truth = np.asarray(observed, dtype=float)
    if drawn.ndim != 2:
        raise ValueError("volume.volume_scores: samples must have shape (rows, draws)")
    if drawn.shape[0] != truth.size:
        raise ValueError(
            f"volume.volume_scores: samples and observations differ in length: {drawn.shape[0]} against "
            f"{truth.size}"
        )
    quantiles = {level: np.quantile(drawn, [(1 - level) / 2, 1 - (1 - level) / 2], axis=1)
                 for level in LEVELS}
    return {
        "rows": int(truth.size),
        "draws": int(drawn.shape[1]),
        "mae": float(np.mean(np.abs(drawn.mean(axis=1) - truth))),
        "crps": float(row_crps(drawn, truth).mean()),
        "coverage_50": float(np.mean((truth >= quantiles[0.50][0]) & (truth <= quantiles[0.50][1]))),
        "coverage_90": float(np.mean((truth >= quantiles[0.90][0]) & (truth <= quantiles[0.90][1]))),
        "mean_width_90": float(np.mean(quantiles[0.90][1] - quantiles[0.90][0])),
    }


def volume_comparison(train_panel, predict_panel, observed_panel, *, draws=DEFAULT_DRAWS, seed=0,
                      window=DEFAULT_WINDOW):
    """Score the pooled count law and the team-history reference on the same predicted team games.

    The three frames are the bundles a split produces: the training team games, the predicted team games
    with no label, and the withheld labels. The comparison refuses a prediction panel that still carries
    a label, because that means the wrong bundle was passed, and it refuses an observed panel whose team
    games are not the predicted ones, because two models compared on different rows prove nothing.

    `pooled` is the league-wide negative binomial: one mean and one dispersion for every team game.
    `team_history` gives each team its own mean over its most recent training team games and reuses the
    pooled dispersion, which is the plan's empirical-noise version of a rolling baseline.
    """
    require_columns(train_panel, PANEL_SCHEMA, "volume_comparison")
    require_columns(predict_panel, ("season", "week", "game_id", "team"), "volume_comparison")
    require_columns(observed_panel, (*TEAM_GAME_KEYS, "targets"), "volume_comparison")
    carried = [column for column in LABEL_COLUMNS if column in predict_panel.columns]
    if carried:
        raise ValueError(
            f"volume_comparison: the prediction panel carries the label columns {carried}; pass the "
            "labels in the observed panel instead"
        )

    keys = predict_panel.select(*TEAM_GAME_KEYS)
    matched = observed_panel.select(*TEAM_GAME_KEYS)
    if keys.height != matched.height or keys.join(matched, on=list(TEAM_GAME_KEYS), how="anti").height:
        raise ValueError(
            "volume_comparison: the observed panel is about other team games than the prediction panel: "
            f"{keys.height} predicted against {matched.height} observed"
        )
    aligned = keys.join(observed_panel, on=list(TEAM_GAME_KEYS), how="left")

    training = _counts(train_panel)
    pooled = fit_negative_binomial(training)
    league_mean = float(training.mean())
    means = team_means(train_panel, window=window)
    joined = predict_panel.join(means, on="team", how="left")
    team_mean = joined["mean"].fill_null(league_mean).to_numpy().astype(float)
    truth = _counts(aligned)

    rng = np.random.default_rng(seed)
    models = {
        "pooled": sample_negative_binomial(np.full(truth.size, pooled["mean"]), pooled["dispersion"],
                                          draws=draws, rng=rng),
        "team_history": sample_negative_binomial(team_mean, pooled["dispersion"], draws=draws, rng=rng),
    }
    return {
        "rows": int(truth.size),
        "draws": int(draws),
        "window": int(window),
        "league_mean": league_mean,
        "pooled": pooled,
        "models": {name: volume_scores(samples, truth) for name, samples in models.items()},
        "predictions": {"team_history_mean": team_mean, "pooled_mean": pooled["mean"]},
    }
