"""Hierarchical parametric QB chain.

Structure (the WR conditional hierarchy of ``hierarchical.py``, rebuilt for the
passing position):

    attempts      = hurdle: Bernoulli(q_attempts(X)), then, when it fires, a
                    >= 1 conditioned NegBin(mu_attempts(X), k_attempts); the
                    logistic probability is recalibrated by an intercept fitted
                    on the calibration season, so the ~5% zero-attempt weeks of
                    backup and kneel-down quarterbacks are reproduced
    completions   ~ BetaBinomial(attempts, p_completion(X), rho_completion)
    passing_yards = sum of ``completions`` signed per-completion gains
                    (Gaussian gain mixture, mean shifted to a conditional
                     yards-per-completion model)
    passing_tds   = one Bernoulli per completion through P(touchdown | gain), a
                    monotone curve fitted on the calibration season, with the
                    row's expected rate set by a shrunk conditional rate model
    interceptions ~ Poisson(attempts * lambda_interception(X)), shrunk
    carries       = hurdle: Bernoulli(q(X)), then a >= 1 truncated
                    NegBin(mu_carries(X), k_carries)
    rushing_yards = sum of ``carries`` signed per-carry gains (own mixture, own
                    conditional yards-per-carry model)
    rushing_tds   = one Bernoulli per carry through a gain-conditional curve

Every mean is a LightGBM boosted model over the same feature matrix the other
candidates see, and both hurdle probabilities (attempts, carries) are
conditional mean models that are refit on ``train``.  The shape layer
(dispersion, hurdle calibration intercept and level, beta-binomial correlation,
gain mixtures, touchdown curves, shrinkage strengths) is estimated either on an
internal time split of the training fold or supplied frozen from a strictly
prior season.

Structural invariants, enforced by construction:

* a zero-attempt draw forces completions, passing yards, passing touchdowns
  and interceptions to zero;
* ``0 <= completions <= attempts`` (binomial over the drawn attempt count);
* passing yards and passing touchdowns only exist when ``completions > 0``;
* ``0 <= passing_tds <= completions`` and ``0 <= interceptions <= attempts``;
* rushing yards and rushing touchdowns only exist when ``carries > 0``;
* yards are signed: per-completion and per-carry gains may be negative, and
  nothing is clipped at zero except the counts that are genuinely non-negative.

Scoring rule (see ``hierarchical_football.scoring``)::

    0.04 * passing_yards + 4 * passing_tds - 2 * interceptions
    + 0.1 * rushing_yards + 6 * rushing_tds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln

from .features import LGBM_PARAMS, _design

STAT_COLUMNS: tuple[str, ...] = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
)
STAT_INDEX: dict[str, int] = {name: index for index, name in enumerate(STAT_COLUMNS)}

POINTS_PER_PASSING_YARD = 0.04
POINTS_PER_PASSING_TD = 4.0
POINTS_PER_INTERCEPTION = -2.0
POINTS_PER_RUSHING_YARD = 0.1
POINTS_PER_RUSHING_TD = 6.0

RHO_BOUNDS = (1e-4, 0.60)
SHAPE_BOUNDS = (0.03, 800.0)
CALIBRATION_FRACTION = 0.20
GAIN_COMPONENTS = 2
MIN_GAIN_ROWS = 40
MIN_MIXTURE_WEIGHT = 0.02
GAIN_REG_COVAR = 0.25
GAIN_EM_ITERATIONS = 200
MAX_BETWEEN_FRACTION = 0.75
MIN_GAIN_VARIANCE = 0.25
TD_BINS = 6
TD_PRIOR_STRENGTH = 8.0
TD_SCALE_BOUNDS = (0.1, 10.0)
TD_MAX_PROBABILITY = 0.95
SHRINKAGE_BOUNDS = (0.05, 5000.0)
HURDLE_BOUNDS = (-8.0, 8.0)
TRUNCATION_ITERATIONS = 20
TRUNCATION_ROUNDS = 40


def qb_fantasy_points(
    stats: np.ndarray, *, passing_td_points: float = POINTS_PER_PASSING_TD
) -> np.ndarray:
    """Apply the QB scoring rule to a raw stat sample.

    ``stats`` is any array whose last axis follows ``STAT_COLUMNS``; the shape is
    preserved.  This is the module-level scoring function the runner can use to
    verify that the sampled stat lines reproduce the panel's ``fantasy_points``.

    The default ``passing_td_points`` is the 4 points used by the panel.  Pass
    6.0 for the repository's 6-point-passing-touchdown convention; every other
    coefficient is unchanged.
    """
    stats = np.asarray(stats, dtype=float)
    return (
        POINTS_PER_PASSING_YARD * stats[..., STAT_INDEX["passing_yards"]]
        + float(passing_td_points) * stats[..., STAT_INDEX["passing_tds"]]
        + POINTS_PER_INTERCEPTION * stats[..., STAT_INDEX["interceptions"]]
        + POINTS_PER_RUSHING_YARD * stats[..., STAT_INDEX["rushing_yards"]]
        + POINTS_PER_RUSHING_TD * stats[..., STAT_INDEX["rushing_tds"]]
    )


def _stage_params(objective: str, n_estimators: int = 200) -> dict[str, Any]:
    params = dict(LGBM_PARAMS)
    params.update({"objective": objective, "n_estimators": n_estimators})
    return params


def _fit_regressor(x_train, y_train, params, categorical, *, weights: np.ndarray | None = None):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params)
    model.fit(x_train, y_train, sample_weight=weights, categorical_feature=categorical or "auto")
    return model


def _time_order(frame: pl.DataFrame) -> np.ndarray:
    """Row indices ordered oldest to newest by (season, week)."""
    return (
        frame.select("season", "week")
        .with_row_index()
        .sort(["season", "week"])
        .get_column("index")
        .to_numpy()
    )


def _negbin_nll(k: float, y: np.ndarray, mu: np.ndarray) -> float:
    """Negative log-likelihood of the mean/dispersion negative binomial."""
    k = float(k)
    if not np.isfinite(k) or k <= 0:
        return 1e18
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    p = k / (k + mu)
    loglik = (
        gammaln(y + k)
        - gammaln(k)
        - gammaln(y + 1.0)
        + k * np.log(p)
        + y * np.log(np.maximum(1.0 - p, 1e-12))
    )
    total = -float(np.sum(loglik))
    return total if np.isfinite(total) else 1e18


def _beta_binomial_nll(rho: float, successes: np.ndarray, trials: np.ndarray, p: np.ndarray) -> float:
    rho = float(rho)
    if not np.isfinite(rho) or rho <= 0:
        return 1e18
    a = np.maximum(np.asarray(p, dtype=float) * (1.0 - rho) / rho, 1e-9)
    b = np.maximum((1.0 - np.asarray(p, dtype=float)) * (1.0 - rho) / rho, 1e-9)
    loglik = betaln(
        np.asarray(successes, dtype=float) + a,
        np.asarray(trials, dtype=float) - np.asarray(successes, dtype=float) + b,
    ) - betaln(a, b)
    total = -float(np.sum(loglik))
    return total if np.isfinite(total) else 1e18


def _mle_scalar(objective, bounds: tuple[float, float], *, grid: int = 200) -> float:
    """Coarse log grid, then a bounded local refine. Returns the minimiser."""
    lo, hi = bounds
    points = np.exp(np.linspace(np.log(lo), np.log(hi), grid))
    values = np.array([float(objective(float(point))) for point in points])
    best_index = int(np.argmin(values))
    best = float(points[best_index])
    best_value = float(values[best_index])
    try:
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
        if np.isfinite(result.fun) and float(result.fun) <= best_value:
            return float(result.x)
    except Exception:  # pragma: no cover - numerical fallback
        pass
    return best


def _minimise_scalar_linear(objective, bounds: tuple[float, float], *, grid: int = 321) -> float:
    """Same contract as ``_mle_scalar`` but for a parameter that can be negative."""
    lo, hi = bounds
    points = np.linspace(lo, hi, grid)
    values = np.array([float(objective(float(point))) for point in points])
    best_index = int(np.argmin(values))
    best = float(points[best_index])
    best_value = float(values[best_index])
    try:
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
        if np.isfinite(result.fun) and float(result.fun) <= best_value:
            return float(result.x)
    except Exception:  # pragma: no cover - numerical fallback
        pass
    return best


def _logistic_shift(probability: np.ndarray, intercept: float) -> np.ndarray:
    """Recalibrate a probability by shifting its log-odds by a fitted intercept."""
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    logit = np.log(probability / (1.0 - probability)) + float(intercept)
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))


def _bernoulli_logistic_nll(intercept: float, probability: np.ndarray, outcomes: np.ndarray) -> float:
    """Bernoulli NLL of the intercept-recalibrated probability."""
    calibrated = _logistic_shift(probability, intercept)
    outcomes = np.asarray(outcomes, dtype=float)
    loglik = outcomes * np.log(calibrated) + (1.0 - outcomes) * np.log(1.0 - calibrated)
    total = -float(np.sum(loglik))
    return total if np.isfinite(total) else 1e18


def _stats(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    missing = [name for name in STAT_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing stat columns: {missing}")
    return {name: frame[name].to_numpy().astype(float) for name in STAT_COLUMNS}


def _expand_trials(x_frame, targets: np.ndarray, successes: np.ndarray):
    """Repeat each row into ``targets`` Bernoulli trials (successes first)."""
    targets = np.maximum(np.rint(np.asarray(targets, dtype=float)), 0.0).astype(int)
    successes = np.clip(np.rint(np.asarray(successes, dtype=float)), 0, targets)
    keep = targets > 0
    if not np.any(keep):
        return x_frame.iloc[[]], np.zeros(0, dtype=float)
    row_index = np.arange(len(targets))[keep]
    n_ok = successes[keep].astype(int)
    n_no = (targets[keep] - successes[keep]).astype(int)
    index = np.concatenate([np.repeat(row_index, n_ok), np.repeat(row_index, n_no)])
    label = np.concatenate([np.ones(int(n_ok.sum())), np.zeros(int(n_no.sum()))])
    return x_frame.iloc[index], label


def _negbin_dispersion(mu: np.ndarray, y: np.ndarray) -> float:
    """Dispersion of a mean-parametrised negative binomial from the marginal variance.

    A plug-in MLE on the model's own conditional means charges the mean-model
    error to the dispersion twice: once in the calibration residual and again
    when the sampler re-uses the same conditional mean for every draw.  Matching
    the observed marginal variance to ``Var(mu) + E[mu] + E[mu**2] / k`` removes
    that double count.  When the mean spread plus Poisson noise already reaches
    the observed variance the estimator returns the Poisson limit.
    """
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return SHAPE_BOUNDS[1]
    within = float(np.var(y)) - float(np.var(mu)) - float(np.mean(mu))
    if not np.isfinite(within) or within <= 1e-6:
        return SHAPE_BOUNDS[1]
    return float(np.clip(float(np.mean(mu**2)) / within, *SHAPE_BOUNDS))


def _shrink(rate: np.ndarray, exposure: np.ndarray, prior: float, strength: float) -> np.ndarray:
    """Empirical-Bayes shrink of a conditional rate toward the global prior."""
    rate = np.asarray(rate, dtype=float)
    exposure = np.maximum(np.asarray(exposure, dtype=float), 0.0)
    prior = float(prior)
    strength = max(float(strength), 1e-6)
    return (rate * exposure + strength * prior) / (exposure + strength)


def _project_moments(
    weights: np.ndarray,
    means: np.ndarray,
    sds: np.ndarray,
    target_mean: float,
    target_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Force a mixture to reproduce a target per-gain mean and variance.

    The mean is matched by a shift.  The variance is split into the between
    component (spread of the component means) and the within component (the
    component spreads); the between part is compressed if it alone would exceed
    the target, and the within part absorbs the remainder.
    """
    weights = np.maximum(np.asarray(weights, dtype=float), MIN_MIXTURE_WEIGHT)
    weights = weights / weights.sum()
    means = np.asarray(means, dtype=float)
    sds = np.maximum(np.asarray(sds, dtype=float), 1e-3)
    target_variance = max(float(target_variance), MIN_GAIN_VARIANCE)
    means = means + (float(target_mean) - float(np.sum(weights * means)))
    centre = float(np.sum(weights * means))
    between = float(np.sum(weights * means**2) - centre**2)
    limit = MAX_BETWEEN_FRACTION * target_variance
    if between > limit:
        factor = float(np.sqrt(limit / max(between, 1e-12)))
        means = centre + (means - centre) * factor
        between = float(np.sum(weights * means**2) - float(np.sum(weights * means)) ** 2)
    within = max(target_variance - between, MIN_GAIN_VARIANCE)
    current = float(np.sum(weights * sds**2))
    if current > 1e-12:
        sds = sds * float(np.sqrt(within / current))
    else:
        sds = np.full(sds.shape, float(np.sqrt(within)))
    return weights, means, sds


def _gain_moments(
    totals: np.ndarray, counts: np.ndarray, mu_per_gain: np.ndarray
) -> tuple[float, float]:
    """Per-gain mean and per-gain variance implied by aggregated observations.

    ``mu_per_gain`` is the conditional mean model's prediction of the *single
    gain*, so a row with ``counts[i]`` gains is centred at
    ``counts[i] * mu_per_gain[i]``.  Using the conditional mean keeps across-row
    mean differences out of the per-gain noise estimate.
    """
    totals = np.asarray(totals, dtype=float)
    counts = np.asarray(counts, dtype=float)
    mu_per_gain = np.asarray(mu_per_gain, dtype=float)
    ok = (counts >= 1.0) & np.isfinite(totals)
    if not np.any(ok):
        return 0.0, MIN_GAIN_VARIANCE
    mean = float(totals[ok].sum() / counts[ok].sum())
    residual = (totals[ok] - counts[ok] * mu_per_gain[ok]) ** 2 / counts[ok]
    variance = float(np.mean(residual))
    if not np.isfinite(variance):
        variance = MIN_GAIN_VARIANCE
    return mean, max(variance, MIN_GAIN_VARIANCE)


@dataclass
class GainMixture:
    """Signed per-completion (or per-carry) gain law: a small Gaussian mixture.

    The mixture is fitted to *aggregated* observations: row ``i`` reports the sum
    of ``counts[i]`` i.i.d. gains, not a single gain.  Rows with one completion
    identify the spread, rows with many completions identify the level, and the
    EM below accounts for the ``counts`` scaling of both moments.  That matters
    for quarterbacks: only ~15 rows per season throw exactly one completion.
    """

    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.sum(self.weights * self.means))

    @classmethod
    def fit(cls, gains: np.ndarray) -> "GainMixture":
        """Fit from individual single-gain observations."""
        gains = np.asarray(gains, dtype=float)
        return cls.fit_aggregated(gains, np.ones_like(gains))

    @property
    def variance(self) -> float:
        """Implied per-gain variance of the mixture."""
        return float(np.sum(self.weights * (self.sds**2 + self.means**2)) - self.mean**2)

    @classmethod
    def fit_aggregated(
        cls,
        totals: np.ndarray,
        counts: np.ndarray,
        *,
        target_mean: float | None = None,
        target_variance: float | None = None,
        components: int = GAIN_COMPONENTS,
        iterations: int = GAIN_EM_ITERATIONS,
    ) -> "GainMixture":
        """Fit from row sums and the number of gains each sum contains.

        ``target_mean`` and ``target_variance`` are the per-gain moments the
        fitted law must reproduce.  They are estimated from the same rows
        (see ``_gain_moments``) and the EM is projected onto them after every
        M-step, so a sampled sum of ``c`` gains always has mean ``c * mean`` and
        variance ``c * variance`` -- the compound model keeps the observed
        spread of the yardage totals.
        """
        totals = np.asarray(totals, dtype=float)
        counts = np.asarray(counts, dtype=float)
        ok = np.isfinite(totals) & np.isfinite(counts) & (counts >= 1.0)
        totals = totals[ok]
        counts = counts[ok]
        pooled = float(totals.sum() / counts.sum()) if counts.sum() > 0 else 0.0
        if target_mean is None:
            target_mean = pooled
        if target_variance is None:
            residual = (totals - counts * target_mean) ** 2 / np.maximum(counts, 1.0)
            target_variance = float(residual.mean()) if residual.size else 1.0
        target_variance = max(float(target_variance), 1e-6)
        if totals.size < MIN_GAIN_ROWS or float(counts.sum()) < MIN_GAIN_ROWS:
            return cls(
                np.array([1.0]),
                np.array([float(target_mean)]),
                np.array([float(np.sqrt(target_variance))]),
            )

        per_unit = totals / counts
        order = np.argsort(per_unit)
        halves = (order[: order.size // 2], order[order.size // 2 :])
        means = np.array([float(per_unit[group].mean()) for group in halves])
        sds = np.array(
            [
                max(
                    float(
                        np.std((totals[group] - counts[group] * means[index]) / np.sqrt(counts[group]))
                    ),
                    0.5,
                )
                for index, group in enumerate(halves)
            ]
        )
        weights = np.full(components, 1.0 / components)
        if components != 2:  # keep the generic path correct without extra columns
            means = np.linspace(float(per_unit.min()), float(per_unit.max()), components)
            sds = np.full(components, float(np.std(per_unit)) + 0.5)
        previous = -np.inf
        scale_counts = np.sqrt(counts)[:, None]
        for _ in range(iterations):
            spread = scale_counts * sds[None, :]
            residual = totals[:, None] - counts[:, None] * means[None, :]
            logp = (
                -0.5 * (residual / spread) ** 2
                - np.log(spread)
                + np.log(np.maximum(weights, 1e-12))[None, :]
            )
            logp -= logp.max(axis=1, keepdims=True)
            responsibilities = np.exp(logp)
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            loglik = float(np.log(responsibilities.sum(axis=1)).sum())
            mass = responsibilities.sum(axis=0)
            weights = np.maximum(mass / totals.size, MIN_MIXTURE_WEIGHT)
            weights = weights / weights.sum()
            for index in range(components):
                weighted = responsibilities[:, index] * counts
                total_weight = float(weighted.sum())
                if total_weight <= 1e-9:
                    continue
                means[index] = float((responsibilities[:, index] * totals).sum() / total_weight)
                # y_i has variance counts_i * s_j**2 given component j, so the
                # likelihood weights each squared residual by 1 / counts_i.
                residual = (totals - counts * means[index]) ** 2 / counts
                variance = float((responsibilities[:, index] * residual).sum() / mass[index])
                sds[index] = float(np.sqrt(max(variance, 1e-6) + GAIN_REG_COVAR))
            weights, means, sds = _project_moments(
                weights, means, sds, target_mean, target_variance
            )
            if abs(loglik - previous) < 1e-8 * max(abs(loglik), 1.0):
                break
            previous = loglik
        return cls(*_project_moments(weights, means, sds, target_mean, target_variance))

    def sample_gains(
        self, sizes: np.ndarray, shifts: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat i.i.d. gains plus the reduceat offsets for each group."""
        sizes = np.asarray(sizes, dtype=int)
        shifts = np.asarray(shifts, dtype=float)
        if sizes.size == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=int)
        total = int(sizes.sum())
        if total == 0:
            return np.zeros(0, dtype=float), np.zeros(sizes.size, dtype=int)
        components = rng.choice(len(self.weights), size=total, p=self.weights)
        draws = self.means[components] + self.sds[components] * rng.normal(size=total)
        draws = draws + np.repeat(shifts, sizes)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        return draws, offsets

    def sample_totals(self, sizes: np.ndarray, shifts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sum of ``sizes[i]`` i.i.d. gains with an additive ``shifts[i]`` offset."""
        sizes = np.asarray(sizes, dtype=int)
        if sizes.size == 0:
            return np.zeros(0, dtype=float)
        draws, offsets = self.sample_gains(sizes, shifts, rng)
        if draws.size == 0:
            return np.zeros(sizes.size, dtype=float)
        return np.add.reduceat(draws, offsets)


def _td_gain_curve(
    gains: np.ndarray, trials: np.ndarray, successes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Monotone P(touchdown | gain) on quantile bins, Laplace smoothed.

    ``gains`` holds the average gain of the ``trials`` opportunities in each
    calibration row, and ``successes`` the touchdowns among them.  Using the row
    average (instead of an individual gain) is the only feasible read for
    quarterbacks: single-completion rows are too rare to bin.  The curve is a
    shape only -- its level is reset per row by the shrunk rate model -- so the
    averaging weakens the slope without biasing the touchdown rate.
    """
    gains = np.asarray(gains, dtype=float)
    trials = np.asarray(trials, dtype=float)
    successes = np.asarray(successes, dtype=float)
    if gains.size < MIN_GAIN_ROWS or float(trials.sum()) < MIN_GAIN_ROWS:
        return np.zeros(0), np.zeros(0)
    expanded = np.repeat(gains, np.maximum(np.rint(trials), 0.0).astype(int))
    if expanded.size < MIN_GAIN_ROWS:
        return np.zeros(0), np.zeros(0)
    edges = np.unique(np.quantile(expanded, np.linspace(0.0, 1.0, TD_BINS + 1)))
    if edges.size < 3:
        return np.zeros(0), np.zeros(0)
    global_rate = float(successes.sum() / max(trials.sum(), 1.0))
    bins = np.clip(np.searchsorted(edges, gains, side="right") - 1, 0, edges.size - 2)
    rates = np.empty(edges.size - 1, dtype=float)
    for index in range(edges.size - 1):
        mask = bins == index
        rates[index] = (
            float(successes[mask].sum()) + TD_PRIOR_STRENGTH * global_rate
        ) / (float(trials[mask].sum()) + TD_PRIOR_STRENGTH)
    rates = np.maximum.accumulate(rates)  # longer gains must not score less often
    return edges, rates


def _truncation_mean(mu_positive: np.ndarray, k: float) -> np.ndarray:
    """Untruncated NB mean whose ``>= 1`` truncation has mean ``mu_positive``.

    For Y ~ NegBin(k, mu) the truncated mean is ``mu / (1 - P(Y = 0))`` with
    ``P(Y = 0) = (k / (k + mu)) ** k``, so the fixed point below restores the
    conditional mean the hurdle model was fitted on.
    """
    target = np.maximum(np.asarray(mu_positive, dtype=float), 1e-3)
    mu = target.copy()
    for _ in range(TRUNCATION_ITERATIONS):
        mu = target * (1.0 - np.power(k / (k + mu), k))
    return np.maximum(mu, 1e-6)


def _truncated_negbin_sample(
    mu_positive: np.ndarray, k: float, size: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    """>= 1 draws of a negative binomial, via bounded rejection (P(0) shrinks fast)."""
    mu = _truncation_mean(mu_positive, k)
    probability = k / (k + mu)
    out = rng.negative_binomial(k, probability, size=size).astype(float)
    for _ in range(TRUNCATION_ROUNDS):
        zero = out <= 0.0
        if not bool(zero.any()):
            break
        count = int(zero.sum())
        out[zero] = rng.negative_binomial(k, probability[zero], size=count).astype(float)
    return np.maximum(out, 1.0)


@dataclass
class ShapeLayer:
    """The low-dimensional part of the QB chain: dispersion, gain law, rates."""

    k_attempts: float
    attempt_hurdle_rate: float
    attempt_hurdle_intercept: float
    rho_completion: float
    gain_mixture: GainMixture
    global_ypc: float
    td_rate_global: float
    td_edges: np.ndarray
    td_rates: np.ndarray
    intercept_rate_global: float
    td_shrinkage: float
    intercept_shrinkage: float
    k_carries: float
    rush_gain_mixture: GainMixture
    global_ypcarry: float
    rush_td_rate_global: float
    rush_td_edges: np.ndarray
    rush_td_rates: np.ndarray

    @property
    def gain_mean(self) -> float:
        return self.gain_mixture.mean

    @property
    def rush_gain_mean(self) -> float:
        return self.rush_gain_mixture.mean

    def has_td_curve(self) -> bool:
        return self.td_edges.size >= 3 and self.td_rates.size >= 2

    def has_rush_td_curve(self) -> bool:
        return self.rush_td_edges.size >= 3 and self.rush_td_rates.size >= 2

    @staticmethod
    def _expected_rate(
        mixture: GainMixture,
        edges: np.ndarray,
        rates: np.ndarray,
        mu_gain: np.ndarray,
        shift_to_conditional_mean: bool = True,
    ) -> np.ndarray:
        """E[P(touchdown | gain)] over the mixture, at each row's mean gain."""
        shift = np.asarray(mu_gain, dtype=float) - mixture.mean if shift_to_conditional_mean else np.zeros_like(
            np.asarray(mu_gain, dtype=float)
        )
        component_means = mixture.means[:, None] + shift[None, :]
        bins = np.clip(np.searchsorted(edges, component_means, side="right") - 1, 0, rates.size - 1)
        return (mixture.weights[:, None] * rates[bins]).sum(axis=0)

    def td_scale(self, mu_ypc: np.ndarray, td_rate: np.ndarray) -> np.ndarray:
        """Per-row multiplier so E[P(touchdown | gain)] equals the row's rate."""
        expected = self._expected_rate(
            self.gain_mixture, np.asarray(self.td_edges, dtype=float),
            np.asarray(self.td_rates, dtype=float), mu_ypc,
        )
        scale = np.asarray(td_rate, dtype=float) / np.maximum(expected, 1e-4)
        return np.clip(scale, *TD_SCALE_BOUNDS)

    def rush_td_scale(self, mu_ypcarry: np.ndarray, rush_td_rate: np.ndarray) -> np.ndarray:
        expected = self._expected_rate(
            self.rush_gain_mixture, np.asarray(self.rush_td_edges, dtype=float),
            np.asarray(self.rush_td_rates, dtype=float), mu_ypcarry,
        )
        scale = np.asarray(rush_td_rate, dtype=float) / np.maximum(expected, 1e-4)
        return np.clip(scale, *TD_SCALE_BOUNDS)

    def scalars(self) -> dict[str, float]:
        """Every fitted shape scalar, for the report."""
        return {
            "k_attempts": float(self.k_attempts),
            "attempt_hurdle_rate": float(self.attempt_hurdle_rate),
            "attempt_hurdle_intercept": float(self.attempt_hurdle_intercept),
            "rho_completion": float(self.rho_completion),
            "td_rate_global": float(self.td_rate_global),
            "gain_mean": float(self.gain_mean),
            "gain_variance": float(self.gain_mixture.variance),
            "intercept_rate_global": float(self.intercept_rate_global),
            "td_shrinkage": float(self.td_shrinkage),
            "intercept_shrinkage": float(self.intercept_shrinkage),
            "global_ypc": float(self.global_ypc),
            "k_carries": float(self.k_carries),
            "rush_gain_mean": float(self.rush_gain_mean),
            "rush_gain_variance": float(self.rush_gain_mixture.variance),
            "global_ypcarry": float(self.global_ypcarry),
            "rush_td_rate_global": float(self.rush_td_rate_global),
            "gain_sd_1": float(self.gain_mixture.sds[0]),
            "gain_sd_2": float(self.gain_mixture.sds[-1]),
            "rush_gain_sd_1": float(self.rush_gain_mixture.sds[0]),
            "rush_gain_sd_2": float(self.rush_gain_mixture.sds[-1]),
        }


def _fit_mean_models(
    x, stats: dict[str, np.ndarray], *, n_estimators: int, categorical
) -> dict[str, Any]:
    """Fit every conditional mean of the chain on one design matrix."""
    with_completion = stats["completions"] > 0
    with_attempt = stats["attempts"] > 0
    with_carry = stats["carries"] > 0
    if not np.any(with_completion):
        raise ValueError("no completed passes in the training frame")
    if not np.any(with_attempt):
        raise ValueError("no pass attempts in the training frame")
    if not np.any(with_carry):
        raise ValueError("no carries in the training frame")
    trial_x, trial_y = _expand_trials(x, stats["attempts"], stats["completions"])
    if trial_y.size == 0:
        raise ValueError("no completion trials in the training frame")
    return {
        "attempt_hurdle": _fit_regressor(
            x,
            (stats["attempts"] > 0).astype(float),
            _stage_params("binary", n_estimators),
            categorical,
        ),
        "attempts_positive": _fit_regressor(
            x[with_attempt],
            stats["attempts"][with_attempt],
            _stage_params("poisson", n_estimators),
            categorical,
        ),
        "completion": _fit_regressor(
            trial_x, trial_y, _stage_params("binary", n_estimators), categorical
        ),
        "ypc": _fit_regressor(
            x[with_completion],
            stats["passing_yards"][with_completion] / stats["completions"][with_completion],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["completions"][with_completion],
        ),
        "td": _fit_regressor(
            x[with_completion],
            stats["passing_tds"][with_completion] / stats["completions"][with_completion],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["completions"][with_completion],
        ),
        "interception": _fit_regressor(
            x[with_attempt],
            stats["interceptions"][with_attempt] / stats["attempts"][with_attempt],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["attempts"][with_attempt],
        ),
        "carry": _fit_regressor(
            x, (stats["carries"] > 0).astype(float), _stage_params("binary", n_estimators), categorical
        ),
        "carries": _fit_regressor(
            x[with_carry],
            stats["carries"][with_carry],
            _stage_params("poisson", n_estimators),
            categorical,
        ),
        "ypcarry": _fit_regressor(
            x[with_carry],
            stats["rushing_yards"][with_carry] / stats["carries"][with_carry],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["carries"][with_carry],
        ),
    }


def _td_shrinkage_nll(
    strength: float, model_rate: np.ndarray, exposure: np.ndarray, successes: np.ndarray, prior: float
) -> float:
    """Binomial NLL of the shrunk touchdown rate on the calibration rows."""
    trials = np.maximum(np.rint(np.asarray(exposure, dtype=float)), 0.0)
    successes = np.clip(np.rint(np.asarray(successes, dtype=float)), 0.0, trials)
    shrunk = np.clip(_shrink(np.asarray(model_rate, dtype=float), trials, prior, strength), 1e-9, 1.0 - 1e-9)
    loglik = successes * np.log(shrunk) + (trials - successes) * np.log(1.0 - shrunk)
    total = -float(np.sum(loglik))
    return total if np.isfinite(total) else 1e18


def _interception_shrinkage_nll(
    strength: float, model_rate: np.ndarray, exposure: np.ndarray, successes: np.ndarray, prior: float
) -> float:
    """Poisson NLL of ``interceptions ~ attempts * rate`` on the calibration rows."""
    attempts = np.maximum(np.asarray(exposure, dtype=float), 1e-6)
    observed = np.asarray(successes, dtype=float)
    lam = np.maximum(_shrink(np.asarray(model_rate, dtype=float), attempts, prior, strength), 1e-9) * attempts
    loglik = observed * np.log(lam) - lam - gammaln(observed + 1.0)
    total = -float(np.sum(loglik))
    return total if np.isfinite(total) else 1e18


def _estimate_shapes(
    models: dict[str, Any], x_cal, stats_cal: dict[str, np.ndarray]
) -> ShapeLayer:
    """Estimate the shape layer on rows the mean models have not seen."""
    # --- attempts hurdle --------------------------------------------------
    # q(X) decides whether the quarterback throws at all; the raw logistic
    # probability is over-confident (mean 0.992 against a 0.955 positive rate on
    # the 2023 calibration rows), so its log-odds are shifted by an intercept
    # fitted by Bernoulli MLE on the calibration season.  That keeps the ranking
    # (AUC 0.88) and repairs the level.
    attempt_outcomes = (stats_cal["attempts"] > 0).astype(float)
    q_attempts_raw = np.clip(
        models["attempt_hurdle"].predict(x_cal).astype(float), 1e-6, 1.0 - 1e-6
    )
    attempt_hurdle_intercept = _minimise_scalar_linear(
        lambda shift: _bernoulli_logistic_nll(shift, q_attempts_raw, attempt_outcomes), HURDLE_BOUNDS
    )
    q_attempts = _logistic_shift(q_attempts_raw, attempt_hurdle_intercept)
    attempt_hurdle_rate = float(np.mean(attempt_outcomes))
    # The conditional NegBin is fitted on the rows that did throw, and its
    # dispersion is matched to the calibration season's marginal variance
    # (see _negbin_dispersion) instead of the wide-biased plug-in MLE.
    with_attempt = stats_cal["attempts"] > 0
    mu_attempts = np.maximum(models["attempts_positive"].predict(x_cal).astype(float), 1e-3)
    k_attempts = _negbin_dispersion(mu_attempts[with_attempt], stats_cal["attempts"][with_attempt])
    # Everything downstream uses the hurdle's expected exposure.
    expected_attempts = q_attempts * mu_attempts

    p_completion = np.clip(models["completion"].predict(x_cal).astype(float), 1e-4, 1.0 - 1e-4)
    trials = stats_cal["attempts"]
    successes = stats_cal["completions"]
    ok = (trials >= 1) & (successes <= trials) & (successes >= 0)
    rho_completion = _mle_scalar(
        lambda rho: _beta_binomial_nll(rho, successes[ok], trials[ok], p_completion[ok]), RHO_BOUNDS
    )

    with_completion = stats_cal["completions"] > 0
    model_ypc = models["ypc"].predict(x_cal).astype(float)
    gain_mean, gain_variance = _gain_moments(
        stats_cal["passing_yards"][with_completion],
        stats_cal["completions"][with_completion],
        model_ypc[with_completion],
    )
    gain_mixture = GainMixture.fit_aggregated(
        stats_cal["passing_yards"][with_completion],
        stats_cal["completions"][with_completion],
        target_mean=gain_mean,
        target_variance=gain_variance,
    )
    global_ypc = float(
        stats_cal["passing_yards"][with_completion].sum()
        / max(stats_cal["completions"][with_completion].sum(), 1.0)
    )
    td_edges, td_rates = _td_gain_curve(
        stats_cal["passing_yards"][with_completion] / stats_cal["completions"][with_completion],
        stats_cal["completions"][with_completion],
        stats_cal["passing_tds"][with_completion],
    )
    # The touchdown rate model is shrunk toward the calibration rate.  The
    # strength is itself estimated (binomial NLL on the calibration rows).
    expected_completions = np.maximum(expected_attempts * p_completion, 0.25)
    td_rate_global = float(
        stats_cal["passing_tds"].sum() / max(stats_cal["completions"].sum(), 1.0)
    )
    model_td_rate = np.maximum(models["td"].predict(x_cal).astype(float), 0.0)
    td_shrinkage = _mle_scalar(
        lambda strength: _td_shrinkage_nll(
            strength, model_td_rate, expected_completions, stats_cal["passing_tds"], td_rate_global
        ),
        SHRINKAGE_BOUNDS,
    )

    intercept_rate_global = float(
        stats_cal["interceptions"].sum() / max(stats_cal["attempts"].sum(), 1.0)
    )
    model_interception = np.maximum(models["interception"].predict(x_cal).astype(float), 0.0)
    intercept_shrinkage = _mle_scalar(
        lambda strength: _interception_shrinkage_nll(
            strength,
            model_interception[with_attempt],
            expected_attempts[with_attempt],
            stats_cal["interceptions"][with_attempt],
            intercept_rate_global,
        ),
        SHRINKAGE_BOUNDS,
    )

    with_carry = stats_cal["carries"] > 0
    mu_carries = np.maximum(models["carries"].predict(x_cal[with_carry]).astype(float), 1e-3)
    # Here the dispersion is fitted conditionally on a positive carry count, on
    # rows the mean model has not seen; the plug-in mean error is small there, so
    # the ordinary MLE already reproduces the observed spread of carries.
    k_carries = _mle_scalar(
        lambda k: _negbin_nll(k, stats_cal["carries"][with_carry], mu_carries), SHAPE_BOUNDS
    )
    model_ypcarry = models["ypcarry"].predict(x_cal).astype(float)
    rush_gain_mean, rush_gain_variance = _gain_moments(
        stats_cal["rushing_yards"][with_carry],
        stats_cal["carries"][with_carry],
        model_ypcarry[with_carry],
    )
    rush_gain_mixture = GainMixture.fit_aggregated(
        stats_cal["rushing_yards"][with_carry],
        stats_cal["carries"][with_carry],
        target_mean=rush_gain_mean,
        target_variance=rush_gain_variance,
    )
    global_ypcarry = float(
        stats_cal["rushing_yards"][with_carry].sum() / max(stats_cal["carries"][with_carry].sum(), 1.0)
    )
    rush_td_edges, rush_td_rates = _td_gain_curve(
        stats_cal["rushing_yards"][with_carry] / stats_cal["carries"][with_carry],
        stats_cal["carries"][with_carry],
        stats_cal["rushing_tds"][with_carry],
    )
    rush_td_rate_global = float(
        stats_cal["rushing_tds"].sum() / max(stats_cal["carries"].sum(), 1.0)
    )
    return ShapeLayer(
        k_attempts=k_attempts,
        attempt_hurdle_rate=attempt_hurdle_rate,
        attempt_hurdle_intercept=attempt_hurdle_intercept,
        rho_completion=rho_completion,
        gain_mixture=gain_mixture,
        global_ypc=global_ypc,
        td_rate_global=td_rate_global,
        td_edges=td_edges,
        td_rates=td_rates,
        intercept_rate_global=intercept_rate_global,
        td_shrinkage=td_shrinkage,
        intercept_shrinkage=intercept_shrinkage,
        k_carries=k_carries,
        rush_gain_mixture=rush_gain_mixture,
        global_ypcarry=global_ypcarry,
        rush_td_rate_global=rush_td_rate_global,
        rush_td_edges=rush_td_edges,
        rush_td_rates=rush_td_rates,
    )


def estimate_frozen_shapes(
    frame: pl.DataFrame,
    season: int,
    feature_columns: tuple[str, ...],
    *,
    n_estimators: int = 200,
) -> ShapeLayer:
    """Shape parameters from the season immediately before the holdout.

    Means are fitted on ``frame[season <= season - 2]`` and predicted on the
    calibration season ``season - 1``, so nothing from ``season`` or later can
    enter the shape estimate.
    """
    calendar = frame.filter(pl.col("season") == season - 1)
    history = frame.filter(pl.col("season") <= season - 2)
    if calendar.is_empty():
        raise ValueError(f"no calibration rows in season {season - 1}")
    if history.is_empty():
        raise ValueError(f"no history before season {season - 1}")
    columns = tuple(name for name in feature_columns if name in frame.columns)
    x_history, x_calendar, categorical = _design(history, calendar, columns)
    models = _fit_mean_models(
        x_history, _stats(history), n_estimators=n_estimators, categorical=categorical
    )
    return _estimate_shapes(models, x_calendar, _stats(calendar))


@dataclass
class QBChain:
    """Fit and sample the QB production chain on one training fold."""

    feature_columns: tuple[str, ...]
    n_estimators: int = 200
    calibration_fraction: float = CALIBRATION_FRACTION

    def fit(self, train: pl.DataFrame, *, frozen: ShapeLayer | None = None) -> "FittedQBChain":
        columns = tuple(name for name in self.feature_columns if name in train.columns)
        if not columns:
            raise ValueError("no usable feature columns")
        stats_all = _stats(train)

        if frozen is None:
            order = _time_order(train)
            n_cal = max(int(round(self.calibration_fraction * len(order))), 120)
            n_cal = min(n_cal, len(order) - 200) if len(order) > 400 else max(len(order) // 5, 1)
            if n_cal < 20:
                raise ValueError("training fold too small for an internal calibration split")
            cal_index = np.sort(order[-n_cal:])
            fit_index = np.sort(order[:-n_cal])
            x_fit, x_cal, categorical = _design(
                train[fit_index.tolist()], train[cal_index.tolist()], columns
            )
            calibration_models = _fit_mean_models(
                x_fit,
                _stats(train[fit_index.tolist()]),
                n_estimators=self.n_estimators,
                categorical=categorical,
            )
            shapes = _estimate_shapes(calibration_models, x_cal, _stats(train[cal_index.tolist()]))
        else:
            # Frozen path: only the conditional means are re-estimated.
            shapes = frozen

        x_all, _, categorical_all = _design(train, train.head(0), columns)
        models = _fit_mean_models(
            x_all, stats_all, n_estimators=self.n_estimators, categorical=categorical_all
        )
        return FittedQBChain(
            columns=columns,
            reference=train.select(*columns),
            models=models,
            shapes=shapes,
        )


@dataclass
class FittedQBChain:
    """Fitted stage models plus the shape layer."""

    columns: tuple[str, ...]
    reference: pl.DataFrame
    models: dict[str, Any]
    shapes: ShapeLayer

    @property
    def k_attempts(self) -> float:
        return self.shapes.k_attempts

    @property
    def rho_completion(self) -> float:
        return self.shapes.rho_completion

    @property
    def gain_mixture(self) -> GainMixture:
        return self.shapes.gain_mixture

    @property
    def td_rate_global(self) -> float:
        return self.shapes.td_rate_global

    @property
    def intercept_rate_global(self) -> float:
        return self.shapes.intercept_rate_global

    def diagnostics(self) -> dict[str, float]:
        """The fitted shape-layer scalars, for the report."""
        return self.shapes.scalars()

    def sample(self, test: pl.DataFrame, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw coherent QB stat lines. Returns (n_test, n_draws, len(STAT_COLUMNS))."""
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1")
        _, x_test, _ = _design(self.reference, test, self.columns)
        n = x_test.shape[0]
        if n == 0:
            return np.zeros((0, n_draws, len(STAT_COLUMNS)))
        shapes = self.shapes

        q_attempts = _logistic_shift(
            np.clip(self.models["attempt_hurdle"].predict(x_test).astype(float), 1e-6, 1.0 - 1e-6),
            shapes.attempt_hurdle_intercept,
        )
        mu_attempts = np.maximum(
            self.models["attempts_positive"].predict(x_test).astype(float), 1e-3
        )
        p_completion = np.clip(self.models["completion"].predict(x_test).astype(float), 1e-4, 1.0 - 1e-4)
        mu_ypc = self.models["ypc"].predict(x_test).astype(float)
        td_rate = np.maximum(self.models["td"].predict(x_test).astype(float), 0.0)
        intercept_rate = np.maximum(self.models["interception"].predict(x_test).astype(float), 0.0)
        q_carry = np.clip(self.models["carry"].predict(x_test).astype(float), 1e-6, 1.0 - 1e-6)
        mu_carries = np.maximum(self.models["carries"].predict(x_test).astype(float), 1e-3)
        mu_ypcarry = self.models["ypcarry"].predict(x_test).astype(float)

        # Empirical-Bayes shrink of both rare-event rates toward the calibration
        # level, with the row's expected exposure as the weight.
        expected_attempts = q_attempts * mu_attempts
        expected_completions = np.maximum(expected_attempts * p_completion, 0.25)
        td_rate = _shrink(td_rate, expected_completions, shapes.td_rate_global, shapes.td_shrinkage)
        intercept_rate = _shrink(
            intercept_rate, expected_attempts, shapes.intercept_rate_global, shapes.intercept_shrinkage
        )
        # --- attempts: hurdle, then a >= 1 negative binomial -------------------
        attempts = np.zeros((n_draws, n), dtype=float)
        throwing = rng.random((n_draws, n)) < q_attempts[None, :]
        if np.any(throwing):
            exposures = np.broadcast_to(mu_attempts[None, :], (n_draws, n))[throwing]
            attempts[throwing] = _truncated_negbin_sample(
                exposures, shapes.k_attempts, (int(throwing.sum()),), rng
            )

        # --- completions | attempts -------------------------------------------
        rho = shapes.rho_completion
        if rho <= RHO_BOUNDS[0] * 1.001:
            theta = np.broadcast_to(p_completion[None, :], (n_draws, n))
        else:
            a = np.maximum(p_completion * (1.0 - rho) / rho, 1e-6)
            b = np.maximum((1.0 - p_completion) * (1.0 - rho) / rho, 1e-6)
            theta = rng.beta(a[None, :], b[None, :], size=(n_draws, n))
        completions = np.rint(
            rng.binomial(attempts.astype(np.int64), np.clip(theta, 0.0, 1.0))
        ).astype(float)

        # --- interceptions | attempts (Poisson, capped at the attempt count) ---
        interceptions = np.minimum(
            rng.poisson(attempts * intercept_rate[None, :]).astype(float), attempts
        )

        # --- per-completion process: yards and touchdowns ---------------------
        passing_yards = np.zeros((n_draws, n), dtype=float)
        passing_tds = np.zeros((n_draws, n), dtype=float)
        completed = completions > 0
        if np.any(completed):
            row_of = np.broadcast_to(np.arange(n)[None, :], completions.shape)[completed]
            sizes = completions[completed].astype(int)
            shifts = (
                np.broadcast_to(mu_ypc[None, :], completions.shape)[completed]
                - shapes.gain_mixture.mean
            )
            gains, offsets = shapes.gain_mixture.sample_gains(sizes, shifts, rng)
            if gains.size:
                passing_yards[completed] = np.add.reduceat(gains, offsets)
                if shapes.has_td_curve():
                    edges = np.asarray(shapes.td_edges, dtype=float)
                    rates = np.asarray(shapes.td_rates, dtype=float)
                    bins = np.clip(np.searchsorted(edges, gains, side="right") - 1, 0, rates.size - 1)
                    scale = shapes.td_scale(mu_ypc, td_rate)[row_of]
                    probability = np.clip(
                        rates[bins] * np.repeat(scale, sizes), 0.0, TD_MAX_PROBABILITY
                    )
                    flags = (rng.random(gains.size) < probability).astype(float)
                    passing_tds[completed] = np.add.reduceat(flags, offsets)
        if not shapes.has_td_curve():
            passing_tds = np.rint(
                rng.binomial(
                    completions.astype(np.int64),
                    np.broadcast_to(np.clip(td_rate, 0.0, TD_MAX_PROBABILITY)[None, :], completions.shape),
                )
            ).astype(float)

        # --- rushing hurdle, then the per-carry process -----------------------
        carries = np.zeros((n_draws, n), dtype=float)
        rushing_yards = np.zeros((n_draws, n), dtype=float)
        rushing_tds = np.zeros((n_draws, n), dtype=float)
        participating = rng.random((n_draws, n)) < q_carry[None, :]
        if np.any(participating):
            exposures = np.broadcast_to(mu_carries[None, :], (n_draws, n))[participating]
            carries[participating] = _truncated_negbin_sample(
                exposures, shapes.k_carries, (int(participating.sum()),), rng
            )
        active_rush = carries > 0
        if np.any(active_rush):
            row_of = np.broadcast_to(np.arange(n)[None, :], carries.shape)[active_rush]
            sizes = carries[active_rush].astype(int)
            shifts = (
                np.broadcast_to(mu_ypcarry[None, :], carries.shape)[active_rush]
                - shapes.rush_gain_mixture.mean
            )
            gains, offsets = shapes.rush_gain_mixture.sample_gains(sizes, shifts, rng)
            if gains.size:
                rushing_yards[active_rush] = np.add.reduceat(gains, offsets)
                if shapes.has_rush_td_curve():
                    edges = np.asarray(shapes.rush_td_edges, dtype=float)
                    rates = np.asarray(shapes.rush_td_rates, dtype=float)
                    bins = np.clip(np.searchsorted(edges, gains, side="right") - 1, 0, rates.size - 1)
                    scale = shapes.rush_td_scale(mu_ypcarry, np.full(n, shapes.rush_td_rate_global))[row_of]
                    probability = np.clip(
                        rates[bins] * np.repeat(scale, sizes), 0.0, TD_MAX_PROBABILITY
                    )
                    flags = (rng.random(gains.size) < probability).astype(float)
                    rushing_tds[active_rush] = np.add.reduceat(flags, offsets)
        if not shapes.has_rush_td_curve():
            rushing_tds = np.rint(
                rng.binomial(
                    carries.astype(np.int64),
                    np.full(carries.shape, min(shapes.rush_td_rate_global, TD_MAX_PROBABILITY)),
                )
            ).astype(float)

        joint = np.stack(
            [
                attempts,
                completions,
                passing_yards,
                passing_tds,
                interceptions,
                carries,
                rushing_yards,
                rushing_tds,
            ],
            axis=-1,
        )
        return np.transpose(joint, (1, 0, 2))


__all__ = [
    "STAT_COLUMNS",
    "STAT_INDEX",
    "QBChain",
    "FittedQBChain",
    "ShapeLayer",
    "GainMixture",
    "estimate_frozen_shapes",
    "qb_fantasy_points",
]
