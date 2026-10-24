"""Hierarchical parametric RB chain.

Structure (the conditional hierarchy of ``hierarchical.py`` and ``chains_qb.py``,
rebuilt for the one position that owns two production branches):

    carries         = hurdle: Bernoulli(q(X)), then a >= 1 truncated
                      NegBin(mu_carries(X), k_carries)
    rushing_yards   = sum of ``carries`` signed per-carry gains (Gaussian gain
                      mixture, mean shifted to a conditional yards-per-carry
                      model).  Rushing yards can be negative.
    rushing_tds     = one Bernoulli per carry through P(touchdown | gain), a
                      monotone curve fitted on the calibration season whose level
                      is reset per row by a shrunk conditional rate model
    targets         ~ NegBin(mu_targets(X), k_targets)
    receptions      ~ BetaBinomial(targets, p_catch(X), rho_catch)
    receiving_yards = sum of ``receptions`` signed per-catch gains (own mixture,
                      own conditional yards-per-catch model)
    receiving_tds   = one Bernoulli per catch through a gain-conditional curve

Branch dependence -- the coupling this position is required to model
====================================================================

The rushing and receiving branches are positively dependent in reality: an
every-down back earns carries and targets from the same backfield role, and one
game script pushes both counts the same way.  Drawing the two branches
independently makes the chain understate the joint ceiling, because a running
back who is used heavily on the ground is also used more as a receiver.

This module takes option (a) of the position brief: the two branches are coupled
through a **shared latent role/game-script factor**.  Option (b) -- conditioning
the receiving means on the *sampled* carries -- was rejected because a noisy draw
used as a regressor attenuates the true conditional dependence and would need a
generated-regressor correction to be honest.

Concretely, every (row, draw) cell shares one standard normal variate ``z``::

    volume(z) = exp(loading * z - loading**2 / 2)
    carries | participating ~ truncated NegBin(mu_carries(X) * volume(z), k_carries)
    targets                 ~ NegBin(mu_targets(X) * volume(z), k_targets)

The exponential tilt is centred, so ``E[volume(z)] = 1``: the shared shock moves
volume between games without moving the marginal expectations.  The single
loading is *estimated on the calibration season* by a one-dimensional grid search
over a small simulation that matches the model-implied correlation between
carries and targets to the observed one (``_estimate_latent_loading``).

Coupling and marginal spread are estimated jointly, not one after the other.
Adding a shared shock would otherwise inflate the marginal variance of both
counts, so ``_targets_dispersion`` and ``_carries_dispersion`` re-solve each
dispersion at the chosen loading from the observed marginal variance.  This is
the marginal-variance matching of ``chains_qb._negbin_dispersion`` generalised to
a log-normally mixed mean (and, for carries, to the hurdle).

Every mean is a LightGBM boosted model over the same feature matrix the other
candidates see.  The shape layer (dispersions, beta-binomial correlation, gain
mixtures, touchdown curves, shrinkage strengths, the coupling loading) is
estimated either on an internal time split of the training fold or supplied
frozen from a strictly prior season.

Structural invariants, enforced by construction:

* ``carries >= 0`` and ``carries >= 1`` whenever the hurdle fires;
* ``0 <= receptions <= targets`` (binomial over the drawn target count);
* rushing yards and rushing touchdowns only exist when ``carries > 0``;
* receiving yards and receiving touchdowns only exist when ``receptions > 0``;
* yards are signed: per-carry and per-catch gains may be negative, and nothing is
  clipped at zero except the counts that are genuinely non-negative.

Scoring rule (see ``hierarchical_football.scoring``)::

    receptions + 0.1 * (rushing_yards + receiving_yards)
    + 6 * (rushing_tds + receiving_tds)
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
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
)
STAT_INDEX: dict[str, int] = {name: index for index, name in enumerate(STAT_COLUMNS)}

POINTS_PER_RECEPTION = 1.0
POINTS_PER_YARD = 0.1
POINTS_PER_TOUCHDOWN = 6.0

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
TRUNCATION_ITERATIONS = 20
TRUNCATION_ROUNDS = 40
LATENT_BOUNDS = (0.0, 0.90)
LATENT_GRID = 10
LATENT_DRAWS = 96
MIN_LATENT_ROWS = 120
LATENT_SEED = 20250901


def rb_fantasy_points(stats: np.ndarray, *, ppr: float = POINTS_PER_RECEPTION) -> np.ndarray:
    """Apply the RB scoring rule to a raw stat sample.

    ``stats`` is any array whose last axis follows ``STAT_COLUMNS``; the shape is
    preserved.  The default is the panel's full PPR (one point per reception);
    pass 0.0 or 0.5 for the standard and half-PPR variants.
    """
    stats = np.asarray(stats, dtype=float)
    return (
        float(ppr) * stats[..., STAT_INDEX["receptions"]]
        + POINTS_PER_YARD
        * (stats[..., STAT_INDEX["rushing_yards"]] + stats[..., STAT_INDEX["receiving_yards"]])
        + POINTS_PER_TOUCHDOWN
        * (stats[..., STAT_INDEX["rushing_tds"]] + stats[..., STAT_INDEX["receiving_tds"]])
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


# ---------------------------------------------------------------------------
# The shared latent role/game-script factor
# ---------------------------------------------------------------------------


def _volume_factor(loading: float, z: np.ndarray) -> np.ndarray:
    """Centred exponential tilt shared by both branches; ``E[volume(z)] = 1``."""
    loading = float(loading)
    return np.exp(loading * np.asarray(z, dtype=float) - 0.5 * loading**2)


def _variance_factor(loading: float) -> float:
    """``E[volume(z) ** 2]`` for a standard normal ``z``."""
    return float(np.exp(float(loading) ** 2))


def _positive_variance(mu_positive: np.ndarray, k: float) -> np.ndarray:
    """Variance of the ``>= 1`` truncated negative binomial at mean ``mu_positive``.

    The untruncated law is recovered first (``_truncation_mean``), then
    ``E[Y**2 | Y >= 1] = E[Y**2] / (1 - P(Y = 0))`` gives the second moment of
    the variable the sampler actually draws.  Ignoring the truncation, as a plain
    ``mu + mu**2 / k`` does, overstates the spread of the positive counts at a
    fixed conditional mean and buys back the error with a dispersion that is far
    too small.
    """
    mu = np.maximum(np.asarray(mu_positive, dtype=float), 1e-3)
    k = float(k)
    untruncated = _truncation_mean(mu, k)
    zero_mass = np.power(k / (k + untruncated), k)
    var_untruncated = untruncated + untruncated**2 / k
    second_moment = (var_untruncated + untruncated**2) / np.maximum(1.0 - zero_mass, 1e-9)
    return np.maximum(second_moment - mu**2, 1e-9)


def _pooled_targets_variance(k: float, mu: np.ndarray, loading: float) -> float:
    """Model variance of targets for one dispersion, pooled over rows.

    Per row the log-normally mixed mean gives ``mu + mu**2 * (factor - 1)
    + mu**2 * factor / k``; pooling adds the spread of the conditional means.
    """
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    factor = _variance_factor(loading)
    return float(
        np.mean(mu) + np.mean(mu**2) * (factor - 1.0) + np.mean(mu**2) * factor / float(k) + np.var(mu)
    )


def _pooled_carries_variance(
    k: float, q: np.ndarray, mu_positive: np.ndarray, loading: float
) -> float:
    """Model variance of carries for one dispersion, pooled over rows.

    ``carries = Bernoulli(q) * Y`` with ``Y`` the ``>= 1`` truncated count, so the
    row variance is ``q * (Var(Y) + E[Y**2]) - (q * E[Y])**2``.  The latent factor
    scales the mean square of ``Y``; the truncation ratio is treated as level-free
    and applied to the mixed second moment.
    """
    q = np.clip(np.asarray(q, dtype=float), 1e-6, 1.0 - 1e-6)
    mu = np.maximum(np.asarray(mu_positive, dtype=float), 1e-3)
    k = float(k)
    factor = _variance_factor(loading)
    var_positive = _positive_variance(mu, k)
    var_untruncated = mu + mu**2 / k
    ratio = var_positive / np.maximum(var_untruncated, 1e-9)
    second_moment = ratio * (mu + mu**2 * factor / k) + mu**2 * factor
    per_row = q * second_moment - (q * mu) ** 2
    return float(np.mean(per_row) + np.var(q * mu))


def _match_dispersion(
    objective, observed_variance: float, *, bounds: tuple[float, float] = SHAPE_BOUNDS, iterations: int = 30
) -> float:
    """Dispersion whose model variance equals the observed one.

    The variance of a mean-parametrised count law falls monotonically in the
    dispersion, so a geometric bisection finds the root without a likelihood.
    """
    lo, hi = (float(bounds[0]), float(bounds[1]))
    observed = float(observed_variance)
    if not np.isfinite(observed) or observed <= 0.0:
        return hi
    if float(objective(lo)) <= observed:
        return lo
    if float(objective(hi)) >= observed:
        return hi
    for _ in range(int(iterations)):
        mid = float(np.sqrt(lo * hi))
        if float(objective(mid)) > observed:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _targets_dispersion(mu: np.ndarray, y: np.ndarray, loading: float = 0.0) -> float:
    """NegBin dispersion of targets from the marginal variance, given the coupling.

    Solves ``_pooled_targets_variance(k) = Var(y)``.  At ``loading = 0`` the root
    is ``E[mu**2] / (Var(y) - E[mu] - Var(mu))``, the marginal-variance estimator
    ``chains_qb`` uses for attempts; the coupling adds
    ``E[mu**2] * (factor - 1)`` to the variance the dispersion has to absorb.  A
    mean spread that already reaches the observed variance -- with or without the
    coupling -- returns the Poisson limit.
    """
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return SHAPE_BOUNDS[1]
    return _match_dispersion(lambda k: _pooled_targets_variance(k, mu, loading), float(np.var(y)))


def _carries_dispersion(
    q: np.ndarray, mu_positive: np.ndarray, y: np.ndarray, loading: float = 0.0
) -> float:
    """NegBin dispersion of the positive carry count, given hurdle and coupling.

    The hurdle, the spread of ``q * mu`` and the truncation of the positive count
    are all removed from the observed variance before the dispersion absorbs the
    remainder, so carries keep the observed marginal spread instead of the
    inflated one a plug-in fit on conditional means would produce.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return SHAPE_BOUNDS[1]
    return _match_dispersion(
        lambda k: _pooled_carries_variance(k, q, mu_positive, loading), float(np.var(y))
    )


def _branch_correlation(
    q: np.ndarray,
    mu_carries: np.ndarray,
    mu_targets: np.ndarray,
    k_carries: float,
    k_targets: float,
    loading: float,
    rng: np.random.Generator,
    draws: int = LATENT_DRAWS,
) -> float:
    """Simulated ``corr(carries, targets)`` for one candidate loading.

    Only the two count stages enter: the observed branch correlation is a
    property of the counts, and the catch process adds no cross-branch noise.
    """
    q = np.clip(np.asarray(q, dtype=float), 1e-6, 1.0 - 1e-6)
    mu_carries = np.maximum(np.asarray(mu_carries, dtype=float), 1e-6)
    mu_targets = np.maximum(np.asarray(mu_targets, dtype=float), 1e-6)
    n = q.size
    volume = _volume_factor(loading, rng.normal(size=(draws, n)))
    mean_carries = volume * mu_carries[None, :]
    mean_targets = volume * mu_targets[None, :]

    carries = np.zeros((draws, n), dtype=float)
    participating = rng.random((draws, n)) < q[None, :]
    if bool(participating.any()):
        carries[participating] = _truncated_negbin_sample(
            mean_carries[participating], float(k_carries), (int(participating.sum()),), rng
        )
    probability = float(k_targets) / (float(k_targets) + mean_targets)
    targets = rng.negative_binomial(float(k_targets), probability, size=(draws, n)).astype(float)
    flat_carries = carries.ravel()
    flat_targets = targets.ravel()
    if flat_carries.std() <= 0.0 or flat_targets.std() <= 0.0:
        return 0.0
    return float(np.corrcoef(flat_carries, flat_targets)[0, 1])


def _estimate_latent_loading(
    q: np.ndarray,
    mu_carries: np.ndarray,
    mu_targets: np.ndarray,
    carries: np.ndarray,
    targets: np.ndarray,
    *,
    rng: np.random.Generator,
    grid: int = LATENT_GRID,
    draws: int = LATENT_DRAWS,
) -> tuple[float, float, float]:
    """Loading on the shared factor, matched on the calibration season.

    Returns ``(loading, model_correlation, observed_correlation)``.  The loading
    is chosen so the model-implied correlation between carries and targets
    reproduces the calibration-season value; at every candidate the two
    dispersions are re-solved by marginal-variance matching, so the search
    compares couplings at a fixed marginal spread rather than trading one moment
    against the other.  A calibration block too small to estimate the correlation
    falls back to independent branches and reports ``nan`` for both correlations.
    """
    carries = np.asarray(carries, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if carries.size < MIN_LATENT_ROWS or targets.size < MIN_LATENT_ROWS:
        return 0.0, float("nan"), float("nan")
    if carries.std() <= 0.0 or targets.std() <= 0.0:
        return 0.0, float("nan"), float("nan")
    observed = float(np.corrcoef(carries, targets)[0, 1])
    best_loading = 0.0
    best_correlation = float("nan")
    best_gap = float("inf")
    for loading in np.linspace(float(LATENT_BOUNDS[0]), float(LATENT_BOUNDS[1]), int(grid)):
        k_carries = _carries_dispersion(q, mu_carries, carries, float(loading))
        k_targets = _targets_dispersion(mu_targets, targets, float(loading))
        simulated = _branch_correlation(
            q, mu_carries, mu_targets, k_carries, k_targets, float(loading), rng, draws
        )
        gap = abs(simulated - observed)
        if gap < best_gap:
            best_gap = gap
            best_loading = float(loading)
            best_correlation = simulated
    return best_loading, best_correlation, observed


@dataclass
class GainMixture:
    """Signed per-completion (or per-carry) gain law: a small Gaussian mixture.

    The mixture is fitted to *aggregated* observations: row ``i`` reports the sum
    of ``counts[i]`` i.i.d. gains, not a single gain.  Rows with one carry (or one
    catch) identify the spread, rows with many identify the level, and the EM
    below accounts for the ``counts`` scaling of both moments.  That matters for
    running backs: most carries arrive bundled in 10-20 carry games.
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
    calibration row, and ``successes`` the touchdowns among them.  The curve is
    a shape only -- its level is reset per row by the shrunk rate model -- so
    averaging the row's gains weakens the slope without biasing the rate.
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


@dataclass
class ShapeLayer:
    """The low-dimensional part of the RB chain: dispersions, gain laws, rates."""

    k_carries: float
    k_targets: float
    rho_catch: float
    latent_loading: float
    branch_correlation: float
    observed_branch_correlation: float
    global_ypcarry: float
    rush_gain_mixture: GainMixture
    rush_td_rate_global: float
    rush_td_edges: np.ndarray
    rush_td_rates: np.ndarray
    rush_td_shrinkage: float
    global_ypr: float
    catch_gain_mixture: GainMixture
    catch_td_rate_global: float
    catch_td_edges: np.ndarray
    catch_td_rates: np.ndarray
    catch_td_shrinkage: float

    @property
    def gain_mean(self) -> float:
        """Per-catch gain mean, the convention the WR and TE chains use."""
        return self.catch_gain_mixture.mean

    @property
    def gain_variance(self) -> float:
        return self.catch_gain_mixture.variance

    @property
    def rush_gain_mean(self) -> float:
        return self.rush_gain_mixture.mean

    @property
    def rush_gain_variance(self) -> float:
        return self.rush_gain_mixture.variance

    @property
    def td_rate_global(self) -> float:
        """Receiving touchdowns per reception on the calibration season."""
        return self.catch_td_rate_global

    def has_rush_td_curve(self) -> bool:
        return self.rush_td_edges.size >= 3 and self.rush_td_rates.size >= 2

    def has_catch_td_curve(self) -> bool:
        return self.catch_td_edges.size >= 3 and self.catch_td_rates.size >= 2

    @staticmethod
    def _expected_rate(
        mixture: GainMixture, edges: np.ndarray, rates: np.ndarray, mu_gain: np.ndarray
    ) -> np.ndarray:
        """E[P(touchdown | gain)] over the mixture, at each row's mean gain."""
        shift = np.asarray(mu_gain, dtype=float) - mixture.mean
        component_means = mixture.means[:, None] + shift[None, :]
        bins = np.clip(
            np.searchsorted(edges, component_means, side="right") - 1, 0, rates.size - 1
        )
        return (mixture.weights[:, None] * rates[bins]).sum(axis=0)

    def rush_td_scale(self, mu_ypcarry: np.ndarray, td_rate: np.ndarray) -> np.ndarray:
        """Per-row multiplier so E[P(touchdown | gain)] equals the row's rate."""
        expected = self._expected_rate(
            self.rush_gain_mixture,
            np.asarray(self.rush_td_edges, dtype=float),
            np.asarray(self.rush_td_rates, dtype=float),
            mu_ypcarry,
        )
        return np.clip(np.asarray(td_rate, dtype=float) / np.maximum(expected, 1e-4), *TD_SCALE_BOUNDS)

    def catch_td_scale(self, mu_ypr: np.ndarray, td_rate: np.ndarray) -> np.ndarray:
        expected = self._expected_rate(
            self.catch_gain_mixture,
            np.asarray(self.catch_td_edges, dtype=float),
            np.asarray(self.catch_td_rates, dtype=float),
            mu_ypr,
        )
        return np.clip(np.asarray(td_rate, dtype=float) / np.maximum(expected, 1e-4), *TD_SCALE_BOUNDS)

    def scalars(self) -> dict[str, float]:
        """Every fitted shape scalar, for the report."""
        return {
            "k_carries": float(self.k_carries),
            "k_targets": float(self.k_targets),
            "rho_catch": float(self.rho_catch),
            "latent_loading": float(self.latent_loading),
            "branch_correlation": float(self.branch_correlation),
            "observed_branch_correlation": float(self.observed_branch_correlation),
            "global_ypcarry": float(self.global_ypcarry),
            "rush_gain_mean": float(self.rush_gain_mean),
            "rush_gain_variance": float(self.rush_gain_variance),
            "rush_td_rate_global": float(self.rush_td_rate_global),
            "rush_td_shrinkage": float(self.rush_td_shrinkage),
            "global_ypr": float(self.global_ypr),
            "gain_mean": float(self.gain_mean),
            "gain_variance": float(self.gain_variance),
            "td_rate_global": float(self.td_rate_global),
            "catch_td_shrinkage": float(self.catch_td_shrinkage),
            "gain_sd_1": float(self.catch_gain_mixture.sds[0]),
            "gain_sd_2": float(self.catch_gain_mixture.sds[-1]),
            "rush_gain_sd_1": float(self.rush_gain_mixture.sds[0]),
            "rush_gain_sd_2": float(self.rush_gain_mixture.sds[-1]),
        }


def _fit_mean_models(
    x, stats: dict[str, np.ndarray], *, n_estimators: int, categorical
) -> dict[str, Any]:
    """Fit every conditional mean of the two branches on one design matrix."""
    with_carry = stats["carries"] > 0
    with_target = stats["targets"] > 0
    with_catch = stats["receptions"] > 0
    if not np.any(with_carry):
        raise ValueError("no carries in the training frame")
    if not np.any(with_target):
        raise ValueError("no targets in the training frame")
    if not np.any(with_catch):
        raise ValueError("no receptions in the training frame")
    trial_x, trial_y = _expand_trials(x, stats["targets"], stats["receptions"])
    if trial_y.size == 0:
        raise ValueError("no catch trials in the training frame")
    return {
        "carry_hurdle": _fit_regressor(
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
        "rush_td": _fit_regressor(
            x[with_carry],
            stats["rushing_tds"][with_carry] / stats["carries"][with_carry],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["carries"][with_carry],
        ),
        "targets": _fit_regressor(
            x, stats["targets"], _stage_params("poisson", n_estimators), categorical
        ),
        "catch": _fit_regressor(
            trial_x, trial_y, _stage_params("binary", n_estimators), categorical
        ),
        "ypcatch": _fit_regressor(
            x[with_catch],
            stats["receiving_yards"][with_catch] / stats["receptions"][with_catch],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["receptions"][with_catch],
        ),
        "catch_td": _fit_regressor(
            x[with_catch],
            stats["receiving_tds"][with_catch] / stats["receptions"][with_catch],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["receptions"][with_catch],
        ),
    }


def _rate_shrinkage_nll(
    strength: float, model_rate: np.ndarray, exposure: np.ndarray, successes: np.ndarray, prior: float
) -> float:
    """Binomial NLL of the shrunk per-opportunity rate on the calibration rows."""
    trials = np.maximum(np.rint(np.asarray(exposure, dtype=float)), 0.0)
    successes = np.clip(np.rint(np.asarray(successes, dtype=float)), 0.0, trials)
    shrunk = np.clip(
        _shrink(np.asarray(model_rate, dtype=float), trials, prior, strength), 1e-9, 1.0 - 1e-9
    )
    loglik = successes * np.log(shrunk) + (trials - successes) * np.log(1.0 - shrunk)
    total = -float(np.sum(loglik))
    return total if np.isfinite(total) else 1e18


def _estimate_shapes(models: dict[str, Any], x_cal, stats_cal: dict[str, np.ndarray]) -> ShapeLayer:
    """Estimate the shape layer on rows the mean models have not seen."""
    carries_cal = stats_cal["carries"]
    targets_cal = stats_cal["targets"]
    receptions_cal = stats_cal["receptions"]

    q = np.clip(models["carry_hurdle"].predict(x_cal).astype(float), 1e-6, 1.0 - 1e-6)
    mu_carries = np.maximum(models["carries"].predict(x_cal).astype(float), 1e-3)
    mu_targets = np.maximum(models["targets"].predict(x_cal).astype(float), 1e-3)

    # The coupling loading and the two dispersions are one joint estimate: each
    # candidate loading re-solves both dispersions by marginal-variance matching
    # before the implied branch correlation is measured.
    loading, branch_correlation, observed_correlation = _estimate_latent_loading(
        q,
        mu_carries,
        mu_targets,
        carries_cal,
        targets_cal,
        rng=np.random.default_rng(LATENT_SEED),
    )
    k_carries = _carries_dispersion(q, mu_carries, carries_cal, loading)
    k_targets = _targets_dispersion(mu_targets, targets_cal, loading)

    p_catch = np.clip(models["catch"].predict(x_cal).astype(float), 1e-4, 1.0 - 1e-4)
    ok = (targets_cal >= 1) & (receptions_cal <= targets_cal) & (receptions_cal >= 0)
    rho_catch = _mle_scalar(
        lambda rho: _beta_binomial_nll(rho, receptions_cal[ok], targets_cal[ok], p_catch[ok]),
        RHO_BOUNDS,
    )

    # --- rushing branch ---------------------------------------------------
    with_carry = carries_cal > 0
    model_ypcarry = models["ypcarry"].predict(x_cal).astype(float)
    rush_gain_mean, rush_gain_variance = _gain_moments(
        stats_cal["rushing_yards"][with_carry], carries_cal[with_carry], model_ypcarry[with_carry]
    )
    rush_gain_mixture = GainMixture.fit_aggregated(
        stats_cal["rushing_yards"][with_carry],
        carries_cal[with_carry],
        target_mean=rush_gain_mean,
        target_variance=rush_gain_variance,
    )
    global_ypcarry = float(
        stats_cal["rushing_yards"].sum() / max(float(carries_cal.sum()), 1.0)
    )
    rush_td_edges, rush_td_rates = _td_gain_curve(
        stats_cal["rushing_yards"][with_carry] / carries_cal[with_carry],
        carries_cal[with_carry],
        stats_cal["rushing_tds"][with_carry],
    )
    rush_td_rate_global = float(
        stats_cal["rushing_tds"].sum() / max(float(carries_cal.sum()), 1.0)
    )
    # The per-carry touchdown rate model is shrunk toward the calibration rate;
    # the strength is itself estimated by a binomial NLL on the calibration rows.
    expected_carries = np.maximum(q * mu_carries, 0.05)
    model_rush_td = np.maximum(models["rush_td"].predict(x_cal).astype(float), 0.0)
    rush_td_shrinkage = _mle_scalar(
        lambda strength: _rate_shrinkage_nll(
            strength, model_rush_td, expected_carries, stats_cal["rushing_tds"], rush_td_rate_global
        ),
        SHRINKAGE_BOUNDS,
    )

    # --- receiving branch -------------------------------------------------
    with_catch = receptions_cal > 0
    model_ypcatch = models["ypcatch"].predict(x_cal).astype(float)
    catch_gain_mean, catch_gain_variance = _gain_moments(
        stats_cal["receiving_yards"][with_catch],
        receptions_cal[with_catch],
        model_ypcatch[with_catch],
    )
    catch_gain_mixture = GainMixture.fit_aggregated(
        stats_cal["receiving_yards"][with_catch],
        receptions_cal[with_catch],
        target_mean=catch_gain_mean,
        target_variance=catch_gain_variance,
    )
    global_ypr = float(stats_cal["receiving_yards"].sum() / max(float(receptions_cal.sum()), 1.0))
    catch_td_edges, catch_td_rates = _td_gain_curve(
        stats_cal["receiving_yards"][with_catch] / receptions_cal[with_catch],
        receptions_cal[with_catch],
        stats_cal["receiving_tds"][with_catch],
    )
    catch_td_rate_global = float(
        stats_cal["receiving_tds"].sum() / max(float(receptions_cal.sum()), 1.0)
    )
    expected_receptions = np.maximum(mu_targets * p_catch, 0.05)
    model_catch_td = np.maximum(models["catch_td"].predict(x_cal).astype(float), 0.0)
    catch_td_shrinkage = _mle_scalar(
        lambda strength: _rate_shrinkage_nll(
            strength,
            model_catch_td,
            expected_receptions,
            stats_cal["receiving_tds"],
            catch_td_rate_global,
        ),
        SHRINKAGE_BOUNDS,
    )
    return ShapeLayer(
        k_carries=k_carries,
        k_targets=k_targets,
        rho_catch=rho_catch,
        latent_loading=float(loading),
        branch_correlation=branch_correlation,
        observed_branch_correlation=observed_correlation,
        global_ypcarry=global_ypcarry,
        rush_gain_mixture=rush_gain_mixture,
        rush_td_rate_global=rush_td_rate_global,
        rush_td_edges=rush_td_edges,
        rush_td_rates=rush_td_rates,
        rush_td_shrinkage=rush_td_shrinkage,
        global_ypr=global_ypr,
        catch_gain_mixture=catch_gain_mixture,
        catch_td_rate_global=catch_td_rate_global,
        catch_td_edges=catch_td_edges,
        catch_td_rates=catch_td_rates,
        catch_td_shrinkage=catch_td_shrinkage,
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
    enter the shape estimate -- including the branch-coupling loading.
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


def _compound_process(
    counts: np.ndarray,
    mu_gain: np.ndarray,
    mixture: GainMixture,
    edges: np.ndarray,
    rates: np.ndarray,
    scale: np.ndarray,
    fallback_rate: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-opportunity yards and touchdowns for one branch.

    Yards are the sum of ``counts`` signed gains from ``mixture``, centred on the
    row's conditional gain model.  Touchdowns are one Bernoulli per gain through
    the gain-conditional curve at the row's scaled level.  When the curve is
    absent (too few calibration gains to bin) touchdowns fall back to one
    Bernoulli per opportunity at the row's shrunk rate; both paths keep the
    ``touchdowns <= opportunities`` invariant and produce no yards without
    opportunities.
    """
    counts = np.asarray(counts, dtype=float)
    pattern = counts.shape
    yards = np.zeros(pattern, dtype=float)
    touchdowns = np.zeros(pattern, dtype=float)
    curved = edges.size >= 3 and rates.size >= 2
    active = counts > 0
    if bool(active.any()):
        row_of = np.broadcast_to(np.arange(pattern[1])[None, :], pattern)[active]
        sizes = counts[active].astype(int)
        shifts = (
            np.broadcast_to(np.asarray(mu_gain, dtype=float)[None, :], pattern)[active] - mixture.mean
        )
        gains, offsets = mixture.sample_gains(sizes, shifts, rng)
        if gains.size:
            yards[active] = np.add.reduceat(gains, offsets)
            if curved:
                bins = np.clip(np.searchsorted(edges, gains, side="right") - 1, 0, rates.size - 1)
                probability = np.clip(
                    rates[bins] * np.repeat(scale[row_of], sizes), 0.0, TD_MAX_PROBABILITY
                )
                flags = (rng.random(gains.size) < probability).astype(float)
                touchdowns[active] = np.add.reduceat(flags, offsets)
    if not curved:
        probability = np.broadcast_to(
            np.clip(np.asarray(fallback_rate, dtype=float)[None, :], 0.0, TD_MAX_PROBABILITY), pattern
        )
        touchdowns = np.rint(
            rng.binomial(counts.astype(np.int64), probability)
        ).astype(float)
    return yards, touchdowns


@dataclass
class RBChain:
    """Fit and sample the RB production chain on one training fold."""

    feature_columns: tuple[str, ...]
    n_estimators: int = 200
    calibration_fraction: float = CALIBRATION_FRACTION

    def fit(self, train: pl.DataFrame, *, frozen: ShapeLayer | None = None) -> "FittedRBChain":
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
            # Frozen path: the shape layer (and the coupling loading it carries)
            # comes from the prior season and only the conditional means are refit.
            shapes = frozen

        x_all, _, categorical_all = _design(train, train.head(0), columns)
        models = _fit_mean_models(
            x_all, stats_all, n_estimators=self.n_estimators, categorical=categorical_all
        )
        return FittedRBChain(
            columns=columns,
            reference=train.select(*columns),
            models=models,
            shapes=shapes,
        )


@dataclass
class FittedRBChain:
    """Fitted stage models plus the shape layer."""

    columns: tuple[str, ...]
    reference: pl.DataFrame
    models: dict[str, Any]
    shapes: ShapeLayer

    @property
    def k_carries(self) -> float:
        return self.shapes.k_carries

    @property
    def k_targets(self) -> float:
        return self.shapes.k_targets

    @property
    def rho_catch(self) -> float:
        return self.shapes.rho_catch

    @property
    def latent_loading(self) -> float:
        return self.shapes.latent_loading

    @property
    def branch_correlation(self) -> float:
        return self.shapes.branch_correlation

    @property
    def gain_mixture(self) -> GainMixture:
        return self.shapes.catch_gain_mixture

    @property
    def rush_gain_mixture(self) -> GainMixture:
        return self.shapes.rush_gain_mixture

    @property
    def td_rate_global(self) -> float:
        return self.shapes.catch_td_rate_global

    @property
    def rush_td_rate_global(self) -> float:
        return self.shapes.rush_td_rate_global

    def diagnostics(self) -> dict[str, float]:
        """The fitted shape-layer scalars, for the report."""
        return self.shapes.scalars()

    def sample(self, test: pl.DataFrame, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw coherent RB stat lines. Returns (n_test, n_draws, len(STAT_COLUMNS))."""
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1")
        _, x_test, _ = _design(self.reference, test, self.columns)
        n = x_test.shape[0]
        if n == 0:
            return np.zeros((0, n_draws, len(STAT_COLUMNS)))
        shapes = self.shapes

        q = np.clip(self.models["carry_hurdle"].predict(x_test).astype(float), 1e-6, 1.0 - 1e-6)
        mu_carries = np.maximum(self.models["carries"].predict(x_test).astype(float), 1e-3)
        mu_ypcarry = self.models["ypcarry"].predict(x_test).astype(float)
        rush_td_rate = np.maximum(self.models["rush_td"].predict(x_test).astype(float), 0.0)
        mu_targets = np.maximum(self.models["targets"].predict(x_test).astype(float), 1e-3)
        p_catch = np.clip(self.models["catch"].predict(x_test).astype(float), 1e-4, 1.0 - 1e-4)
        mu_ypcatch = self.models["ypcatch"].predict(x_test).astype(float)
        catch_td_rate = np.maximum(self.models["catch_td"].predict(x_test).astype(float), 0.0)

        # Empirical-Bayes shrink of both touchdown rates toward the calibration
        # level, with the row's expected exposure as the weight.
        expected_carries = np.maximum(q * mu_carries, 0.05)
        rush_td_rate = _shrink(
            rush_td_rate, expected_carries, shapes.rush_td_rate_global, shapes.rush_td_shrinkage
        )
        expected_receptions = np.maximum(mu_targets * p_catch, 0.05)
        catch_td_rate = _shrink(
            catch_td_rate, expected_receptions, shapes.catch_td_rate_global, shapes.catch_td_shrinkage
        )

        # --- the shared role/game-script factor -------------------------------
        # One variate per (row, draw) drives both volume means, so a "big role"
        # draw is big on the ground and through the air at the same time.
        volume = _volume_factor(shapes.latent_loading, rng.normal(size=(n_draws, n)))
        mean_carries = mu_carries[None, :] * volume
        mean_targets = mu_targets[None, :] * volume

        # --- carries: hurdle, then a >= 1 truncated NegBin --------------------
        carries = np.zeros((n_draws, n), dtype=float)
        participating = rng.random((n_draws, n)) < q[None, :]
        if bool(participating.any()):
            carries[participating] = _truncated_negbin_sample(
                mean_carries[participating], shapes.k_carries, (int(participating.sum()),), rng
            )

        # --- targets: NegBin on the same latent factor ------------------------
        targets = rng.negative_binomial(
            shapes.k_targets,
            shapes.k_targets / (shapes.k_targets + mean_targets),
            size=(n_draws, n),
        ).astype(float)

        # --- receptions | targets: BetaBinomial -------------------------------
        rho = shapes.rho_catch
        if rho <= RHO_BOUNDS[0] * 1.001:
            theta = np.broadcast_to(p_catch[None, :], (n_draws, n))
        else:
            a = np.maximum(p_catch * (1.0 - rho) / rho, 1e-6)
            b = np.maximum((1.0 - p_catch) * (1.0 - rho) / rho, 1e-6)
            theta = rng.beta(a[None, :], b[None, :], size=(n_draws, n))
        receptions = np.rint(
            rng.binomial(targets.astype(np.int64), np.clip(theta, 0.0, 1.0))
        ).astype(float)

        # --- per-opportunity processes ----------------------------------------
        scale_rush = (
            shapes.rush_td_scale(mu_ypcarry, rush_td_rate)
            if shapes.has_rush_td_curve()
            else np.ones(n, dtype=float)
        )
        rushing_yards, rushing_tds = _compound_process(
            carries,
            mu_ypcarry,
            shapes.rush_gain_mixture,
            np.asarray(shapes.rush_td_edges, dtype=float),
            np.asarray(shapes.rush_td_rates, dtype=float),
            scale_rush,
            rush_td_rate,
            rng,
        )
        scale_catch = (
            shapes.catch_td_scale(mu_ypcatch, catch_td_rate)
            if shapes.has_catch_td_curve()
            else np.ones(n, dtype=float)
        )
        receiving_yards, receiving_tds = _compound_process(
            receptions,
            mu_ypcatch,
            shapes.catch_gain_mixture,
            np.asarray(shapes.catch_td_edges, dtype=float),
            np.asarray(shapes.catch_td_rates, dtype=float),
            scale_catch,
            catch_td_rate,
            rng,
        )

        joint = np.stack(
            [
                carries,
                rushing_yards,
                rushing_tds,
                targets,
                receptions,
                receiving_yards,
                receiving_tds,
            ],
            axis=-1,
        )
        return np.transpose(joint, (1, 0, 2))


__all__ = [
    "STAT_COLUMNS",
    "STAT_INDEX",
    "RBChain",
    "FittedRBChain",
    "ShapeLayer",
    "GainMixture",
    "estimate_frozen_shapes",
    "rb_fantasy_points",
]
