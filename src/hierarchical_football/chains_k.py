"""Hierarchical parametric kicker chain.

The plan's "frozen categorical" case: unlike the skill-position chains, the
kicker's per-band make rates and its distance-band mix are stable across
seasons, so they are estimated once on the calibration season and then held
fixed.  Only the volumes are conditional on the feature matrix.

Structure::

    fg_att         = hurdle: Bernoulli(p_attempt(X)) for "did he attempt a
                     field goal at all", then a >= 1 truncated
                     NegBin(mu_fg_att(X), k_fg_att) for the attempts given at
                     least one
    band           = each attempt's distance band (under 40, 40-49, 50-59, 60+)
                     is drawn from a FROZEN categorical distribution, the
                     calibration season's attempt shares (see `_band_counters`,
                     which reads the per-band make and miss counters of the
                     panel)
    fg_made_*      = Binomial(attempts in that band, frozen per-band make rate)
    fg_made        = sum of the four bands by construction
    pat_att        = NegBin(mu_pat_att(X), k_pat_att)
    pat_made       = Binomial(pat_att, frozen pat rate)

The scoring rule (see `hierarchical_football.scoring`) needs
"field goals under 40", which is not a panel column: it is the remainder
``fg_made - fg_made_40_49 - fg_made_50_59 - fg_made_60_``.  Deriving the under-40
band as a binomial on the under-40 band -- instead of differencing the sampled
totals -- makes ``fg_made = sum of the four bands`` true draw by draw, so the
remainder can never go negative.  ``verify_panel_scoring`` checks the same
scoring rule against the panel's own ``fantasy_points`` column before anything
else is trusted.

Structural invariants, enforced by construction:

* ``0 <= fg_att`` and every band count is an integer, so
  ``0 <= fg_made <= fg_att``;
* ``fg_made == fg_made_40_49 + fg_made_50_59 + fg_made_60_ + under-40``, hence
  the derived under-40 remainder is non-negative;
* each band's makes never exceed that band's attempts;
* ``0 <= pat_made <= pat_att``;
* both counts are zero whenever the corresponding attempt count is zero.


Why the hurdle is not optional: the panel's marginal ``fg_att`` is *under*-
dispersed (mean 1.94, variance 1.63 over 2019-2024, and 2.23 against 1.31 on the
positive rows alone), and a negative binomial cannot go below Poisson.  A plain
NegBin therefore over-predicts the zero mass and cannot match the count
distribution: on the 2023 calibration season it put ``P(fg_att = 0)`` at 0.162
against an observed 0.127, and scored -1.41 nats per row against -1.29 for the
hurdle, whose zero mass is exact by construction.  ``KickerChain(hurdle=False)``
keeps the plain-NegBin comparison available, and ``attempt_loglik`` plus the
tests measure both.

Two limits of the frozen layer, measured on the same panel.  The 60+ band sees
only single-digit attempts in a calibration season (5 to 14 per season over
2019-2024), so its frozen rate is noisy (0.14 to 0.63); the band carries 6
points per make but only ~1% of the attempts, so its contribution to the mean
stays under 0.05 points.  And because the volumes are under-dispersed, both
fitted dispersions land on the Poisson limit, which leaves the sampled
``fg_att`` slightly wider than the observed counts (variance 2.04 against 1.70
on the 2023 calibration season): the negative-binomial family cannot go below
Poisson, so nothing inside this family is tighter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .features import LGBM_PARAMS, _design

STAT_COLUMNS: tuple[str, ...] = (
    "fg_att",
    "fg_made",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "pat_att",
    "pat_made",
)
STAT_INDEX: dict[str, int] = {name: index for index, name in enumerate(STAT_COLUMNS)}

# The four distance bands, in scoring order.  The panel only carries the makes
# per band; the attempts per band come from the make plus miss counters.
BANDS: tuple[str, ...] = ("under_40", "40_49", "50_59", "60_")
BAND_COLUMNS: dict[str, tuple[str, ...]] = {
    "under_40": ("0_19", "20_29", "30_39"),
    "40_49": ("40_49",),
    "50_59": ("50_59",),
    "60_": ("60_",),
}
POINTS_PER_BAND: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0)
POINTS_PER_PAT = 1.0

# Pooled league values over 2019-2024.  They are used only where the calibration
# season cannot identify a quantity at all, i.e. where a band saw no attempt.
DEFAULT_BAND_MIX: tuple[float, ...] = (0.514, 0.291, 0.187, 0.009)
DEFAULT_BAND_RATES: tuple[float, ...] = (0.962, 0.795, 0.702, 0.404)

SHAPE_BOUNDS = (0.03, 800.0)
CALIBRATION_FRACTION = 0.20
TRUNCATION_ITERATIONS = 20
TRUNCATION_ROUNDS = 40
MIN_PROBABILITY = 1e-6


def kicker_fantasy_points(stats: np.ndarray, *, pat_points: float = POINTS_PER_PAT) -> np.ndarray:
    """Apply the kicker scoring rule to a raw stat sample.

    ``stats`` is any array whose last axis follows ``STAT_COLUMNS``; the shape is
    preserved.  The under-40 band is the remainder of ``fg_made`` after the three
    named bands, exactly as ``hierarchical_football.scoring.kicking`` and
    the panel itself compute it.
    """
    stats = np.asarray(stats, dtype=float)
    made = stats[..., STAT_INDEX["fg_made"]]
    forty = stats[..., STAT_INDEX["fg_made_40_49"]]
    fifty = stats[..., STAT_INDEX["fg_made_50_59"]]
    sixty = stats[..., STAT_INDEX["fg_made_60_"]]
    under_40 = made - forty - fifty - sixty
    return (
        POINTS_PER_BAND[0] * under_40
        + POINTS_PER_BAND[1] * forty
        + POINTS_PER_BAND[2] * fifty
        + POINTS_PER_BAND[3] * sixty
        + float(pat_points) * stats[..., STAT_INDEX["pat_made"]]
    )


def verify_panel_scoring(frame: pl.DataFrame, *, atol: float = 1e-6) -> dict[str, Any]:
    """Check ``kicker_fantasy_points`` against the panel's ``fantasy_points`` column.

    The panel is the source of truth: the comparison is only meaningful if the
    stat vector scores back to the stored fantasy points for every row.
    """
    missing = [name for name in STAT_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"panel is missing stat columns: {missing}")
    if "fantasy_points" not in frame.columns:
        raise ValueError("panel is missing the fantasy_points column")
    stats = frame.select(list(STAT_COLUMNS)).to_numpy().astype(float)
    reconstructed = np.asarray(kicker_fantasy_points(stats), dtype=float)
    stored = frame["fantasy_points"].to_numpy().astype(float)
    error = np.abs(reconstructed - stored)
    under_40 = stats[:, STAT_INDEX["fg_made"]] - stats[:, STAT_INDEX["fg_made_40_49"]] - stats[
        :, STAT_INDEX["fg_made_50_59"]
    ] - stats[:, STAT_INDEX["fg_made_60_"]]
    return {
        "rows": frame.height,
        "max_abs_error": float(np.max(error)) if error.size else 0.0,
        "mismatched_rows": int(np.sum(error > atol)),
        "min_under_40": float(np.min(under_40)) if under_40.size else 0.0,
        "mean_fantasy_points": float(stored.mean()) if stored.size else 0.0,
    }


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


def _negbin_zero_probability(mu: np.ndarray, k: float) -> np.ndarray:
    """P(Y = 0) of the mean/dispersion negative binomial."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    k = float(max(k, 1e-6))
    return np.power(k / (k + mu), k)


def _negbin_dispersion(mu: np.ndarray, y: np.ndarray) -> float:
    """Dispersion of a mean-parametrised negative binomial from the marginal variance.

    A plug-in MLE on the model's own conditional means charges the mean-model
    error to the dispersion twice: once in the calibration residual and again
    when the sampler re-uses the same conditional mean for every draw.  Matching
    the observed marginal variance to ``Var(mu) + E[mu] + E[mu**2] / k`` removes
    that double count, and returns the Poisson limit when the mean spread plus
    Poisson noise already reaches the observed variance.
    """
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return SHAPE_BOUNDS[1]
    within = float(np.var(y)) - float(np.var(mu)) - float(np.mean(mu))
    if not np.isfinite(within) or within <= 1e-6:
        return SHAPE_BOUNDS[1]
    return float(np.clip(float(np.mean(mu**2)) / within, *SHAPE_BOUNDS))


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


def _column_sum(frame: pl.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame[column].cast(pl.Float64, strict=False).fill_null(0.0).sum())


def _band_counters(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-band field-goal attempts and makes over one whole frame.

    Attempts come from the make plus miss counters of the panel (blocked kicks
    are not distance-classified, so they stay out of both).  When those counters
    are absent the frame cannot identify a per-band rate, so the makes are
    rescaled with the pooled league rates and the estimate degrades to them.
    """
    made = np.array(
        [
            sum(_column_sum(frame, f"fg_made_{part}") for part in BAND_COLUMNS[band])
            for band in BANDS
        ],
        dtype=float,
    )
    counters = [f"fg_missed_{part}" for parts in BAND_COLUMNS.values() for part in parts]
    if all(name in frame.columns for name in counters):
        missed = np.array(
            [
                sum(_column_sum(frame, f"fg_missed_{part}") for part in BAND_COLUMNS[band])
                for band in BANDS
            ],
            dtype=float,
        )
        return made + missed, made
    return made / np.asarray(DEFAULT_BAND_RATES, dtype=float), made


def _band_mix(attempts: np.ndarray) -> np.ndarray:
    """Frozen categorical probabilities from per-band attempt counts."""
    attempts = np.maximum(np.asarray(attempts, dtype=float), 0.0)
    total = float(attempts.sum())
    if total <= 0:
        return np.asarray(DEFAULT_BAND_MIX, dtype=float)
    probabilities = attempts / total
    empty = probabilities <= 0.0
    if bool(empty.any()):
        probabilities = np.where(empty, np.asarray(DEFAULT_BAND_MIX, dtype=float), probabilities)
    return probabilities / probabilities.sum()


def _band_rates(attempts: np.ndarray, made: np.ndarray) -> np.ndarray:
    """Frozen per-band make rates from the calibration season's own counters."""
    attempts = np.maximum(np.asarray(attempts, dtype=float), 0.0)
    made = np.minimum(np.maximum(np.asarray(made, dtype=float), 0.0), attempts)
    rates = np.where(attempts > 0.0, made / np.maximum(attempts, 1.0), DEFAULT_BAND_RATES)
    return np.clip(rates, 0.0, 1.0)


def _truncation_mean(mu_positive: np.ndarray, k: float) -> np.ndarray:
    """Untruncated NB mean whose ``>= 1`` truncation has mean ``mu_positive``.

    For Y ~ NegBin(k, mu) the truncated mean is ``mu / (1 - P(Y = 0))`` with
    ``P(Y = 0) = (k / (k + mu)) ** k``, so the fixed point below restores the
    conditional mean the hurdle stage was fitted on.
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


def _sample_bands(attempts: np.ndarray, probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Split each row's attempts across the frozen distance bands.

    A categorical draw per attempt is reproduced by a chain of binomials on the
    remaining attempts, which keeps every band count an integer and their sum
    equal to the drawn attempt count.
    """
    attempts = np.asarray(attempts, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    out = np.zeros(attempts.shape + (len(BANDS),), dtype=np.int64)
    remaining = attempts.copy()
    cumulative = 0.0
    for index in range(len(BANDS) - 1):
        share = float(probabilities[index]) / max(1.0 - cumulative, 1e-9)
        drawn = rng.binomial(remaining, float(np.clip(share, 0.0, 1.0)))
        out[..., index] = drawn
        remaining = remaining - drawn
        cumulative += float(probabilities[index])
    out[..., len(BANDS) - 1] = remaining
    return out


@dataclass
class ShapeLayer:
    """The frozen part of the kicker chain: band mix, band rates, pat rate, shapes.

    ``p_attempt`` is the calibration mean of the hurdle probability, for the
    report; the row-level probability always comes from the fitted model.
    """

    band_probabilities: np.ndarray
    band_rates: np.ndarray
    pat_rate: float
    p_attempt: float
    k_fg_att: float
    k_pat_att: float
    hurdle: bool = True

    def band_probability(self, band: str) -> float:
        return float(self.band_probabilities[BANDS.index(band)])

    def band_rate(self, band: str) -> float:
        return float(self.band_rates[BANDS.index(band)])

    def scalars(self) -> dict[str, float]:
        """Every fitted shape scalar as a plain float, for the cross-position report."""
        out: dict[str, float] = {
            "band_probabilities_sum": float(np.sum(self.band_probabilities)),
            "p_attempt": float(self.p_attempt),
            "pat_rate": float(self.pat_rate),
            "k_fg_att": float(self.k_fg_att),
            "k_pat_att": float(self.k_pat_att),
            "hurdle": float(self.hurdle),
        }
        for index, band in enumerate(BANDS):
            label = band.rstrip("_")
            out[f"band_prob_{label}"] = float(self.band_probabilities[index])
            out[f"band_rate_{label}"] = float(self.band_rates[index])
        out["pat_rate"] = float(self.pat_rate)
        return out


def _fit_mean_models(
    x, stats: dict[str, np.ndarray], *, n_estimators: int, categorical, hurdle: bool
) -> dict[str, Any]:
    """Fit the conditional volumes of the chain on one design matrix."""
    positive = stats["fg_att"] > 0
    if not np.any(positive):
        raise ValueError("no field-goal attempts in the training frame")
    models: dict[str, Any] = {
        "pat_att": _fit_regressor(
            x, stats["pat_att"], _stage_params("poisson", n_estimators), categorical
        )
    }
    if hurdle:
        models["attempt"] = _fit_regressor(
            x, positive.astype(float), _stage_params("binary", n_estimators), categorical
        )
        models["fg_att"] = _fit_regressor(
            x[positive],
            stats["fg_att"][positive],
            _stage_params("poisson", n_estimators),
            categorical,
        )
    else:
        models["fg_att"] = _fit_regressor(
            x, stats["fg_att"], _stage_params("poisson", n_estimators), categorical
        )
    return models


def _estimate_shapes(
    models: dict[str, Any],
    x_cal,
    stats_cal: dict[str, np.ndarray],
    calibration: pl.DataFrame,
    *,
    hurdle: bool,
) -> ShapeLayer:
    """Estimate the frozen layer on rows the mean models have not seen."""
    attempts, made = _band_counters(calibration)
    band_probabilities = _band_mix(attempts)
    band_rates = _band_rates(attempts, made)

    mu_fg_att = np.maximum(models["fg_att"].predict(x_cal).astype(float), 1e-3)
    if hurdle:
        p_attempt = np.clip(models["attempt"].predict(x_cal).astype(float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY)
        p_attempt_mean = float(np.mean(p_attempt))
        # The count law only describes the rows that did attempt, and the mean
        # model was fitted on strictly earlier seasons, so the ordinary MLE on
        # the calibration rows does not double-charge the mean error here.
        positive = stats_cal["fg_att"] > 0
        if not np.any(positive):
            raise ValueError("no field-goal attempts in the calibration frame")
        k_fg_att = _mle_scalar(
            lambda k: _negbin_nll(k, stats_cal["fg_att"][positive], mu_fg_att[positive]), SHAPE_BOUNDS
        )
    else:
        # Plain negative binomial: the zero mass is whatever the count law
        # implies, and the dispersion estimate follows the marginal variance.
        k_fg_att = _negbin_dispersion(mu_fg_att, stats_cal["fg_att"])
        p_attempt_mean = float(np.mean(1.0 - _negbin_zero_probability(mu_fg_att, k_fg_att)))

    mu_pat_att = np.maximum(models["pat_att"].predict(x_cal).astype(float), 1e-3)
    k_pat_att = _negbin_dispersion(mu_pat_att, stats_cal["pat_att"])

    pat_attempts = float(stats_cal["pat_att"].sum())
    pat_rate = (
        float(np.clip(stats_cal["pat_made"].sum() / pat_attempts, 0.0, 1.0))
        if pat_attempts > 0
        else 1.0
    )
    return ShapeLayer(
        band_probabilities=band_probabilities,
        band_rates=band_rates,
        pat_rate=pat_rate,
        p_attempt=p_attempt_mean,
        k_fg_att=k_fg_att,
        k_pat_att=k_pat_att,
        hurdle=hurdle,
    )


def estimate_frozen_shapes(
    frame: pl.DataFrame,
    season: int,
    feature_columns: tuple[str, ...],
    *,
    n_estimators: int = 200,
    hurdle: bool = True,
) -> ShapeLayer:
    """Shape parameters from the season immediately before the holdout.

    Means are fitted on ``frame[season <= season - 2]`` and predicted on the
    calibration season ``season - 1``, whose counters also fix the band mix and
    the per-band make rates, so nothing from ``season`` or later can enter the
    shape estimate.
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
        x_history,
        _stats(history),
        n_estimators=n_estimators,
        categorical=categorical,
        hurdle=hurdle,
    )
    return _estimate_shapes(
        models, x_calendar, _stats(calendar), calendar, hurdle=hurdle
    )


@dataclass
class KickerChain:
    """Fit and sample the kicker production chain on one training fold."""

    feature_columns: tuple[str, ...]
    n_estimators: int = 200
    calibration_fraction: float = CALIBRATION_FRACTION
    hurdle: bool = True

    def fit(self, train: pl.DataFrame, *, frozen: ShapeLayer | None = None) -> "FittedKickerChain":
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
                hurdle=self.hurdle,
            )
            shapes = _estimate_shapes(
                calibration_models,
                x_cal,
                _stats(train[cal_index.tolist()]),
                train[cal_index.tolist()],
                hurdle=self.hurdle,
            )
        else:
            # Frozen path: only the conditional volumes are re-estimated, and the
            # frozen layer keeps the hurdle choice it was estimated with.
            shapes = frozen

        x_all, _, categorical_all = _design(train, train.head(0), columns)
        models = _fit_mean_models(
            x_all,
            stats_all,
            n_estimators=self.n_estimators,
            categorical=categorical_all,
            hurdle=shapes.hurdle,
        )
        return FittedKickerChain(
            columns=columns,
            reference=train.select(*columns),
            models=models,
            shapes=shapes,
        )


@dataclass
class FittedKickerChain:
    """Fitted stage models plus the frozen shape layer."""

    columns: tuple[str, ...]
    reference: pl.DataFrame
    models: dict[str, Any]
    shapes: ShapeLayer

    @property
    def band_probabilities(self) -> np.ndarray:
        return self.shapes.band_probabilities

    @property
    def band_rates(self) -> np.ndarray:
        return self.shapes.band_rates

    @property
    def pat_rate(self) -> float:
        return self.shapes.pat_rate

    @property
    def k_pat_att(self) -> float:
        return self.shapes.k_pat_att

    def diagnostics(self) -> dict[str, Any]:
        """The frozen layer: band mix, band rates, pat rate, and the fitted shapes.

        ``band_probabilities`` and ``band_rates`` are the four-vectors; every
        other entry is a plain float, which is what the cross-position runner
        keeps (it casts each value and drops what does not cast).
        """
        out: dict[str, Any] = dict(self.shapes.scalars())
        out["band_probabilities"] = [float(value) for value in self.shapes.band_probabilities]
        out["band_rates"] = [float(value) for value in self.shapes.band_rates]
        return out

    def attempt_volumes(self, test: pl.DataFrame) -> dict[str, np.ndarray]:
        """Row-level conditional volumes, before any draw (for validation and tests)."""
        _, x_test, _ = _design(self.reference, test, self.columns)
        out = {
            "mu_fg_att": np.maximum(self.models["fg_att"].predict(x_test).astype(float), 1e-3),
            "mu_pat_att": np.maximum(self.models["pat_att"].predict(x_test).astype(float), 1e-3),
        }
        if "attempt" in self.models:
            out["p_attempt"] = np.clip(
                self.models["attempt"].predict(x_test).astype(float), MIN_PROBABILITY, 1.0 - MIN_PROBABILITY
            )
        else:
            out["p_attempt"] = 1.0 - _negbin_zero_probability(out["mu_fg_att"], self.shapes.k_fg_att)
        return out

    def zero_attempt_probability(self, test: pl.DataFrame) -> np.ndarray:
        """P(fg_att = 0) per row under whichever attempt law was fitted."""
        volumes = self.attempt_volumes(test)
        if "attempt" in self.models:
            return 1.0 - volumes["p_attempt"]
        return _negbin_zero_probability(volumes["mu_fg_att"], self.shapes.k_fg_att)

    def attempt_loglik(self, test: pl.DataFrame) -> np.ndarray:
        """Per-row log-likelihood of the observed ``fg_att`` under the attempt law.

        This is the evidence behind the module docstring: on the same rows, the
        hurdle beats the plain negative binomial because the panel's attempts are
        under-dispersed and only the hurdle can set the zero mass directly.
        """
        if "fg_att" not in test.columns:
            raise ValueError("the frame must carry the observed fg_att column")
        observed = np.rint(test["fg_att"].to_numpy().astype(float))
        volumes = self.attempt_volumes(test)
        k = self.shapes.k_fg_att
        mu = _truncation_mean(volumes["mu_fg_att"], k) if self.shapes.hurdle else volumes["mu_fg_att"]
        probability = k / (k + mu)
        loglik = (
            gammaln(observed + k)
            - gammaln(k)
            - gammaln(observed + 1.0)
            + k * np.log(probability)
            + observed * np.log(np.maximum(1.0 - probability, 1e-12))
        )
        if not self.shapes.hurdle:
            return loglik
        q = np.clip(volumes["p_attempt"], 1e-12, 1.0 - 1e-12)
        positive = observed > 0
        return np.where(positive, np.log(q) + loglik, np.log(np.maximum(1.0 - q, 1e-12)))

    def sample(self, test: pl.DataFrame, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw coherent kicker stat lines. Returns (n_test, n_draws, len(STAT_COLUMNS))."""
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1")
        _, x_test, _ = _design(self.reference, test, self.columns)
        n = x_test.shape[0]
        if n == 0:
            return np.zeros((0, n_draws, len(STAT_COLUMNS)))
        shapes = self.shapes

        mu_fg_att = np.maximum(self.models["fg_att"].predict(x_test).astype(float), 1e-3)
        mu_pat_att = np.maximum(self.models["pat_att"].predict(x_test).astype(float), 1e-3)

        # --- field-goal attempts ---------------------------------------------
        fg_att = np.zeros((n_draws, n), dtype=float)
        if "attempt" in self.models:
            p_attempt = np.clip(
                self.models["attempt"].predict(x_test).astype(float),
                MIN_PROBABILITY,
                1.0 - MIN_PROBABILITY,
            )
            participating = rng.random((n_draws, n)) < p_attempt[None, :]
            if bool(participating.any()):
                exposures = np.broadcast_to(mu_fg_att[None, :], (n_draws, n))[participating]
                fg_att[participating] = _truncated_negbin_sample(
                    exposures, shapes.k_fg_att, (int(participating.sum()),), rng
                )
        else:
            k = shapes.k_fg_att
            fg_att = rng.negative_binomial(
                k, k / (k + mu_fg_att)[None, :], size=(n_draws, n)
            ).astype(float)

        # --- band mix, then makes per band ------------------------------------
        bands = _sample_bands(fg_att, shapes.band_probabilities, rng)
        makes = np.zeros(bands.shape, dtype=np.int64)
        for index in range(len(BANDS)):
            makes[..., index] = rng.binomial(bands[..., index], float(shapes.band_rates[index]))
        fg_made = makes.sum(axis=-1).astype(float)

        # --- extra points ------------------------------------------------------
        k_pat = shapes.k_pat_att
        pat_att = rng.negative_binomial(
            k_pat, k_pat / (k_pat + mu_pat_att)[None, :], size=(n_draws, n)
        ).astype(float)
        pat_made = rng.binomial(pat_att.astype(np.int64), float(shapes.pat_rate)).astype(float)

        joint = np.stack(
            [
                fg_att,
                fg_made,
                makes[..., 1].astype(float),
                makes[..., 2].astype(float),
                makes[..., 3].astype(float),
                pat_att,
                pat_made,
            ],
            axis=-1,
        )
        return np.transpose(joint, (1, 0, 2))


__all__ = [
    "STAT_COLUMNS",
    "STAT_INDEX",
    "BANDS",
    "BAND_COLUMNS",
    "ShapeLayer",
    "KickerChain",
    "FittedKickerChain",
    "estimate_frozen_shapes",
    "kicker_fantasy_points",
    "verify_panel_scoring",
]
