"""Hierarchical parametric DST (team defense and special teams) chain.

The DST stat line is a set of low-count defensive events plus a points-allowed
total that is scored through a coarse, nonlinear tier table.  Unlike the
scrimmage chains there is no volume stage: every one of the eight columns is a
non-negative count on a team-week, and five of them are almost always zero.
The chain therefore models each column explicitly and freezes the quantities
that a single training fold cannot estimate:

    def_sacks         ~ NegBin(mu_sacks(X), k_sacks)
    def_interceptions ~ NegBin(mu_interceptions(X), k_interceptions)
    def_fumbles       ~ hurdle-NegBin.  A FROZEN zero mass ``P(def_fumbles = 0)``
                        plus a ``>= 1`` truncated NegBin whose positive-branch
                        mean ``mu_fumbles(X) / P(def_fumbles > 0)`` restores the
                        unconditional conditional mean model.
    def_tds           = a FROZEN per-takeaway rate applied to the DRAWN
                        takeaways ``def_interceptions + def_fumbles``.  The zero
                        mass is therefore exposure-driven: a defence that records
                        no takeaway cannot score a defensive touchdown.  No
                        separate ``def_tds`` mean model is fitted, because the
                        calibration season holds too few positives (mean ~ 0.07).
    def_safeties      ~ hurdle-NegBin with a frozen zero mass.
    blocked_kicks     ~ hurdle-NegBin with a frozen zero mass.
    special_teams_tds ~ fully frozen hurdle: a frozen zero mass plus a frozen
                        positive count.  Nothing in the panel observes return
                        opportunities, so a fitted mean would only memorise noise.
    points_allowed    ~ NegBin(mu_points_allowed(X), k_points_allowed), mapped
                        through the points-allowed tier table AFTER sampling.

Every count column has its OWN dispersion, estimated on the calibration season
(the season immediately before the holdout when the shape layer is frozen):

* the two frequent columns (``def_sacks``, ``def_interceptions``) and
  ``points_allowed`` use marginal-variance matching, which removes the double
  count of the mean-model error that a plug-in MLE would charge to the
  dispersion twice;
* the rare hurdle columns use an ordinary MLE on the POSITIVE rows only, because
  the positive branch is what the dispersion describes there.

Why the tier stage is separate
------------------------------
The points-allowed scoring table maps a *sampled* ``points_allowed`` through::

    0 -> 10,  1-6 -> 7,  7-13 -> 4,  14-20 -> 1,
    21-27 -> 0,  28-34 -> -1,  35+ -> -4

The map is nonlinear and discontinuous (step width 7 points per tier), so a
model with a perfect conditional mean for ``points_allowed`` can still put the
wrong mass in each tier and therefore get the fantasy points wrong.  The tier
map is applied after sampling and the model's own tier histogram is recorded by
``diagnostics`` (``tier_mass_*``) so it can be compared, week by week, with the
observed tier histogram.  ``tier_histogram`` is the shared helper for both.

Structural invariants, enforced by construction:

* all eight columns are non-negative;
* all eight columns are integer-valued;
* ``def_tds`` never exceeds the drawn takeaway count ``def_interceptions +
  def_fumbles`` (a defensive score needs a takeaway);
* ``points_allowed`` is drawn as a count and then mapped, so the tier and the
  sampled total are always coherent.

Scoring rule (see ``hierarchical_football.scoring``)::

    1 * def_sacks + 2 * (def_interceptions + def_fumbles)
    + 6 * (def_tds + special_teams_tds)
    + 2 * (def_safeties + blocked_kicks)
    + points_allowed_tier(points_allowed)
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
    "def_sacks",
    "def_interceptions",
    "def_fumbles",
    "def_tds",
    "def_safeties",
    "blocked_kicks",
    "special_teams_tds",
    "points_allowed",
)
STAT_INDEX: dict[str, int] = {name: index for index, name in enumerate(STAT_COLUMNS)}

# The seven count columns, in the order the sampler draws them.
COUNT_COLUMNS: tuple[str, ...] = (
    "def_sacks",
    "def_interceptions",
    "def_fumbles",
    "def_tds",
    "def_safeties",
    "blocked_kicks",
    "special_teams_tds",
)
# Columns with their own LightGBM conditional-mean model.
FITTED_MEAN_COLUMNS: tuple[str, ...] = (
    "def_sacks",
    "def_interceptions",
    "def_fumbles",
    "def_safeties",
    "blocked_kicks",
    "points_allowed",
)
# Columns that add an explicit, frozen zero mass in front of the count law.
HURDLE_COLUMNS: tuple[str, ...] = (
    "def_fumbles",
    "def_safeties",
    "blocked_kicks",
    "special_teams_tds",
)
# The defensive score is a frozen per-takeaway rate over the drawn takeaways.
EXPOSURE_COLUMN: str = "def_tds"
EXPOSURE_SOURCES: tuple[str, ...] = ("def_interceptions", "def_fumbles")
# Columns whose whole law is frozen (no conditional mean is fitted).
FROZEN_COLUMNS: tuple[str, ...] = ("special_teams_tds",)
POINTS_ALLOWED_COLUMN: str = "points_allowed"

POINTS_PER_SACK = 1.0
POINTS_PER_TAKEAWAY = 2.0
POINTS_PER_TOUCHDOWN = 6.0
POINTS_PER_SAFETY = 2.0
POINTS_PER_BLOCKED_KICK = 2.0

TIER_LABELS: tuple[str, ...] = ("0", "1-6", "7-13", "14-20", "21-27", "28-34", "35+")
TIER_EDGES: tuple[float, ...] = (0.0, 6.0, 13.0, 20.0, 27.0, 34.0)
TIER_POINTS: tuple[float, ...] = (10.0, 7.0, 4.0, 1.0, 0.0, -1.0, -4.0)

SHAPE_BOUNDS = (0.03, 800.0)
CALIBRATION_FRACTION = 0.20
MIN_CALIBRATION_ROWS = 20
ZERO_MASS_BOUNDS = (1e-6, 1.0 - 1e-6)
TRUNCATION_ITERATIONS = 20
TRUNCATION_ROUNDS = 40


def dst_fantasy_points(stats: np.ndarray) -> np.ndarray:
    """Apply the DST scoring rule to a raw stat sample.

    ``stats`` is any array whose last axis follows ``STAT_COLUMNS``; the shape is
    preserved.  This is the module-level scoring function the runner uses to
    verify that the sampled stat lines reproduce the panel's ``fantasy_points``
    column.
    """
    stats = np.asarray(stats, dtype=float)
    return (
        POINTS_PER_SACK * stats[..., STAT_INDEX["def_sacks"]]
        + POINTS_PER_TAKEAWAY
        * (stats[..., STAT_INDEX["def_interceptions"]] + stats[..., STAT_INDEX["def_fumbles"]])
        + POINTS_PER_TOUCHDOWN
        * (stats[..., STAT_INDEX["def_tds"]] + stats[..., STAT_INDEX["special_teams_tds"]])
        + POINTS_PER_SAFETY * stats[..., STAT_INDEX["def_safeties"]]
        + POINTS_PER_BLOCKED_KICK * stats[..., STAT_INDEX["blocked_kicks"]]
        + points_allowed_tier(stats[..., STAT_INDEX["points_allowed"]])
    )


def points_allowed_tier(points_allowed: np.ndarray) -> np.ndarray:
    """Map a points-allowed total through the seven-step tier table.

    ``0 -> 10``, ``1-6 -> 7``, ``7-13 -> 4``, ``14-20 -> 1``, ``21-27 -> 0``,
    ``28-34 -> -1`` and ``35+ -> -4``.  The shape of the input is preserved.
    """
    points_allowed = np.asarray(points_allowed, dtype=float)
    return np.select(
        [points_allowed <= 0, points_allowed <= 6, points_allowed <= 13,
         points_allowed <= 20, points_allowed <= 27, points_allowed <= 34],
        list(TIER_POINTS[:6]),
        default=TIER_POINTS[6],
    )


def tier_index(points_allowed: np.ndarray) -> np.ndarray:
    """Tier index in ``0..6`` (the position of the label in ``TIER_LABELS``)."""
    points_allowed = np.asarray(points_allowed, dtype=float)
    return np.select(
        [points_allowed <= 0, points_allowed <= 6, points_allowed <= 13,
         points_allowed <= 20, points_allowed <= 27, points_allowed <= 34],
        [0, 1, 2, 3, 4, 5],
        default=6,
    ).astype(int)


def tier_histogram(points_allowed: np.ndarray, *, normalize: bool = False) -> dict[str, float]:
    """Counts (or probabilities) per points-allowed tier, keyed by ``TIER_LABELS``.

    The seven tiers are always returned, in order, so a model histogram and an
    observed histogram can be compared entry by entry.
    """
    index = tier_index(np.asarray(points_allowed, dtype=float)).ravel()
    counts = np.bincount(index, minlength=len(TIER_LABELS)).astype(float)
    if normalize:
        total = float(counts.sum())
        counts = counts / total if total > 0 else counts
    return {label: float(counts[i]) for i, label in enumerate(TIER_LABELS)}


def tier_comparison(model_points_allowed: np.ndarray, observed_points_allowed: np.ndarray) -> dict[str, Any]:
    """Compare the model's tier histogram with the observed one, entry by entry.

    Returns both normalised histograms over ``TIER_LABELS``, their difference,
    and the total-variation distance ``0.5 * sum |model - observed|``.  The
    points-allowed tier map is nonlinear, so a correct conditional mean is not
    enough: the tier mass is what the fantasy-point distribution actually reads.
    """
    model = tier_histogram(model_points_allowed, normalize=True)
    observed = tier_histogram(observed_points_allowed, normalize=True)
    difference = {label: float(model[label] - observed[label]) for label in TIER_LABELS}
    total_variation = float(0.5 * sum(abs(value) for value in difference.values()))
    return {
        "model": model,
        "observed": observed,
        "difference": difference,
        "total_variation": total_variation,
        "matches": bool(total_variation <= 0.05),
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


def _negbin_dispersion(mu: np.ndarray, y: np.ndarray) -> float:
    """Dispersion of a mean-parametrised negative binomial from the marginal variance.

    A plug-in MLE on the model's own conditional means charges the mean-model error
    to the dispersion twice: once in the calibration residual and again when the
    sampler re-uses the same conditional mean for every draw.  Matching the observed
    marginal variance to ``Var(mu) + E[mu] + E[mu**2] / k`` removes that double
    count.  When the mean spread plus Poisson noise already reaches the observed
    variance the estimator returns the Poisson limit.
    """
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return SHAPE_BOUNDS[1]
    within = float(np.var(y)) - float(np.var(mu)) - float(np.mean(mu))
    if not np.isfinite(within) or within <= 1e-6:
        return SHAPE_BOUNDS[1]
    return float(np.clip(float(np.mean(mu**2)) / within, *SHAPE_BOUNDS))


def _positive_dispersion(mu_positive: np.ndarray, y_positive: np.ndarray) -> float:
    """Dispersion of the positive branch of a hurdle, by MLE on the positive rows."""
    y_positive = np.asarray(y_positive, dtype=float)
    mu_positive = np.maximum(np.asarray(mu_positive, dtype=float), 1e-6)
    if y_positive.size < 2:
        return SHAPE_BOUNDS[1]
    return _mle_scalar(lambda k: _negbin_nll(k, y_positive, mu_positive), SHAPE_BOUNDS)


def _implied_zero_mass(mu: np.ndarray, k: float) -> np.ndarray:
    """``P(Y = 0)`` for a mean-parametrised NegBin at each row's mean."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    k = max(float(k), 1e-6)
    return np.power(k / (k + mu), k)


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
class CountLaw:
    """One column's count law: how its mean is set, plus the frozen shape pieces.

    ``mean_source`` is one of

    * ``"model"``    -- the LightGBM Poisson mean ``mu(X)`` fitted for the column;
    * ``"exposure"`` -- a frozen per-opportunity ``rate`` times a drawn exposure
                        count (only ``def_tds``, over the drawn takeaways);
    * ``"frozen"``   -- a frozen positive count ``positive_mean`` with a frozen
                        zero mass (only ``special_teams_tds``).

    ``hurdle`` adds the explicit frozen zero mass ``zero_mass`` in front of the
    count law.  ``k`` is the column's own negative-binomial dispersion.
    """

    name: str
    mean_source: str
    hurdle: bool
    k: float
    zero_mass: float | None = None
    rate: float | None = None
    positive_mean: float | None = None


@dataclass
class ShapeLayer:
    """The low-dimensional part of the DST chain: dispersions and frozen masses."""

    laws: dict[str, CountLaw]
    k_points_allowed: float
    p_zero: dict[str, float]

    def law(self, name: str) -> CountLaw:
        return self.laws[name]

    def scalars(self) -> dict[str, float]:
        """Every fitted shape scalar, for the report."""
        out: dict[str, float] = {"k_points_allowed": float(self.k_points_allowed)}
        for name in COUNT_COLUMNS:
            law = self.laws[name]
            out[f"k_{name}"] = float(law.k)
            if law.zero_mass is not None:
                out[f"zero_mass_{name}"] = float(law.zero_mass)
            if law.rate is not None:
                out[f"rate_{name}"] = float(law.rate)
            if law.positive_mean is not None:
                out[f"positive_mean_{name}"] = float(law.positive_mean)
        out["td_rate_per_takeaway"] = float(self.laws[EXPOSURE_COLUMN].rate)
        for name, value in self.p_zero.items():
            out[f"p_zero_{name}"] = float(value)
        return out


def _fit_mean_models(
    x, stats: dict[str, np.ndarray], *, n_estimators: int, categorical
) -> dict[str, Any]:
    """Fit one Poisson conditional mean per column that has its own mean model."""
    return {
        name: _fit_regressor(
            x, stats[name], _stage_params("poisson", n_estimators), categorical
        )
        for name in FITTED_MEAN_COLUMNS
    }


def _clip_zero_mass(value: float) -> float:
    return float(np.clip(float(value), *ZERO_MASS_BOUNDS))


def _estimate_shapes(
    models: dict[str, Any], x_cal, stats_cal: dict[str, np.ndarray]
) -> ShapeLayer:
    """Estimate the shape layer on rows the mean models have not seen."""
    mu = {
        name: np.maximum(models[name].predict(x_cal).astype(float), 1e-6)
        for name in FITTED_MEAN_COLUMNS
    }
    laws: dict[str, CountLaw] = {}

    # --- frequent counts: a plain negative binomial per column -----------------
    for name in ("def_sacks", "def_interceptions"):
        laws[name] = CountLaw(
            name=name,
            mean_source="model",
            hurdle=False,
            k=_negbin_dispersion(mu[name], stats_cal[name]),
        )

    # --- rare counts with a fitted mean and a FROZEN zero mass -----------------
    for name in ("def_fumbles", "def_safeties", "blocked_kicks"):
        y = np.asarray(stats_cal[name], dtype=float)
        zero_mass = _clip_zero_mass(float(np.mean(y == 0.0)))
        positive = y > 0.0
        mu_positive = mu[name] / (1.0 - zero_mass)
        k = (
            _positive_dispersion(mu_positive[positive], y[positive])
            if bool(positive.any())
            else SHAPE_BOUNDS[1]
        )
        laws[name] = CountLaw(
            name=name,
            mean_source="model",
            hurdle=True,
            k=k,
            zero_mass=zero_mass,
        )

    # --- defensive touchdowns: a FROZEN per-takeaway rate over drawn takeaways --
    takeaways = np.asarray(stats_cal["def_interceptions"], dtype=float) + np.asarray(
        stats_cal["def_fumbles"], dtype=float
    )
    rate = float(
        np.clip(float(np.sum(stats_cal[EXPOSURE_COLUMN])) / max(float(takeaways.sum()), 1.0), 0.0, 1.0)
    )
    mu_td_cal = rate * takeaways
    laws[EXPOSURE_COLUMN] = CountLaw(
        name=EXPOSURE_COLUMN,
        mean_source="exposure",
        hurdle=False,
        k=_negbin_dispersion(mu_td_cal, stats_cal[EXPOSURE_COLUMN]),
        rate=rate,
    )

    # --- special-teams touchdowns: fully frozen hurdle --------------------------
    y = np.asarray(stats_cal["special_teams_tds"], dtype=float)
    zero_mass = _clip_zero_mass(float(np.mean(y == 0.0)))
    positive = y > 0.0
    positive_mean = float(y[positive].mean()) if bool(positive.any()) else 1.0
    k_st = (
        _positive_dispersion(np.full(int(positive.sum()), positive_mean), y[positive])
        if bool(positive.any())
        else SHAPE_BOUNDS[1]
    )
    laws["special_teams_tds"] = CountLaw(
        name="special_teams_tds",
        mean_source="frozen",
        hurdle=True,
        k=k_st,
        zero_mass=zero_mass,
        positive_mean=positive_mean,
    )

    # --- points allowed: its own dispersion, the tier map is applied later ------
    k_points_allowed = _negbin_dispersion(mu[POINTS_ALLOWED_COLUMN], stats_cal[POINTS_ALLOWED_COLUMN])

    # --- model-implied zero masses for the report -------------------------------
    mu_td_model = rate * (mu["def_interceptions"] + mu["def_fumbles"])
    p_zero = {
        "def_sacks": float(np.mean(_implied_zero_mass(mu["def_sacks"], laws["def_sacks"].k))),
        "def_interceptions": float(
            np.mean(_implied_zero_mass(mu["def_interceptions"], laws["def_interceptions"].k))
        ),
        "def_fumbles": float(laws["def_fumbles"].zero_mass),
        "def_tds": float(np.mean(_implied_zero_mass(mu_td_model, laws[EXPOSURE_COLUMN].k))),
        "def_safeties": float(laws["def_safeties"].zero_mass),
        "blocked_kicks": float(laws["blocked_kicks"].zero_mass),
        "special_teams_tds": float(laws["special_teams_tds"].zero_mass),
        "points_allowed": float(
            np.mean(_implied_zero_mass(mu[POINTS_ALLOWED_COLUMN], k_points_allowed))
        ),
    }
    return ShapeLayer(laws=laws, k_points_allowed=k_points_allowed, p_zero=p_zero)


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
class DefenseChain:
    """Fit and sample the DST production chain on one training fold."""

    feature_columns: tuple[str, ...]
    n_estimators: int = 200
    calibration_fraction: float = CALIBRATION_FRACTION

    def fit(self, train: pl.DataFrame, *, frozen: ShapeLayer | None = None) -> "FittedDefenseChain":
        columns = tuple(name for name in self.feature_columns if name in train.columns)
        if not columns:
            raise ValueError("no usable feature columns")
        stats_all = _stats(train)

        if frozen is None:
            order = _time_order(train)
            n_cal = max(int(round(self.calibration_fraction * len(order))), 120)
            n_cal = min(n_cal, len(order) - 200) if len(order) > 400 else max(len(order) // 5, 1)
            if n_cal < MIN_CALIBRATION_ROWS:
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
        return FittedDefenseChain(
            columns=columns,
            reference=train.select(*columns),
            models=models,
            shapes=shapes,
        )


@dataclass
class FittedDefenseChain:
    """Fitted stage models plus the shape layer."""

    columns: tuple[str, ...]
    reference: pl.DataFrame
    models: dict[str, Any]
    shapes: ShapeLayer
    tier_mass: dict[str, float] | None = None

    @property
    def k_points_allowed(self) -> float:
        return self.shapes.k_points_allowed

    def law(self, name: str) -> CountLaw:
        return self.shapes.law(name)

    def diagnostics(self) -> dict[str, float]:
        """The fitted shape scalars plus, once ``sample`` has run, the tier mass."""
        out = dict(self.shapes.scalars())
        for label, value in (self.tier_mass or {}).items():
            out[f"tier_mass_{label}"] = float(value)
        return out

    @staticmethod
    def _negbin(mu: np.ndarray, k: float, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        """Mean-parametrised negative binomial draws on ``shape``."""
        mu = np.broadcast_to(np.asarray(mu, dtype=float), shape)
        out = np.zeros(shape, dtype=float)
        positive = mu > 1e-12
        if bool(positive.any()):
            out[positive] = rng.negative_binomial(
                k, k / (k + mu[positive]), size=int(positive.sum())
            ).astype(float)
        return out

    @staticmethod
    def _hurdle(
        mu: np.ndarray,
        k: float,
        zero_mass: np.ndarray | float,
        shape: tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Frozen-zero-mass hurdle with a ``>= 1`` truncated NegBin positive branch."""
        mu = np.broadcast_to(np.asarray(mu, dtype=float), shape)
        zero_mass = np.broadcast_to(np.asarray(zero_mass, dtype=float), (shape[1],))
        positive_mass = np.maximum(1.0 - zero_mass, 1e-6)
        mu_positive = mu / positive_mass
        out = np.zeros(shape, dtype=float)
        nonzero = rng.random(shape) >= zero_mass[None, :]
        if bool(nonzero.any()):
            out[nonzero] = _truncated_negbin_sample(
                mu_positive[nonzero], k, (int(nonzero.sum()),), rng
            )
        return out

    def sample(self, test: pl.DataFrame, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw coherent DST stat lines. Returns (n_test, n_draws, len(STAT_COLUMNS))."""
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1")
        _, x_test, _ = _design(self.reference, test, self.columns)
        n = x_test.shape[0]
        if n == 0:
            return np.zeros((0, n_draws, len(STAT_COLUMNS)))
        shapes = self.shapes
        shape = (n_draws, n)
        mu = {
            name: np.maximum(self.models[name].predict(x_test).astype(float), 1e-6)
            for name in FITTED_MEAN_COLUMNS
        }

        draws: dict[str, np.ndarray] = {}
        # --- frequent counts: a plain negative binomial per column -------------
        for name in ("def_sacks", "def_interceptions"):
            law = shapes.law(name)
            draws[name] = self._negbin(mu[name], law.k, shape, rng)

        # --- rare counts: frozen zero mass, truncated-NegBin positive branch ---
        for name in ("def_fumbles", "def_safeties", "blocked_kicks"):
            law = shapes.law(name)
            draws[name] = self._hurdle(mu[name], law.k, np.float64(law.zero_mass), shape, rng)

        # --- defensive touchdowns: frozen per-takeaway rate on DRAWN takeaways -
        law = shapes.law(EXPOSURE_COLUMN)
        exposure = draws["def_interceptions"] + draws["def_fumbles"]
        draws[EXPOSURE_COLUMN] = np.minimum(
            self._negbin(law.rate * exposure, law.k, shape, rng), exposure
        )

        # --- special-teams touchdowns: fully frozen hurdle ----------------------
        law = shapes.law("special_teams_tds")
        mu_st = np.full(n, max(1.0 - float(law.zero_mass), 1e-6) * float(law.positive_mean))
        draws["special_teams_tds"] = self._hurdle(mu_st, law.k, np.float64(law.zero_mass), shape, rng)

        # --- points allowed: a count, mapped through the tier table afterwards --
        draws[POINTS_ALLOWED_COLUMN] = self._negbin(
            mu[POINTS_ALLOWED_COLUMN], shapes.k_points_allowed, shape, rng
        )

        joint = np.stack([draws[name] for name in STAT_COLUMNS], axis=-1)
        joint = np.rint(np.maximum(joint, 0.0))
        # The model's own tier histogram, so ``diagnostics`` can expose it and the
        # caller can compare it with the observed holdout tier histogram.
        self.tier_mass = tier_histogram(draws[POINTS_ALLOWED_COLUMN], normalize=True)
        return np.transpose(joint, (1, 0, 2))


__all__ = [
    "STAT_COLUMNS",
    "STAT_INDEX",
    "COUNT_COLUMNS",
    "FITTED_MEAN_COLUMNS",
    "HURDLE_COLUMNS",
    "EXPOSURE_COLUMN",
    "TIER_LABELS",
    "TIER_EDGES",
    "TIER_POINTS",
    "CountLaw",
    "ShapeLayer",
    "DefenseChain",
    "FittedDefenseChain",
    "dst_fantasy_points",
    "points_allowed_tier",
    "tier_index",
    "tier_histogram",
    "tier_comparison",
    "estimate_frozen_shapes",
]
