"""Hierarchical parametric WR model.

Structure (the plan's Option 1 conditional hierarchy, extended to the rare
rushing component so the full-PPR scoring rule is reproduced):

    targets         ~ NegBin(mu_targets(X), k)
    receptions      ~ BetaBinomial(targets, p_catch(X), rho)
    receiving_yards = sum of ``receptions`` signed per-catch gains
                      (Gaussian gain mixture, mean matched to the conditional
                       yards-per-reception model)
    receiving_tds   depends on the per-catch process:
                      ``td_mode="separate"`` Poisson(receptions * lambda_td(X))
                      ``td_mode="joint"``    Bernoulli over the per-catch gain,
                                             P(touchdown | gain) fitted on the
                                             calibration season
    carries > 0     ~ Bernoulli(q(X)); when positive, (carries, rushing_yards,
                      rushing_tds) are drawn jointly from the training hurdle pool
    rushing_yards   = 0 and rushing_tds = 0 when carries == 0

Every mean is a LightGBM boosted model over the same feature matrix the two
forest candidates see.  The shape parameters (overdispersion, gain mixture,
touchdown curve) are estimated either on an internal 80/20 time split of the
training fold, or supplied frozen from a strictly prior season.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln

from .common import STAT_COLUMNS
from .features import LGBM_PARAMS, _design

RHO_BOUNDS = (1e-4, 0.60)
SHAPE_BOUNDS = (0.03, 800.0)
CALIBRATION_FRACTION = 0.20
GAIN_COMPONENTS = 2
MIN_GAIN_ROWS = 40
TD_SHRINKAGE_EXPOSURE = 5.0
TD_BINS = 6
TD_PRIOR_STRENGTH = 8.0
TD_SCALE_BOUNDS = (0.1, 10.0)
TD_MAX_PROBABILITY = 0.95
TD_MODES = ("separate", "joint")


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


@dataclass
class GainMixture:
    """Signed per-catch gain distribution: a small Gaussian mixture."""

    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.sum(self.weights * self.means))

    @classmethod
    def fit(cls, gains: np.ndarray) -> "GainMixture":
        gains = np.asarray(gains, dtype=float)
        gains = gains[np.isfinite(gains)]
        if gains.size < MIN_GAIN_ROWS or float(np.std(gains)) < 1e-6:
            pooled = float(gains.mean()) if gains.size else 0.0
            spread = float(gains.std()) if gains.size > 1 else 10.0
            return cls(np.array([1.0]), np.array([pooled]), np.array([max(spread, 1e-3)]))
        try:
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(
                n_components=GAIN_COMPONENTS, random_state=0, n_init=2, reg_covar=1e-2
            ).fit(gains[:, None])
            order = np.argsort(model.means_[:, 0])
            return cls(
                weights=np.asarray(model.weights_)[order],
                means=np.asarray(model.means_[order, 0]),
                sds=np.sqrt(np.asarray(model.covariances_[order, 0, 0])),
            )
        except Exception:  # pragma: no cover - defensive fallback
            return cls(np.array([1.0]), np.array([float(gains.mean())]), np.array([float(gains.std())]))

    def sample_gains(
        self, sizes: np.ndarray, shifts: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat i.i.d. per-catch gains plus the reduceat offsets for each group."""
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


def _td_gain_curve(gains: np.ndarray, touchdowns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone P(touchdown | per-catch gain) on quantile bins, Laplace smoothed."""
    gains = np.asarray(gains, dtype=float)
    touchdowns = np.asarray(touchdowns, dtype=float)
    if gains.size < MIN_GAIN_ROWS:
        return np.zeros(0), np.zeros(0)
    edges = np.unique(np.quantile(gains, np.linspace(0.0, 1.0, TD_BINS + 1)))
    if edges.size < 3:
        return np.zeros(0), np.zeros(0)
    global_rate = float(touchdowns.mean())
    bins = np.clip(np.searchsorted(edges, gains, side="right") - 1, 0, edges.size - 2)
    rates = np.empty(edges.size - 1, dtype=float)
    for index in range(edges.size - 1):
        mask = bins == index
        count = int(mask.sum())
        successes = float(touchdowns[mask].sum())
        rates[index] = (successes + TD_PRIOR_STRENGTH * global_rate) / (count + TD_PRIOR_STRENGTH)
    # Longer catches should not score less often: enforce a monotone shape.
    rates = np.maximum.accumulate(rates)
    return edges, rates


@dataclass
class ShapeParams:
    """The low-dimensional part of the chain: dispersion, gain law, touchdown law."""

    k_targets: float
    rho_catch: float
    gain_mixture: GainMixture
    global_ypr: float
    td_rate_global: float
    td_mode: str = "separate"
    td_edges: np.ndarray | None = None
    td_rates: np.ndarray | None = None

    def has_td_curve(self) -> bool:
        return (
            self.td_mode == "joint"
            and self.td_edges is not None
            and self.td_rates is not None
            and self.td_edges.size >= 3
        )

    def td_scale(self, mu_ypr: np.ndarray, td_rate: np.ndarray) -> np.ndarray:
        """Per-row multiplier so E[P(touchdown | gain)] equals the row's rate."""
        edges = np.asarray(self.td_edges, dtype=float)
        rates = np.asarray(self.td_rates, dtype=float)
        shift = np.asarray(mu_ypr, dtype=float) - self.gain_mixture.mean
        component_means = self.gain_mixture.means[:, None] + shift[None, :]
        bins = np.clip(np.searchsorted(edges, component_means, side="right") - 1, 0, rates.size - 1)
        expected = (self.gain_mixture.weights[:, None] * rates[bins]).sum(axis=0)
        scale = np.asarray(td_rate, dtype=float) / np.maximum(expected, 1e-4)
        return np.clip(scale, *TD_SCALE_BOUNDS)


def _fit_mean_models(x, stats: dict[str, np.ndarray], *, n_estimators: int, categorical) -> dict[str, Any]:
    """Fit every conditional mean on one design matrix."""
    rows_with_catch = stats["receptions"] > 0
    catch_x, catch_y = _expand_trials(x, stats["targets"], stats["receptions"])
    if catch_y.size == 0:
        raise ValueError("no catch trials in the training frame")
    return {
        "targets": _fit_regressor(
            x, stats["targets"], _stage_params("poisson", n_estimators), categorical
        ),
        "catch": _fit_regressor(
            catch_x, catch_y, _stage_params("binary", n_estimators), categorical
        ),
        "ypr": _fit_regressor(
            x[rows_with_catch],
            stats["receiving_yards"][rows_with_catch] / stats["receptions"][rows_with_catch],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["receptions"][rows_with_catch],
        ),
        "td": _fit_regressor(
            x[rows_with_catch],
            stats["receiving_tds"][rows_with_catch] / stats["receptions"][rows_with_catch],
            _stage_params("regression_l2", n_estimators),
            categorical,
            weights=stats["receptions"][rows_with_catch],
        ),
        "rush": _fit_regressor(
            x,
            (stats["carries"] > 0).astype(float),
            _stage_params("binary", n_estimators),
            categorical,
        ),
    }


def _estimate_shapes(
    models: dict[str, Any],
    x_cal,
    stats_cal: dict[str, np.ndarray],
    *,
    td_mode: str,
) -> ShapeParams:
    """Estimate the shape layer on rows the mean models have not seen."""
    mu_targets = np.maximum(models["targets"].predict(x_cal).astype(float), 1e-3)
    k_targets = _mle_scalar(lambda k: _negbin_nll(k, stats_cal["targets"], mu_targets), SHAPE_BOUNDS)

    p_catch = np.clip(models["catch"].predict(x_cal).astype(float), 1e-4, 1 - 1e-4)
    trials = stats_cal["targets"]
    successes = stats_cal["receptions"]
    ok = (trials >= 1) & (successes <= trials)
    rho_catch = _mle_scalar(
        lambda rho: _beta_binomial_nll(rho, successes[ok], trials[ok], p_catch[ok]), RHO_BOUNDS
    )

    single = stats_cal["receptions"] == 1
    gain_mixture = GainMixture.fit(stats_cal["receiving_yards"][single])
    with_catch = stats_cal["receptions"] > 0
    global_ypr = float(
        stats_cal["receiving_yards"][with_catch].sum() / max(stats_cal["receptions"][with_catch].sum(), 1.0)
    )
    td_rate_global = float(
        stats_cal["receiving_tds"].sum() / max(stats_cal["receptions"].sum(), 1.0)
    )
    edges = np.zeros(0)
    rates = np.zeros(0)
    if td_mode == "joint":
        edges, rates = _td_gain_curve(
            stats_cal["receiving_yards"][single], stats_cal["receiving_tds"][single]
        )
    return ShapeParams(
        k_targets=k_targets,
        rho_catch=rho_catch,
        gain_mixture=gain_mixture,
        global_ypr=global_ypr,
        td_rate_global=td_rate_global,
        td_mode=td_mode,
        td_edges=edges,
        td_rates=rates,
    )


def estimate_frozen_shapes(
    frame: pl.DataFrame,
    season: int,
    feature_columns: tuple[str, ...],
    *,
    n_estimators: int = 200,
    td_mode: str = "separate",
) -> ShapeParams:
    """Shape parameters from the season immediately before the holdout.

    Means are fitted on all earlier seasons and predicted on the calibration
    season, so nothing from the holdout season enters the estimate.
    """
    cal = frame.filter(pl.col("season") == season - 1)
    history = frame.filter(pl.col("season") <= season - 2)
    if cal.is_empty():
        raise ValueError(f"no calibration rows in season {season - 1}")
    if history.is_empty():
        raise ValueError(f"no history before season {season - 1}")
    columns = tuple(name for name in feature_columns if name in frame.columns)
    x_hist, x_cal, categorical = _design(history, cal, columns)
    models = _fit_mean_models(
        x_hist, _stats(history), n_estimators=n_estimators, categorical=categorical
    )
    return _estimate_shapes(models, x_cal, _stats(cal), td_mode=td_mode)


@dataclass
class HierarchicalParametric:
    """Fit and sample the WR production chain on one training fold."""

    feature_columns: tuple[str, ...]
    n_estimators: int = 200
    calibration_fraction: float = CALIBRATION_FRACTION
    td_mode: str = "separate"

    def fit(self, train: pl.DataFrame, *, frozen: ShapeParams | None = None) -> "FittedHierarchy":
        if self.td_mode not in TD_MODES:
            raise ValueError(f"td_mode must be one of {TD_MODES}")
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
            fit_models = _fit_mean_models(
                x_fit, _stats(train[fit_index.tolist()]), n_estimators=self.n_estimators, categorical=categorical
            )
            shapes = _estimate_shapes(
                fit_models,
                x_cal,
                _stats(train[cal_index.tolist()]),
                td_mode=self.td_mode,
            )
        else:
            shapes = frozen

        x_all, _, categorical_all = _design(train, train.head(0), columns)
        models = _fit_mean_models(
            x_all, stats_all, n_estimators=self.n_estimators, categorical=categorical_all
        )
        rush_mask = stats_all["carries"] > 0
        rush_pool = np.column_stack(
            [
                stats_all["carries"][rush_mask],
                stats_all["rushing_yards"][rush_mask],
                stats_all["rushing_tds"][rush_mask],
            ]
        ).astype(float)
        return FittedHierarchy(
            columns=columns,
            reference=train.select(*columns),
            models=models,
            shapes=shapes,
            rush_pool=rush_pool,
        )


@dataclass
class FittedHierarchy:
    """Fitted stage models plus the shape layer."""

    columns: tuple[str, ...]
    reference: pl.DataFrame
    models: dict[str, Any]
    shapes: ShapeParams
    rush_pool: np.ndarray

    @property
    def k_targets(self) -> float:
        return self.shapes.k_targets

    @property
    def rho_catch(self) -> float:
        return self.shapes.rho_catch

    @property
    def gain_mixture(self) -> GainMixture:
        return self.shapes.gain_mixture

    @property
    def global_ypr(self) -> float:
        return self.shapes.global_ypr

    @property
    def global_td_rate(self) -> float:
        return self.shapes.td_rate_global

    def diagnostics(self) -> dict[str, float]:
        """Shape-layer scalars for the cross-position report."""
        shapes = self.shapes
        out = {
            "k_targets": shapes.k_targets,
            "rho_catch": shapes.rho_catch,
            "global_td_rate": shapes.td_rate_global,
            "global_ypr": shapes.global_ypr,
            "gain_mean": shapes.gain_mixture.mean,
            "gain_components": float(len(shapes.gain_mixture.weights)),
        }
        if shapes.td_rates is not None and len(shapes.td_rates):
            out["td_curve_top"] = float(np.max(shapes.td_rates))
            out["td_curve_bottom"] = float(np.min(shapes.td_rates))
        return out

    def sample(self, test: pl.DataFrame, *, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """Draw coherent stat lines. Returns (n_test, n_draws, len(STAT_COLUMNS))."""
        if n_draws < 1:
            raise ValueError("n_draws must be >= 1")
        _, x_test, _ = _design(self.reference, test, self.columns)
        n = x_test.shape[0]
        if n == 0:
            return np.zeros((0, n_draws, len(STAT_COLUMNS)))

        mu_targets = np.maximum(self.models["targets"].predict(x_test).astype(float), 1e-3)
        p = np.clip(self.models["catch"].predict(x_test).astype(float), 1e-4, 1 - 1e-4)
        mu_ypr = np.maximum(self.models["ypr"].predict(x_test).astype(float), 0.0)
        td_rate = np.maximum(self.models["td"].predict(x_test).astype(float), 0.0)
        # Empirical-Bayes shrink of the per-reception touchdown rate toward the
        # calibration rate, with expected receptions as the exposure.
        exposure = np.maximum(mu_targets * p, 0.25)
        td_rate = (td_rate * exposure + TD_SHRINKAGE_EXPOSURE * self.shapes.td_rate_global) / (
            exposure + TD_SHRINKAGE_EXPOSURE
        )
        q_rush = np.clip(self.models["rush"].predict(x_test).astype(float), 1e-6, 1 - 1e-6)

        # --- targets ---------------------------------------------------------
        k = self.shapes.k_targets
        targets = rng.negative_binomial(k, k / (k + mu_targets)[None, :], size=(n_draws, n)).astype(float)

        # --- receptions | targets --------------------------------------------
        rho = self.shapes.rho_catch
        if rho <= RHO_BOUNDS[0] * 1.001:
            theta = np.broadcast_to(p[None, :], (n_draws, n))
        else:
            a = np.maximum(p * (1.0 - rho) / rho, 1e-6)
            b = np.maximum((1.0 - p) * (1.0 - rho) / rho, 1e-6)
            theta = rng.beta(a[None, :], b[None, :], size=(n_draws, n))
        receptions = np.rint(
            rng.binomial(targets.astype(np.int64), np.clip(theta, 0.0, 1.0))
        ).astype(float)

        # --- per-catch process: yards (and touchdowns in joint mode) ----------
        yards = np.zeros((n_draws, n), dtype=float)
        tds = np.zeros((n_draws, n), dtype=float)
        active = receptions > 0
        if np.any(active):
            row_of = np.broadcast_to(np.arange(n)[None, :], receptions.shape)[active]
            sizes = receptions[active].astype(int)
            shifts = np.broadcast_to(mu_ypr[None, :], receptions.shape)[active] - self.shapes.gain_mixture.mean
            gains, offsets = self.shapes.gain_mixture.sample_gains(sizes, shifts, rng)
            if gains.size:
                yards[active] = np.add.reduceat(gains, offsets)
                if self.shapes.has_td_curve():
                    edges = np.asarray(self.shapes.td_edges, dtype=float)
                    rates = np.asarray(self.shapes.td_rates, dtype=float)
                    bins = np.clip(
                        np.searchsorted(edges, gains, side="right") - 1, 0, rates.size - 1
                    )
                    scale = self.shapes.td_scale(mu_ypr, td_rate)[row_of]
                    probability = np.clip(rates[bins] * np.repeat(scale, sizes), 0.0, TD_MAX_PROBABILITY)
                    flags = (rng.random(gains.size) < probability).astype(float)
                    tds[active] = np.add.reduceat(flags, offsets)
        if not self.shapes.has_td_curve():
            tds = rng.poisson(receptions * td_rate[None, :]).astype(float)

        # --- rushing hurdle ---------------------------------------------------
        carries = np.zeros((n_draws, n), dtype=float)
        rush_yards = np.zeros((n_draws, n), dtype=float)
        rush_tds = np.zeros((n_draws, n), dtype=float)
        participating = rng.random((n_draws, n)) < q_rush[None, :]
        if np.any(participating):
            pool = self.rush_pool if self.rush_pool.shape[0] else np.zeros((1, 3))
            picks = rng.integers(0, pool.shape[0], size=int(participating.sum()))
            carries[participating] = pool[picks, 0]
            rush_yards[participating] = pool[picks, 1]
            rush_tds[participating] = pool[picks, 2]

        joint = np.stack([targets, receptions, yards, tds, carries, rush_yards, rush_tds], axis=-1)
        return np.transpose(joint, (1, 0, 2))
