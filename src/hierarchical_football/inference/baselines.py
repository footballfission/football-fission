"""Regularized baseline estimators for the canonical inference tables.

Each table gets a nested pair of baselines: pre-play context alone, then context plus identified
players. Both are fitted the same way on the same rows, because the difference between them is the
only thing that can be read as evidence about identity; an unrelated pair of models would not be.

A player term here is a conditional, regularized association. The design is one-hot under an L2
penalty, which centres the terms of one identity block on zero, so a term reads as a deviation from
the average identified player, and thin evidence pulls it back toward that average instead of
handing it a rate of its own. A player who only ever appears inside one level of the support
environment cannot be separated from that environment at all. `identity_terms` reports every term
with its exposure and its nesting, and it is sorted by name rather than by size, because an order
by size would look like a ranking of players.

Nothing in this module splits a table, samples plays, or picks a winning model. The caller
declares the chronological split and reports the comparison.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss

from .contracts import (
    CONTRACTS,
    PASSING_CONTRACT,
    RUSHING_CONTRACT,
    UNKNOWN,
    column_labels,
    label,
    validate,
)

PASSING_CONTINUOUS = ("air_yards", "yardline_100", "ydstogo")
PASSING_CATEGORICAL = ("down", "pass_location")
PASSING_CONTEXT_COMPONENTS = {
    "linear": (), "depth": ("depth",), "position": ("position",), "opponent": ("opponent",),
    "depth_position": ("depth", "position"), "depth_opponent": ("depth", "opponent"),
    "position_opponent": ("position", "opponent"), "expanded": ("depth", "position", "opponent"),
}
PASSING_CONTEXTS = tuple(PASSING_CONTEXT_COMPONENTS)
DEPTH_KNOTS = (0.0, 5.0, 10.0, 20.0, 30.0)
RUSHING_CONTINUOUS = ("yardline_100", "ydstogo")
RUSHING_CATEGORICAL = ("down", "run_location", "run_gap")
MISSING_SUFFIX = "_missing"
SUPPORT_ENVIRONMENT = ("team", "run_location", "run_gap")
DEFAULT_PENALTY = 1.0
# A loose tolerance stops the logistic solver along the flat direction between an identity block
# and the unpenalized intercept, which leaves every level term offset by an arbitrary constant.
DEFAULT_TOLERANCE = 1e-8
# Weak penalties on several hundred correlated columns need more iterations than the solver's own
# default allows: on the 2022 passing table an identity design at penalty 0.03 converges in 2021
# iterations. A fit that needs the extra work gets it, and the report says how many it took.
DEFAULT_MAX_ITER = 5000


def _levels(found_labels):
    """Declared levels of one categorical column: sorted, with UNKNOWN kept last."""
    levels = sorted(set(found_labels) - {UNKNOWN})
    if UNKNOWN in found_labels:
        levels.append(UNKNOWN)
    return tuple(levels)


def _numeric(frame, column):
    return frame[column].cast(pl.Float64, strict=False).to_numpy().astype(float)


def _composite_labels(frame, columns):
    parts = [column_labels(frame, column) for column in columns]
    return ["|".join(part) for part in zip(*parts)]


def _one_hot(labels, levels, *, name):
    """Zero-one block over fixed levels, plus the count of labels no level covers."""
    position = {level: index for index, level in enumerate(levels)}
    block = np.zeros((len(labels), len(levels)), dtype=float)
    rows, columns, unseen = [], [], 0
    for row, label in enumerate(labels):
        index = position.get(label)
        if index is None:
            unseen += 1
        else:
            rows.append(row)
            columns.append(index)
    if rows:
        block[np.asarray(rows), np.asarray(columns)] = 1.0
    return block, [f"{name}={level}" for level in levels], unseen


def _nesting(labels_by_identity, context_labels):
    """Distinct context levels each identity level appears with."""
    counted = {}
    for column, labels in labels_by_identity.items():
        seen = {}
        for player, context in zip(labels, context_labels):
            seen.setdefault(player, set()).add(context)
        counted[column] = {player: len(contexts) for player, contexts in seen.items()}
    return counted


def _spec(continuous, categorical, composites, identities, hinges):
    return {
        "continuous": tuple(continuous),
        "categorical": tuple(categorical),
        "composites": {name: tuple(columns) for name, columns in (composites or {}).items()},
        "identities": tuple(identities),
        "hinges": dict(hinges or {}),
    }


def _require_identity_columns(table, identities, contract):
    absent = [column for column in identities if column not in table.columns]
    if absent:
        raise ValueError(f"{contract.name}: identity columns {absent} are not in the table")


@dataclass
class Design:
    """A design matrix and the reference statistics that produced it.

    `continuous` holds the reference mean, standard deviation, and median per column,
    `categorical` and `composites` hold the reference level sets, and `identities` adds the exposure of
    every identified player. `nesting` counts the distinct levels of the first composite every player
    appears with, which is what turns a term into either a separated term or one that is confounded with
    its environment. `notes` counts what the reference could not cover.
    """
    name: str
    spec: dict
    matrix: np.ndarray
    columns: tuple
    continuous: dict
    categorical: dict
    composites: dict
    identities: dict
    nesting: dict
    rows: int
    notes: dict


def build_design(frame, *, continuous=(), categorical=(), composites=None, identities=(),
                 reference=None, name="design", hinges=None):
    """Standardized one-hot design matrix, built from `frame` or from an earlier reference.

    With `reference=None` the frame supplies the means, standard deviations, and level sets. Give
    the returned design as `reference` to score a later frame on the same columns: a level the
    reference never saw contributes a zero row and is counted in `notes` rather than adding a
    column, and a null continuous value is filled with the reference median and counted.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{name}: expected a polars DataFrame, found {type(frame).__name__}")
    spec = (reference.spec if reference is not None else
            _spec(continuous, categorical, composites, identities, hinges))
    # Fixed hinge bases are recomputed from source values, never trusted from input columns.
    for derived, (source, knot) in spec.get("hinges", {}).items():
        raw = _numeric(frame, source)
        frame = frame.with_columns(pl.Series(derived, np.maximum(raw - knot, 0.0)))
    blocks, columns = [], []
    stats = {"continuous": {}, "categorical": {}, "composites": {}, "identities": {}}
    notes = {"filled_continuous": {}, "unseen_categorical_rows": {}, "unseen_identity_rows": {}}
    for column in spec["continuous"]:
        raw = _numeric(frame, column)
        nulls = int(np.isnan(raw).sum())
        if reference is None:
            observed = raw[~np.isnan(raw)]
            if not observed.size:
                raise ValueError(f"{name}: {column} has no observed value to standardise against")
            mean = float(observed.mean())
            deviation = float(observed.std())
            median = float(np.median(observed))
        else:
            entry = reference.continuous[column]
            mean, deviation, median = entry["mean"], entry["sd"], entry["median"]
        values = np.where(np.isnan(raw), median, raw)
        if nulls:
            notes["filled_continuous"][column] = nulls
        scaled = (values - mean) / deviation if deviation else values - mean
        blocks.append(scaled.reshape(-1, 1))
        columns.append(column)
        stats["continuous"][column] = {
            "mean": mean, "sd": deviation, "median": median,
            "missing_column": (reference.continuous[column]["missing_column"] if reference is not None
                               else nulls > 0),
            "filled": nulls,
        }
    for column in spec["continuous"]:
        if stats["continuous"][column]["missing_column"]:
            blocks.append(np.isnan(_numeric(frame, column)).astype(float).reshape(-1, 1))
            columns.append(f"{column}{MISSING_SUFFIX}")
    for group, sources in spec["composites"].items():
        labels = _composite_labels(frame, sources)
        levels = _levels(labels) if reference is None else reference.composites[group]["levels"]
        block, names, unseen = _one_hot(labels, levels, name=group)
        blocks.append(block)
        columns.extend(names)
        stats["composites"][group] = {"sources": sources, "levels": levels, "unseen_rows": unseen}
        if unseen:
            notes["unseen_categorical_rows"][group] = unseen
    for column in spec["categorical"]:
        labels = column_labels(frame, column)
        levels = _levels(labels) if reference is None else reference.categorical[column]["levels"]
        block, names, unseen = _one_hot(labels, levels, name=column)
        blocks.append(block)
        columns.extend(names)
        stats["categorical"][column] = {"levels": levels, "unseen_rows": unseen}
        if unseen:
            notes["unseen_categorical_rows"][column] = unseen
    identity_labels = {column: column_labels(frame, column) for column in spec["identities"]}
    for column, labels in identity_labels.items():
        if reference is None:
            levels = tuple(sorted(set(labels)))
            exposure = dict(Counter(labels))
        else:
            levels = reference.identities[column]["levels"]
            exposure = reference.identities[column]["rows"]
        block, names, unseen = _one_hot(labels, levels, name=column)
        blocks.append(block)
        columns.extend(names)
        stats["identities"][column] = {"levels": levels, "rows": exposure, "unseen_rows": unseen}
        if unseen:
            notes["unseen_identity_rows"][column] = unseen
    if reference is not None:
        nesting = reference.nesting
    elif spec["identities"] and spec["composites"]:
        context = _composite_labels(frame, spec["composites"][next(iter(spec["composites"]))])
        nesting = _nesting(identity_labels, context)
    elif spec["identities"]:
        nesting = {column: {level: None for level in stats["identities"][column]["levels"]}
                   for column in spec["identities"]}
    else:
        nesting = {}
    matrix = np.hstack(blocks) if blocks else np.zeros((frame.height, 0))
    return Design(name=name, spec=spec, matrix=matrix, columns=tuple(columns), rows=frame.height,
                  continuous=stats["continuous"], categorical=stats["categorical"],
                  composites=stats["composites"], identities=stats["identities"],
                  nesting=nesting, notes=notes)


def convergence(estimator, max_iter):
    """Whether a solver reached its own stopping rule, and how many iterations it took.

    A fit that stops on the iteration cap is still a fit, and its coefficients are still returned;
    that is exactly why the fact has to be recorded rather than left in the solver. A weak penalty on
    several hundred correlated columns is where this happens in practice.
    """
    iterations = getattr(estimator, "n_iter_", None)
    if iterations is None:            # a closed-form solver has no iteration count to report
        return {"converged": True, "iterations": None}
    used = int(np.max(np.asarray(iterations)))
    if max_iter is None:              # no cap was named, so there is nothing to fall short of
        return {"converged": True, "iterations": used}
    return {"converged": used < int(max_iter), "iterations": used}


def warn_if_not_converged(state, name):
    """Warn when a fit stopped at its iteration cap, and pass the state through for reporting.

    A capped fit is still returned, because the weak end of a penalty grid has to stay measurable; the
    warning and the `converged` flag are how a caller finds out that the coefficients are not at an
    optimum.
    """
    if not state["converged"]:
        warnings.warn(f"{name}: the solver stopped at its iteration cap after "
                      f"{state['iterations']} iterations, so the coefficients are not at the optimum",
                      RuntimeWarning, stacklevel=3)
    return state


def _composed(composites, support):
    composed = dict(composites or {})
    if support:
        composed = {"support": SUPPORT_ENVIRONMENT, **composed}
    return composed


def passing_context_sources(context):
    """Raw continuous and categorical inputs shared by designs and domain audits.

    Fixed depth hinges are transformations of air_yards, already in continuous.
    """
    if context not in PASSING_CONTEXT_COMPONENTS:
        raise ValueError(f'unknown passing context {context!r}')
    components = PASSING_CONTEXT_COMPONENTS[context]
    return {'continuous': PASSING_CONTINUOUS,
            'categorical': (PASSING_CATEGORICAL + (("position",) if "position" in components else ()) +
                            (("defteam",) if "opponent" in components else ()))}


def passing_design(table, *, identities=(), composites=None, reference=None, context="linear"):
    """Design for the completion baselines on a canonical `passing_targets` table.

    `composites` composes context columns into one categorical column, for example
    `{"position": ("position",)}` for a per-position mean, which the plan prefers over one shared
    position intercept. `identities` names the columns whose levels enter as their own one-hot block,
    typically `("qb_id", "receiver_id")`.

    An earlier design passed as `reference` is reused instead of validated afresh, because it carries
    the means and level sets the later frame has to be scored on.
    """
    if reference is None:
        validate(table, PASSING_CONTRACT)
    if context not in PASSING_CONTEXTS:
        raise ValueError(f"unknown passing context {context!r}; expected {PASSING_CONTEXTS}")
    _require_identity_columns(table, identities, PASSING_CONTRACT)
    components = PASSING_CONTEXT_COMPONENTS[context]
    hinges = ({f"air_yards_above_{knot:g}": ("air_yards", knot) for knot in DEPTH_KNOTS}
              if "depth" in components else {})
    sources = passing_context_sources(context)
    design = build_design(table, continuous=sources['continuous'] + tuple(hinges), categorical=sources['categorical'],
                          composites=composites, identities=identities, reference=reference,
                          name="passing", hinges=hinges)
    if reference is None:
        design.spec["passing_context"] = context
    return design


def rushing_design(table, *, identities=(), composites=None, support=False, reference=None):
    """Design for the rushing baselines on a canonical `rushing_plays` table.

    `support=False` is the default: the context columns alone. `support=True` adds the team, location,
    and gap composition as one `support` column, which is the support environment of the plan and the
    layer the runner block of `partial_pooling` sits on. It is an environment label, not a line's
    quality, and it holds a team, a scheme, and an opponent at once.
    """
    if reference is None:
        validate(table, RUSHING_CONTRACT)
    _require_identity_columns(table, identities, RUSHING_CONTRACT)
    return build_design(table, continuous=RUSHING_CONTINUOUS, categorical=RUSHING_CATEGORICAL,
                        composites=_composed(composites, support), identities=identities,
                        reference=reference, name="rushing")


def penalty_value(penalty):
    """A penalty as a positive float, or a refusal.

    Both fitters take a penalty where zero or a negative value would silently mean "no regularisation"
    or an invalid objective, so the rule is enforced at the boundary rather than inside the solver.
    """
    if not penalty > 0:
        raise ValueError(f"baselines.penalty_value: penalty must be positive, found {penalty}")
    return float(penalty)


def baseline_name(prefix, identities, design):
    """The name a fit reports: `prefix:context[+composite][+identity(columns)]`.

    The name is a contract, because a recorded table or a comparison key cites it; a model with the
    same design and the same penalty has to produce the same string. `partial_pooling` extends it with
    the two penalties it splits.
    """
    parts = ["context", *design.composites]
    context = design.spec.get("passing_context", "expanded" if design.spec.get("hinges") else "linear")
    if context != "linear":
        parts[0] = f"context[{context}]"
    if identities:
        parts.append("identity(" + ",".join(identities) + ")")
    return prefix + ":" + "+".join(parts)


@dataclass
class FittedBaseline:
    """A fitted regularized baseline, its design, and its in-sample error.

    `in_sample` is the error on the rows the model was fitted on. It is reported to catch a broken
    fit, never as evidence about identity: a model with a player column per player fits its own
    rows well by construction.
    """
    name: str
    task: str
    target: str
    contract: str
    penalty: float
    rows: int
    estimator: object
    design: Design
    in_sample: dict
    block_penalties: dict = field(default_factory=dict)

    def holdout_design(self, table):
        """Design for a later frame, on the fitted columns and reference statistics."""
        contract = CONTRACTS[self.contract]
        validate(table, contract)
        _require_identity_columns(table, self.design.spec["identities"], contract)
        return build_design(table, reference=self.design, name=self.name)

    def predict(self, table):
        """Fitted probability of a completion, or fitted rushing yards, per row of `table`."""
        matrix = self.holdout_design(table).matrix
        if self.task == "completion":
            return self.estimator.predict_proba(matrix)[:, 1]
        return self.estimator.predict(matrix)

    def coefficients(self):
        """Fitted term per design column, in design order."""
        return {column: float(value) for column, value in
                zip(self.design.columns, np.ravel(self.estimator.coef_))}

    def identity_terms(self, *, identity=None, limit=None):
        """One entry per identified player: term, rows, share, and context nesting.

        `nested` is `True` when the player only ever appears inside one level of the first composite
        column, which is the case where the term and the environment cannot be told apart.

        `identity` keeps only one block, for example `"qb_id"` out of a passer-and-receiver design, and
        `limit` truncates the list without changing its order. Entries are ordered by identity column
        and player name, never by term size: an order by size would read as a ranking of players, and a
        regularized term is not one.
        """
        coefficients = self.coefficients()
        terms = []
        for column, entry in self.design.identities.items():
            if identity is not None and column != identity:
                continue
            for level in entry["levels"]:
                contexts = self.design.nesting.get(column, {}).get(level)
                rows = int(entry["rows"].get(level, 0))
                terms.append({
                    "identity": column,
                    "player": level,
                    "term": coefficients.get(f"{column}={level}"),
                    "rows": rows,
                    "share": round(rows / self.rows, 6) if self.rows else None,
                    "distinct_contexts": contexts,
                    "nested": None if contexts is None else contexts == 1,
                })
        terms.sort(key=lambda term: (term["identity"], term["player"]))
        return terms if limit is None else terms[:limit]

    def report(self):
        """Fit facts: rows, penalty, in-sample error, and what the reference could not cover."""
        return {
            "name": self.name,
            "task": self.task,
            "target": self.target,
            "contract": self.contract,
            "rows": self.rows,
            "features": len(self.design.columns),
            "penalty": self.penalty,
            "block_penalties": dict(self.block_penalties),
            "in_sample": dict(self.in_sample),
            "identity_levels": {column: len(entry["levels"])
                                for column, entry in self.design.identities.items()},
            "composites": {name: {"sources": entry["sources"], "levels": len(entry["levels"])}
                           for name, entry in self.design.composites.items()},
            "filled_continuous": dict(self.design.notes["filled_continuous"]),
        }


def fit_completion(table, *, identities=(), composites=None, penalty=DEFAULT_PENALTY,
                   max_iter=DEFAULT_MAX_ITER, tolerance=DEFAULT_TOLERANCE, context="linear"):
    """Regularized logistic baseline for `complete_pass`.

    The context is the pre-play context the contract declares: target depth, field position,
    distance to go, down, and charted pass location. No completion-derived column is admitted, so
    this is a conversion baseline. Pass `identities=("qb_id", "receiver_id")` for the second
    baseline, and read its terms as the plan asks: a target-only conversion model is not a
    receiver-talent model, because it never sees the routes a receiver was asked to run.

    The L2 penalty centres the terms of one identity block on zero, so a term is a deviation from
    the average identified player rather than an absolute rate, and the tolerance is tight enough
    that the solver does not stop short of that parametrisation.

    `penalty` is the inverse of the L2 strength, because the solver's `C` is `1 / penalty`: a larger
    penalty shrinks every term harder.
    """
    penalty = penalty_value(penalty)
    design = passing_design(table, identities=identities, composites=composites, context=context)
    target = table["complete_pass"].cast(pl.Float64, strict=False).to_numpy().astype(float)
    if len(np.unique(target)) < 2:
        raise ValueError("passing_targets: complete_pass holds one outcome only, and a logistic "
                         "baseline needs both a completion and an incompletion")
    # The penalty is scikit-learn's L2 default: naming it explicitly is deprecated in newer
    # versions and only warns, and every version this project supports defaults to L2.
    estimator = LogisticRegression(C=1.0 / penalty, solver="lbfgs", max_iter=max_iter, tol=tolerance)
    estimator.fit(design.matrix, target)
    probability = estimator.predict_proba(design.matrix)[:, 1]
    in_sample = {
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(target, probability)),
        **warn_if_not_converged(convergence(estimator, max_iter), "passing_targets:context"),
    }
    return FittedBaseline(name=baseline_name("passing", identities, design), task="completion",
                          target="complete_pass", contract=PASSING_CONTRACT.name, penalty=penalty,
                          rows=table.height, estimator=estimator, design=design, in_sample=in_sample)


def fit_rushing(table, *, identities=(), composites=None, support=False, penalty=DEFAULT_PENALTY):
    """Regularized ridge baseline for signed `rushing_yards`.

    Fitting the mean of a skewed, heavy-tailed outcome with a squared penalty is a pragmatic first
    step, not the last one: the plan leaves robust, mixture, and quantile models open. `penalty` is
    the ridge alpha applied to the fitted terms, so a larger penalty shrinks them harder. Add
    `identities=("rusher_id",)` for the runner baseline. The contract declares `qb_id` as an optional
    column and `prepare` does not write it, so a quarterback block needs a caller-supplied attribution
    before it can be fitted at all; the error names the missing column if it is not there.
    """
    penalty = penalty_value(penalty)
    design = rushing_design(table, identities=identities, composites=composites, support=support)
    target = _numeric(table, "rushing_yards")
    estimator = Ridge(alpha=penalty)
    estimator.fit(design.matrix, target)
    residual = target - estimator.predict(design.matrix)
    in_sample = {"mae": float(np.mean(np.abs(residual))),
                 "rmse": float(np.sqrt(np.mean(residual ** 2))),
                 **convergence(estimator, None)}
    return FittedBaseline(name=baseline_name("rushing", identities, design), task="yards",
                          target="rushing_yards", contract=RUSHING_CONTRACT.name, penalty=penalty,
                          rows=table.height, estimator=estimator, design=design, in_sample=in_sample)
