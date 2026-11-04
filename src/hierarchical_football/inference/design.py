"""Standardized contrasts under a reference population, with the support they rest on.

A fitted baseline answers a conditional question: what does this player do in the contexts he
actually saw? A contrast asks a different one - what does the model predict for two players under
one shared distribution of pre-play contexts Q? The answer is only as good as the overlap between Q
and what each player actually did, so every contrast here travels with a support report that names
the part of Q a player never faced.

Nothing here is a causal effect. Holding the context fixed removes the observed context difference
between two players. It does not remove the coach, the coverage, the route assignment, the health,
or the game script, none of which are in the model. A standardized contrast is a statement about the
fitted model under Q, and no more.

The population is a slice of a canonical table rather than a fresh grid, because the plan asks for
an observed-support mix and because a canonical frame is what the design builder can score. Cell
labels turn a row count into an overlap measure: the share of Q that a player's own rows cover, and
the share that they do not, which is the extrapolation the caller has to declare.

Two scales are offered because the plan names two quantities. A catch rate under Q averages the
fitted probability; an adjusted log-odds deviation averages the linear predictor. They are not
interchangeable: averaging a sigmoid over a population is not the sigmoid of the average, so a fixed
partner changes a probability contrast even in a model that is additive on the log-odds scale.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl

from .contracts import PASSING_CONTRACT, UNKNOWN, label, missing, validate
from .partial_pooling import fit_block_completion

AIR_YARDS_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
FIELD_EDGES = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
DISTANCE_EDGES = (1.0, 4.0, 8.0, 15.0, 25.0)

# The cell schemes the plan names, as the keyword arguments `context_cells` and `player_support` take.
# A scheme is `{"continuous": ((column, edges), ...), "categorical": (column, ...)}`: every column in
# `continuous` is banded on its own edges and enters the cell label as one band, every column in
# `categorical` enters as its own value. `edges` is an ascending tuple whose interior boundaries split
# the bands, so `AIR_YARDS_EDGES` produces the five labels "0-5", "5-10", "10-15", "15-20", "20-30".
# A value outside the outer edges lands in the nearest end band, and a null lands in `UNKNOWN`.
# A scheme is passed with `**`, for example `context_cells(table, **PASSING_CELLS)`.
PASSING_CELLS = {"continuous": (("air_yards", AIR_YARDS_EDGES),),
                 "categorical": ("down", "pass_location")}
RUSHING_CELLS = {"continuous": (("yardline_100", FIELD_EDGES), ("ydstogo", DISTANCE_EDGES)),
                 "categorical": ("down", "run_location", "run_gap")}
DEFAULT_MIN_ROWS = 5
SCALES = ("probability", "log_odds")


def shared_passing_population(windows, population, *, identity, left, right, fixed=None,
                              min_rows=DEFAULT_MIN_ROWS):
    """Keep observed Q rows in cells supported by both joint arms in every fitting window.

    Support uses the declared coarse depth/down/direction cells. Fixed partners are filtered
    jointly with the varied player, not counted from unrelated plays. Outcomes never choose
    cells. Passing this screen does not establish overlap on every modeled covariate.
    """
    if not windows or type(min_rows) is not int or min_rows < 1:
        raise ValueError("shared population needs fitting windows and a positive integer min_rows")
    if identity not in ("qb_id", "receiver_id") or left == right:
        raise ValueError("shared population needs two distinct passing identities")
    fixed = dict(fixed or {})
    if identity in fixed or set(fixed) - {"qb_id", "receiver_id"}:
        raise ValueError("fixed levels must name only the other passing identity")
    if any(not isinstance(value, str) or not value.strip() for value in (left, right, *fixed.values())):
        raise ValueError("player levels must be nonempty strings")
    validate(population, PASSING_CONTRACT)
    if not population.height:
        raise ValueError("reference population must hold observed rows")
    source_labels = context_cells(population, **PASSING_CELLS)
    shared = set(source_labels)
    exposure = {}
    for name, table in windows.items():
        validate(table, PASSING_CONTRACT)
        exposure[name] = {}
        for side, player in (("left", left), ("right", right)):
            arm = table
            for column, level in {**fixed, identity: player}.items():
                arm = arm.filter(pl.col(column) == level)
            counted = Counter(context_cells(arm, **PASSING_CELLS))
            supported = {cell for cell, count in counted.items() if count >= min_rows}
            shared &= supported
            exposure[name][side] = {"rows": arm.height, "cells": len(counted),
                                     "supported_cells": len(supported), "rows_per_cell": dict(counted)}
    selected = population.filter(pl.Series([cell in shared for cell in source_labels]))
    return {"population": selected, "report": {
        "status": "ok" if selected.height else "no_shared_support", "identity": identity,
        "left": left, "right": right, "fixed": fixed, "min_rows": min_rows,
        "source_rows": population.height, "retained_rows": selected.height,
        "retained_share": selected.height / population.height,
        "source_cells": len(set(source_labels)), "shared_cells": len(shared),
        "retained_cells": sorted(shared), "exposure": exposure,
        "cell_definition": PASSING_CELLS,
        "note": "coarse observed joint support across all windows; not full positivity or causal identification"}}


def _passing_component(table, identity, player):
    graph = {}
    for qb, receiver in table.select("qb_id", "receiver_id").unique().iter_rows():
        q, r = ("qb_id", qb), ("receiver_id", receiver)
        graph.setdefault(q, set()).add(r)
        graph.setdefault(r, set()).add(q)
    reached, frontier = set(), [(identity, player)]
    while frontier:
        node = frontier.pop()
        if node not in reached:
            reached.add(node)
            frontier.extend(graph.get(node, set()) - reached)
    return reached


def bootstrap_contrast(train, population, *, identity, left, right, fixed=None,
                       identity_penalties, context="linear", context_penalty=1.0,
                       draws=200, seed=0, level=0.95, scale="probability", opponent_penalty=None,
                       min_rows=DEFAULT_MIN_ROWS):
    """Refitted game-bootstrap interval conditional on penalties, context and a fixed Q.

    No interval is emitted for disconnected players or incomplete resamples. This measures
    sampling variation of a regularized contrast, not causal identification, selection
    uncertainty, uncertainty in Q, or a Bayesian posterior.
    """
    if identity not in ("qb_id", "receiver_id"):
        raise ValueError("bootstrap_contrast identity must be qb_id or receiver_id")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 20 or not 0 < level < 1:
        raise ValueError("bootstrap_contrast needs at least 20 draws and 0 < level < 1")
    if type(min_rows) is not int or min_rows < 1:
        raise ValueError("bootstrap_contrast needs a positive integer min_rows")
    fixed = dict(fixed or {})
    if set(fixed) - {"qb_id", "receiver_id"}:
        raise ValueError("fixed levels must be passing identities")
    requested = [(identity, left), (identity, right), *fixed.items()]
    for column, player in requested:
        if player not in set(train[column]):
            raise ValueError(f"bootstrap_contrast requires observed player {column}={player}")
    fit_args = dict(identity_penalties=identity_penalties, context=context,
                    context_penalty=context_penalty, opponent_penalty=opponent_penalty)
    fitted = fit_block_completion(train, **fit_args)
    games = sorted(set(train["game_id"]))
    if len(games) < 2:
        raise ValueError("bootstrap_contrast needs at least two games")
    contrast_args = dict(identity=identity, left=left, right=right, fixed=fixed, scale=scale, min_rows=min_rows)
    contrast = standardized_contrast(fitted, population, fitting_table=train,
                                     **PASSING_CELLS, **contrast_args)
    # A connected graph is necessary, not sufficient, for separating these identities.
    reached = _passing_component(train, identity, left)
    connected = all(node in reached for node in requested)
    report = {"contrast": contrast, "interval": None, "draws": draws, "seed": seed,
              "level": level, "games": len(games), "few_games": len(games) < 10,
              "successful_draws": 0, "failures": {"missing_player": 0, "single_outcome": 0,
                                                   "not_converged": 0, "disconnected_players": 0},
              "identity_penalties": dict(identity_penalties), "context": context,
              "context_penalty": context_penalty,
              "opponent_penalty": opponent_penalty,
              "note": "conditional game bootstrap; fixed penalties and reference population; "
                      "not causal, not a posterior, and not an identification interval"}
    if not connected or not fitted.in_sample["converged"]:
        report["status"] = "disconnected_players" if not connected else "not_converged"
        return report
    if any(arm["support"]["extrapolated_share"] > 1e-9 for arm in contrast["arms"].values()):
        report["status"] = "unsupported_context"
        return report
    blocks = {game: train.filter(pl.col("game_id") == game) for game in games}
    rng, differences = np.random.default_rng(seed), []
    for _ in range(draws):
        # Re-key repeated games so canonical play uniqueness survives cluster resampling.
        sampled = pl.concat([blocks[games[int(index)]].with_columns(
            pl.lit(f"bootstrap_game_{copy}").alias("game_id"))
            for copy, index in enumerate(rng.integers(0, len(games), size=len(games)))])
        if any(player not in set(sampled[column]) for column, player in requested):
            report["failures"]["missing_player"] += 1
            continue
        if sampled["complete_pass"].n_unique() < 2:
            report["failures"]["single_outcome"] += 1
            continue
        reached = _passing_component(sampled, identity, left)
        if any(node not in reached for node in requested):
            report["failures"]["disconnected_players"] += 1
            continue
        refit = fit_block_completion(sampled, **fit_args)
        if not refit.in_sample["converged"]:
            report["failures"]["not_converged"] += 1
            continue
        differences.append(standardized_contrast(refit, population, **contrast_args)["difference"])
    report["successful_draws"] = len(differences)
    report["status"] = "ok" if len(differences) == draws else "incomplete_resamples"
    if report["status"] == "ok":
        tail = (1 - level) / 2
        low, high = np.quantile(differences, [tail, 1 - tail])
        report["interval"] = {"estimate": contrast["difference"], "lower": float(low),
                               "upper": float(high), "level": level}
    return report


def _bucket(value, edges):
    """The band a numeric context value falls in, or UNKNOWN when there is no value to band."""
    if missing(value):
        return UNKNOWN
    position = np.digitize(float(value), edges[1:-1])
    return f"{edges[position]:g}-{edges[position + 1]:g}"


def context_cells(table, *, continuous=(), categorical=()):
    """One cell label per row: the pre-play context that a contrast is standardized over.

    A cell is the combination of the named bands and labels, so two rows share a cell when they are
    comparable situations. A missing value keeps its own `UNKNOWN` cell rather than being dropped,
    because a model that had no context for a row still produced a prediction for it.
    """
    if not isinstance(table, pl.DataFrame):
        raise TypeError(f"context_cells: expected a polars DataFrame, found {type(table).__name__}")
    expected = [column for column, _ in continuous] + list(categorical)
    absent = [column for column in expected if column not in table.columns]
    if absent:
        raise ValueError(f"context_cells: columns {absent} are not in the table")
    parts = []
    for column, edges in continuous:
        parts.append([f"{column}={_bucket(value, edges)}" for value in table[column].to_list()])
    for column in categorical:
        parts.append([f"{column}={label(value)}" for value in table[column].to_list()])
    return ["|".join(row) for row in zip(*parts)]


def cell_weights(labels):
    """Share of rows in each cell, largest first."""
    counted = {}
    for label in labels:
        counted[label] = counted.get(label, 0) + 1
    total = len(labels)
    return {label: count / total for label, count in
            sorted(counted.items(), key=lambda item: (-item[1], item[0]))}


def support_report(table, *, continuous=(), categorical=(), min_rows=DEFAULT_MIN_ROWS):
    """How much of one population sits in cells that hold enough rows to lean on.

    `thin_share` is the part of the population in cells below `min_rows`, which is where a contrast
    is being read off a handful of plays.
    """
    labels = context_cells(table, continuous=continuous, categorical=categorical)
    weights = cell_weights(labels)
    rows = dict(Counter(labels))
    thin = {label: weight for label, weight in weights.items() if rows[label] < min_rows}
    return {
        "rows": table.height,
        "cells": len(weights),
        "min_rows": min_rows,
        "weights": weights,
        "rows_per_cell": rows,
        "thin_share": float(sum(thin.values())),
        "thin_cells": len(thin),
        "largest_cell_share": next(iter(weights.values())) if weights else None,
    }


def player_support(fitting_table, population, *, identity, player, continuous=(), categorical=(),
                   min_rows=DEFAULT_MIN_ROWS):
    """The share of a population a player's own rows cover, and the share they do not.

    `extrapolated_share` is the part of Q the player never faced at all, so a contrast there is the
    model speaking outside the player's support. Zero rows for the player means every cell is
    extrapolated and no contrast should be reported for him.
    """
    for name, frame in (("fitting_table", fitting_table), ("population", population)):
        if identity not in frame.columns:
            raise ValueError(f"player_support: {identity} is not in the {name}")
    rows = fitting_table.filter(pl.col(identity) == player)
    if rows.height == 0:
        raise ValueError(f"player_support: {player} has no rows in {identity}")
    within = context_cells(rows, continuous=continuous, categorical=categorical)
    weights = cell_weights(context_cells(population, continuous=continuous, categorical=categorical))
    covered = {}
    for label in within:
        covered[label] = covered.get(label, 0) + 1
    supported = sum(weight for label, weight in weights.items() if covered.get(label, 0) > 0)
    well = sum(weight for label, weight in weights.items() if covered.get(label, 0) >= min_rows)
    absent = sorted(label for label in weights if label not in covered)
    return {
        "identity": identity,
        "player": player,
        "rows": rows.height,
        "cells_faced": len(covered),
        "population_cells": len(weights),
        "population_rows": population.height,
        "min_rows": min_rows,
        "supported_share": float(supported),
        "well_supported_share": float(well),
        "extrapolated_share": float(1.0 - supported),
        "unsupported_cells": absent,
    }


def _forced(population, levels):
    """A population with every named identity column set to the level asked for."""
    if not levels:
        return population
    absent = [column for column in levels if column not in population.columns]
    if absent:
        raise ValueError(f"design._forced: identity columns {absent} are not in the population")
    return population.with_columns([pl.lit(level).alias(column) for column, level in levels.items()])


def _arm_values(fitted, frame, scale):
    """The fitted value of every population row, on the requested scale."""
    if scale not in SCALES:
        raise ValueError(f"design._arm_values: unknown scale {scale!r}; expected one of "
                         f"{list(SCALES)}")
    if scale == "probability":
        return np.asarray(fitted.predict(frame), dtype=float)
    if fitted.task != "completion":
        raise ValueError("design._arm_values: a log-odds arm needs a completion model; yards are "
                         "already additive")
    matrix = fitted.holdout_design(frame).matrix
    return np.asarray(fitted.estimator.decision_function(matrix), dtype=float).ravel()


def standardized_prediction(fitted, population, *, levels=None, fitting_table=None, continuous=(),
                            categorical=(), min_rows=DEFAULT_MIN_ROWS, scale="probability"):
    """Mean fitted outcome over a population, with any identity level forced onto every row.

    `levels` fixes identity columns to one player or one reference level, which is what turns a
    conditional model into a standardized prediction: the same Q for every arm of a contrast.

    `scale="probability"` averages the fitted rate, which is the plan's catch-rate contrast.
    `scale="log_odds"` averages the linear predictor, which is the plan's adjusted log-odds
    deviation and is the scale on which an additive model stays additive: the mean log-odds of a
    player under Q does not depend on which partner is fixed, while the mean probability does.
    """
    if population.height == 0:
        raise ValueError("standardized_prediction: the population holds no rows")
    frame = _forced(population, levels)
    values = _arm_values(fitted, frame, scale)
    report = {
        "rows": population.height,
        "scale": scale,
        "prediction": float(values.mean()),
        "spread": float(values.std()),
        "levels": dict(levels or {}),
        "unseen_identities": sorted(
            f"{column}={level}" for column, level in (levels or {}).items()
            if f"{column}={level}" not in fitted.design.columns),
    }
    if fitting_table is not None and levels and len(levels) == 1:
        column, level = next(iter(levels.items()))
        report["support"] = player_support(fitting_table, population, identity=column, player=level,
                                           continuous=continuous, categorical=categorical,
                                           min_rows=min_rows)
    return report


def standardized_contrast(fitted, population, *, identity, left, right, fixed=None,
                          fitting_table=None, continuous=(), categorical=(),
                          min_rows=DEFAULT_MIN_ROWS, scale="probability"):
    """Difference between two players under one population, with the support behind each arm.

    The difference is `left - right` on the requested scale. Pass `fixed` to hold the other
    identities at reference levels, for example one passer for a receiver contrast. On the
    probability scale a fixed partner changes the difference even in an additive model, because the
    average of a sigmoid is not the sigmoid of an average; on the log-odds scale it does not. Either
    way the fixed levels are part of the estimand and are reported back.
    """
    if identity not in population.columns:
        raise ValueError(f"standardized_contrast: {identity} is not in the population")
    base = dict(fixed or {})
    if identity in base:
        raise ValueError(f"standardized_contrast: {identity} is both fixed and varied")
    arms = {side: standardized_prediction(fitted, population, levels={**base, identity: level},
                                          fitting_table=fitting_table, continuous=continuous,
                                          categorical=categorical, min_rows=min_rows, scale=scale)
            for side, level in (("left", left), ("right", right))}
    if fitting_table is not None and base:
        # A fixed partner makes two levels per arm, so the arm could not report the support of the one
        # that varies; with no fixed level the arm already carries it.
        for arm, level in (("left", left), ("right", right)):
            arms[arm]["support"] = player_support(fitting_table, population, identity=identity,
                                                  player=level, continuous=continuous,
                                                  categorical=categorical, min_rows=min_rows)
    return {
        "identity": identity,
        "left": left,
        "right": right,
        "fixed": base,
        "scale": scale,
        "left_prediction": arms["left"]["prediction"],
        "right_prediction": arms["right"]["prediction"],
        "difference": arms["left"]["prediction"] - arms["right"]["prediction"],
        "population_rows": population.height,
        "arms": arms,
    }


def contrast_sensitivity(fitter, population, *, identity, left, right, penalties, fixed=None,
                         fitting_table=None, continuous=(), categorical=(),
                         min_rows=DEFAULT_MIN_ROWS, scale="probability"):
    """How far a contrast moves when the penalty moves, reported as a sensitivity range.

    This is not a confidence interval and must not be read as one: it measures one modelling choice,
    the penalty, and says nothing about sampling error or about a player's true effect. It is here
    because a point estimate would otherwise be read as certain.
    """
    values = {}
    for penalty in penalties:
        contrast = standardized_contrast(fitter(penalty), population, identity=identity, left=left,
                                         right=right, fixed=fixed, fitting_table=fitting_table,
                                         continuous=continuous, categorical=categorical,
                                         min_rows=min_rows, scale=scale)
        values[float(penalty)] = contrast["difference"]
    ordered = sorted(values.values())
    return {
        "identity": identity,
        "left": left,
        "right": right,
        "scale": scale,
        "penalties": values,
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "spread": ordered[-1] - ordered[0],
        "penalties_tried": len(values),
        "note": "penalty sensitivity range, not a confidence interval",
    }
