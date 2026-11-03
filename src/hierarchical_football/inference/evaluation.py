"""Holdout evaluation for the inference baselines: metrics, calibration, and clustered intervals.

A play is not an independent observation. Plays are clustered inside games, and a player's rows are
clustered inside the weeks and seasons he played, so a standard error computed over rows understates
the uncertainty of every comparison here. The resampling unit in this module is therefore the game,
or whatever cluster column the caller names, and a paired comparison resamples the same clusters for
both models.

The estimand is a row-level average loss over a season drawn from the same games. That is why a drawn
cluster contributes all of its rows and the pooled mean is taken over the drawn rows rather than the
average of per-cluster means: games do not all carry the same number of plays.

Nothing here fits a model, chooses a penalty, or names a winner: a comparison reports one paired
difference per model against the one it was measured against, and what a difference means is a decision
for the research protocol rather than for whichever statistic is largest.

One limit is stated rather than hidden: the percentile bootstrap is narrow when there are very few
clusters, because few distinct resamples cannot reproduce the sampling spread of a small group of
games. The cluster count travels with every interval so a reader can see it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .contracts import PASSING_CONTRACT, validate

TASKS = ("completion", "yards")


def require_task(task):
    """The metric family a loss is measured in, or a refusal naming the two that exist.

    One vocabulary is shared by the metric functions, the bootstrap and the model fitter, so a typo
    fails at the boundary rather than somewhere inside a bootstrap loop.
    """
    if task not in TASKS:
        raise ValueError(f"evaluation.require_task: unknown task {task!r}; expected one of {list(TASKS)}")
    return task
DEFAULT_DRAWS = 2000
DEFAULT_LEVEL = 0.95
TRANSFER_COHORTS = ("known_pair", "new_pair", "unseen_qb", "unseen_receiver", "unseen_both")
TAIL_YARDS = 10.0
COMPLETION_TAIL = 2.0


def identity_increments(losses, *, clusters, draws=DEFAULT_DRAWS, seed=0, level=DEFAULT_LEVEL):
    """Paired incremental loss of receivers beyond QBs, and QBs beyond receivers.

    Model keys are qb, receiver and both. Negative differences favor the combined fit.
    Comparisons concern refitted blocks at the declared penalties, not causal player credit.
    """
    required = ("qb", "receiver", "both")
    if not set(required) <= set(losses):
        raise ValueError("identity_increments requires qb, receiver and both losses")
    clusters = list(clusters)
    arrays = {name: np.asarray(losses[name], dtype=float) for name in required}
    if any(values.shape != (len(clusters),) or not values.size or not np.isfinite(values).all()
           for values in arrays.values()):
        raise ValueError("identity_increments needs finite aligned losses and one game per row")
    if type(draws) is not int or draws < 1 or not 0 < level < 1:
        raise ValueError("identity_increments needs positive draws and 0 < level < 1")
    games = len(set(clusters))
    comparisons = {}
    for key, reference in (("receiver_given_qb", "qb"), ("qb_given_receiver", "receiver")):
        interval = (bootstrap_difference(arrays["both"], arrays[reference], clusters=clusters,
                                          draws=draws, seed=seed, level=level) if games >= 2 else None)
        comparisons[key] = interval
    return {"rows": len(clusters), "games": games, "few_games": games < 10,
            "candidate_model": "both", "reference_models": {"receiver_given_qb": "qb",
                                                              "qb_given_receiver": "receiver"},
            "comparisons": comparisons}


def passing_transfer(train, holdout, *, predictions, reference_model=None,
                     draws=DEFAULT_DRAWS, seed=0, level=DEFAULT_LEVEL):
    """Score disjoint partnership cohorts defined only by earlier training identities.

    A new pair consists of two previously observed players who never appeared together in
    training. Unseen players are separate cohorts. This is natural chronological transfer,
    not a randomized teammate swap or a deliberately held-out edge experiment.
    """
    for table in (train, holdout):
        validate(table, PASSING_CONTRACT)
        if not table.height:
            raise ValueError("passing_transfer needs nonempty training and holdout tables")
    if train["season"].max() >= holdout["season"].min():
        raise ValueError("passing_transfer training seasons must be earlier than holdout")
    if not predictions:
        raise ValueError("passing_transfer needs predictions")
    truth = holdout["complete_pass"].to_numpy().astype(float)
    estimates = {name: np.asarray(values, dtype=float) for name, values in predictions.items()}
    for values in estimates.values():
        if values.shape != truth.shape:
            raise ValueError("passing_transfer needs one prediction per holdout row")
        row_losses(truth, values, task="completion")
    qbs, receivers = set(train["qb_id"]), set(train["receiver_id"])
    pairs = set(train.select("qb_id", "receiver_id").iter_rows())
    assignments = []
    for qb, receiver in holdout.select("qb_id", "receiver_id").iter_rows():
        unseen_qb, unseen_receiver = qb not in qbs, receiver not in receivers
        if unseen_qb and unseen_receiver:
            cohort = "unseen_both"
        elif unseen_qb:
            cohort = "unseen_qb"
        elif unseen_receiver:
            cohort = "unseen_receiver"
        else:
            cohort = "known_pair" if (qb, receiver) in pairs else "new_pair"
        assignments.append(cohort)
    clusters = np.asarray(holdout["game_id"].to_list())
    cohorts = {}
    for cohort in TRANSFER_COHORTS:
        mask = np.asarray(assignments) == cohort
        rows, games = int(mask.sum()), len(set(clusters[mask]))
        scores = {name: completion_metrics(truth[mask], values[mask])
                  for name, values in estimates.items()} if rows else None
        losses = {name: row_losses(truth[mask], values[mask], task="completion")
                  for name, values in estimates.items()} if rows else None
        comparison = (clustered_comparison(losses, clusters=clusters[mask],
                                           reference_model=reference_model, draws=draws,
                                           seed=seed, level=level) if games >= 2 else None)
        cohorts[cohort] = {"rows": rows, "games": games, "few_games": games < 10,
                           "metrics": scores, "comparison": comparison,
                           "identity_increments": (identity_increments(
                               losses, clusters=clusters[mask], draws=draws, seed=seed, level=level)
                               if rows and {"qb", "receiver", "both"} <= set(losses) else None)}
    return {"assignments": assignments, "cohorts": cohorts}


def chronological_split(table, *, train_seasons, holdout_season, contract=PASSING_CONTRACT):
    """Split one canonical table into earlier seasons to fit and a later season to score.

    The holdout season must be strictly later than every training season, and both sides must hold
    rows. A random split of plays would put rows from one game on both sides of the comparison, and
    a holdout season that is not later would turn a forecast into a fit.
    """
    validate(table, contract)
    training_seasons = tuple(int(season) for season in train_seasons)
    if not training_seasons:
        raise ValueError("chronological_split: name at least one training season")
    holdout = int(holdout_season)
    overlapping = sorted(season for season in training_seasons if season >= holdout)
    if overlapping:
        raise ValueError(f"chronological_split: training seasons {overlapping} are not earlier than "
                         f"the holdout season {holdout}")
    training = table.filter(pl.col("season").is_in(list(training_seasons)))
    held_out = table.filter(pl.col("season") == holdout)
    if training.height == 0:
        raise ValueError(f"chronological_split: no rows in seasons {list(training_seasons)}")
    if held_out.height == 0:
        raise ValueError(f"chronological_split: no rows in the holdout season {holdout}")
    return {"train": training, "holdout": held_out, "train_seasons": training_seasons,
            "holdout_season": holdout}


def row_losses(observed, predicted, *, task):
    """Per-row loss: the log loss of a completion, or the absolute error of a carry.

    The per-row losses are exposed because every interval in this module is built by resampling
    rows inside their clusters, which needs the loss of each row rather than one pooled average.
    """
    require_task(task)
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError(f"evaluation.row_losses: observed and predicted differ in length: "
                         f"{truth.shape} against {estimate.shape}")
    if task == "yards":
        return np.abs(truth - estimate)
    if not np.all(np.isfinite(estimate)) or np.any((estimate < 0) | (estimate > 1)):
        raise ValueError("evaluation.row_losses: completion probabilities must be finite and inside "
                         "[0, 1]")
    clipped = np.clip(estimate, 1e-12, 1 - 1e-12)
    return -(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))


def completion_metrics(observed, probability, *, bins=10, tail=COMPLETION_TAIL):
    """Log loss, Brier score, a reliability table, and the share of heavily punished rows.

    The tail share is the part of the season where a confident prediction met the other outcome.
    Two models can share a log loss and disagree about how often they are badly wrong, which is why
    the share sits beside the mean rather than inside it.
    """
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(probability, dtype=float)
    losses = row_losses(truth, estimate, task="completion")
    return {
        "rows": int(truth.size),
        "log_loss": float(losses.mean()),
        "brier": float(np.mean((estimate - truth) ** 2)),
        "mean_probability": float(estimate.mean()),
        "observed_rate": float(truth.mean()),
        f"tail_share_at_{int(tail)}": float(np.mean(losses >= tail)),
        "calibration": calibration_table(truth, estimate, bins=bins),
    }


def calibration_table(observed, probability, *, bins=10):
    """Observed completion rate against fitted probability, in equal-width probability bins.

    Only bins that hold rows are reported, so an empty part of the probability range is absent
    rather than shown as a well-calibrated zero.
    """
    if bins < 1:
        raise ValueError("evaluation.calibration_table: needs at least one bin")
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(probability, dtype=float)
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("evaluation.calibration_table: needs one observed value per fitted probability")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(estimate, edges[1:-1]), 0, bins - 1)
    table = []
    for position in range(bins):
        selected = index == position
        rows = int(selected.sum())
        if not rows:
            continue
        table.append({
            "lower": float(edges[position]),
            "upper": float(edges[position + 1]),
            "rows": rows,
            "predicted": float(estimate[selected].mean()),
            "observed": float(truth[selected].mean()),
        })
    return table


def yardage_metrics(observed, predicted, *, tail=TAIL_YARDS):
    """Mean absolute error, root mean squared error, and the share of large residuals.

    A squared penalty on a skewed outcome is dominated by the tail, so the extreme residuals are
    counted beside the mean: two models can share an MAE and disagree about who is badly wrong.
    """
    truth = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    residual = truth - estimate
    return {
        "rows": int(truth.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        f"tail_share_at_{int(tail)}": float(np.mean(np.abs(residual) >= tail)),
    }


def _cluster_positions(clusters):
    grouped = {}
    for row, cluster in enumerate(clusters):
        grouped.setdefault(cluster, []).append(row)
    positions = [np.asarray(rows) for rows in grouped.values()]
    for block in positions:
        block.setflags(write=False)
    return positions


def _drawn_means(losses, positions, rng, draws):
    """Mean loss over rows of the drawn clusters, for each bootstrap replicate.

    A cluster's rows enter as its total loss and its row count, so a replicate is two inner products
    over the drawn multiplicities rather than a concatenation of the drawn rows. The mean is taken
    over the pooled rows, which is what a row-level estimand needs when games differ in size.
    """
    totals = np.array([losses[block].sum() for block in positions], dtype=float)
    counts = np.array([block.size for block in positions], dtype=float)
    broad = len(positions)
    drawn = np.empty(draws, dtype=float)
    for draw in range(draws):
        picked = rng.integers(0, broad, size=broad)
        multiplicity = np.bincount(picked, minlength=broad)
        drawn[draw] = float(multiplicity @ totals) / float(multiplicity @ counts)
    return drawn


def _interval(drawn, estimate, *, level):
    tail = (1.0 - level) / 2.0 * 100.0
    lower, upper = np.percentile(drawn, [tail, 100.0 - tail])
    return {"estimate": float(estimate), "lower": float(lower), "upper": float(upper),
            "level": float(level), "draws": int(drawn.size)}


def clustered_metric(losses, *, clusters, draws=DEFAULT_DRAWS, seed=0, level=DEFAULT_LEVEL):
    """Percentile interval for one model's mean row loss, resampling whole clusters.

    Few clusters make the interval optimistic in the same way few games make any season's number
    noisy; the cluster count is reported so that a reader can see it.
    """
    rows = np.asarray(losses, dtype=float)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("clustered_metric needs one loss per row")
    positions = _cluster_positions(list(clusters))
    if len(positions) < 2:
        raise ValueError("clustered_metric needs at least two clusters to resample")
    rng = np.random.default_rng(seed)
    drawn = _drawn_means(rows, positions, rng, draws)
    return {**_interval(drawn, rows.mean(), level=level), "clusters": len(positions),
            "rows": int(rows.size)}


def bootstrap_difference(first, second, *, clusters, draws=DEFAULT_DRAWS, seed=0, level=DEFAULT_LEVEL):
    """Paired cluster bootstrap of `mean(first) - mean(second)`, in the same loss units.

    Both models are scored on the same drawn clusters, so a play that is hard for one model is hard
    for the other in every replicate. That pairing is what makes the interval about the difference
    between the models rather than about the two seasons they were scored on.

    A positive difference means `first` carries the larger loss, so it is the worse model; an
    interval that stays above zero is evidence against "no difference" in the losses.
    """
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("bootstrap_difference needs two arrays with one loss per row")
    positions = _cluster_positions(list(clusters))
    if len(positions) < 2:
        raise ValueError("bootstrap_difference needs at least two clusters to resample")
    rng = np.random.default_rng(seed)
    difference = left - right
    drawn = _drawn_means(difference, positions, rng, draws)
    return {**_interval(drawn, difference.mean(), level=level), "clusters": len(positions),
            "rows": int(left.size)}


def clustered_comparison(losses, *, clusters, reference_model=None, draws=DEFAULT_DRAWS, seed=0,
                         level=DEFAULT_LEVEL):
    """Interval per model plus the paired difference of every later model against the reference.

    The model every other one is measured against is the first in insertion order unless the caller
    names another, which is how the context-only baseline stays the thing every added block is measured
    against. It is named `reference_model` rather than `reference`, because `reference` already means
    "the earlier design whose means and levels are reused" in `baselines`. A positive difference means
    that model is worse than the one it was measured against.
    """
    if not losses:
        raise ValueError("clustered_comparison needs at least one model")
    names = list(losses)
    reference_model = names[0] if reference_model is None else reference_model
    if reference_model not in losses:
        raise ValueError(f"evaluation.clustered_comparison: unknown reference model "
                         f"{reference_model!r}; expected one of {names}")
    metrics = {name: clustered_metric(losses[name], clusters=clusters, draws=draws, seed=seed,
                                      level=level) for name in names}
    differences = {name: bootstrap_difference(losses[name], losses[reference_model],
                                              clusters=clusters, draws=draws, seed=seed, level=level)
                   for name in names if name != reference_model}
    return {"metrics": metrics, "differences": differences, "reference_model": reference_model}
