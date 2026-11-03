"""Reproducible development studies of context-adjusted completion associations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import polars as pl

from .audits import pairing_support, table_coverage
from .baselines import PASSING_CONTEXTS, fit_completion
from .contracts import PASSING_CONTRACT, validate
from .design import bootstrap_contrast
from .evaluation import chronological_split, clustered_comparison, completion_metrics, identity_increments, passing_transfer, row_losses
from .partial_pooling import fit_block_completion, select_passing_penalties


@dataclass(frozen=True)
class PassingStudySpec:
    """An explicit development protocol; reserved seasons may never enter a declared split.

    Context and identity penalties are selected once, then held fixed across expanding
    outer folds. Reservation is an exclusion rule, not proof that a season was never inspected.
    """
    train_seasons: tuple[int, ...]
    selection_season: int
    evaluation_seasons: tuple[int, ...]
    penalties: tuple[float, ...] = (10.0, 100.0, 1000.0)
    contexts: tuple[str, ...] = ("linear", "expanded")
    context_penalty: float = 1.0
    draws: int = 2000
    seed: int = 0
    level: float = 0.95
    reserved_seasons: tuple[int, ...] = ()
    purpose: str = "development"

    def __post_init__(self):
        for key in ("train_seasons", "evaluation_seasons", "reserved_seasons", "penalties", "contexts"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for key in ("train_seasons", "evaluation_seasons", "reserved_seasons"):
            values = getattr(self, key)
            if any(type(value) is not int for value in values) or tuple(sorted(set(values))) != values:
                raise ValueError(f"{key} must contain unique increasing integer seasons")
        if not self.train_seasons or not self.evaluation_seasons or type(self.selection_season) is not int:
            raise ValueError("declare training, selection and evaluation seasons")
        if not max(self.train_seasons) < self.selection_season < min(self.evaluation_seasons):
            raise ValueError("selection must follow training and precede evaluation")
        if set(self.reserved_seasons) & set(self.requested_seasons):
            raise ValueError("reserved seasons cannot be fitted, selected or evaluated")
        if self.purpose != "development":
            raise ValueError("only development studies are supported; untouched status is unverified")
        if (not self.penalties or any(type(p) not in (int, float) or not math.isfinite(p) or p <= 0
                                      for p in (*self.penalties, self.context_penalty))):
            raise ValueError("penalties must be nonempty, finite and positive")
        if not self.contexts or len(set(self.contexts)) != len(self.contexts) or set(self.contexts) - set(PASSING_CONTEXTS):
            raise ValueError("contexts must be unique declared passing contexts")
        if type(self.draws) is not int or self.draws < 20 or type(self.seed) is not int or self.seed < 0:
            raise ValueError("declare at least 20 draws and a nonnegative integer seed")
        if type(self.level) not in (int, float) or not 0 < self.level < 1:
            raise ValueError("interval level must be between zero and one")

    @property
    def requested_seasons(self):
        return (*self.train_seasons, self.selection_season, *self.evaluation_seasons)


def run_passing_study(table, *, spec, contrasts=(), contrast_draws=200):
    """Select on an inner season, refit expanding folds, score identical later targets.

    Optional preselected contrasts use the training targets of a declared position (or all
    training targets) as fixed Q. No holdout target or outcome enters Q or penalty selection.
    """
    if not isinstance(spec, PassingStudySpec):
        raise ValueError("run_passing_study requires a validated PassingStudySpec")
    table = table.filter(pl.col("season").is_in(spec.requested_seasons))
    validate(table, PASSING_CONTRACT)
    absent = sorted(set(spec.requested_seasons) - set(table["season"]))
    if absent:
        raise ValueError(f"missing declared seasons {absent}")
    table = table.sort(["season", "week", "game_id", "play_id"])
    selection = select_passing_penalties(
        table, train_seasons=spec.train_seasons, holdout_season=spec.selection_season,
        penalties=spec.penalties, contexts=spec.contexts, context_penalty=spec.context_penalty)
    chosen = selection["selected"]
    penalties = {key: chosen[key] for key in ("qb_id", "receiver_id")}
    folds, predictions = [], []
    for season in spec.evaluation_seasons:
        train_seasons = [year for year in spec.requested_seasons if year < season]
        split = chronological_split(table, train_seasons=train_seasons, holdout_season=season)
        train, holdout = split["train"], split["holdout"]
        models = {"linear_context": fit_completion(train, penalty=spec.context_penalty),
                  "context": fit_completion(train, context=chosen["context"], penalty=spec.context_penalty)}
        for name, identities in (("qb", ("qb_id",)), ("receiver", ("receiver_id",)),
                                  ("both", ("qb_id", "receiver_id"))):
            models[name] = fit_block_completion(
                train, identities=identities, identity_penalties={key: penalties[key] for key in identities},
                context=chosen["context"], context_penalty=spec.context_penalty)
        if any(not fit.in_sample["converged"] for fit in models.values()):
            raise ValueError(f"nonconverged outer fit in season {season}; no study result emitted")
        probabilities = {name: fitted.predict(holdout) for name, fitted in models.items()}
        truth = holdout["complete_pass"].to_numpy().astype(float)
        losses = {name: row_losses(truth, values, task="completion") for name, values in probabilities.items()}
        comparison = clustered_comparison(losses, clusters=holdout["game_id"].to_list(),
                                           reference_model="context", draws=spec.draws,
                                           seed=spec.seed, level=spec.level)
        transfer = passing_transfer(train, holdout, predictions=probabilities, reference_model="context",
                                    draws=spec.draws, seed=spec.seed, level=spec.level)
        frame = holdout.select("season", "week", "game_id", "play_id", "qb_id", "receiver_id", "complete_pass")
        predictions.append(frame.with_columns(
            [pl.Series("p_" + name, values) for name, values in probabilities.items()] +
            [pl.Series("transfer_cohort", transfer.pop("assignments"))]))
        contrast_reports = []
        for request in contrasts:
            args = dict(request)
            position = args.pop("population_position", None)
            population = train if position is None else train.filter(pl.col("position") == position)
            result = bootstrap_contrast(train, population, identity_penalties=penalties,
                                         context=chosen["context"], context_penalty=spec.context_penalty,
                                         draws=contrast_draws, seed=spec.seed, level=spec.level, **args)
            result["reference_population"] = {"source": "training_targets", "position": position,
                                               "seasons": train_seasons, "rows": population.height}
            contrast_reports.append(result)
        folds.append({"holdout_season": season, "train_seasons": train_seasons,
                      "rows": {"train": train.height, "holdout": holdout.height},
                      "models": {name: fitted.report() for name, fitted in models.items()},
                      "metrics": {name: completion_metrics(truth, values) for name, values in probabilities.items()},
                      "comparison": comparison, "transfer": transfer,
                      "identity_increments": identity_increments(
                          losses, clusters=holdout["game_id"].to_list(), draws=spec.draws,
                          seed=spec.seed, level=spec.level),
                      "pairing_support": pairing_support(train), "contrasts": contrast_reports})
    return {"report": {"schema_version": 1, "purpose": spec.purpose, "specification": asdict(spec),
                       "untouched_holdout_verified": False, "selection": selection,
                       "coverage": table_coverage(table, PASSING_CONTRACT), "folds": folds,
                       "interpretation": "target-conditional associations; no causal credit or skill ranking"},
            "predictions": pl.concat(predictions)}
