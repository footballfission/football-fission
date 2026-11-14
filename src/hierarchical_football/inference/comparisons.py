"""Supported retrospective player contrasts across disjoint historical windows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math

import numpy as np
import polars as pl

from .baselines import PASSING_CONTEXT_COMPONENTS
from .contracts import PASSING_CONTRACT, validate
from .context_support import model_context_population
from .design import bootstrap_contrast, shared_passing_population, standardized_contrast
from .partial_pooling import fit_block_completion


@dataclass(frozen=True)
class PassingComparisonSpec:
    """Team-scoped retrospective specification; no fitting or selection on scored holdouts."""
    team: str
    identity: str
    left: str
    right: str
    partner: str
    windows: tuple[tuple[int, ...], ...]
    position: str | None = None
    context: str = "expanded"
    qb_penalty: float = 100.0
    receiver_penalty: float = 100.0
    opponent_penalty: float | None = 100.0
    sensitivity_penalties: tuple[float, ...] = (10.0, 100.0, 1000.0)
    min_rows: int = 5
    draws: int = 200
    seed: int = 0
    level: float = 0.95
    scale: str = "probability"
    reference_policy: str = "coarse"

    def __post_init__(self):
        object.__setattr__(self, "windows", tuple(tuple(window) for window in self.windows))
        object.__setattr__(self, "sensitivity_penalties", tuple(self.sensitivity_penalties))
        if any(not isinstance(value, str) or not value.strip() for value in
               (self.team, self.identity, self.left, self.right, self.partner)):
            raise ValueError("comparison labels must be nonempty strings")
        if self.identity not in ("qb_id", "receiver_id") or self.left == self.right:
            raise ValueError("comparison needs distinct players in one passing identity block")
        if len(self.windows) < 2:
            raise ValueError("comparison needs at least two historical windows")
        previous = None
        for window in self.windows:
            if (not window or any(type(year) is not int for year in window) or
                    tuple(sorted(set(window))) != window or
                    (previous is not None and previous >= min(window))):
                raise ValueError("windows must contain unique increasing seasons and be disjoint in time")
            previous = max(window)
        if self.context not in PASSING_CONTEXT_COMPONENTS:
            raise ValueError("unknown comparison context")
        if self.position not in (None, "WR", "TE", "RB", "QB", "FB", "UNKNOWN"):
            raise ValueError("unknown reference position")
        penalties = (self.qb_penalty, self.receiver_penalty, *self.sensitivity_penalties)
        if (not self.sensitivity_penalties or len(set(self.sensitivity_penalties)) != len(self.sensitivity_penalties)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in penalties)):
            raise ValueError("comparison penalties must be finite and positive with a unique sensitivity grid")
        if self.opponent_penalty is not None and (
                type(self.opponent_penalty) not in (int, float) or not math.isfinite(self.opponent_penalty)
                or self.opponent_penalty <= 0 or "opponent" not in PASSING_CONTEXT_COMPONENTS[self.context]):
            raise ValueError("opponent penalty needs a finite positive value and opponent context")
        if (type(self.min_rows) is not int or self.min_rows < 1 or type(self.draws) is not int or self.draws < 20
                or type(self.seed) is not int or self.seed < 0 or not 0 < self.level < 1):
            raise ValueError("invalid comparison support or bootstrap settings")
        if self.scale not in ("probability", "log_odds"):
            raise ValueError("comparison scale must be probability or log_odds")
        if self.reference_policy not in ("coarse", "model_context"):
            raise ValueError("reference_policy must be coarse or model_context")

    @property
    def requested_seasons(self):
        return tuple(year for window in self.windows for year in window)


def run_passing_comparison(table, *, spec):
    """Fit each historical window with fixed priors and evaluate one shared, earlier Q.

    Priors are specified, never selected here. Later exposure may restrict shared support,
    so this is retrospective explanation, not an out-of-time predictive test.
    """
    if not isinstance(spec, PassingComparisonSpec):
        raise ValueError("comparison requires a validated PassingComparisonSpec")
    table = table.filter((pl.col("team") == spec.team) & pl.col("season").is_in(spec.requested_seasons))
    validate(table, PASSING_CONTRACT)
    missing = sorted(set(spec.requested_seasons) - set(table["season"]))
    if missing:
        raise ValueError(f"missing declared team seasons {missing}")
    table = table.sort(["season", "week", "game_id", "play_id"])
    windows = {"-".join(map(str, years)): table.filter(pl.col("season").is_in(years)) for years in spec.windows}
    reference = next(iter(windows.values()))
    if spec.position is not None:
        reference = reference.filter(pl.col("position") == spec.position)
    fixed = {"qb_id" if spec.identity == "receiver_id" else "receiver_id": spec.partner}
    contrast_args = dict(identity=spec.identity, left=spec.left, right=spec.right, fixed=fixed, scale=spec.scale)
    report = {"schema_version": 1, "purpose": "retrospective_comparison", "forecast_evaluation": False,
              "specification": asdict(spec), "windows": [], "summary": None,
              "context_support": None, "reference_rows": 0, "reference_share": None,
              "reference_seasons": list(spec.windows[0]),
              "interpretation": "team-scoped target conversion associations; no portable talent or causal credit"}
    if not reference.height:
        report.update(status="no_reference_rows", support=None)
        return {"report": report, "reference_population": reference}
    common = shared_passing_population(windows, reference, identity=spec.identity, left=spec.left,
                                       right=spec.right, fixed=fixed, min_rows=spec.min_rows)
    population = common["population"]
    report.update(status=common["report"]["status"], support=common["report"])
    if not population.height:
        report["reference_share"] = 0.0
        return {"report": report, "reference_population": population}
    domain = model_context_population(windows, population, context=spec.context)
    report["context_support"] = domain["report"]
    if spec.reference_policy == "model_context":
        population = domain["population"]
    report.update(reference_rows=population.height, reference_share=population.height / reference.height)
    if not population.height:
        report["status"] = "no_shared_model_context"
        return {"report": report, "reference_population": population}
    seeds = np.random.SeedSequence(spec.seed).spawn(len(windows))
    for (name, train), seed_sequence in zip(windows.items(), seeds):
        seed = int(seed_sequence.generate_state(1)[0])
        bootstrap = bootstrap_contrast(
            train, population, **contrast_args, context=spec.context, context_penalty=1.0,
            identity_penalties={"qb_id": spec.qb_penalty, "receiver_id": spec.receiver_penalty},
            opponent_penalty=spec.opponent_penalty, draws=spec.draws, seed=seed, level=spec.level,
            min_rows=spec.min_rows)
        sensitivities = []
        for qb, receiver in product(spec.sensitivity_penalties, repeat=2):
            fit = fit_block_completion(train, context=spec.context, opponent_penalty=spec.opponent_penalty,
                                       identity_penalties={"qb_id": qb, "receiver_id": receiver})
            converged = fit.in_sample["converged"]
            difference = (standardized_contrast(fit, population, **contrast_args)["difference"]
                          if converged else None)
            sensitivities.append({"qb_penalty": qb, "receiver_penalty": receiver,
                                   "difference": difference, "converged": converged})
        complete = all(row["converged"] for row in sensitivities)
        values = [row["difference"] for row in sensitivities] if complete else []
        notes = fit.holdout_design(population).notes
        report["windows"].append({"name": name, "seasons": sorted(set(train["season"])),
            "rows": train.height, "games": train["game_id"].n_unique(), "bootstrap": bootstrap,
            "reference_context": {key: notes[key] for key in ("unseen_categorical_rows", "filled_continuous")},
            "sensitivity": {"fits": sensitivities, "complete": complete,
                            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                            "note": "prior sensitivity range, not a confidence interval"}})
    differences = [window["bootstrap"]["contrast"]["difference"] for window in report["windows"]]
    intervals = [window["bootstrap"]["interval"] for window in report["windows"]]
    valid_points = all(window["bootstrap"]["status"] in ("ok", "incomplete_resamples")
                       for window in report["windows"])
    report["status"] = ("ok" if all(window["bootstrap"]["status"] == "ok" and window["sensitivity"]["complete"]
                                     for window in report["windows"]) else "incomplete_estimation")
    uncertainty_windows = {}
    for window in report['windows']:
        bootstrap = window['bootstrap']
        audit = bootstrap['estimability']
        estimable = (audit['log_odds_contrast']['estimable'] if spec.scale == 'log_odds'
                     else audit['probability_arms_estimable'])
        uncertainty_windows[window['name']] = {
            'requested_scale_estimable': estimable, 'interval_status': bootstrap['status'],
            'evaluated_draws': bootstrap['resample_context']['evaluated_draws'],
            'context_failure_draws': bootstrap['resample_context']['draws_with_failures']}
    report["summary"] = {
        'uncertainty': {'interval_target': 'conditional_regularized_estimator',
                        'skill_interval_coverage_verified': False,
                        'all_original_requested_scale_estimable': all(
                            row['requested_scale_estimable'] for row in uncertainty_windows.values()),
                        'windows': uncertainty_windows},
        "point_direction_consistent": ((all(value > 0 for value in differences) or all(value < 0 for value in differences))
                                       if valid_points else None),
        "interval_direction_consistent": ((all(value["lower"] > 0 for value in intervals) or
                                           all(value["upper"] < 0 for value in intervals))
                                          if all(value is not None for value in intervals) else None),
        "note": "direction checks are descriptive; no interval for between-window change or claim of stable skill"}
    return {"report": report, "reference_population": population}
