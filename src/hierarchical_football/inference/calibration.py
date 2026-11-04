"""Known-truth Monte Carlo checks of the existing conditional contrast estimator."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from statistics import NormalDist

import numpy as np
import polars as pl

from .design import bootstrap_contrast
from .simulation import SCENARIOS, simulate_passing


@dataclass(frozen=True)
class CalibrationSpec:
    trials: int = 30
    draws: int = 100
    games: int = 20
    repeats: int = 4
    penalties: tuple = (1.0, 100.0, 1000.0)
    scenarios: tuple = ('crossed', 'nested', 'depth_confounded')
    seed: int = 0
    level: float = 0.95

    def __post_init__(self):
        for name, minimum in (('trials', 2), ('draws', 20), ('games', 2), ('repeats', 1), ('seed', 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if (isinstance(self.level, bool) or not isinstance(self.level, (int, float))
                or not 0 < self.level < 1):
            raise ValueError('level must lie between zero and one')
        if (not isinstance(self.penalties, (tuple, list)) or not self.penalties
                or any(isinstance(p, bool) or not isinstance(p, (int, float))
                       or not math.isfinite(p) or p <= 0 for p in self.penalties)
                or len(set(self.penalties)) != len(self.penalties)):
            raise ValueError('penalties must be distinct positive finite numbers')
        if (not isinstance(self.scenarios, (tuple, list)) or not self.scenarios
                or any(s not in SCENARIOS for s in self.scenarios)
                or len(set(self.scenarios)) != len(self.scenarios)):
            raise ValueError('scenarios must be distinct declared synthetic designs')
        object.__setattr__(self, 'penalties', tuple(float(p) for p in self.penalties))
        object.__setattr__(self, 'scenarios', tuple(self.scenarios))


def _wilson(successes, count):
    if not count:
        return None
    z = NormalDist().inv_cdf(0.975)
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return {'lower': max(0.0, center - radius), 'upper': min(1.0, center + radius)}


def _summarize(rows):
    estimates = [row['estimate'] - row['truth'] for row in rows if row['estimate'] is not None]
    intervals = [row for row in rows if row['lower'] is not None]
    covered = sum(row['covered'] for row in intervals)
    return {'scenario': rows[0]['scenario'], 'penalty': rows[0]['penalty'], 'trials': len(rows),
            'point_estimates': len(estimates),
            'bias': float(np.mean(estimates)) if estimates else None,
            'bias_monte_carlo_se': float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
            if len(estimates) > 1 else None,
            'rmse': float(np.sqrt(np.mean(np.square(estimates)))) if estimates else None,
            'available_intervals': len(intervals), 'interval_availability': len(intervals) / len(rows),
            'covered_intervals': covered, 'covered_over_all_trials': covered / len(rows),
            'coverage_given_available': covered / len(intervals) if intervals else None,
            'coverage_wilson_95': _wilson(covered, len(intervals)),
            'mean_interval_width': float(np.mean([r['upper'] - r['lower'] for r in intervals]))
            if intervals else None, 'statuses': dict(Counter(row['status'] for row in rows)),
            'estimability_statuses': dict(Counter(row['estimability_status'] or 'not_evaluated' for row in rows))}


def run_calibration(spec):
    """Repeat synthetic datasets, pairing each dataset and bootstrap seed across priors.

    Both identity blocks use the declared penalty; linear context penalty is 1.
    Q and priors are fixed within each refitted game bootstrap. Failed trials
    remain in the table and availability denominator. Unexpected errors propagate.
    """
    if not isinstance(spec, CalibrationSpec):
        raise ValueError('run_calibration requires a CalibrationSpec')
    seeds = iter(np.random.SeedSequence(spec.seed).spawn(len(spec.scenarios) * spec.trials))
    rows, truths, generating_models = [], {}, {}
    for scenario in spec.scenarios:
        for trial in range(spec.trials):
            dataset_seed, bootstrap_seed = [int(s) for s in next(seeds).generate_state(2)]
            data = simulate_passing(games=spec.games, repeats=spec.repeats,
                                    seed=dataset_seed, scenario=scenario)
            truth = data['truth']
            truths[scenario] = truth
            generating_models[scenario] = data['generating_model']
            for penalty in spec.penalties:
                if data['train']['complete_pass'].n_unique() < 2:
                    result = {'status': 'single_outcome', 'interval': None}
                else:
                    result = bootstrap_contrast(
                        data['train'], data['population'], identity=truth['identity'],
                        left=truth['left'], right=truth['right'], fixed=truth['fixed'],
                        identity_penalties={'qb_id': penalty, 'receiver_id': penalty},
                        context='linear', context_penalty=1.0, draws=spec.draws,
                        seed=bootstrap_seed, level=spec.level, min_rows=1)
                interval = result['interval']
                estimability = result.get('estimability', {})
                rows.append({'scenario': scenario, 'trial': trial, 'penalty': penalty,
                             'dataset_seed': dataset_seed, 'bootstrap_seed': bootstrap_seed,
                             'rows': data['train'].height, 'truth': truth['difference'],
                             'status': result['status'],
                             'estimability_status': estimability.get('status'),
                             'log_odds_estimable': estimability.get('log_odds_contrast', {}).get('estimable'),
                             'probability_arms_estimable': estimability.get('probability_arms_estimable'),
                             'estimate': result.get('contrast', {}).get('difference'),
                             'lower': interval['lower'] if interval else None,
                             'upper': interval['upper'] if interval else None,
                             'covered': interval['lower'] <= truth['difference'] <= interval['upper']
                             if interval else None,
                             'context_failure_draws': result.get('resample_context', {}).get('draws_with_failures')})
    summaries = [_summarize([r for r in rows if r['scenario'] == scenario and r['penalty'] == penalty])
                 for scenario in spec.scenarios for penalty in spec.penalties]
    frame = pl.DataFrame(rows, schema_overrides={
        'estimate': pl.Float64, 'lower': pl.Float64, 'upper': pl.Float64,
        'covered': pl.Boolean, 'context_failure_draws': pl.Int64,
        'estimability_status': pl.String, 'log_odds_estimable': pl.Boolean,
        'probability_arms_estimable': pl.Boolean})
    return {'trials': frame, 'report': {
        'schema_version': 1, 'synthetic': True, 'real_player_coverage_verified': False,
        'specification': asdict(spec), 'truth_by_scenario': truths,
        'generating_models': generating_models, 'summaries': summaries,
        'interpretation': 'Known-DGP contrast coverage, not real-player validation. Coverage is conditional '
                          'on interval availability; unavailable trials remain in the availability denominator. '
                          'covered_over_all_trials counts only returned intervals containing truth. '
                          'Wilson 95% bounds describe Monte Carlo coverage uncertainty, not player uncertainty. '
                          'Estimability diagnostics describe the original design; not_evaluated is not a pass. '
                          'Fixed priors, fixed Q, linear context; no tuning or causal identification.'}}
