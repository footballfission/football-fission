"""Seeded synthetic targets with known contrasts, never real-player evidence."""
from __future__ import annotations

import numpy as np
import polars as pl

from .contracts import PASSING_CONTRACT, validate

SCENARIOS = ('crossed', 'nested', 'depth_confounded')
QB_EFFECTS = {'synthetic_q1': 0.25, 'synthetic_q2': -0.25}
RECEIVER_EFFECTS = {'synthetic_r1': 0.6, 'synthetic_r2': -0.6}


def _probability(qb, receiver, depth):
    logit = 0.4 + QB_EFFECTS[qb] + RECEIVER_EFFECTS[receiver] - 0.04 * (depth - 10)
    return float(1 / (1 + np.exp(-logit)))


def simulate_passing(*, games=20, repeats=4, seed=0, scenario='crossed'):
    """Generate Bernoulli targets under a fully declared additive logistic model.

    Crossed data contain every QB/receiver/depth pairing in every game. Nested
    data contain only q1/r1 and q2/r2. Depth-confounded data cross partners but
    deterministically assign depth 6 to r1 and 9 to r2, inside one coarse cell.
    Q is a fixed equal-weight depth mix, not selected using simulated outcomes.
    Games have no latent effects; this isolates estimator bias and design limits.
    """
    if (type(games) is not int or games < 2 or type(repeats) is not int or repeats < 1
            or type(seed) is not int or seed < 0 or scenario not in SCENARIOS):
        raise ValueError('simulation needs games >= 2, repeats >= 1, nonnegative seed and a known scenario')
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(games):
        for qi, qb in enumerate(QB_EFFECTS):
            for ri, receiver in enumerate(RECEIVER_EFFECTS):
                if scenario == 'nested' and qi != ri:
                    continue
                depths = (6.0 if ri == 0 else 9.0,) if scenario == 'depth_confounded' else (5.0, 15.0)
                for depth in depths:
                    for _ in range(repeats):
                        rows.append(dict(
                            season=2024, week=game + 1, game_id=f'synthetic_game_{game}',
                            play_id=len(rows) + 1, team='synthetic_offense', defteam='synthetic_defense',
                            qb_id=qb, receiver_id=receiver,
                            complete_pass=int(rng.random() < _probability(qb, receiver, depth)),
                            air_yards=depth, yardline_100=40.0, ydstogo=10.0, down=1,
                            pass_location='middle', position='WR'))
    train = pl.DataFrame(rows, schema_overrides={'complete_pass': pl.Int8})
    # Select an observed row at each reference depth; its observed outcome is irrelevant.
    depths = (6.0, 9.0) if scenario == 'depth_confounded' else (5.0, 15.0)
    population = pl.concat([train.filter(pl.col('air_yards') == depth).head(1) for depth in depths])
    fixed_qb = 'synthetic_q1'
    left, right = 'synthetic_r1', 'synthetic_r2'
    left_rate = float(np.mean([_probability(fixed_qb, left, depth) for depth in depths]))
    right_rate = float(np.mean([_probability(fixed_qb, right, depth) for depth in depths]))
    validate(train, PASSING_CONTRACT)
    validate(population, PASSING_CONTRACT)
    return {'synthetic': True, 'scenario': scenario, 'train': train, 'population': population,
            'truth': {'identity': 'receiver_id', 'left': left, 'right': right,
                      'fixed': {'qb_id': fixed_qb}, 'scale': 'probability',
                      'left_prediction': left_rate, 'right_prediction': right_rate,
                      'difference': left_rate - right_rate},
            'generating_model': {'intercept': 0.4, 'depth_center': 10.0, 'depth_slope': -0.04,
                                 'qb_effects': dict(QB_EFFECTS),
                                 'receiver_effects': dict(RECEIVER_EFFECTS),
                                 'outcomes': 'independent Bernoulli; no latent game effects',
                                 'reference': 'equal-weight fixed depth mix with synthetic_q1'}}
