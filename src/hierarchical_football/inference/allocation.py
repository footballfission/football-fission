"""Joint target allocation conditional on supplied rosters, shares and team draws."""
from __future__ import annotations

import numpy as np
import polars as pl

from .panels import CANDIDATE_KEYS, OTHER, TEAM_GAME_KEYS
from .prepare import require_columns, require_unique
from .splits import LABEL_COLUMNS


def _probabilities(values, length, name):
    result = np.asarray(values, dtype=float)
    if (result.shape != (length,) or np.any(~np.isfinite(result))
            or np.any(result < 0) or np.any(result > 1)):
        raise ValueError(f'{name} needs one finite probability in [0,1] per input row')
    return result


def sample_target_allocation(panel, candidates, team_draws, *, shares, other_shares, seed=0):
    """Allocate each supplied team/draw total jointly, preserving input row order.

    Shares align with candidate rows; OTHER shares align with panel rows. Each
    pool must sum to one within 1e-10, with numerical roundoff normalized before
    sampling. Internal key sorting makes seeded results invariant to arbitrary
    input row order. Draw columns retain their supplied identity. A zero-share
    player always gets zero. With no candidates, all probability must be OTHER.
    No share, availability or volume model is fitted here.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError('allocation seed must be a nonnegative integer')
    require_columns(panel, TEAM_GAME_KEYS, 'allocation team games')
    require_columns(candidates, (*CANDIDATE_KEYS, 'player_id'), 'allocation candidates')
    for frame in (panel, candidates):
        if set(frame.columns) & set(LABEL_COLUMNS):
            raise ValueError('allocation prediction inputs must not contain outcome labels')
        for column in TEAM_GAME_KEYS:
            if (frame.schema[column] not in (pl.String, pl.Categorical, pl.Enum)
                    or frame[column].is_null().any()
                    or (frame[column].cast(pl.String).str.strip_chars().str.len_chars() == 0).any()):
                raise ValueError('allocation team/game keys must be nonempty text')
    require_unique(panel, TEAM_GAME_KEYS, 'allocation team games')
    require_unique(candidates, CANDIDATE_KEYS, 'allocation candidates')
    require_unique(candidates, (*TEAM_GAME_KEYS, 'player_id'), 'allocation candidate players')
    if (not candidates.schema['candidate_index'].is_integer()
            or (candidates['candidate_index'] < 0).any()):
        raise ValueError('allocation candidate_index must be nonnegative integers')
    if (candidates.schema['player_id'] not in (pl.String, pl.Categorical, pl.Enum)
            or candidates['player_id'].is_null().any()
            or (candidates['player_id'].cast(pl.String).str.strip_chars().str.len_chars() == 0).any()
            or (candidates['player_id'] == OTHER).any()):
        raise ValueError('allocation player IDs must be nonempty text distinct from OTHER')
    totals = np.asarray(team_draws)
    if (not panel.height or totals.ndim != 2 or totals.shape[0] != panel.height or totals.shape[1] < 1
            or not np.issubdtype(totals.dtype, np.integer) or np.any(totals < 0)
            or int(totals.max()) > np.iinfo(np.int64).max):
        raise ValueError('allocation needs a nonnegative integer matrix of team rows by draws')
    totals = totals.astype(np.int64, copy=False)
    probabilities = _probabilities(shares, candidates.height, 'candidate shares')
    residual = _probabilities(other_shares, panel.height, 'OTHER shares')
    pools = {tuple(key): index for index, key in enumerate(panel.select(TEAM_GAME_KEYS).iter_rows())}
    positions = {key: [] for key in pools}
    players = candidates['player_id'].to_list()
    for index, key in enumerate(candidates.select(TEAM_GAME_KEYS).iter_rows()):
        if tuple(key) not in pools:
            raise ValueError('allocation candidate has no matching team game')
        positions[tuple(key)].append(index)
    candidate_samples = np.zeros((candidates.height, totals.shape[1]), dtype=np.int64)
    other_samples = np.zeros_like(totals)
    rng = np.random.default_rng(seed)
    for key in sorted(pools):
        row = pools[key]
        members = sorted(positions[key], key=lambda index: players[index])
        p = np.array([*(probabilities[index] for index in members), residual[row]])
        total_probability = float(p.sum())
        if not np.isclose(total_probability, 1.0, rtol=0, atol=1e-10):
            raise ValueError('allocation candidate and OTHER probabilities must sum to one for each team game')
        p = p / total_probability
        sampled = rng.multinomial(totals[row], p)
        candidate_samples[members] = sampled[:, :-1].T
        other_samples[row] = sampled[:, -1]
    return {'candidate_samples': candidate_samples, 'other_samples': other_samples,
            'candidate_keys': candidates.select(*CANDIDATE_KEYS, 'player_id'),
            'team_game_keys': panel.select(TEAM_GAME_KEYS), 'seed': seed,
            'condition': 'supplied_roster_probabilities_and_team_target_draws',
            'note': 'fixed-share multinomial allocation; no availability, share-parameter or volume '
                    'uncertainty beyond supplied inputs; no pregame information certification'}
