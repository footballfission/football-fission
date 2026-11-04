"""Proper conditional target-count scores, including residual and impossible events."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.special import gammaln, xlogy

from .allocation import _probabilities
from .panels import CANDIDATE_KEYS, TEAM_GAME_KEYS, conservation
from .prepare import require_columns, require_unique
from .splits import LABEL_COLUMNS

PLAYER_KEYS = (*CANDIDATE_KEYS, 'player_id')


def _matching_keys(predicted, observed, keys, name):
    for frame in (predicted, observed):
        require_columns(frame, keys, name)
        require_unique(frame, keys, name)
        for column in keys:
            if column != 'candidate_index' and any(not isinstance(v, str) or not v.strip() for v in frame[column]):
                raise ValueError(f'{name} needs nonempty text identities')
    if (predicted.select(keys).join(observed.select(keys), on=list(keys), how='anti').height
            or observed.select(keys).join(predicted.select(keys), on=list(keys), how='anti').height):
        raise ValueError(f'{name} prediction and observation keys do not match exactly')


def _observed_counts(frame, columns):
    require_columns(frame, columns, 'allocation score counts')
    for column in columns:
        if not frame.schema[column].is_numeric():
            raise ValueError('allocation score counts must be finite nonnegative integers')
        values = frame[column].cast(pl.Float64).to_numpy()
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError('allocation score counts must be finite nonnegative integers')


def allocation_log_scores(prediction, observed_panel, observed_candidates):
    """Score candidate and OTHER count mass conditional on each observed team total.

    Prediction keys must exactly match observed identities; row order is not
    meaningful. Proper multinomial mass includes the count coefficient. An
    observed positive count at probability zero has infinite loss, retained in
    the frame. Its report summaries are null with explicit infinite-loss status,
    rather than clipping probabilities or averaging only finite games.
    """
    required = {'candidate_keys', 'team_game_keys', 'shares', 'other_shares'}
    if not isinstance(prediction, dict) or not required <= set(prediction):
        raise ValueError('allocation scores need keyed candidate and OTHER probabilities')
    team_keys, player_keys = prediction['team_game_keys'], prediction['candidate_keys']
    if not team_keys.height:
        raise ValueError('allocation scores need at least one team game')
    for frame in (team_keys, player_keys):
        if set(frame.columns) & set(LABEL_COLUMNS):
            raise ValueError('allocation prediction keys must not contain outcome labels')
    _matching_keys(team_keys, observed_panel, TEAM_GAME_KEYS, 'allocation score team games')
    _matching_keys(player_keys, observed_candidates, PLAYER_KEYS, 'allocation score players')
    for frame in (player_keys, observed_candidates):
        require_unique(frame, CANDIDATE_KEYS, 'allocation score candidate indices')
        require_unique(frame, (*TEAM_GAME_KEYS, 'player_id'), 'allocation score candidate players')
    _observed_counts(observed_panel, ('targets', 'candidates', 'candidate_targets', 'other_targets'))
    _observed_counts(observed_candidates, ('targets',))
    conservation(observed_panel, observed_candidates)
    shares = _probabilities(prediction['shares'], player_keys.height, 'allocation score candidate shares')
    other = _probabilities(prediction['other_shares'], team_keys.height, 'allocation score OTHER shares')
    joined = player_keys.select(PLAYER_KEYS).with_columns(pl.Series('_probability', shares)).join(
        observed_candidates.select(*PLAYER_KEYS, pl.col('targets').alias('_count')),
        on=list(PLAYER_KEYS), how='left', validate='1:1')
    groups = {}
    for game, team, player, probability, count in joined.select(
            *TEAM_GAME_KEYS, 'player_id', '_probability', '_count').iter_rows():
        groups.setdefault((game, team), []).append((player, probability, count))
    observations = {tuple(row[:2]): (int(row[2]), int(row[3])) for row in
                    observed_panel.select(*TEAM_GAME_KEYS, 'targets', 'other_targets').iter_rows()}
    rows = []
    for index, key in enumerate(team_keys.select(TEAM_GAME_KEYS).iter_rows()):
        members = sorted(groups.get(tuple(key), []))
        probabilities = np.array([*(value[1] for value in members), other[index]])
        total_probability = math.fsum(probabilities)
        if not np.isclose(total_probability, 1.0, rtol=0, atol=1e-10):
            raise ValueError('allocation score candidate and OTHER probabilities must sum to one')
        probabilities = probabilities / total_probability
        total, residual_count = observations[tuple(key)]
        counts = np.array([*(value[2] for value in members), residual_count], dtype=float)
        impossible = bool(np.any((counts > 0) & (probabilities == 0)))
        cross_entropy = -math.fsum(xlogy(counts, probabilities))
        if impossible:
            nll = float('inf')
        else:
            coefficient = float(gammaln(total + 1) - math.fsum(gammaln(counts + 1)))
            nll = cross_entropy - coefficient
            if not math.isfinite(nll):
                raise ValueError('allocation counts exceed finite numerical score support')
            nll = max(0.0, nll)  # A deterministic mass can round slightly above one.
        rows.append({'game_id': key[0], 'team': key[1], 'targets': total,
                     'negative_log_likelihood': float(nll), 'target_log_loss_sum': float(cross_entropy),
                     'zero_probability_event': impossible})
    failures = sum(row['zero_probability_event'] for row in rows)
    targets = sum(row['targets'] for row in rows)
    mean_nll = math.fsum(row['negative_log_likelihood'] for row in rows) / len(rows) if not failures else None
    target_loss = (math.fsum(row['target_log_loss_sum'] for row in rows) / targets
                   if targets and not failures else None)
    return {'scores': pl.DataFrame(rows), 'report': {
        'status': 'infinite_loss' if failures else 'ok', 'team_games': len(rows), 'targets': targets,
        'zero_probability_games': failures, 'mean_negative_log_likelihood': mean_nll,
        'target_weighted_log_loss': target_loss,
        'interpretation': 'Proper multinomial count mass conditional on observed team target totals and '
                          'the supplied roster; no volume/availability evaluation or pregame certification. '
                          'Infinite-loss games remain in the score frame and force null aggregate scores; '
                          'no probability clipping or finite-game-only averaging.'}}
