"""Training-only position-pooled rates for proportional target redistribution."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import polars as pl

from .panels import CANDIDATE_KEYS, CANDIDATE_POSITIONS, OTHER, TEAM_GAME_KEYS, conservation
from .prepare import require_columns, require_unique
from .splits import LABEL_COLUMNS


def _roster_keys(panel, candidates):
    require_columns(panel, (*TEAM_GAME_KEYS, 'season', 'week'), 'allocation baseline panel')
    require_columns(candidates, (*CANDIDATE_KEYS, 'season', 'week', 'player_id', 'position'),
                    'allocation baseline candidates')
    if not panel.height:
        raise ValueError('allocation baseline needs team games')
    require_unique(panel, TEAM_GAME_KEYS, 'allocation baseline panel')
    require_unique(panel, ('season', 'week', 'team'), 'allocation baseline team weeks')
    require_unique(candidates, CANDIDATE_KEYS, 'allocation baseline candidates')
    require_unique(candidates, (*TEAM_GAME_KEYS, 'player_id'), 'allocation baseline players')
    for frame in (panel, candidates):
        for name in ('season', 'week'):
            if not frame.schema[name].is_integer() or frame[name].is_null().any() or (frame[name] < 1).any():
                raise ValueError('allocation baseline needs positive integer season/week metadata')
        for name in TEAM_GAME_KEYS:
            if any(not isinstance(value, str) or not value.strip() for value in frame[name]):
                raise ValueError('allocation baseline team/game keys must be nonempty text')
    if (not candidates.schema['candidate_index'].is_integer()
            or (candidates['candidate_index'] < 0).any()
            or any(not isinstance(value, str) or not value.strip() or value == OTHER
                   for value in candidates['player_id'])):
        raise ValueError('allocation baseline needs valid candidate indices and player IDs')
    if set(candidates['position']) - set(CANDIDATE_POSITIONS):
        raise ValueError('allocation baseline candidates must have WR, RB or TE positions')
    calendar = panel.select('game_id', 'season', 'week').unique()
    if calendar['game_id'].n_unique() != calendar.height:
        raise ValueError('allocation baseline game metadata disagree across offenses')
    matched = candidates.select(*TEAM_GAME_KEYS, 'season', 'week').join(
        panel.select(*TEAM_GAME_KEYS, pl.col('season').alias('_panel_season'),
                     pl.col('week').alias('_panel_week')), on=list(TEAM_GAME_KEYS), how='left', validate='m:1')
    if matched.filter((pl.col('season') != pl.col('_panel_season')).fill_null(True)
                      | (pl.col('week') != pl.col('_panel_week')).fill_null(True)).height:
        raise ValueError('allocation baseline candidates must match panel keys and week metadata')


def _counts(frame, names):
    require_columns(frame, names, 'allocation baseline counts')
    for name in names:
        if not frame.schema[name].is_numeric():
            raise ValueError('allocation baseline counts must be nonnegative finite integers')
        values = frame[name].cast(pl.Float64).to_numpy()
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError('allocation baseline counts must be nonnegative finite integers')


@dataclass(frozen=True)
class FittedProportionalAllocation:
    prior_exposure: float
    global_share: float
    position_shares: dict
    player_counts: dict
    other_rate: float
    train_through: tuple
    training_games: frozenset
    training_summary: dict

    def predict(self, panel, candidates, *, include_player_history=True):
        """Normalize historical rates over the supplied future roster plus OTHER."""
        if type(include_player_history) is not bool:
            raise ValueError('include_player_history must be boolean')
        if (set(panel.columns) | set(candidates.columns)) & set(LABEL_COLUMNS):
            raise ValueError('allocation baseline prediction inputs must be label-free')
        _roster_keys(panel, candidates)
        if (any(tuple(row) <= self.train_through for row in panel.select('season', 'week').iter_rows())
                or set(panel['game_id']) & self.training_games):
            raise ValueError('allocation baseline prediction must follow training weeks without reused games')
        pools = {tuple(key): i for i, key in enumerate(panel.select(TEAM_GAME_KEYS).iter_rows())}
        rates, memberships = [], []
        pool_rates = [[self.other_rate] for _ in range(panel.height)]
        unseen_player_rows = unseen_positions = 0
        unseen_players = set()
        for game, team, player, position in candidates.select(*TEAM_GAME_KEYS, 'player_id', 'position').iter_rows():
            prior = self.position_shares.get(position, self.global_share)
            targets, exposure = (self.player_counts.get(player, (0.0, 0.0))
                                 if include_player_history else (0.0, 0.0))
            rates.append((targets + self.prior_exposure * prior) / (exposure + self.prior_exposure))
            memberships.append(pools[(game, team)])
            pool_rates[memberships[-1]].append(rates[-1])
            if player not in self.player_counts:
                unseen_players.add(player)
                unseen_player_rows += 1
            unseen_positions += int(position not in self.position_shares)
        rates = np.asarray(rates, dtype=float)
        memberships = np.asarray(memberships, dtype=int)
        # Stable summation preserves keyed probabilities under arbitrary roster order.
        denominators = np.array([math.fsum(pool) for pool in pool_rates])
        return {'shares': rates / denominators[memberships],
                'other_shares': self.other_rate / denominators,
                'candidate_keys': candidates.select(*CANDIDATE_KEYS, 'player_id'),
                'team_game_keys': panel.select(TEAM_GAME_KEYS),
                'unseen_players': len(unseen_players), 'unseen_player_rows': unseen_player_rows,
                'unseen_position_rows': unseen_positions,
                'include_player_history': include_player_history,
                'condition': 'supplied_roster_with_training_only_target_exposure',
                'note': 'proportional redistribution baseline; no learned availability or substitution '
                        'response; weekly ACT status is not pregame timestamp certification'}

    def report(self):
        return {'model': 'position_pooled_proportional_allocation', 'prior_exposure': self.prior_exposure,
                'base_exposure': 1.0, 'global_share': self.global_share,
                'position_shares': dict(self.position_shares), 'other_rate': self.other_rate,
                'players': len(self.player_counts), 'training': dict(self.training_summary),
                'train_through': {'season': self.train_through[0], 'week': self.train_through[1]},
                'interpretation': 'historical target rates per eligible team-target exposure; '
                                  'fixed-rate roster scenario baseline, not player skill or causal replacement credit'}


def fit_proportional_allocation(panel, candidates, *, prior_exposure=100.0):
    """Fit positive target/exposure rates using only the declared training roster games.

    Exposure is team targets in games where a player was an eligible candidate.
    The global mean has one exposure unit toward an equal-slot prior; position
    and player rates shrink by prior_exposure toward training-only pooled means.
    OTHER has one exposure unit toward the same equal-slot prior.
    """
    if (isinstance(prior_exposure, bool) or not isinstance(prior_exposure, (int, float))
            or not math.isfinite(prior_exposure) or prior_exposure <= 0):
        raise ValueError('prior_exposure must be finite and positive')
    _roster_keys(panel, candidates)
    _counts(panel, ('targets', 'candidates', 'candidate_targets', 'other_targets'))
    _counts(candidates, ('targets',))
    summary = conservation(panel, candidates)
    exposed = candidates.join(panel.select(*TEAM_GAME_KEYS, pl.col('targets').alias('_exposure')),
                              on=list(TEAM_GAME_KEYS), how='left', validate='m:1')
    player_counts, position_counts = {}, {}
    for player, position, targets, exposure in exposed.select('player_id', 'position', 'targets', '_exposure').iter_rows():
        for key, store in ((player, player_counts), (position, position_counts)):
            previous_targets, previous_exposure = store.get(key, (0.0, 0.0))
            store[key] = (previous_targets + targets, previous_exposure + exposure)
    total_exposure = sum(exposure for _, exposure in player_counts.values())
    if not total_exposure:
        raise ValueError('allocation baseline needs positive eligible team-target exposure')
    uniform_prior = 1 / (candidates.height / panel.height + 1)
    global_share = (sum(targets for targets, _ in player_counts.values()) + uniform_prior) / (total_exposure + 1)
    position_shares = {position: (targets + prior_exposure * global_share) / (exposure + prior_exposure)
                       for position, (targets, exposure) in sorted(position_counts.items()) if exposure > 0}
    other_rate = (summary['other_targets'] + uniform_prior) / (summary['targets'] + 1)
    cutoff = max(tuple(row) for row in panel.select('season', 'week').iter_rows())
    return FittedProportionalAllocation(float(prior_exposure), float(global_share), position_shares,
                                        player_counts, float(other_rate), cutoff,
                                        frozenset(panel['game_id']), summary)
