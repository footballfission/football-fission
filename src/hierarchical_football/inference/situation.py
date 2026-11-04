"""Audited joins of observed pre-play situation onto canonical passing targets."""
from __future__ import annotations

import polars as pl

from .contracts import PASSING_CONTRACT, PLAY_KEYS, validate
from .prepare import require_columns, require_unique

SITUATION_FLAGS = ('shotgun', 'no_huddle')
SITUATION_CONTINUOUS = ('score_differential', 'game_seconds_remaining')
SITUATION_COLUMNS = SITUATION_FLAGS + SITUATION_CONTINUOUS
SITUATION_SOURCE_COLUMNS = PLAY_KEYS + ('season', 'posteam') + SITUATION_COLUMNS
FIELD_DEFINITIONS = 'https://nflfastr.com/reference/fast_scraper.html'
_ORDER, _SEASON, _TEAM, _MATCH = ('__situation_order', '__situation_season',
                                 '__situation_team', '__situation_match')


def attach_passing_situation(table, pbp):
    """Return unchanged target rows/outcomes with four explicitly selected context fields.

    Source keys must be unique integral play IDs. Every target must match the
    source season and offense. Null context is preserved and reported; observed
    nonfinite values, invalid binary flags and negative clocks are refused.
    No source outcome, post-play score or fitted probability is read.
    """
    validate(table, PASSING_CONTRACT)
    require_columns(pbp, SITUATION_SOURCE_COLUMNS, 'situation play-by-play')
    existing = set(table.columns) & set((*SITUATION_COLUMNS, _ORDER, _SEASON, _TEAM, _MATCH))
    if existing:
        raise ValueError(f'situation join refuses existing columns {sorted(existing)}')
    require_unique(pbp, PLAY_KEYS, 'situation play-by-play')
    ids = pbp['play_id']
    if not ids.dtype.is_numeric() or not ids.is_finite().all() or (ids != ids.floor()).any():
        raise ValueError('situation play_id must be finite integral numeric keys')
    for column in SITUATION_COLUMNS:
        dtype = pbp.schema[column]
        if not (dtype.is_numeric() or dtype in (pl.Boolean, pl.Null)):
            raise ValueError(f'situation {column} must be numeric or null')
    source = pbp.select(
        pl.col('game_id').cast(pl.String), pl.col('play_id').cast(pl.Int64),
        pl.col('season').alias(_SEASON), pl.col('posteam').alias(_TEAM),
        *(pl.col(column).cast(pl.Float64) for column in SITUATION_COLUMNS),
        pl.lit(True).alias(_MATCH))
    require_unique(source, PLAY_KEYS, 'normalized situation play-by-play')
    joined = table.with_row_index(_ORDER).join(source, on=list(PLAY_KEYS), how='left', validate='1:1')
    matches = (pl.col(_MATCH).fill_null(False) & (pl.col(_SEASON) == pl.col('season')).fill_null(False)
               & (pl.col(_TEAM) == pl.col('team')).fill_null(False))
    if joined.filter(~matches).height:
        raise ValueError('situation source must match every target key, season and offense')
    for column in SITUATION_COLUMNS:
        values = joined[column].drop_nulls()
        if not values.is_finite().all():
            raise ValueError(f'situation {column} has nonfinite observed values')
        if column in SITUATION_FLAGS and not values.is_in([0.0, 1.0]).all():
            raise ValueError(f'situation {column} must be binary or null')
    if (joined['game_seconds_remaining'].drop_nulls() < 0).any():
        raise ValueError('situation game_seconds_remaining cannot be negative')
    result = joined.sort(_ORDER).select(
        *table.columns,
        *(pl.col(column).cast(pl.Int8) for column in SITUATION_FLAGS),
        *SITUATION_CONTINUOUS)
    return result, {'source_rows': pbp.height, 'target_rows': table.height, 'matched_rows': result.height,
                    'columns': list(SITUATION_COLUMNS),
                    'missing_context': {column: result[column].null_count() for column in SITUATION_COLUMNS},
                    'field_definitions': FIELD_DEFINITIONS,
                    'note': 'observed pre-play situation; no outcome or post-play score added; no imputation'}
