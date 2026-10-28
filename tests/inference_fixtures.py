"""Canonical row and frame builders shared by the inference check modules.

The two canonical tables have one shape, so the modules that exercise them build their rows from one
place. A fixture that drifts from the contract in five modules is five chances to test the wrong frame;
this file is checked by the same suite as everything else, because every one of those modules fails
without it.

The preparation checks keep their own play-by-play rows: those describe source partitions, which carry
many more columns than the canonical tables do.
"""
from __future__ import annotations

import polars as pl

PASSING_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, team="A", defteam="B", qb_id="qb1",
    receiver_id="wr1", complete_pass=1, air_yards=8.0, yardline_100=40.0, ydstogo=10.0, down=1,
    pass_location="left", position="WR",
)
RUSHING_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, team="A", defteam="B", rusher_id="rb1",
    rushing_yards=4.0, run_location="middle", run_gap="guard", yardline_100=40.0, ydstogo=10.0,
    down=1, position="RB",
)

PASSING_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, qb_id=pl.String, receiver_id=pl.String, complete_pass=pl.Int8,
    air_yards=pl.Float64, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    pass_location=pl.String, position=pl.String,
)
RUSHING_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, rusher_id=pl.String, rushing_yards=pl.Float64, run_location=pl.String,
    run_gap=pl.String, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    position=pl.String,
)


def passing_frame(*changes):
    """One canonical passing row per change, with a unique play key in every row."""
    rows = [{**PASSING_ROW, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=PASSING_SCHEMA)


def rushing_frame(*changes):
    """One canonical rushing row per change, with a unique play key in every row."""
    rows = [{**RUSHING_ROW, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=RUSHING_SCHEMA)


def empty_passing_frame(**extra):
    """A canonical passing frame with no rows, optionally with charted columns attached."""
    schema = {**PASSING_SCHEMA, **extra}
    return pl.DataFrame(schema=[(column, dtype) for column, dtype in schema.items()])


GAME_ROW = dict(
    season=2024, week=1, game_id="g1", game_date="2024-09-08", home_team="AAA", away_team="BBB",
    season_type="REG",
)
ROSTER_ROW = dict(
    season=2024, week=1, team="AAA", gsis_id="wr1", position="WR", status="ACT", game_type="REG",
)

GAME_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, game_date=pl.String, home_team=pl.String,
    away_team=pl.String, season_type=pl.String,
)
ROSTER_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, team=pl.String, gsis_id=pl.String, position=pl.String,
    status=pl.String, game_type=pl.String,
)


def game_frame(*changes):
    """One play-by-play game row per change, keyed by game."""
    rows = [{**GAME_ROW, "game_id": f"g{index + 1}", **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=GAME_SCHEMA)


def roster_frame(*changes):
    """One weekly roster row per change, keyed by player."""
    rows = [{**ROSTER_ROW, "gsis_id": f"wr{index + 1}", **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=ROSTER_SCHEMA)


PANEL_ROW = dict(
    season=2024, week=1, game_date="2024-09-08", game_id="g1", team="AAA", opponent="BBB",
    is_home=True, targets=3, candidates=2, candidate_targets=2, other_targets=1,
)
CANDIDATE_ROW = dict(
    season=2024, week=1, game_id="g1", team="AAA", candidate_index=0, player_id="wr1",
    position="WR", targets=2,
)

PANEL_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_date=pl.String, game_id=pl.String, team=pl.String,
    opponent=pl.String, is_home=pl.Boolean, targets=pl.Int32, candidates=pl.Int32,
    candidate_targets=pl.Int32, other_targets=pl.Int32,
)
CANDIDATE_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, team=pl.String, candidate_index=pl.Int32,
    player_id=pl.String, position=pl.String, targets=pl.Int32,
)


def panel_frame(*changes):
    """One team-game panel row per change, with the keys the caller names."""
    return pl.DataFrame([{**PANEL_ROW, **change} for change in changes], schema_overrides=PANEL_SCHEMA)


def candidate_frame(*changes):
    """One candidate row per change, with the keys the caller names."""
    return pl.DataFrame([{**CANDIDATE_ROW, **change} for change in changes],
                        schema_overrides=CANDIDATE_SCHEMA)
