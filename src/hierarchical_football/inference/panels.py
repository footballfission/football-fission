"""Team-game receiving panels: the unit of observation and the candidates inside it.

Two models in the plan read one object. The volume model asks how many targets a team's offense
generates in a game; the allocation model asks how that total splits across the receivers who were
available. This module materializes the object: one row per team game with its observed target total,
and one row per available candidate with the targets he drew.

A candidate is a player the weekly roster lists as active for that team and week with a position in
`CANDIDATE_POSITIONS`. Everyone else who drew a target lands in the `OTHER` residual: a lineman
reporting eligible, a quarterback catching his own deflection, a receiver the roster feed does not
place on the team that week. `OTHER` is a category in the allocation model, not a rounding error, so
it is measured and never silently dropped.

Conservation is the panel's central invariant. For every team game the candidate targets plus the
`OTHER` targets equal the team's target total, and a build that cannot satisfy it raises instead of
writing a frame that lost a target. The plan's blocking gate is exactly this equality.

The candidate index orders players by player id. It never orders them by targets, because an index
sorted by the label would carry the answer in the row key and would change whenever a player's week
changed.
"""
from __future__ import annotations

import polars as pl

from .contracts import PASSING_CONTRACT, UNKNOWN, canonical_position, validate
from .prepare import require_columns, require_unique, roster_positions

TEAM_GAME_KEYS = ("game_id", "team")
CANDIDATE_KEYS = ("game_id", "team", "candidate_index")

# Positions the allocation model covers. A tight end lined up at fullback is still a tight end, but
# a fullback is a different role with its own usage pattern, so the plan's three positions are named.
CANDIDATE_POSITIONS = ("WR", "RB", "TE")
OTHER = "OTHER"
ACTIVE = "ACT"

GAME_SOURCE_COLUMNS = ("season", "week", "game_id", "game_date", "home_team", "away_team", "season_type")
GAME_COLUMNS = GAME_SOURCE_COLUMNS[:-1]
ROSTER_STATUS_COLUMN = "status"

# The team-game row and its derived split. `candidates`, `candidate_targets`, and `other_targets` are
# arithmetic on the rows below, kept beside them because conservation is the panel's gate: a reader of
# the artifact can check the equality without re-deriving it.
PANEL_COLUMNS = ("season", "week", "game_date", "game_id", "team", "opponent", "is_home", "targets")
PANEL_SCHEMA = PANEL_COLUMNS + ("candidates", "candidate_targets", "other_targets")
CANDIDATE_COLUMNS = ("season", "week", "game_id", "team", "candidate_index", "player_id",
                    "position", "targets")


def game_context(pbp):
    """One row per regular-season game, with its date and its two teams.

    Postseason games are dropped because the plan models the regular season. A game whose plays
    disagree about their week or date is refused, because the split orders team games by week and a
    game that appears in two weeks would be counted twice.
    """
    require_columns(pbp, GAME_SOURCE_COLUMNS, "game_context")
    regular = pbp.filter(pl.col("season_type").cast(pl.String).str.to_uppercase().eq("REG"))
    games = regular.select(*GAME_COLUMNS).unique()
    require_unique(games, ("game_id",), "games")
    return games


def team_games(passing, games):
    """One row per team game: the targets its offense recorded and who it played.

    `passing` is the canonical passing table, so one row is one recorded target. A team game whose
    plays are in the cache but whose offense recorded no in-scope target keeps a zero, because the
    cache and the total come from the same feed: there is no game here whose outcome is unknown.
    A team game the cache does not contain gets no row at all.
    """
    validate(passing, PASSING_CONTRACT)
    require_columns(games, GAME_COLUMNS, "team_games")
    require_unique(games, ("game_id",), "games")

    home = games.select(
        "season", "week", "game_date", "game_id",
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
    ).with_columns(pl.lit(True).alias("is_home"))
    away = games.select(
        "season", "week", "game_date", "game_id",
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
    ).with_columns(pl.lit(False).alias("is_home"))
    rows = pl.concat([home, away], how="vertical")
    require_unique(rows, ("season", "week", "team"), "team games")

    counts = passing.group_by("game_id", "team").agg(
        pl.len().alias("targets"),
        pl.col("season").first().alias("passing_season"),
        pl.col("week").first().alias("passing_week"),
        pl.col("season").n_unique().alias("passing_seasons"),
        pl.col("week").n_unique().alias("passing_weeks"),
    )
    mixed = counts.filter((pl.col("passing_seasons") > 1) | (pl.col("passing_weeks") > 1))
    if mixed.height:
        first = mixed.row(0, named=True)
        raise ValueError(
            "team_games: the passing table puts one team game in more than one week: "
            f"game {first['game_id']}, team {first['team']}"
        )
    unmatched = counts.join(rows.select("game_id", "team"), on=["game_id", "team"], how="anti")
    if unmatched.height:
        first = unmatched.row(0, named=True)
        raise ValueError(
            "team_games: the passing table names a team game the games feed does not contain: "
            f"game {first['game_id']}, team {first['team']} ({unmatched.height} rows)"
        )

    panel = rows.join(counts, on=["game_id", "team"], how="left")
    disagreement = panel.filter(
        pl.col("targets").is_not_null()
        & ((pl.col("passing_season") != pl.col("season")) | (pl.col("passing_week") != pl.col("week")))
    )
    if disagreement.height:
        first = disagreement.row(0, named=True)
        raise ValueError(
            "team_games: the passing table and the games feed disagree about when a team game was "
            f"played: game {first['game_id']}, team {first['team']}"
        )
    panel = panel.with_columns(pl.col("targets").fill_null(0).cast(pl.Int32)).drop(
        "passing_season", "passing_week", "passing_seasons", "passing_weeks"
    )
    return panel.select(PANEL_COLUMNS)


def available_players(rosters):
    """One row per player the weekly roster lists as active, with a canonical position.

    Availability is the weekly roster's active status, so practice-squad, reserve, and inactive rows
    are not candidates. Those players cannot draw a target, and treating a reserve as an available
    receiver would put a zero in a pool he was never part of. Only the three modelled positions are
    kept; every other position is a candidate for `OTHER`, not for the allocator.
    """
    require_columns(rosters, (ROSTER_STATUS_COLUMN,), "available players")
    active = rosters.filter(pl.col(ROSTER_STATUS_COLUMN).cast(pl.String).str.to_uppercase().eq(ACTIVE))
    players = roster_positions(active)
    return players.filter(pl.col("position").is_in(list(CANDIDATE_POSITIONS)))


def candidates(passing, rosters, panel):
    """One row per team game and available candidate, with the targets he drew.

    A candidate who drew no target keeps an observed zero: he was available and the feed recorded
    every target in the game, so the zero is a measurement rather than a missing value. A candidate
    whose team game is absent from the panel is not returned, because the panel is the unit of
    observation and a candidate without one belongs to a game the cache does not contain.
    """
    validate(passing, PASSING_CONTRACT)
    require_columns(panel, TEAM_GAME_KEYS + ("season", "week"), "candidates")
    require_unique(panel, ("season", "week", "team"), "team games")

    players = available_players(rosters)
    keyed = panel.select("season", "week", "game_id", "team")
    joined = keyed.join(players, on=["season", "week", "team"], how="inner")
    targets = passing.group_by("season", "week", "game_id", "team", "receiver_id").agg(
        pl.len().alias("targets")
    ).rename({"receiver_id": "player_id"})
    joined = joined.join(targets, on=["season", "week", "game_id", "team", "player_id"], how="left")
    joined = joined.with_columns(pl.col("targets").fill_null(0).cast(pl.Int32))
    joined = joined.sort("game_id", "team", "player_id")
    joined = joined.with_columns(
        (pl.col("player_id").cum_count().over("game_id", "team") - 1).cast(pl.Int32).alias("candidate_index")
    )
    return joined.select(CANDIDATE_COLUMNS)


def conservation(panel, candidates_frame):
    """The team target total split into candidate and OTHER targets, with every violation named.

    Returns the rows checked, the targets they carry, and the targets the candidates explain. A
    negative `other_targets`, a total the candidates overdraw, or a candidate count the panel does
    not agree with all raise, because each means the roster join and the play join disagree about
    who was on the field.
    """
    require_columns(panel, ("game_id", "team", "targets", "candidates", "candidate_targets", "other_targets"),
                    "conservation")
    require_columns(candidates_frame, TEAM_GAME_KEYS + ("targets",), "conservation")
    require_unique(panel, TEAM_GAME_KEYS, 'conservation team games')
    orphan = candidates_frame.join(panel.select(TEAM_GAME_KEYS), on=list(TEAM_GAME_KEYS), how='anti')
    if orphan.height:
        raise ValueError('conservation: candidate group has no matching team game')
    counted = candidates_frame.group_by("game_id", "team").agg(
        pl.len().alias("_observed_candidates"), pl.col("targets").sum().alias("_observed_candidate_targets")
    )
    merged = panel.join(counted, on=["game_id", "team"], how="left", validate='1:1')
    merged = merged.with_columns(
        pl.col("_observed_candidates").fill_null(0),
        pl.col("_observed_candidate_targets").fill_null(0),
    )
    mismatch = merged.filter(
        (pl.col('candidates') != pl.col('_observed_candidates')).fill_null(True)
        | (pl.col('candidate_targets') != pl.col('_observed_candidate_targets')).fill_null(True))
    if mismatch.height:
        raise ValueError('conservation: stored candidate counts or targets disagree with candidate rows')
    negative = merged.filter(pl.col("other_targets") < 0)
    if negative.height:
        first = negative.row(0, named=True)
        raise ValueError(
            "conservation: candidates explain more targets than the team recorded: "
            f"game {first['game_id']}, team {first['team']}, targets {first['targets']}, "
            f"candidates {first['candidate_targets']}"
        )
    broken = merged.filter(pl.col("targets") != pl.col("candidate_targets") + pl.col("other_targets"))
    if broken.height:
        first = broken.row(0, named=True)
        raise ValueError(
            "conservation: candidate and OTHER targets do not add up to the team total: "
            f"game {first['game_id']}, team {first['team']}, targets {first['targets']}, "
            f"candidates {first['candidate_targets']}, OTHER {first['other_targets']}"
        )
    return {
        "team_games": merged.height,
        "targets": int(merged["targets"].sum()),
        "candidate_targets": int(merged["candidate_targets"].sum()),
        "other_targets": int(merged["other_targets"].sum()),
        "team_games_with_other": merged.filter(pl.col("other_targets") > 0).height,
        "candidates": counted["_observed_candidates"].sum(),
    }


def team_game_panel(passing, pbp, rosters):
    """Build the panel and its candidates, and refuse to return them if a target went missing.

    Returns `(panel, candidates)`. The panel carries one row per team game in the cached regular
    season, its target total, and the split of that total into candidate and OTHER targets; the
    candidates frame carries one row per available candidate with the targets he drew.
    """
    games = game_context(pbp)
    panel = team_games(passing, games)
    candidates_frame = candidates(passing, rosters, panel)
    counted = candidates_frame.group_by("game_id", "team").agg(
        pl.len().alias("candidates"), pl.col("targets").sum().alias("candidate_targets")
    )
    panel = panel.join(counted, on=["game_id", "team"], how="left")
    panel = panel.with_columns(
        pl.col("candidates").fill_null(0).cast(pl.Int32),
        pl.col("candidate_targets").fill_null(0).cast(pl.Int32),
    )
    panel = panel.with_columns(
        (pl.col("targets") - pl.col("candidate_targets")).cast(pl.Int32).alias("other_targets")
    )
    conservation(panel, candidates_frame)
    return panel.select(PANEL_SCHEMA), candidates_frame
