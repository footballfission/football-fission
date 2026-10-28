"""Canonical passing and rushing tables from locally cached play-by-play.

Preparation owns the selection policy: it keeps the plays the plan declares in scope, counts
every excluded play by reason, and writes the label `UNKNOWN` where the source does not say.
It never deduplicates, imputes, or drops a row silently.
"""
from __future__ import annotations

import polars as pl

from ..data.store import atomic_json, content_hash, read_table, utc_now, write_frame
from .audits import pairing_support, table_coverage
from .contracts import (GAPS, LOCATIONS, PASSING_CONTRACT, PLAY_KEYS, POSITIONS, RUSHING_CONTRACT,
                        UNKNOWN, canonical_position, validate)

REASON = "exclusion_reason"

# Source fields the passing selection reads. A cached partition that lacks one is refused
# rather than quietly treated as if the play were not a pass.
PASSING_SOURCE_COLUMNS = (
    "season", "week", "game_id", "play_id", "season_type", "posteam", "defteam", "play_type",
    "two_point_attempt", "aborted_play", "sack", "qb_scramble", "pass_attempt",
    "passer_player_id", "receiver_player_id", "complete_pass", "air_yards", "yardline_100",
    "ydstogo", "down", "pass_location",
)
ROSTER_COLUMNS = ("season", "week", "team", "position")
CONTEXT_COLUMNS = ("air_yards", "yardline_100", "ydstogo", "down")

RUSHING_SOURCE_COLUMNS = (
    "season", "week", "game_id", "play_id", "season_type", "posteam", "defteam", "play_type",
    "rush_attempt", "qb_kneel", "qb_spike", "qb_scramble", "two_point_attempt", "aborted_play",
    "rusher_player_id", "rushing_yards", "run_location", "run_gap", "yardline_100", "ydstogo",
    "down",
)
RUSHING_CONTEXT_COLUMNS = ("yardline_100", "ydstogo", "down")

# Who counts as the runner. A carry by a quarterback or an untagged player is a different
# question from a carry by a running back, so the scope is named rather than assumed.
RUSHER_SCOPES = {
    "backfield": ("RB", "FB"),
    "non_qb": ("RB", "FB", "WR", "TE"),
    "all": POSITIONS,
}


def require_columns(frame, columns, name):
    """Refuse a source frame that lacks a column the selection reads.

    `name` is the frame being described, so the message says which input was wrong rather than only
    which column is absent.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def require_unique(frame, keys, name):
    """Refuse a source frame whose play key repeats or is null.

    A duplicate play key means the upstream join already failed, so preparation stops instead of
    producing a table that counts one play twice.
    """
    keyed = frame.select(list(keys))
    duplicates = keyed.is_duplicated().sum()
    if duplicates:
        raise ValueError(f"{name}: {duplicates} rows share a duplicate key {list(keys)}")
    nulls = int(keyed.null_count().to_numpy().sum())
    if nulls:
        raise ValueError(f"{name}: {nulls} null values in keys {list(keys)}")


def _flag(column):
    """True where a play-by-play flag is set, treating a missing flag as not set."""
    return pl.col(column).cast(pl.Float64, strict=False).fill_null(0).ne(0)


def _declared_label(column, allowed):
    """A declared label, or UNKNOWN. A missing or unrecognised value is never a guess.

    It is named for what it returns so that it does not collide with `contracts.label`, which turns one
    scalar into one label. This one is a Polars expression over a column.
    """
    value = pl.col(column).cast(pl.String).str.strip_chars().str.to_lowercase()
    return pl.when(value.is_in(list(allowed))).then(value).otherwise(pl.lit(UNKNOWN))


def roster_positions(rosters):
    """One canonical position per player, team, and week, from a weekly roster feed."""
    require_columns(rosters, (*ROSTER_COLUMNS, "gsis_id"), "weekly rosters")
    frame = rosters
    if "game_type" in frame.columns:
        frame = frame.filter(pl.col("game_type").cast(pl.String).str.to_uppercase().eq("REG"))
    frame = frame.filter(pl.col("gsis_id").is_not_null())
    positions = frame.select(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        pl.col("team").cast(pl.String), pl.col("gsis_id").cast(pl.String).alias("player_id"),
        pl.col("position").map_elements(canonical_position, return_dtype=pl.String)
        .fill_null(UNKNOWN).alias("position"),
    )
    require_unique(positions, ("season", "week", "team", "player_id"), "weekly roster positions")
    return positions


def _passing_reasons():
    """Exclusion reasons in policy order; the first matching reason is the one counted."""
    return (
        ("postseason", pl.col("season_type").cast(pl.String).str.to_uppercase().fill_null("").ne("REG")),
        ("two_point_attempt", _flag("two_point_attempt")),
        ("aborted_play", _flag("aborted_play")),
        ("sack", _flag("sack")),
        ("scramble", _flag("qb_scramble")),
        ("not_a_pass_play", pl.col("play_type").cast(pl.String).fill_null("").ne("pass")),
        ("no_pass_attempt", _flag("pass_attempt").not_()),
        ("missing_quarterback", pl.col("passer_player_id").is_null()),
        ("missing_receiver", pl.col("receiver_player_id").is_null()),
        ("missing_team", pl.col("posteam").is_null() | pl.col("defteam").is_null()),
        ("invalid_outcome",
         pl.col("complete_pass").cast(pl.Float64, strict=False).is_in([0.0, 1.0]).fill_null(False).not_()),
    )


def _tag(frame, reasons):
    return frame.with_columns(
        pl.coalesce([pl.when(condition).then(pl.lit(reason)) for reason, condition in reasons]).alias(REASON))


def _reason_counts(tagged, reasons):
    counts = {reason: 0 for reason, _ in reasons}
    counts.update({str(reason): int(rows) for reason, rows in tagged.group_by(REASON).len().iter_rows()
                   if reason is not None})
    return {reason: counts[reason] for reason, _ in reasons}


def _context_report(table, columns):
    report = {}
    for column in columns:
        nulls = int(table[column].null_count())
        report[column] = {"nulls": nulls, "share": round(nulls / table.height, 4) if table.height else None}
    return report


def passing_targets(pbp, *, positions=None):
    """The canonical targeted-pass table plus the record of what preparation removed.

    A kept play is a regular-season pass attempt with a known passer, a known receiver, a
    known team, and a known completion. Sacks, scrambles, throwaways, two-point attempts, and
    aborted plays are excluded because they are not a target a receiver could have converted.
    """
    if not isinstance(pbp, pl.DataFrame):
        raise TypeError(f"play-by-play: expected a polars DataFrame, found {type(pbp).__name__}")
    require_columns(pbp, PASSING_SOURCE_COLUMNS, "play-by-play")
    require_unique(pbp, PLAY_KEYS, "play-by-play")
    reasons = _passing_reasons()
    tagged = _tag(pbp, reasons)
    kept = tagged.filter(pl.col(REASON).is_null())
    table = kept.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_id").cast(pl.String),
        pl.col("play_id").cast(pl.Int64),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("defteam"),
        pl.col("passer_player_id").cast(pl.String).alias("qb_id"),
        pl.col("receiver_player_id").cast(pl.String).alias("receiver_id"),
        pl.col("complete_pass").cast(pl.Int8),
        pl.col("air_yards").cast(pl.Float64),
        pl.col("yardline_100").cast(pl.Float64),
        pl.col("ydstogo").cast(pl.Float64),
        pl.col("down").cast(pl.Int32),
        _declared_label("pass_location", LOCATIONS).alias("pass_location"),
        pl.col("receiver_player_id").cast(pl.String).alias("player_id"),
    )
    matched = 0
    if positions is None:
        table = table.with_columns(pl.lit(None, dtype=pl.String).alias("position"))
    else:
        table = table.join(positions, on=["season", "week", "team", "player_id"], how="left",
                           validate="m:1")
        matched = int(table["position"].is_not_null().sum())
    table = table.with_columns(pl.col("position").fill_null(UNKNOWN)).drop("player_id")
    validate(table, PASSING_CONTRACT)
    seasons = sorted(table["season"].unique().to_list())
    excluded = _reason_counts(tagged, reasons)
    by_season = {}
    for season in sorted(tagged["season"].unique().to_list()):
        block = tagged.filter(pl.col("season") == season)
        by_season[str(int(season))] = {
            "source_rows": block.height,
            "target_rows": int(block[REASON].is_null().sum()),
            "excluded": _reason_counts(block, reasons),
        }
    report = {
        "source_rows": pbp.height,
        "target_rows": table.height,
        "excluded_rows": tagged.height - table.height,
        "exclusions": excluded,
        "exclusions_by_season": by_season,
        "missing_context": _context_report(table, CONTEXT_COLUMNS),
        "unknown_pass_location": int(table["pass_location"].eq(UNKNOWN).sum()),
        "position_rows_matched": matched,
        "position_rows_unknown": int(table["position"].eq(UNKNOWN).sum()),
        "seasons": [int(season) for season in seasons],
    }
    return table, report


def _rushing_reasons(scope_labels):
    """Exclusion reasons in policy order, including carries outside the named rusher scope."""
    return (
        ("postseason", pl.col("season_type").cast(pl.String).str.to_uppercase().fill_null("").ne("REG")),
        ("two_point_attempt", _flag("two_point_attempt")),
        ("aborted_play", _flag("aborted_play")),
        ("kneel", _flag("qb_kneel")),
        ("spike", _flag("qb_spike")),
        ("scramble", _flag("qb_scramble")),
        ("not_a_run_play", pl.col("play_type").cast(pl.String).fill_null("").ne("run")),
        ("no_rush_attempt", _flag("rush_attempt").not_()),
        ("missing_rusher", pl.col("rusher_player_id").is_null()),
        ("missing_team", pl.col("posteam").is_null() | pl.col("defteam").is_null()),
        ("missing_yardage", pl.col("rushing_yards").cast(pl.Float64, strict=False).is_null()),
        ("outside_rusher_scope", pl.col("position").is_in(list(scope_labels)).not_()),
    )


def _rusher_scope(scope):
    if scope not in RUSHER_SCOPES:
        raise ValueError(f"prepare._rusher_scope: unknown rusher scope {scope!r}; expected one of "
                         f"{sorted(RUSHER_SCOPES)}")
    return RUSHER_SCOPES[scope]


def rushing_plays(pbp, *, positions=None, rusher_scope="backfield"):
    """The canonical rushing attempt table plus the record of what preparation removed.

    A kept play is a designed regular-season carry by a runner inside the named scope, with a
    known team and a recorded signed yardage. Kneels, spikes, and scrambles happen on plays
    recorded as runs but are not rushing attempts, and a carry by a quarterback is a different
    question from a carry by a running back, so it needs its own scope.
    """
    if not isinstance(pbp, pl.DataFrame):
        raise TypeError(f"play-by-play: expected a polars DataFrame, found {type(pbp).__name__}")
    scope_labels = _rusher_scope(rusher_scope)
    if positions is None and rusher_scope != "all":
        raise ValueError(f"prepare.rushing_plays: rusher scope {rusher_scope!r} needs weekly roster "
                         f"positions")
    require_columns(pbp, RUSHING_SOURCE_COLUMNS, "play-by-play")
    require_unique(pbp, PLAY_KEYS, "play-by-play")
    located = pbp.with_columns(pl.col("rusher_player_id").cast(pl.String).alias("player_id"),
                               pl.col("posteam").cast(pl.String).alias("team"))
    if positions is None:
        located = located.with_columns(pl.lit(UNKNOWN, dtype=pl.String).alias("position"))
    else:
        located = located.join(positions, on=["season", "week", "team", "player_id"], how="left",
                               validate="m:1")
        located = located.with_columns(pl.col("position").fill_null(UNKNOWN))
    reasons = _rushing_reasons(scope_labels)
    tagged = _tag(located, reasons)
    kept = tagged.filter(pl.col(REASON).is_null())
    table = kept.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_id").cast(pl.String),
        pl.col("play_id").cast(pl.Int64),
        pl.col("team").cast(pl.String),
        pl.col("defteam").cast(pl.String),
        pl.col("rusher_player_id").cast(pl.String).alias("rusher_id"),
        pl.col("rushing_yards").cast(pl.Float64),
        _declared_label("run_location", LOCATIONS).alias("run_location"),
        _declared_label("run_gap", GAPS).alias("run_gap"),
        pl.col("yardline_100").cast(pl.Float64),
        pl.col("ydstogo").cast(pl.Float64),
        pl.col("down").cast(pl.Int32),
        pl.col("position").cast(pl.String),
    )
    validate(table, RUSHING_CONTRACT)
    by_season = {}
    for season in sorted(tagged["season"].unique().to_list()):
        block = tagged.filter(pl.col("season") == season)
        by_season[str(int(season))] = {
            "source_rows": block.height,
            "target_rows": int(block[REASON].is_null().sum()),
            "excluded": _reason_counts(block, reasons),
        }
    report = {
        "source_rows": pbp.height,
        "target_rows": table.height,
        "excluded_rows": tagged.height - table.height,
        "rusher_scope": rusher_scope,
        "scope_positions": list(scope_labels),
        "exclusions": _reason_counts(tagged, reasons),
        "exclusions_by_season": by_season,
        "missing_context": _context_report(table, RUSHING_CONTEXT_COLUMNS),
        "unknown_run_location": int(table["run_location"].eq(UNKNOWN).sum()),
        "unknown_run_gap": int(table["run_gap"].eq(UNKNOWN).sum()),
        "position_rows_matched": int(table["position"].ne(UNKNOWN).sum()) if positions is not None else 0,
        "position_rows_unknown": int(table["position"].eq(UNKNOWN).sum()),
        # What the scope actually kept, beside what it declared: a declared position the weekly
        # roster feed never carries cannot appear in the table, and this is where that shows up.
        "kept_positions": {label: int(table["position"].eq(label).sum()) for label in scope_labels},
        "seasons": [int(season) for season in sorted(table["season"].unique().to_list())],
    }
    return table, report


def build(raw, output, seasons, *, rusher_scope="backfield"):
    """Write both canonical tables, their audits, and the manifest of the inputs they came from.

    Reading is local only: a missing cached partition raises instead of starting a download, so a
    prepared table always names the partitions and hashes it was built from.
    """
    seasons = sorted({int(season) for season in seasons})
    if not seasons:
        raise ValueError("prepare.build: name at least one season")
    play_by_play = read_table(raw, "pbp", seasons)
    rosters = read_table(raw, "rosters_weekly", seasons, required=False)
    if rosters.is_empty() and rusher_scope != "all":
        raise ValueError(f"prepare.build: rusher scope {rusher_scope!r} needs cached rosters_weekly "
                         f"partitions")
    positions = None if rosters.is_empty() else roster_positions(rosters)
    passing, passing_report = passing_targets(play_by_play, positions=positions)
    rushing, rushing_report = rushing_plays(play_by_play, positions=positions, rusher_scope=rusher_scope)
    report = {
        "built_at": utc_now(),
        "seasons": seasons,
        "rusher_scope": rusher_scope,
        "roster_positions_joined": positions is not None,
        "inputs": [{"table_partition": path.relative_to(raw).as_posix(), "sha256": content_hash(path)}
                   for path in sorted(raw.rglob("part.parquet"))
                   if path.parts[len(raw.parts)] in ("pbp", "rosters_weekly")
                   and any(f"season={season}" in path.parts for season in seasons)],
        "passing": {"preparation": passing_report, "coverage": table_coverage(passing, PASSING_CONTRACT),
                    "pairing_support": pairing_support(passing)},
        "rushing": {"preparation": rushing_report, "coverage": table_coverage(rushing, RUSHING_CONTRACT)},
    }
    write_frame(output / f"{PASSING_CONTRACT.name}.parquet", passing,
                metadata={"table": PASSING_CONTRACT.name, "seasons": seasons})
    write_frame(output / f"{RUSHING_CONTRACT.name}.parquet", rushing,
                metadata={"table": RUSHING_CONTRACT.name, "seasons": seasons, "rusher_scope": rusher_scope})
    atomic_json(output / "prepare_report.json", report)
    return report
