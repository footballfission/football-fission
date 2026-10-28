"""Canonical play-level inference tables and the checks every consumer relies on.

The two tables are the contract between preparation and modeling: one row per play key,
identities present, and every label either declared here or `UNKNOWN`. Validation reports
what is wrong instead of repairing it. It never deduplicates plays, imputes a missing
value, or drops a row.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

PASSING_TARGETS = "passing_targets"
RUSHING_PLAYS = "rushing_plays"
PLAY_KEYS = ("game_id", "play_id")
UNKNOWN = "UNKNOWN"

# Roster tags the weekly feed uses, mapped onto the canonical labels. Anything else is
# UNKNOWN: the table records no position rather than guessing one.
POSITION_ALIASES = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "FB", "WR": "WR", "TE": "TE"}
POSITIONS = ("QB", "RB", "FB", "WR", "TE", UNKNOWN)
LOCATIONS = ("left", "middle", "right", UNKNOWN)
GAPS = ("guard", "tackle", "end", UNKNOWN)

_KINDS = (
    ("int", lambda dtype: dtype.is_integer()),
    ("number", lambda dtype: dtype.is_numeric()),
    ("text", lambda dtype: dtype in (pl.String, pl.Categorical, pl.Enum)),
    ("flag", lambda dtype: dtype == pl.Boolean or dtype.is_integer()),
)


def missing(value):
    """True for the three ways a feed says "nothing here": null, NaN, and an empty string.

    The canonical tables keep an absent label as the declared `UNKNOWN` level rather than a null, so
    every consumer agrees on what "no side", "no gap", or "no position" means, and this is the single
    test they share.
    """
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return not str(value).strip()


def label(value):
    """One categorical cell label: the trimmed text, or `UNKNOWN` when there is nothing to trim."""
    return UNKNOWN if missing(value) else str(value).strip()


def column_labels(frame, column):
    """The canonical label of every row of one column, in row order."""
    return [label(value) for value in frame[column].to_list()]


def canonical_position(tag):
    """Canonical roster position label, or UNKNOWN when the feed identifies no known one."""
    if missing(tag):
        return UNKNOWN
    return POSITION_ALIASES.get(str(tag).strip().upper(), UNKNOWN)


@dataclass(frozen=True)
class Contract:
    """One canonical table: its columns and kinds, its key, and the labels it may carry."""
    name: str
    required: tuple
    not_null: tuple
    labels: tuple
    flags: tuple
    optional: tuple
    keys: tuple = PLAY_KEYS

    @property
    def columns(self):
        """Every column the table must carry, in declaration order."""
        return tuple(column for column, _ in self.required)

    @property
    def optional_columns(self):
        """Charted or defensive columns the plan admits, which a valid table may omit.

        A table without them is valid; whether they exist in the source is a coverage question that
        `audits.table_coverage` answers.
        """
        return tuple(c for c, _ in self.optional)


PASSING_CONTRACT = Contract(
    name=PASSING_TARGETS,
    required=(
        ("season", "int"), ("week", "int"), ("game_id", "text"), ("play_id", "int"),
        ("team", "text"), ("defteam", "text"), ("qb_id", "text"), ("receiver_id", "text"),
        ("complete_pass", "flag"), ("air_yards", "number"), ("yardline_100", "number"),
        ("ydstogo", "number"), ("down", "int"), ("pass_location", "text"), ("position", "text"),
    ),
    not_null=("season", "week", "game_id", "play_id", "team", "defteam", "qb_id",
              "receiver_id", "complete_pass", "position"),
    labels=(("pass_location", LOCATIONS), ("position", POSITIONS)),
    flags=("complete_pass",),
    optional=(
        ("is_catchable_ball", "flag"), ("is_drop", "flag"), ("is_contested_ball", "flag"),
        ("is_screen_pass", "flag"), ("was_pressure", "flag"), ("yards_after_catch", "number"),
    ),
)

RUSHING_CONTRACT = Contract(
    name=RUSHING_PLAYS,
    required=(
        ("season", "int"), ("week", "int"), ("game_id", "text"), ("play_id", "int"),
        ("team", "text"), ("defteam", "text"), ("rusher_id", "text"),
        ("rushing_yards", "number"), ("run_location", "text"), ("run_gap", "text"),
        ("yardline_100", "number"), ("ydstogo", "number"), ("down", "int"), ("position", "text"),
    ),
    not_null=("season", "week", "game_id", "play_id", "team", "defteam", "rusher_id",
              "rushing_yards", "position"),
    labels=(("run_location", LOCATIONS), ("run_gap", GAPS), ("position", POSITIONS)),
    flags=(),
    optional=(
        ("n_defense_box", "int"), ("qb_id", "text"), ("is_rpo", "flag"), ("is_qb_sneak", "flag"),
        ("personnel", "text"), ("formation", "text"),
    ),
)

CONTRACTS = {PASSING_CONTRACT.name: PASSING_CONTRACT, RUSHING_CONTRACT.name: RUSHING_CONTRACT}


def _kind_matches(dtype, kind):
    for name, matches in _KINDS:
        if name == kind:
            return matches(dtype)
    raise ValueError(f"contracts._kind_matches: unknown contract kind {kind!r}")


def validate(frame, contract):
    """Return the frame unchanged, or raise for anything that makes it not this contract.

    It raises for a missing required column, a required column of the wrong kind, a null in a column the
    contract says is never null, a flag outside `0` and `1`, a label outside the declared set, and a
    duplicate or null play key.

    Validation is read-only: it never casts a dtype, fills a value, or drops a row. A duplicate play key
    is an error rather than a row to drop, because silent deduplication would hide a failed join
    upstream.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{contract.name}: expected a polars DataFrame, found {type(frame).__name__}")
    missing = sorted(column for column, _ in contract.required if column not in frame.columns)
    if missing:
        raise ValueError(f"{contract.name}: missing columns {missing}")
    for column, kind in contract.required:
        dtype = frame.schema[column]
        if not _kind_matches(dtype, kind):
            raise ValueError(f"{contract.name}: {column} must be {kind}, found {dtype}")
    for column in contract.not_null:
        nulls = frame[column].null_count()
        if nulls:
            raise ValueError(f"{contract.name}: {nulls} null values in {column}")
    for column in contract.flags:
        found = sorted(set(frame[column].unique().to_list()) - {0, 1, False, True}, key=str)
        if found:
            raise ValueError(f"{contract.name}: unexpected {column} values {found}")
    for column, allowed in contract.labels:
        found = sorted(set(frame[column].unique().to_list()) - set(allowed), key=str)
        if found:
            raise ValueError(f"{contract.name}: unexpected {column} labels {found}")
    duplicates = frame.select(contract.keys).is_duplicated().sum()
    if duplicates:
        raise ValueError(f"{contract.name}: {duplicates} rows share a duplicate play key "
                         f"{list(contract.keys)}")
    return frame
