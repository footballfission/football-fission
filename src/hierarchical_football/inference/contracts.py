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


def canonical_position(tag):
    """Canonical roster position label, or UNKNOWN when the feed identifies no known one."""
    if tag is None or (isinstance(tag, float) and tag != tag):
        return UNKNOWN
    text = str(tag).strip().upper()
    if not text:
        return UNKNOWN
    return POSITION_ALIASES.get(text, UNKNOWN)


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
        return tuple(column for column, _ in self.required)

    @property
    def optional_columns(self):
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
    raise ValueError(f"Unknown contract kind {kind!r}")


def validate(frame, contract):
    """Return the frame unchanged, or raise for missing columns, null keys, or bad labels.

    Validation is read-only. A duplicate play key is an error rather than a row to drop,
    because silent deduplication would hide a failed join upstream.
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
        raise ValueError(f"{contract.name}: {duplicates} duplicate play keys {list(contract.keys)}")
    return frame
