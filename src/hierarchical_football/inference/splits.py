"""Chronological team-game splits with as-of outcome gating.

A model that predicts a week may learn from the weeks before it and from nothing else. This module
turns a team-game panel into three bundles: the team games a model may fit on, the team games it is
asked to predict with their labels withheld, and those labels kept aside for scoring.

The ordering key is `(season, week)`, not the week number, because week 18 of one season is older than
week 1 of the next. A cutoff therefore names a season and a week, and every team game at or before it is
training data.

A team game is one prediction unit, so both teams of a game and every teammate of a team game move
together. The split refuses a panel whose game sits on both sides of the cutoff, because a model fitted
on half of a game and scored on the other half has seen the answer.

The prediction bundle never carries a label. `team_game_split` drops the outcome columns rather than
trusting a caller not to look at them, and the observed bundle returns exactly those columns for exactly
the predicted team games.
"""
from __future__ import annotations

import polars as pl

from .panels import CANDIDATE_COLUMNS, CANDIDATE_KEYS, PANEL_SCHEMA, TEAM_GAME_KEYS
from .prepare import require_columns, require_unique

LABEL_COLUMNS = ("targets", "candidate_targets", "other_targets", "candidates")
OBSERVED_PANEL_COLUMNS = ("game_id", "team", "targets", "candidate_targets", "other_targets", "candidates")
OBSERVED_CANDIDATE_COLUMNS = ("game_id", "team", "candidate_index", "player_id", "targets")


def _cutoff(train_through):
    """The cutoff as a `(season, week)` pair of integers."""
    try:
        season, week = train_through
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"team_game_split: train_through must be a (season, week) pair, found {train_through!r}"
        ) from error
    return int(season), int(week)


def _weeks(frame):
    """The distinct weeks a frame covers, oldest first."""
    pairs = frame.select("season", "week").unique().sort("season", "week")
    return [{"season": int(season), "week": int(week)}
            for season, week in zip(pairs["season"].to_list(), pairs["week"].to_list(), strict=True)]


def _candidates_for(candidates_frame, panel):
    """The candidate rows whose team game appears in a panel slice."""
    return candidates_frame.join(panel.select(*TEAM_GAME_KEYS), on=list(TEAM_GAME_KEYS), how="semi")


def team_game_split(panel, candidates_frame, *, train_through):
    """Split a team-game panel at one week into training, prediction, and observed bundles.

    `train_through` is the last `(season, week)` a model may learn from. The returned mapping holds
    `train`, `predict`, and `observed` bundles plus a `manifest` naming the cutoff, the two week ranges,
    and the label columns withheld from prediction. Both bundles keep the candidate rows of their team
    games, so a model can allocate a pool for a team game it has never seen; the prediction bundle keeps
    no target of any kind.
    """
    require_columns(panel, PANEL_SCHEMA, "team_game_split")
    require_columns(candidates_frame, CANDIDATE_COLUMNS, "team_game_split")
    require_unique(panel, TEAM_GAME_KEYS, "team games")
    require_unique(candidates_frame, CANDIDATE_KEYS, "candidates")
    season, week = _cutoff(train_through)

    older = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") <= week))
    train = panel.filter(older)
    predict = panel.filter(older.not_())
    if train.height == 0:
        raise ValueError(f"team_game_split: no team games at or before season {season} week {week}")
    if predict.height == 0:
        raise ValueError(f"team_game_split: nothing to predict after season {season} week {week}")

    straddling = (panel.with_columns(older.alias("_train")).group_by("game_id")
                  .agg(pl.col("_train").n_unique().alias("sides")).filter(pl.col("sides") > 1))
    if straddling.height:
        raise ValueError(
            "team_game_split: game "
            f"{straddling['game_id'].to_list()[0]} has team games on both sides of the cutoff"
        )

    stray = candidates_frame.join(panel.select(*TEAM_GAME_KEYS), on=list(TEAM_GAME_KEYS), how="anti")
    if stray.height:
        first = stray.row(0, named=True)
        raise ValueError(
            "team_game_split: a candidate for a team game the panel does not contain: "
            f"game {first['game_id']}, team {first['team']}, player {first['player_id']}"
        )

    train_candidates = _candidates_for(candidates_frame, train)
    predict_candidates = _candidates_for(candidates_frame, predict)
    if predict_candidates.height == 0:
        raise ValueError(
            "team_game_split: nothing to predict after season "
            f"{season} week {week}: the predicted team games have no candidates"
        )
    observed_panel = predict.select(*OBSERVED_PANEL_COLUMNS)
    observed_candidates = predict_candidates.select(*OBSERVED_CANDIDATE_COLUMNS)

    return {
        "train": {"panel": train, "candidates": train_candidates},
        "predict": {"panel": predict.drop(*LABEL_COLUMNS),
                    "candidates": predict_candidates.drop("targets")},
        "observed": {"panel": observed_panel, "candidates": observed_candidates},
        "manifest": {
            "train_through": {"season": season, "week": week},
            "train": {"team_games": train.height, "candidates": train_candidates.height,
                      "weeks": _weeks(train)},
            "predict": {"team_games": predict.height, "candidates": predict_candidates.height,
                        "weeks": _weeks(predict)},
            "label_columns_withheld": list(LABEL_COLUMNS),
        },
    }
