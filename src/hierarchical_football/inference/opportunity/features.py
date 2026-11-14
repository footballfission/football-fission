"""Leakage-safe skill vectors, vacated-role features, and opponent defense latents."""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..prepare import require_columns
from .pools import SKILL_POSITIONS

SKILL_DIMS = (
    "lag_target_share",
    "lag_carry_share",
    "lag_dropback_share",
    "log_prior_games",
)
DEFENSE_DIMS = (
    "def_lag_targets_allowed",
    "def_lag_rush_yards_allowed",
    "def_lag_success_allowed",
)
PRIOR_EXPOSURE = 100.0
DEFENSE_PRIOR_GAMES = 4.0


def player_game_shares(candidates, panel):
    """Observed within-pool shares for each candidate-game (labels for building lags)."""
    require_columns(candidates, ("season", "week", "game_id", "team", "pool", "player_id",
                                 "position", "count"), "player game shares")
    require_columns(panel, ("season", "week", "game_id", "team", "pool", "total"), "player game shares panel")
    joined = candidates.join(
        panel.select("game_id", "team", "pool", "total"),
        on=["game_id", "team", "pool"], how="left", validate="m:1",
    )
    return joined.with_columns(
        pl.when(pl.col("total") > 0)
        .then(pl.col("count") / pl.col("total"))
        .otherwise(0.0)
        .alias("share")
    )


def build_skill_history(candidates, panel, *, prior_exposure=PRIOR_EXPOSURE):
    """Lagged shrunk skill vector per player-team-game from strictly earlier games.

    Coordinates are pool shares when the player was an active candidate, shrunk toward
    the position mean with `prior_exposure` (default 100). `player_id` attaches the
    vector only; the latent arm never reads a player intercept from this table.
    """
    shares = player_game_shares(candidates, panel).sort("season", "week", "game_id", "team", "pool")
    # Wide per player-game: one row per player appearance in any pool that week.
    player_games = shares.select(
        "season", "week", "game_id", "team", "player_id", "position"
    ).unique()
    wide = player_games
    for pool, name in (("targets", "target_share"), ("carries", "carry_share"),
                       ("dropbacks", "dropback_share")):
        block = shares.filter(pl.col("pool") == pool).select(
            "season", "week", "game_id", "team", "player_id",
            pl.col("share").alias(name),
            pl.col("count").alias(f"{name}_count"),
            pl.col("total").alias(f"{name}_team_total"),
        )
        wide = wide.join(block, on=["season", "week", "game_id", "team", "player_id"], how="left")
    wide = wide.with_columns([
        pl.col(c).fill_null(0.0) for c in (
            "target_share", "carry_share", "dropback_share",
            "target_share_count", "carry_share_count", "dropback_share_count",
            "target_share_team_total", "carry_share_team_total", "dropback_share_team_total",
        )
    ]).sort("season", "week", "game_id")

    # Position priors from the full history (used only through lagged shrinkage at predict time;
    # we recompute expanding priors chronologically below).
    rows = wide.to_dicts()
    # Accumulator keyed by player and by position.
    player_acc = {}
    position_acc = {pos: {
        "target": [0.0, 0.0], "carry": [0.0, 0.0], "dropback": [0.0, 0.0], "games": 0.0,
    } for pos in SKILL_POSITIONS}
    global_acc = {"target": [0.0, 0.0], "carry": [0.0, 0.0], "dropback": [0.0, 0.0], "games": 0.0}
    out = []

    def _shrunk(acc, prior_rate):
        total, exposure = acc
        return (total + prior_exposure * prior_rate) / (exposure + prior_exposure)

    def _rate(acc):
        total, exposure = acc
        return total / exposure if exposure > 0 else 0.0

    for row in rows:
        player = row["player_id"]
        position = row["position"]
        if position not in position_acc:
            position = "WR"
        # Snapshot lags before adding this game.
        p_acc = player_acc.get(player, {
            "target": [0.0, 0.0], "carry": [0.0, 0.0], "dropback": [0.0, 0.0], "games": 0.0,
        })
        pos = position_acc[position]
        pos_target = _rate(pos["target"]) if pos["target"][1] > 0 else (
            _rate(global_acc["target"]) if global_acc["target"][1] > 0 else 1.0 / 8.0)
        pos_carry = _rate(pos["carry"]) if pos["carry"][1] > 0 else (
            _rate(global_acc["carry"]) if global_acc["carry"][1] > 0 else 1.0 / 8.0)
        pos_drop = _rate(pos["dropback"]) if pos["dropback"][1] > 0 else (
            _rate(global_acc["dropback"]) if global_acc["dropback"][1] > 0 else 1.0 / 2.0)
        out.append({
            "season": row["season"],
            "week": row["week"],
            "game_id": row["game_id"],
            "team": row["team"],
            "player_id": player,
            "position": row["position"],
            "lag_target_share": _shrunk(p_acc["target"], pos_target),
            "lag_carry_share": _shrunk(p_acc["carry"], pos_carry),
            "lag_dropback_share": _shrunk(p_acc["dropback"], pos_drop),
            "log_prior_games": math.log1p(p_acc["games"]),
            "prior_games": p_acc["games"],
        })
        # Update with this game's realized shares (exposure = team pool total when candidate).
        for key, share_key, total_key in (
            ("target", "target_share", "target_share_team_total"),
            ("carry", "carry_share", "carry_share_team_total"),
            ("dropback", "dropback_share", "dropback_share_team_total"),
        ):
            exposure = float(row[total_key])
            contribution = float(row[share_key]) * exposure
            p_acc[key][0] += contribution
            p_acc[key][1] += exposure
            pos[key][0] += contribution
            pos[key][1] += exposure
            global_acc[key][0] += contribution
            global_acc[key][1] += exposure
        p_acc["games"] += 1.0
        pos["games"] += 1.0
        global_acc["games"] += 1.0
        player_acc[player] = p_acc

    return pl.DataFrame(out)


def vacated_role_features(skill_players, skills):
    """Sum of lagged skill mass for teammates active last week but not this week.

    Split by position group so a missing receiver and a missing back move different
    substitution channels. Uses roster ACT status only; unknown ≠ healthy is not claimed.
    """
    require_columns(skill_players, ("season", "week", "team", "player_id", "position"), "vacated roles")
    # Prior-week active skill roster per team.
    prior = skill_players.select(
        "season",
        (pl.col("week") + 1).alias("week"),
        "team",
        pl.col("player_id").alias("prior_player_id"),
        pl.col("position").alias("prior_position"),
    )
    current = skill_players.select("season", "week", "team", "player_id")
    missing = prior.join(
        current, left_on=["season", "week", "team", "prior_player_id"],
        right_on=["season", "week", "team", "player_id"], how="anti",
    )
    # Skills already stores lag_* as pre-game; for vacated players use the same-week
    # skill row keyed by player (vector as of the week they are missing).
    skill_asof = skills.select(
        "season", "week", "player_id",
        "lag_target_share", "lag_carry_share", "lag_dropback_share", "position",
    )
    attached = missing.join(
        skill_asof,
        left_on=["season", "week", "prior_player_id"],
        right_on=["season", "week", "player_id"],
        how="left",
    ).with_columns([
        pl.col("lag_target_share").fill_null(0.0),
        pl.col("lag_carry_share").fill_null(0.0),
        pl.col("lag_dropback_share").fill_null(0.0),
        pl.col("prior_position").alias("position_filled"),
    ])
    attached = attached.with_columns(
        pl.when(pl.col("position_filled").is_in(["WR", "TE"])).then(pl.lit("receiver"))
        .when(pl.col("position_filled").is_in(["RB", "FB"])).then(pl.lit("back"))
        .when(pl.col("position_filled") == "QB").then(pl.lit("qb"))
        .otherwise(pl.lit("other")).alias("group")
    )
    agg = attached.group_by("season", "week", "team").agg(
        pl.col("lag_target_share").filter(pl.col("group") == "receiver").sum().alias("vacated_receiver_target"),
        pl.col("lag_carry_share").filter(pl.col("group") == "back").sum().alias("vacated_back_carry"),
        pl.col("lag_dropback_share").filter(pl.col("group") == "qb").sum().alias("vacated_qb_dropback"),
        pl.col("lag_target_share").sum().alias("vacated_any_target"),
        pl.len().cast(pl.Int32).alias("vacated_skill_count"),
    )
    return agg


def build_defense_latents(events, panel, *, prior_games=DEFENSE_PRIOR_GAMES):
    """Lagged opponent defensive strength, shrunk toward the league mean.

    Encoding (used by both latent and identity arms):
    - `def_lag_targets_allowed`: prior-games mean targets allowed per team-game
    - `def_lag_rush_yards_allowed`: prior-games mean rush yards allowed per team-game
    - `def_lag_success_allowed`: prior-games mean play success rate allowed

    Only earlier (season, week) games enter the average. Values shrink toward the
    expanding league mean with `prior_games` pseudo-observations. Same-game and
    future games are never used.
    """
    require_columns(events, ("season", "week", "game_id", "team", "opponent", "pool",
                             "epa", "success", "yards"), "defense latents")
    # Defense faces the offense: opponent column on events is the defense.
    defense_games = panel.filter(pl.col("pool") == "targets").select(
        "season", "week", "game_id", "team", "opponent", "is_home"
    ).unique()
    # Aggregate allowed rates per defense team-game from events.
    by_game = events.group_by("season", "week", "game_id", "opponent").agg(
        pl.col("pool").eq("targets").sum().cast(pl.Float64).alias("targets_allowed"),
        pl.col("yards").filter(pl.col("pool") == "carries").sum().alias("rush_yards_allowed"),
        pl.col("success").mean().alias("success_allowed"),
        pl.len().cast(pl.Float64).alias("plays"),
    ).rename({"opponent": "defense"})
    by_game = by_game.sort("season", "week", "game_id", "defense")

    rows = by_game.to_dicts()
    team_hist = {}
    league = {"targets": [0.0, 0.0], "rush": [0.0, 0.0], "success": [0.0, 0.0]}
    latents = []

    def _mean(acc):
        total, n = acc
        return total / n if n > 0 else None

    def _shrink(acc, league_acc, default):
        total, n = acc
        league_rate = _mean(league_acc)
        if league_rate is None:
            league_rate = default
        return (total + prior_games * league_rate) / (n + prior_games)

    for row in rows:
        defense = row["defense"]
        hist = team_hist.get(defense, {"targets": [0.0, 0.0], "rush": [0.0, 0.0], "success": [0.0, 0.0]})
        latents.append({
            "season": row["season"],
            "week": row["week"],
            "game_id": row["game_id"],
            "defense": defense,
            "def_lag_targets_allowed": _shrink(hist["targets"], league["targets"], 32.0),
            "def_lag_rush_yards_allowed": _shrink(hist["rush"], league["rush"], 110.0),
            "def_lag_success_allowed": _shrink(hist["success"], league["success"], 0.45),
            "def_prior_games": hist["targets"][1],
        })
        # Update after snapshot.
        for key, value in (
            ("targets", float(row["targets_allowed"])),
            ("rush", float(row["rush_yards_allowed"] or 0.0)),
            ("success", float(row["success_allowed"]) if row["success_allowed"] is not None else 0.45),
        ):
            hist[key][0] += value
            hist[key][1] += 1.0
            league[key][0] += value
            league[key][1] += 1.0
        team_hist[defense] = hist

    defense_frame = pl.DataFrame(latents)
    # Attach to offense panel rows via opponent = defense.
    attached = defense_games.join(
        defense_frame,
        left_on=["season", "week", "game_id", "opponent"],
        right_on=["season", "week", "game_id", "defense"],
        how="left",
    )
    return attached.select(
        "season", "week", "game_id", "team", "opponent", "is_home",
        *DEFENSE_DIMS, "def_prior_games",
    )


def attach_model_features(panel, candidates, skills, vacated, defense, ol_counts=None):
    """Join leakage-safe features onto panel and candidate rows for one or all pools."""
    team_feats = panel.select(
        "season", "week", "game_id", "team", "opponent", "is_home", "pool",
        "total", "candidates", "candidate_total", "other_total",
    ).join(
        defense.select("game_id", "team", *DEFENSE_DIMS, "def_prior_games"),
        on=["game_id", "team"], how="left", validate="m:1",
    ).join(
        vacated, on=["season", "week", "team"], how="left"
    )
    for col in ("vacated_receiver_target", "vacated_back_carry", "vacated_qb_dropback",
                "vacated_any_target", "vacated_skill_count"):
        if col not in team_feats.columns:
            team_feats = team_feats.with_columns(pl.lit(0.0).alias(col))
        else:
            team_feats = team_feats.with_columns(pl.col(col).fill_null(0.0))
    if ol_counts is not None and ol_counts.height:
        team_feats = team_feats.join(ol_counts, on=["season", "week", "team"], how="left")
        team_feats = team_feats.with_columns(
            pl.col("ol_active").fill_null(0),
            pl.col("ol_non_active").fill_null(0),
        )
    else:
        team_feats = team_feats.with_columns(
            pl.lit(0).cast(pl.Int32).alias("ol_active"),
            pl.lit(0).cast(pl.Int32).alias("ol_non_active"),
        )

    player_feats = candidates.join(
        skills.select(
            "season", "week", "game_id", "team", "player_id", *SKILL_DIMS, "prior_games"
        ),
        on=["season", "week", "game_id", "team", "player_id"],
        how="left",
    )
    for col in SKILL_DIMS:
        player_feats = player_feats.with_columns(pl.col(col).fill_null(0.0))
    player_feats = player_feats.with_columns(pl.col("prior_games").fill_null(0.0))
    player_feats = player_feats.join(
        team_feats.select(
            "game_id", "team", "pool", *DEFENSE_DIMS,
            "vacated_receiver_target", "vacated_back_carry", "vacated_qb_dropback",
            "vacated_any_target", "is_home",
        ),
        on=["game_id", "team", "pool"], how="left", validate="m:1",
    )
    return team_feats, player_feats
