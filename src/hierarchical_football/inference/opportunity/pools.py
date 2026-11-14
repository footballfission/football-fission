"""Three conserved opportunity pools: targets, carries, and dropbacks."""
from __future__ import annotations

import polars as pl

from ..panels import OTHER, TEAM_GAME_KEYS, conservation, game_context
from ..prepare import require_columns, require_unique, roster_positions
from ..contracts import canonical_position

SKILL_POSITIONS = ("QB", "RB", "FB", "WR", "TE")
OL_POSITIONS = ("OL", "T", "G", "C", "OT", "OG")
ACTIVE = "ACT"
POOLS = ("targets", "carries", "dropbacks")

# Dropbacks: QBs only. Carries/targets: all skill positions.
POOL_POSITIONS = {
    "targets": SKILL_POSITIONS,
    "carries": SKILL_POSITIONS,
    "dropbacks": ("QB",),
}

PANEL_COLUMNS = (
    "season", "week", "game_date", "game_id", "team", "opponent", "is_home",
    "pool", "total", "candidates", "candidate_total", "other_total",
)
CANDIDATE_COLUMNS = (
    "season", "week", "game_id", "team", "pool", "candidate_index",
    "player_id", "position", "count",
)


def _flag(column):
    return pl.col(column).cast(pl.Float64, strict=False).fill_null(0).ne(0)


def extract_pool_events(pbp):
    """One row per opportunity event with pool label and credited player.

    Targets follow the canonical pass-attempt-with-receiver rule (throwaways out).
    Carries are rush attempts excluding kneels. Dropbacks use the nflverse
    `qb_dropback` flag (attempts, sacks, and scrambles; designed runs excluded).
    """
    require_columns(
        pbp,
        ("season", "week", "game_id", "play_id", "season_type", "posteam", "defteam",
         "pass_attempt", "sack", "qb_scramble", "play_type", "receiver_player_id",
         "passer_player_id", "rush_attempt", "qb_kneel", "rusher_player_id",
         "qb_dropback", "epa", "success", "rushing_yards", "passing_yards"),
        "opportunity pool events",
    )
    regular = pbp.filter(pl.col("season_type").cast(pl.String).str.to_uppercase().eq("REG"))
    targets = regular.filter(
        _flag("pass_attempt")
        & (~_flag("sack"))
        & pl.col("play_type").cast(pl.String).fill_null("").eq("pass")
        & pl.col("receiver_player_id").is_not_null()
        & pl.col("passer_player_id").is_not_null()
        & pl.col("posteam").is_not_null()
    ).select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_id").cast(pl.String),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("opponent"),
        pl.col("receiver_player_id").cast(pl.String).alias("player_id"),
        pl.lit("targets").alias("pool"),
        pl.col("epa").cast(pl.Float64),
        pl.col("success").cast(pl.Float64),
        pl.col("passing_yards").cast(pl.Float64).fill_null(0.0).alias("yards"),
    )
    carries = regular.filter(
        _flag("rush_attempt") & (~_flag("qb_kneel")) & pl.col("rusher_player_id").is_not_null()
        & pl.col("posteam").is_not_null()
    ).select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_id").cast(pl.String),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("opponent"),
        pl.col("rusher_player_id").cast(pl.String).alias("player_id"),
        pl.lit("carries").alias("pool"),
        pl.col("epa").cast(pl.Float64),
        pl.col("success").cast(pl.Float64),
        pl.col("rushing_yards").cast(pl.Float64).fill_null(0.0).alias("yards"),
    )
    dropbacks = regular.filter(
        _flag("qb_dropback") & pl.col("passer_player_id").is_not_null() & pl.col("posteam").is_not_null()
    ).select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("game_id").cast(pl.String),
        pl.col("posteam").cast(pl.String).alias("team"),
        pl.col("defteam").cast(pl.String).alias("opponent"),
        pl.col("passer_player_id").cast(pl.String).alias("player_id"),
        pl.lit("dropbacks").alias("pool"),
        pl.col("epa").cast(pl.Float64),
        pl.col("success").cast(pl.Float64),
        pl.lit(0.0).alias("yards"),
    )
    return pl.concat([targets, carries, dropbacks], how="vertical_relaxed")


def skill_roster(rosters):
    """Active skill players plus offensive-line out/active counts for volume covariates."""
    require_columns(rosters, ("status", "gsis_id", "position", "season", "week", "team"), "skill roster")
    frame = rosters
    if "game_type" in frame.columns:
        frame = frame.filter(pl.col("game_type").cast(pl.String).str.to_uppercase().eq("REG"))
    active = frame.filter(pl.col("status").cast(pl.String).str.to_uppercase().eq(ACTIVE))
    players = roster_positions(active).filter(pl.col("position").is_in(list(SKILL_POSITIONS)))
    ol = frame.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        pl.col("team").cast(pl.String),
        pl.col("gsis_id").cast(pl.String).alias("player_id"),
        pl.col("position").cast(pl.String),
        pl.col("status").cast(pl.String).str.to_uppercase().alias("status"),
    ).filter(pl.col("position").is_in(list(OL_POSITIONS)) | pl.col("position").str.to_uppercase().eq("OL"))
    ol_counts = ol.group_by("season", "week", "team").agg(
        pl.col("status").eq(ACTIVE).sum().cast(pl.Int32).alias("ol_active"),
        (pl.col("status") != ACTIVE).sum().cast(pl.Int32).alias("ol_non_active"),
    )
    return players, ol_counts


def _team_games_for_pool(events, games, pool):
    home = games.select(
        "season", "week", "game_date", "game_id",
        pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
    ).with_columns(pl.lit(True).alias("is_home"), pl.lit(pool).alias("pool"))
    away = games.select(
        "season", "week", "game_date", "game_id",
        pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
    ).with_columns(pl.lit(False).alias("is_home"), pl.lit(pool).alias("pool"))
    rows = pl.concat([home, away], how="vertical")
    counts = events.filter(pl.col("pool") == pool).group_by("game_id", "team").agg(
        pl.len().cast(pl.Int32).alias("total")
    )
    panel = rows.join(counts, on=["game_id", "team"], how="left").with_columns(
        pl.col("total").fill_null(0).cast(pl.Int32)
    )
    return panel


def _candidates_for_pool(events, players, panel, pool):
    positions = POOL_POSITIONS[pool]
    keyed = panel.select("season", "week", "game_id", "team", "pool")
    eligible = players.filter(pl.col("position").is_in(list(positions)))
    joined = keyed.join(eligible, on=["season", "week", "team"], how="inner")
    counts = events.filter(pl.col("pool") == pool).group_by(
        "season", "week", "game_id", "team", "player_id"
    ).agg(pl.len().cast(pl.Int32).alias("count"))
    joined = joined.join(counts, on=["season", "week", "game_id", "team", "player_id"], how="left")
    joined = joined.with_columns(pl.col("count").fill_null(0).cast(pl.Int32))
    joined = joined.sort("game_id", "team", "player_id")
    joined = joined.with_columns(
        (pl.col("player_id").cum_count().over("game_id", "team") - 1).cast(pl.Int32).alias("candidate_index")
    )
    return joined.select(CANDIDATE_COLUMNS)


def _finalize_panel(panel, candidates):
    counted = candidates.group_by("game_id", "team", "pool").agg(
        pl.len().cast(pl.Int32).alias("candidates"),
        pl.col("count").sum().cast(pl.Int32).alias("candidate_total"),
    )
    panel = panel.join(counted, on=["game_id", "team", "pool"], how="left").with_columns(
        pl.col("candidates").fill_null(0).cast(pl.Int32),
        pl.col("candidate_total").fill_null(0).cast(pl.Int32),
    ).with_columns(
        (pl.col("total") - pl.col("candidate_total")).cast(pl.Int32).alias("other_total")
    )
    # Reuse conservation helper by renaming to the targets panel schema temporarily.
    check_panel = panel.select(
        *TEAM_GAME_KEYS,
        pl.col("total").alias("targets"),
        pl.col("candidates"),
        pl.col("candidate_total").alias("candidate_targets"),
        pl.col("other_total").alias("other_targets"),
    )
    check_candidates = candidates.select(
        *TEAM_GAME_KEYS,
        pl.col("count").alias("targets"),
    )
    # conservation expects unique team-games; one pool at a time is unique on TEAM_GAME_KEYS.
    conservation(check_panel, check_candidates)
    return panel.select(PANEL_COLUMNS), candidates


def build_opportunity_panels(pbp, rosters):
    """Build conserved panels and candidates for targets, carries, and dropbacks.

    Returns `(panel, candidates, events, skill_players, ol_counts, audit)`.
    Population is QB/RB/FB/WR/TE. OL counts are volume covariates only.
    """
    games = game_context(pbp)
    events = extract_pool_events(pbp)
    skill_players, ol_counts = skill_roster(rosters)
    panels, candidate_frames = [], []
    audit = {"pools": {}, "skill_positions": list(SKILL_POSITIONS)}
    for pool in POOLS:
        panel = _team_games_for_pool(events, games, pool)
        candidates = _candidates_for_pool(events, skill_players, panel, pool)
        panel, candidates = _finalize_panel(panel, candidates)
        other_mass = float(panel["other_total"].sum()) / max(float(panel["total"].sum()), 1.0)
        audit["pools"][pool] = {
            "team_games": panel.height,
            "total": int(panel["total"].sum()),
            "candidate_total": int(panel["candidate_total"].sum()),
            "other_total": int(panel["other_total"].sum()),
            "other_mass": other_mass,
            "candidate_rows": candidates.height,
            "positions": list(POOL_POSITIONS[pool]),
        }
        panels.append(panel)
        candidate_frames.append(candidates)
    panel = pl.concat(panels).sort("pool", "season", "week", "game_id", "team")
    candidates = pl.concat(candidate_frames).sort(
        "pool", "season", "week", "game_id", "team", "candidate_index"
    )
    require_unique(panel, (*TEAM_GAME_KEYS, "pool"), "opportunity panel")
    require_unique(candidates, (*TEAM_GAME_KEYS, "pool", "candidate_index"), "opportunity candidates")
    audit["team_games_unique"] = panel.select(*TEAM_GAME_KEYS).unique().height
    audit["seasons"] = sorted(panel["season"].unique().to_list())
    return panel, candidates, events, skill_players, ol_counts, audit
