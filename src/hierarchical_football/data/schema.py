"""Explicit source aliases, model targets, and feature policy."""
from ..common import STAT_COLUMNS as WR_COLUMNS
from ..chains_rb import STAT_COLUMNS as RB_COLUMNS
from ..chains_te import STAT_COLUMNS as TE_COLUMNS
from ..chains_qb import STAT_COLUMNS as QB_COLUMNS
from ..chains_k import STAT_COLUMNS as K_COLUMNS
from ..chains_dst import STAT_COLUMNS as DST_COLUMNS

POSITION_COLUMNS = dict(WR=WR_COLUMNS, RB=RB_COLUMNS, TE=TE_COLUMNS,
                        QB=QB_COLUMNS, K=K_COLUMNS, DST=DST_COLUMNS)
BLOCK_COLUMNS = ("def_fg_blocks", "def_pat_blocks", "def_punt_blocks")
BANDS = ("0_19", "20_29", "30_39", "40_49", "50_59", "60_")
KICK_COLUMNS = tuple(f"fg_{kind}_{band}" for kind in ("made", "missed") for band in BANDS)
PLAYER_OUTCOMES = tuple(dict.fromkeys(
    c for cols in POSITION_COLUMNS.values() for c in cols if c not in ("points_allowed", "blocked_kicks")
)) + BLOCK_COLUMNS + tuple(c for c in KICK_COLUMNS if c not in K_COLUMNS)
LAG_HORIZONS = ("last", "ewma_short", "ewma_med", "last16", "season_1", "season_2", "season_3")
LAG_SOURCE_COLUMNS = tuple(dict.fromkeys((*PLAYER_OUTCOMES, "blocked_kicks", "points_allowed",
    "offense_snaps", "offense_snap_pct", "target_share", "carry_share", "fantasy_points")))
EWMA_SHORT_HALF_LIFE_GAMES = 4.0
EWMA_MEDIUM_HALF_LIFE_GAMES = 16.0
LAST_N_GAMES = 16
KEYS = ["season", "week", "team", "player_id"]
TEAM_KEYS = ["season", "week", "team"]
CONTEXT_FEATURES = ("team", "opponent", "is_home", "week")


def model_features(columns):
    """Only generated historical features and deliberately narrow schedule context."""
    allowed = {f"{source}_{horizon}" for source in LAG_SOURCE_COLUMNS for horizon in LAG_HORIZONS}
    allowed.update(("n_prior_games", "days_since_last_game", *CONTEXT_FEATURES))
    return tuple(c for c in columns if c in allowed)
