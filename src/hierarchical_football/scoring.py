"""Fantasy-point scoring functions; these are research rules, not universal platform scoring."""
from __future__ import annotations
from typing import Callable
import numpy as np

def _at(stats: np.ndarray, columns: tuple[str, ...], name: str) -> np.ndarray:
    return np.asarray(stats, dtype=float)[..., columns.index(name)]

def _scrimmage(alpha: float) -> Callable[[np.ndarray, tuple[str, ...]], np.ndarray]:
    def score(stats: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        return (
            alpha * _at(stats, columns, "receptions")
            + 0.1 * (_at(stats, columns, "receiving_yards") + _at(stats, columns, "rushing_yards"))
            + 6.0 * (_at(stats, columns, "receiving_tds") + _at(stats, columns, "rushing_tds"))
        )

    return score

def _passing(four_point_td: bool) -> Callable[[np.ndarray, tuple[str, ...]], np.ndarray]:
    td_points = 4.0 if four_point_td else 6.0

    def score(stats: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
        return (
            0.04 * _at(stats, columns, "passing_yards")
            + td_points * _at(stats, columns, "passing_tds")
            - 2.0 * _at(stats, columns, "interceptions")
            + 0.1 * _at(stats, columns, "rushing_yards")
            + 6.0 * _at(stats, columns, "rushing_tds")
        )

    return score

def _kicking(stats: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    made = _at(stats, columns, "fg_made")
    b40 = _at(stats, columns, "fg_made_40_49")
    b50 = _at(stats, columns, "fg_made_50_59")
    b60 = _at(stats, columns, "fg_made_60_")
    under40 = made - b40 - b50 - b60
    return 3.0 * under40 + 4.0 * b40 + 5.0 * b50 + 6.0 * b60 + _at(stats, columns, "pat_made")

def _allowed_tier(points_allowed: np.ndarray) -> np.ndarray:
    pa = np.asarray(points_allowed, dtype=float)
    return np.select(
        [pa <= 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34],
        [10.0, 7.0, 4.0, 1.0, 0.0, -1.0],
        default=-4.0,
    )

def _defense(stats: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    return (
        _at(stats, columns, "def_sacks")
        + 2.0 * (_at(stats, columns, "def_interceptions") + _at(stats, columns, "def_fumbles"))
        + 6.0 * (_at(stats, columns, "def_tds") + _at(stats, columns, "special_teams_tds"))
        + 2.0 * (_at(stats, columns, "def_safeties") + _at(stats, columns, "blocked_kicks"))
        + _allowed_tier(_at(stats, columns, "points_allowed"))
    )

SCRIMMAGE_COLUMNS = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
)

QB_COLUMNS = (
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "interceptions",
    "carries",
    "rushing_yards",
    "rushing_tds",
)

K_COLUMNS = ("fg_att", "fg_made", "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "pat_att", "pat_made")

DST_COLUMNS = (
    "def_sacks",
    "def_interceptions",
    "def_fumbles",
    "def_tds",
    "def_safeties",
    "blocked_kicks",
    "special_teams_tds",
    "points_allowed",
)

scrimmage = _scrimmage
passing = _passing
kicking = _kicking
defense = _defense
