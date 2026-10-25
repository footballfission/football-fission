"""Roster rows for scheduled weeks that have no roster snapshot."""
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import polars as pl

from hierarchical_football.data.panels import assemble, build
from hierarchical_football.data.store import write_frame

AS_OF = "2025-01-01T00:00:00Z"


def schedule():
    rows = []
    for week, day in ((1, "2024-09-05"), (2, "2024-09-12"), (3, "2024-09-19")):
        rows.append(dict(season=2024, week=week, game_type="REG", game_id=f"g{week}",
                         home_team="A", away_team="B", gameday=day, gametime="20:20",
                         home_score=21, away_score=14))
    return pd.DataFrame(rows)


def stats():
    return pd.DataFrame([
        dict(season=2024, week=1, season_type="REG", team="A", player_id="a1",
             position="WR", targets=5.0, carries=1.0, receptions=3.0, receiving_yards=30.0,
             receiving_tds=0.0, rushing_yards=0.0, rushing_tds=0.0),
        dict(season=2024, week=1, season_type="REG", team="B", player_id="b1",
             position="WR", targets=4.0, carries=0.0, receptions=2.0, receiving_yards=20.0,
             receiving_tds=0.0, rushing_yards=0.0, rushing_tds=0.0),
    ])


def rosters():
    rows = []
    for week, team, player in ((1, "A", "a1"), (1, "A", "a2"), (1, "B", "b1"), (1, "B", "b2"),
                               (2, "A", "a3"), (2, "B", "b1"), (2, "B", "b2"),
                               (3, "B", "b1"), (3, "B", "b2")):
        rows.append(dict(season=2024, week=week, game_type="REG", team=team,
                         gsis_id=player, position="WR", status="ACT", full_name=player))
    return pd.DataFrame(rows)


def assembled(**kwargs):
    players, teams, report = assemble(schedule(), stats(), rosters(), as_of=AS_OF, **kwargs)
    return players, teams, report


def team_week(players, team, week):
    return players.filter((pl.col("team") == team) & (pl.col("week") == week))


@contextmanager
def workspace():
    parent = Path(__file__).resolve().parent
    folder = parent / ("roster_extension_" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError("unexpected cleanup path")
        shutil.rmtree(folder)


class RosterExtension(unittest.TestCase):
    def test_week_without_a_snapshot_receives_carried_rows(self):
        players, _, _ = assembled(extend_rosters=True)
        carried = team_week(players, "A", 3)
        self.assertEqual(carried.height, 1)
        self.assertTrue(carried["roster_carried_forward"].all())
        self.assertTrue(carried["roster_present"].all())
        self.assertEqual(carried["roster_status"][0], "ACT")

    def test_extension_is_off_by_default(self):
        players, _, _ = assembled()
        self.assertEqual(team_week(players, "A", 3).height, 0)
        self.assertIn("roster_carried_forward", players.columns)
        self.assertFalse(players["roster_carried_forward"].any())

    def test_carry_uses_the_most_recent_snapshot(self):
        players, _, _ = assembled(extend_rosters=True)
        carried = team_week(players, "A", 3)
        self.assertEqual(carried["player_id"].to_list(), ["a3"])

    def test_observed_snapshots_are_not_marked_as_carried(self):
        players, _, _ = assembled(extend_rosters=True)
        observed = team_week(players, "B", 3)
        self.assertEqual(observed.height, 2)
        self.assertFalse(observed["roster_carried_forward"].any())
        self.assertFalse(players.filter(pl.col("week") < 3)["roster_carried_forward"].any())

    def test_carried_rows_have_unknown_outcomes(self):
        players, _, _ = assembled(extend_rosters=True)
        carried = team_week(players, "A", 3)
        self.assertTrue(carried["targets"].is_null().all())
        self.assertTrue((carried["outcome_status"] == "unknown").all())

    def test_report_counts_carried_rows_and_extended_weeks(self):
        _, _, report = assembled(extend_rosters=True)
        self.assertEqual(report["carried_forward_roster_rows"], 1)
        self.assertEqual(report["extended_roster_team_weeks"], 1)
        _, _, plain = assembled()
        self.assertEqual(plain["carried_forward_roster_rows"], 0)
        self.assertEqual(plain["extended_roster_team_weeks"], 0)

    def test_build_forwards_the_extension_to_unplayed_weeks(self):
        with workspace() as root:
            raw, out = root / "raw", root / "out"
            for table, frame in (("schedules", schedule()), ("player_stats_week", stats()),
                                 ("rosters_weekly", rosters())):
                write_frame(raw / table / "season=2024" / "part.parquet", pl.from_pandas(frame))
            report = build(raw, out, [2024], positions=("WR",), extend_rosters=True,
                           as_of="2024-09-10T00:00:00Z")
            self.assertEqual(report["carried_forward_roster_rows"], 1)
            prediction = pl.read_parquet(out / "WR" / "prediction.parquet")
            carried = prediction.filter((pl.col("team") == "A") & (pl.col("week") == 3))
            self.assertEqual(carried["player_id"].to_list(), ["a3"])


if __name__ == "__main__":
    unittest.main()
