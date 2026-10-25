"""Offline checks for time boundaries, roster coverage, source failures, and scoring."""
from pathlib import Path
import json
import unittest
from contextlib import contextmanager
import shutil
import uuid

import numpy as np
import pandas as pd
import polars as pl

from hierarchical_football.data.history import add_lagged_player_features
from hierarchical_football.data.ingest import ingest
from hierarchical_football.data.panels import assemble, build, position_panel, schedule_rows
from hierarchical_football.data.schema import PLAYER_OUTCOMES, model_features
from hierarchical_football.data.store import write_frame
from hierarchical_football.data.reports import latest_known_injury_status
from hierarchical_football.data.splits import split_week


def fixtures():
    schedules, stats, rosters = [], [], []
    for week, day in ((1, "2024-09-05"), (2, "2024-09-12"), (3, "2024-09-19")):
        schedules.append(dict(season=2024, week=week, game_type="REG", game_id=f"g{week}",
            home_team="A", away_team="B", gameday=day, gametime="20:20", home_score=21, away_score=14))
        for team, player in (("A", "a"), ("B", "b"), ("A", "unrostered")):
            row = {c: 0.0 for c in PLAYER_OUTCOMES}
            row.update(season=2024, week=week, season_type="REG", team=team, player_id=player,
                       position="WR", targets=float(week + 3), receptions=2., receiving_yards=20.)
            stats.append(row)
        for team, player in (("A", "a"), ("B", "b"), ("A", "backup")):
            rosters.append(dict(season=2024, week=week, game_type="REG", team=team,
                gsis_id=player, position="WR", status="ACT", full_name=player))
    return pd.DataFrame(schedules), pd.DataFrame(stats), pd.DataFrame(rosters)


@contextmanager
def workspace_directory():
    """Keep test artifacts local and portable across restricted Windows environments."""
    parent = Path(__file__).resolve().parent
    folder = parent / ("pipeline_test_" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError("Unexpected test cleanup path")
        shutil.rmtree(folder)


class PipelineTests(unittest.TestCase):
    def assemble(self, **kwargs):
        return assemble(*fixtures(), as_of="2025-01-01T00:00:00Z", **kwargs)

    def test_unknown_is_not_zero(self):
        players, teams, report = self.assemble()
        backup = players.filter(pl.col("player_id") == "backup")
        self.assertTrue(backup["targets"].is_null().all())
        self.assertEqual(report["assumed_zero_rows"], 0)
        self.assertTrue((backup["outcome_status"] == "unknown").all())

    def test_explicit_zero_policy(self):
        players, _, report = self.assemble(zero_missing=True)
        backup = players.filter(pl.col("player_id") == "backup")
        self.assertTrue((backup["targets"] == 0).all())
        self.assertEqual(report["assumed_zero_rows"], 3)

    def test_team_pool_includes_unrostered_players(self):
        _, teams, _ = self.assemble()
        a = teams.filter((pl.col("team") == "A") & (pl.col("week") == 1))
        self.assertEqual(a["team_targets"][0], 8)
        self.assertEqual(a["residual_targets"][0], 4)

    def test_schedule_timezone_and_cutoff(self):
        rows = schedule_rows(fixtures()[0], as_of="2025-01-01T00:00:00Z")
        self.assertEqual(str(rows.kickoff.iloc[0]), "2024-09-06 00:20:00")
        self.assertEqual(str(rows.prediction_timestamp.iloc[0]), "2024-09-04 04:00:00")

    def test_no_invented_kickoff(self):
        schedule, _, _ = fixtures()
        schedule.loc[0, "gametime"] = None
        with self.assertRaisesRegex(ValueError, "kickoff"):
            schedule_rows(schedule)

    def test_duplicate_roster_rejected(self):
        schedule, stats, roster = fixtures()
        roster = pd.concat([roster, roster.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assemble(schedule, stats, roster)

    def test_future_labels_hidden_even_when_feed_contains_scores(self):
        players, teams, _ = assemble(*fixtures(), as_of="2024-09-15T00:00:00Z")
        self.assertTrue(players.filter(pl.col("week") == 3)["targets"].is_null().all())
        self.assertTrue(teams.filter(pl.col("week") == 3)["team_targets"].is_null().all())

    def test_current_outcome_cannot_change_own_lag(self):
        players, teams, _ = self.assemble()
        first = position_panel(players, teams, "WR")
        changed = players.with_columns(pl.when(pl.col("week") == 2).then(999.).otherwise(pl.col("targets")).alias("targets"))
        second = position_panel(changed, teams, "WR")
        for panel in (first, second):
            row = panel.filter((pl.col("player_id") == "a") & (pl.col("week") == 2))
            self.assertEqual(row["targets_last"][0], 4)

    def test_four_hour_boundary_is_strict(self):
        frame = pl.DataFrame(dict(player_id=["a", "a"], season=[2024, 2024],
            kickoff=pd.to_datetime(["2024-09-01T00:00:00", "2024-09-02T00:00:00"]),
            prediction_timestamp=pd.to_datetime(["2024-08-31", "2024-09-01T04:00:00"], format="mixed"),
            targets=[7., 100.]))
        result = add_lagged_player_features(frame, sources=("targets",))
        self.assertIsNone(result["targets_last"][1])

    def test_unknown_weeks_not_counted_as_played_history(self):
        players, teams, _ = self.assemble()
        panel = position_panel(players, teams, "WR")
        backup = panel.filter(pl.col("player_id") == "backup")
        self.assertTrue((backup["n_prior_games"] == 0).all())

    def test_future_rows_receive_history(self):
        players, teams, _ = assemble(*fixtures(), as_of="2024-09-15T00:00:00Z")
        panel = position_panel(players, teams, "WR")
        row = panel.filter((pl.col("player_id") == "a") & (pl.col("week") == 3))
        self.assertEqual(row["targets_last"][0], 5)
        self.assertEqual(row["n_prior_games"][0], 2)
        self.assertIsNone(row["fantasy_points"][0])

    def test_features_exclude_current_outcomes_and_status(self):
        selected = model_features(["targets", "roster_status", "outcome_status", "fantasy_points", "targets_last", "team"])
        self.assertEqual(selected, ("targets_last", "team"))

    def test_report_cutoff_and_unknown_status(self):
        players, _, _ = self.assemble()
        reports = pl.DataFrame(dict(gsis_id=["a", "a"], season=[2024, 2024], week=[1, 1],
            date_modified=["2024-09-03T20:00:00Z", "2024-09-04T04:00:00Z"],
            report_status=["Questionable", "Out"]))
        joined = latest_known_injury_status(players, reports)
        row = joined.filter((pl.col("player_id") == "a") & (pl.col("week") == 1))
        self.assertEqual(row["candidate_injury_status"][0], "Questionable")
        self.assertTrue(joined.filter(pl.col("player_id") == "b")["candidate_injury_status"].is_null().all())

    def test_nonparticipant_row_preserved_in_team_pool(self):
        schedule, stats, roster = fixtures()
        stats.loc[stats.player_id.eq("unrostered"), "player_id"] = None
        players, teams, report = assemble(schedule, stats, roster, as_of="2025-01-01T00:00:00Z")
        self.assertEqual(report["unidentified_stat_rows"], 3)
        self.assertFalse(players["player_id"].is_null().any())
        self.assertEqual(teams.filter((pl.col("team") == "A") & (pl.col("week") == 1))["team_targets"][0], 8)

    def test_custom_cutoff_changes_future_only(self):
        rows = schedule_rows(fixtures()[0], as_of="2024-09-15T00:00:00Z", prediction_time="2024-09-18T12:00:00Z")
        self.assertEqual(str(rows.loc[rows.week.eq(1), "prediction_timestamp"].iloc[0]), "2024-09-04 04:00:00")
        self.assertEqual(str(rows.loc[rows.week.eq(3), "prediction_timestamp"].iloc[0]), "2024-09-18 12:00:00")

    def test_postseason_rows_excluded(self):
        schedule, stats, roster = fixtures()
        extra = stats.iloc[:1].copy()
        extra["season_type"] = "POST"
        extra["targets"] = 999.
        players, teams, _ = assemble(schedule, pd.concat([stats, extra]), roster, as_of="2025-01-01T00:00:00Z")
        self.assertEqual(players.height, 12)
        self.assertEqual(teams.filter((pl.col("team") == "A") & (pl.col("week") == 1))["team_targets"][0], 8)

    def test_duplicate_identified_stats_rejected(self):
        schedule, stats, roster = fixtures()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assemble(schedule, pd.concat([stats, stats.iloc[:1]]), roster)

    def test_report_conflicts_rejected(self):
        players, _, _ = self.assemble()
        reports = pl.DataFrame(dict(gsis_id=["a", "a"], season=[2024, 2024], week=[1, 1],
            date_modified=["2024-09-03T20:00:00Z"] * 2, report_status=["Questionable", "Out"]))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            latest_known_injury_status(players, reports)

    def test_ambiguous_snap_ids_are_not_assigned(self):
        schedule, stats, roster = fixtures()
        snaps = pd.DataFrame(dict(pfr_player_id=["pfr"], season=[2024], week=[1], team=["A"],
                                  offense_snaps=[50], offense_pct=[.9], game_type=["REG"]))
        ids = pd.DataFrame(dict(pfr_id=["pfr", "pfr"], gsis_id=["a", "other"]))
        players, _, report = assemble(schedule, stats, roster, snaps=snaps, ids=ids, as_of="2025-01-01T00:00:00Z")
        self.assertTrue(players["offense_snaps"].is_null().all())
        self.assertEqual(report["ambiguous_pfr_ids_excluded"], 1)

    def test_incomplete_stat_record_rejected(self):
        schedule, stats, roster = fixtures()
        stats.loc[0, "targets"] = np.nan
        with self.assertRaisesRegex(ValueError, "nulls"):
            assemble(schedule, stats, roster)

    def test_ingestion_cache_and_failure_reporting(self):
        with workspace_directory() as folder:
            raw = Path(folder)
            calls = []
            def loader(table, season):
                calls.append((table, season))
                return pl.DataFrame({"season": [season], "value": [1]})
            ingest(raw, [2024], ["schedules"], loader=loader)
            ingest(raw, [2024], ["schedules"], loader=loader)
            self.assertEqual(len(calls), 1)
            old = (raw/"schedules/season=2024/part.parquet").read_bytes()
            def broken(table, season):
                raise OSError("offline source")
            with self.assertRaises(RuntimeError):
                ingest(raw, [2024], ["schedules"], refresh=True, loader=broken)
            self.assertEqual(old, (raw/"schedules/season=2024/part.parquet").read_bytes())
            self.assertEqual(len(json.loads((raw/"ingest_report.json").read_text())["errors"]), 1)

    def test_offline_build_all_positions(self):
        schedule, stats, roster = fixtures()
        with workspace_directory() as folder:
            root = Path(folder)
            for table, frame in (("schedules", schedule), ("player_stats_week", stats), ("rosters_weekly", roster)):
                write_frame(root/"raw"/table/"season=2024/part.parquet", pl.from_pandas(frame))
            report = build(root/"raw", root/"processed", [2024], as_of="2025-01-01T00:00:00Z")
            self.assertEqual(report["positions"]["WR"]["history_rows"], 9)
            self.assertEqual(report["positions"]["DST"]["history_rows"], 6)
            frame = pl.read_parquet(root/"processed/WR/history.parquet")
            self.assertTrue((frame["fantasy_points"] == 4).all())
            split_week(root/"processed", root/"week3", "WR", 2024, 3)
            train = pl.read_parquet(root/"week3/train.parquet")
            predict = pl.read_parquet(root/"week3/prediction.parquet")
            self.assertLess(train["week"].max(), 3)
            self.assertNotIn("targets", predict.columns)
            self.assertNotIn("fantasy_points", predict.columns)


if __name__ == "__main__":
    unittest.main()
