"""Offline checks for the canonical rushing attempt table."""
import unittest

import polars as pl

from hierarchical_football.inference.contracts import RUSHING_CONTRACT, UNKNOWN, validate
from hierarchical_football.inference.prepare import roster_positions, rushing_plays

RUSH = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, season_type="REG", posteam="A",
    defteam="B", play_type="run", rush_attempt=1.0, qb_kneel=0.0, qb_spike=0.0,
    qb_scramble=0.0, two_point_attempt=0.0, aborted_play=0.0, rusher_player_id="rb1",
    rushing_yards=4.0, run_location="left", run_gap="guard", yardline_100=35.0, ydstogo=2.0,
    down=1.0,
)

PBP_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, season_type=pl.String,
    posteam=pl.String, defteam=pl.String, play_type=pl.String, rush_attempt=pl.Float64,
    qb_kneel=pl.Float64, qb_spike=pl.Float64, qb_scramble=pl.Float64, two_point_attempt=pl.Float64,
    aborted_play=pl.Float64, rusher_player_id=pl.String, rushing_yards=pl.Float64,
    run_location=pl.String, run_gap=pl.String, yardline_100=pl.Float64, ydstogo=pl.Float64,
    down=pl.Float64,
)

ROSTER_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, team=pl.String, gsis_id=pl.String, position=pl.String,
    game_type=pl.String,
)


def pbp_frame(*changes):
    rows = [{**RUSH, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=PBP_SCHEMA)


def roster_frame(*changes):
    base = dict(season=2024, week=1, team="A", gsis_id="rb1", position="RB", game_type="REG")
    return pl.DataFrame([{**base, **change} for change in changes], schema_overrides=ROSTER_SCHEMA)


def nearby_positions():
    return roster_positions(roster_frame(
        {"gsis_id": "rb1", "position": "RB"}, {"gsis_id": "qb1", "position": "QB"},
        {"gsis_id": "wr9", "position": "WR"}, {"gsis_id": "hb1", "position": "HB"},
        {"gsis_id": "te3", "position": "TE"}))


def mixed_pbp():
    """Eighteen plays: four backfield carries and one play for each exclusion reason."""
    return pbp_frame(
        {},
        {"rushing_yards": -3.0},
        {"run_location": None},
        {"run_gap": None},
        {"two_point_attempt": 1.0},
        {"aborted_play": 1.0},
        {"qb_kneel": 1.0},
        {"qb_spike": 1.0},
        {"qb_scramble": 1.0},
        {"play_type": "pass", "rush_attempt": 0.0},
        {"rush_attempt": 0.0},
        {"rusher_player_id": None},
        {"posteam": None},
        {"season_type": "POST"},
        {"rusher_player_id": "qb1"},
        {"rusher_player_id": "wr9"},
        {"rusher_player_id": "untagged"},
        {"rushing_yards": None},
    )


def mixed_table(scope="backfield"):
    return rushing_plays(mixed_pbp(), positions=nearby_positions(), rusher_scope=scope)


class RushingPlays(unittest.TestCase):
    def test_kept_plays_are_designed_backfield_carries(self):
        table, report = mixed_table()
        self.assertEqual(table["play_id"].to_list(), [1, 2, 3, 4])
        self.assertEqual(table.columns, list(RUSHING_CONTRACT.columns))
        self.assertEqual(report["source_rows"], 18)
        self.assertEqual(report["excluded_rows"], 14)

    def test_output_satisfies_the_table_contract(self):
        table, _ = mixed_table()
        self.assertTrue(validate(table, RUSHING_CONTRACT).equals(table))

    def test_each_exclusion_is_counted_by_reason(self):
        _, report = mixed_table()
        self.assertEqual(report["exclusions"], {
            "postseason": 1, "two_point_attempt": 1, "aborted_play": 1, "kneel": 1, "spike": 1,
            "scramble": 1, "not_a_run_play": 1, "no_rush_attempt": 1, "missing_rusher": 1,
            "missing_team": 1, "missing_yardage": 1, "outside_rusher_scope": 3,
        })

    def test_exclusions_and_carries_account_for_every_source_play(self):
        table, report = mixed_table()
        self.assertEqual(report["target_rows"] + sum(report["exclusions"].values()), 18)
        self.assertEqual(report["target_rows"], table.height)

    def test_signed_yardage_is_kept(self):
        table, _ = mixed_table()
        self.assertEqual(table["rushing_yards"].to_list(), [4.0, -3.0, 4.0, 4.0])

    def test_missing_location_and_gap_become_unknown(self):
        table, report = mixed_table()
        self.assertEqual(table["run_location"].to_list(), ["left", "left", UNKNOWN, "left"])
        self.assertEqual(table["run_gap"].to_list(), ["guard", "guard", "guard", UNKNOWN])
        self.assertEqual((report["unknown_run_location"], report["unknown_run_gap"]), (1, 1))

    def test_season_breakdown_matches_the_policy(self):
        _, report = mixed_table()
        self.assertEqual(report["seasons"], [2024])
        self.assertEqual(report["exclusions_by_season"]["2024"]["source_rows"], 18)
        self.assertEqual(report["exclusions_by_season"]["2024"]["target_rows"], 4)
        self.assertEqual(report["exclusions_by_season"]["2024"]["excluded"]["kneel"], 1)

    def test_roster_alias_keeps_a_halfback_carry(self):
        table, _ = rushing_plays(pbp_frame({"rusher_player_id": "hb1"}),
                                 positions=nearby_positions())
        self.assertEqual(table["position"].to_list(), ["RB"])

    def test_position_column_records_the_scope_label(self):
        _, report = mixed_table()
        self.assertEqual(report["rusher_scope"], "backfield")
        self.assertEqual(report["scope_positions"], ["RB", "FB"])
        self.assertEqual(report["position_rows_matched"], 4)

    def test_the_report_counts_the_positions_the_scope_actually_kept(self):
        """A declared position the roster feed never carries must show up as an empty count."""
        table, report = mixed_table()
        self.assertEqual(table["position"].unique().to_list(), ["RB"])
        self.assertEqual(report["kept_positions"], {"RB": 4, "FB": 0})

    def test_the_kept_counts_follow_a_wider_scope(self):
        _, report = mixed_table("non_qb")
        self.assertEqual(report["kept_positions"], {"RB": 4, "FB": 0, "WR": 1, "TE": 0})


class RusherScope(unittest.TestCase):
    def test_non_qb_scope_keeps_handoffs_to_receivers(self):
        table, report = mixed_table("non_qb")
        self.assertEqual(table["position"].to_list(), ["RB", "RB", "RB", "RB", "WR"])
        self.assertEqual(report["exclusions"]["outside_rusher_scope"], 2)

    def test_all_scope_keeps_quarterback_and_untagged_carries(self):
        table, _ = mixed_table("all")
        self.assertEqual(table["play_id"].to_list(), [1, 2, 3, 4, 15, 16, 17])
        self.assertEqual(table["position"].to_list(),
                         ["RB", "RB", "RB", "RB", "QB", "WR", UNKNOWN])

    def test_all_scope_works_without_a_roster(self):
        table, report = rushing_plays(pbp_frame({}, {}), rusher_scope="all")
        self.assertEqual(table["position"].to_list(), [UNKNOWN, UNKNOWN])
        self.assertEqual(report["position_rows_matched"], 0)

    def test_position_dependent_scope_needs_a_roster(self):
        with self.assertRaisesRegex(ValueError, "needs weekly roster positions"):
            rushing_plays(pbp_frame({}))

    def test_unknown_scope_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown rusher scope"):
            rushing_plays(pbp_frame({}), positions=nearby_positions(), rusher_scope="skilled")


class SourcePlayByPlay(unittest.TestCase):
    def test_duplicate_source_plays_are_refused(self):
        frame = pl.concat([pbp_frame({}), pbp_frame({})])
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            rushing_plays(frame, positions=nearby_positions())

    def test_missing_source_column_is_named(self):
        with self.assertRaisesRegex(ValueError, r"play-by-play: missing columns \['qb_kneel'\]"):
            rushing_plays(pbp_frame({}).drop("qb_kneel"), positions=nearby_positions())

    def test_non_frames_are_refused(self):
        with self.assertRaises(TypeError):
            rushing_plays(None)

    def test_missing_context_is_reported_and_kept(self):
        table, report = rushing_plays(pbp_frame({"ydstogo": None}, {}),
                                      positions=nearby_positions())
        self.assertEqual(table.height, 2)
        self.assertEqual(report["missing_context"]["ydstogo"], {"nulls": 1, "share": 0.5})


if __name__ == "__main__":
    unittest.main()
