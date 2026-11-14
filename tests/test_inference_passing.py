"""Offline checks for the canonical passing target table."""
import unittest

import polars as pl

from hierarchical_football.inference.contracts import PASSING_CONTRACT, UNKNOWN, validate
from hierarchical_football.inference.prepare import (passing_targets, require_columns,
                                                     require_unique, roster_positions)

PLAY = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, season_type="REG", posteam="A",
    defteam="B", play_type="pass", two_point_attempt=0.0, aborted_play=0.0, sack=0.0,
    qb_scramble=0.0, pass_attempt=1.0, passer_player_id="qb1", receiver_player_id="wr1",
    complete_pass=1.0, air_yards=8.0, yardline_100=40.0, ydstogo=10.0, down=1.0,
    pass_location="left",
)

PBP_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, season_type=pl.String,
    posteam=pl.String, defteam=pl.String, play_type=pl.String, two_point_attempt=pl.Float64,
    aborted_play=pl.Float64, sack=pl.Float64, qb_scramble=pl.Float64, pass_attempt=pl.Float64,
    passer_player_id=pl.String, receiver_player_id=pl.String, complete_pass=pl.Float64,
    air_yards=pl.Float64, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Float64,
    pass_location=pl.String,
)

ROSTER_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, team=pl.String, gsis_id=pl.String, position=pl.String,
    game_type=pl.String,
)


def pbp_frame(*changes):
    rows = [{**PLAY, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=PBP_SCHEMA)


def roster_frame(*changes):
    base = dict(season=2024, week=1, team="A", gsis_id="wr1", position="WR", game_type="REG")
    return pl.DataFrame([{**base, **change} for change in changes], schema_overrides=ROSTER_SCHEMA)


def mixed_pbp():
    """Sixteen plays: five targeted passes and one play for each exclusion reason."""
    return pbp_frame(
        {},
        {"complete_pass": 0.0, "pass_location": "right"},
        {"pass_location": None},
        {"two_point_attempt": 1.0},
        {"sack": 1.0, "pass_attempt": 0.0},
        {"play_type": "run", "qb_scramble": 1.0, "pass_attempt": 0.0},
        {"play_type": "run", "pass_attempt": 0.0},
        {"receiver_player_id": None},
        {"passer_player_id": None},
        {"complete_pass": None},
        {"season_type": "POST"},
        {"aborted_play": 1.0},
        {"pass_attempt": 0.0},
        {"pass_location": " LEFT "},
        {"pass_location": "deep"},
        {"posteam": None},
    )


class PassingTargets(unittest.TestCase):
    def test_kept_plays_are_the_targeted_passes(self):
        table, report = passing_targets(mixed_pbp())
        self.assertEqual(table.height, 5)
        self.assertEqual(table.columns, list(PASSING_CONTRACT.columns))
        self.assertEqual(report["source_rows"], 16)
        self.assertEqual(report["excluded_rows"], 11)

    def test_output_satisfies_the_table_contract(self):
        table, _ = passing_targets(mixed_pbp())
        self.assertTrue(validate(table, PASSING_CONTRACT).equals(table))

    def test_each_exclusion_is_counted_by_reason(self):
        _, report = passing_targets(mixed_pbp())
        self.assertEqual(report["exclusions"], {
            "postseason": 1, "two_point_attempt": 1, "aborted_play": 1, "sack": 1, "scramble": 1,
            "not_a_pass_play": 1, "no_pass_attempt": 1, "missing_quarterback": 1,
            "missing_receiver": 1, "missing_team": 1, "invalid_outcome": 1,
        })

    def test_exclusions_and_targets_account_for_every_source_play(self):
        table, report = passing_targets(mixed_pbp())
        self.assertEqual(report["target_rows"] + sum(report["exclusions"].values()), 16)
        self.assertEqual(report["target_rows"], table.height)

    def test_incomplete_passes_stay_in_the_table(self):
        table, _ = passing_targets(mixed_pbp())
        self.assertEqual(table["complete_pass"].to_list(), [1, 0, 1, 1, 1])

    def test_missing_location_becomes_unknown_instead_of_a_side(self):
        table, report = passing_targets(mixed_pbp())
        self.assertEqual(table["pass_location"].to_list(), ["left", "right", UNKNOWN, "left", UNKNOWN])
        self.assertEqual(report["unknown_pass_location"], 2)

    def test_per_play_identity_and_context_are_carried_through(self):
        table, _ = passing_targets(pbp_frame({}))
        row = table.row(0, named=True)
        self.assertEqual((row["qb_id"], row["receiver_id"]), ("qb1", "wr1"))
        self.assertEqual((row["team"], row["defteam"], row["down"]), ("A", "B", 1))
        self.assertEqual((row["air_yards"], row["yardline_100"], row["ydstogo"]), (8.0, 40.0, 10.0))

    def test_missing_context_is_reported_and_kept(self):
        table, report = passing_targets(pbp_frame({"air_yards": None}, {}))
        self.assertEqual(table.height, 2)
        self.assertEqual(report["missing_context"]["air_yards"], {"nulls": 1, "share": 0.5})
        self.assertIsNone(table["air_yards"][0])

    def test_season_breakdown_matches_the_policy(self):
        _, report = passing_targets(mixed_pbp())
        self.assertEqual(report["seasons"], [2024])
        self.assertEqual(report["exclusions_by_season"]["2024"]["source_rows"], 16)
        self.assertEqual(report["exclusions_by_season"]["2024"]["target_rows"], 5)
        self.assertEqual(report["exclusions_by_season"]["2024"]["excluded"]["postseason"], 1)

    def test_only_the_declared_plays_survive(self):
        table, _ = passing_targets(mixed_pbp())
        self.assertEqual(table["play_id"].to_list(), [1, 2, 3, 14, 15])


class SourcePlayByPlay(unittest.TestCase):
    def test_duplicate_source_plays_are_refused(self):
        frame = pl.concat([pbp_frame({}), pbp_frame({})])
        with self.assertRaisesRegex(ValueError, "share a duplicate key"):
            passing_targets(frame)

    def test_null_play_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "null values in keys"):
            passing_targets(pbp_frame({"play_id": None}))

    def test_missing_source_column_is_named(self):
        with self.assertRaisesRegex(ValueError, r"play-by-play: missing columns \['pass_location'\]"):
            passing_targets(pbp_frame({}).drop("pass_location"))

    def test_non_frames_are_refused(self):
        with self.assertRaises(TypeError):
            passing_targets({})


class RosterPositions(unittest.TestCase):
    def test_aliases_are_canonicalized(self):
        positions = roster_positions(roster_frame(
            {"gsis_id": "wr1", "position": "HB"}, {"gsis_id": "wr2", "position": "TE"},
            {"gsis_id": "wr3", "position": "DB"}, {"gsis_id": "wr4", "position": None}))
        labels = {row["player_id"]: row["position"] for row in positions.iter_rows(named=True)}
        self.assertEqual(labels, {"wr1": "RB", "wr2": "TE", "wr3": UNKNOWN, "wr4": UNKNOWN})

    def test_postseason_roster_rows_are_dropped(self):
        positions = roster_positions(roster_frame({}, {"gsis_id": "wr2", "game_type": "POST"}))
        self.assertEqual(positions["player_id"].to_list(), ["wr1"])

    def test_ambiguous_position_for_one_player_week_is_refused(self):
        with self.assertRaisesRegex(ValueError, "share a duplicate key"):
            roster_positions(roster_frame({"position": "HB"}, {"position": "RB"}))

    def test_position_join_marks_unmatched_receivers_unknown(self):
        positions = roster_positions(roster_frame({"position": "HB"}))
        table, report = passing_targets(
            pbp_frame({"receiver_player_id": "wr1"}, {"receiver_player_id": "missing"}),
            positions=positions)
        self.assertEqual(table["position"].to_list(), ["RB", UNKNOWN])
        self.assertEqual(report["position_rows_matched"], 1)
        self.assertEqual(report["position_rows_unknown"], 1)

    def test_without_a_roster_every_position_is_unknown(self):
        table, report = passing_targets(pbp_frame({}))
        self.assertEqual(table["position"].to_list(), [UNKNOWN])
        self.assertEqual(report["position_rows_unknown"], 1)


class SourceChecks(unittest.TestCase):
    def test_a_missing_source_column_is_named_with_its_frame(self):
        with self.assertRaisesRegex(ValueError, r"pbp: missing columns \['nope'\]"):
            require_columns(pbp_frame({}), ("season", "nope"), "pbp")

    def test_a_repeated_play_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pbp: 2 rows share a duplicate key"):
            require_unique(pbp_frame({}, {"play_id": 1}), ("game_id", "play_id"), "pbp")

    def test_a_null_play_key_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pbp: 1 null values in keys"):
            require_unique(pbp_frame({"game_id": None}), ("game_id", "play_id"), "pbp")


if __name__ == "__main__":
    unittest.main()
