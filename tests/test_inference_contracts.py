"""Offline checks for the canonical inference table contracts."""
import unittest

import polars as pl

from hierarchical_football.inference.contracts import (
    PASSING_CONTRACT,
    PLAY_KEYS,
    RUSHING_CONTRACT,
    UNKNOWN,
    canonical_position,
    validate,
)

PASSING_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=75, team="A", defteam="B",
    qb_id="00-0034857", receiver_id="00-0036322", complete_pass=1, air_yards=12.0,
    yardline_100=45.0, ydstogo=10.0, down=2, pass_location="middle", position="WR",
)

RUSHING_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=99, team="A", defteam="B",
    rusher_id="00-0033280", rushing_yards=6.0, run_location="left", run_gap="guard",
    yardline_100=32.0, ydstogo=3.0, down=1, position="RB",
)

PASSING_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, qb_id=pl.String, receiver_id=pl.String, complete_pass=pl.Int8,
    air_yards=pl.Float64, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    pass_location=pl.String, position=pl.String,
)

RUSHING_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, rusher_id=pl.String, rushing_yards=pl.Float64, run_location=pl.String,
    run_gap=pl.String, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    position=pl.String,
)


def passing_frame(*rows, **overrides):
    return pl.DataFrame([{**PASSING_ROW, **overrides, **row} for row in rows or ({},)],
                        schema_overrides=PASSING_SCHEMA)


def rushing_frame(*rows, **overrides):
    return pl.DataFrame([{**RUSHING_ROW, **overrides, **row} for row in rows or ({},)],
                        schema_overrides=RUSHING_SCHEMA)


class ContractDefinition(unittest.TestCase):
    def test_both_tables_share_the_play_key(self):
        self.assertEqual(PLAY_KEYS, ("game_id", "play_id"))
        self.assertEqual(PASSING_CONTRACT.keys, PLAY_KEYS)
        self.assertEqual(RUSHING_CONTRACT.keys, PLAY_KEYS)

    def test_declared_columns_follow_the_plan(self):
        self.assertEqual(PASSING_CONTRACT.columns, (
            "season", "week", "game_id", "play_id", "team", "defteam", "qb_id", "receiver_id",
            "complete_pass", "air_yards", "yardline_100", "ydstogo", "down", "pass_location",
            "position"))
        self.assertEqual(RUSHING_CONTRACT.columns, (
            "season", "week", "game_id", "play_id", "team", "defteam", "rusher_id",
            "rushing_yards", "run_location", "run_gap", "yardline_100", "ydstogo", "down",
            "position"))

    def test_charted_columns_stay_optional(self):
        self.assertEqual(PASSING_CONTRACT.optional_columns, (
            "is_catchable_ball", "is_drop", "is_contested_ball", "is_screen_pass", "was_pressure",
            "yards_after_catch"))
        self.assertNotIn("n_defense_box", RUSHING_CONTRACT.columns)
        self.assertIn("n_defense_box", RUSHING_CONTRACT.optional_columns)


class PassingContract(unittest.TestCase):
    def test_valid_table_is_returned_unchanged(self):
        frame = passing_frame()
        self.assertTrue(validate(frame, PASSING_CONTRACT).equals(frame))

    def test_missing_column_is_named(self):
        frame = passing_frame().drop("air_yards")
        with self.assertRaisesRegex(ValueError, r"passing_targets: missing columns \['air_yards'\]"):
            validate(frame, PASSING_CONTRACT)

    def test_duplicate_play_keys_are_rejected_not_deduplicated(self):
        frame = passing_frame({}, {})
        with self.assertRaisesRegex(ValueError, "duplicate play keys"):
            validate(frame, PASSING_CONTRACT)

    def test_two_plays_in_one_game_are_allowed(self):
        frame = passing_frame({"play_id": 75}, {"play_id": 76})
        self.assertEqual(validate(frame, PASSING_CONTRACT).height, 2)

    def test_null_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "receiver_id"):
            validate(passing_frame(receiver_id=None), PASSING_CONTRACT)

    def test_null_outcome_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete_pass"):
            validate(passing_frame(complete_pass=None), PASSING_CONTRACT)

    def test_completion_must_be_a_flag(self):
        with self.assertRaisesRegex(ValueError, "complete_pass"):
            validate(passing_frame(complete_pass=2), PASSING_CONTRACT)

    def test_unknown_labels_are_allowed(self):
        frame = passing_frame(pass_location=UNKNOWN, position=UNKNOWN)
        self.assertEqual(validate(frame, PASSING_CONTRACT).height, 1)

    def test_specific_side_is_not_imputed_for_a_missing_location(self):
        with self.assertRaisesRegex(ValueError, "pass_location"):
            validate(passing_frame(pass_location=None), PASSING_CONTRACT)

    def test_unexpected_location_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"unexpected pass_location labels \['LEFT'\]"):
            validate(passing_frame(pass_location="LEFT"), PASSING_CONTRACT)

    def test_position_labels_stay_inside_the_declared_set(self):
        with self.assertRaisesRegex(ValueError, "position"):
            validate(passing_frame(position="K"), PASSING_CONTRACT)

    def test_wrong_dtype_is_rejected(self):
        frame = passing_frame().with_columns(pl.col("air_yards").cast(pl.String))
        with self.assertRaisesRegex(ValueError, "air_yards"):
            validate(frame, PASSING_CONTRACT)

    def test_unknown_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rushing_plays"):
            validate(passing_frame(), RUSHING_CONTRACT)


class RushingContract(unittest.TestCase):
    def test_valid_table_is_returned_unchanged(self):
        frame = rushing_frame()
        self.assertTrue(validate(frame, RUSHING_CONTRACT).equals(frame))

    def test_negative_yardage_is_kept(self):
        frame = rushing_frame(rushing_yards=-4.0)
        self.assertEqual(validate(frame, RUSHING_CONTRACT)["rushing_yards"][0], -4.0)

    def test_null_rusher_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rusher_id"):
            validate(rushing_frame(rusher_id=None), RUSHING_CONTRACT)

    def test_undefined_gap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"unexpected run_gap labels \['edge'\]"):
            validate(rushing_frame(run_gap="edge"), RUSHING_CONTRACT)

    def test_unknown_gap_is_allowed(self):
        frame = rushing_frame(run_gap=UNKNOWN, run_location=UNKNOWN)
        self.assertEqual(validate(frame, RUSHING_CONTRACT).height, 1)


class PositionTags(unittest.TestCase):
    def test_roster_aliases_map_to_canonical_labels(self):
        for tag, expected in (("QB", "QB"), ("HB", "RB"), ("rb", "RB"), (" FB ", "FB"),
                              ("WR", "WR"), ("te", "TE"), ("DB", UNKNOWN), ("K", UNKNOWN),
                              (None, UNKNOWN), (float("nan"), UNKNOWN)):
            self.assertEqual(canonical_position(tag), expected)


class FrameType(unittest.TestCase):
    def test_non_frames_are_rejected(self):
        for value in ({}, [], "passing_targets"):
            with self.assertRaises(TypeError):
                validate(value, PASSING_CONTRACT)


if __name__ == "__main__":
    unittest.main()
