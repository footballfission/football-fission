"""Offline checks for the chronological team-game split."""
import unittest

import polars as pl

from hierarchical_football.inference.panels import PANEL_COLUMNS
from hierarchical_football.inference.splits import LABEL_COLUMNS, team_game_split

from tests.inference_fixtures import candidate_frame, panel_frame

# Two regular-season weeks in 2023 and two in 2024. The first pair is the training history for a cutoff
# at the end of 2023, and the last two weeks are the team games a model would be asked to predict.
PANEL = (
    {"season": 2023, "week": 18, "game_id": "g1", "team": "AAA", "opponent": "BBB",
     "candidate_targets": 28, "other_targets": 2, "targets": 30, "candidates": 2},
    {"season": 2023, "week": 18, "game_id": "g1", "team": "BBB", "opponent": "AAA", "is_home": False,
     "candidate_targets": 30, "other_targets": 0, "targets": 30, "candidates": 2},
    {"season": 2024, "week": 1, "game_id": "g2", "team": "CCC", "opponent": "DDD",
     "candidate_targets": 18, "other_targets": 2, "targets": 20, "candidates": 1},
    {"season": 2024, "week": 1, "game_id": "g2", "team": "DDD", "opponent": "CCC", "is_home": False,
     "candidate_targets": 25, "other_targets": 0, "targets": 25, "candidates": 1},
    {"season": 2024, "week": 2, "game_id": "g3", "team": "EEE", "opponent": "FFF",
     "candidate_targets": 31, "other_targets": 2, "targets": 33, "candidates": 1},
    {"season": 2024, "week": 2, "game_id": "g3", "team": "FFF", "opponent": "EEE", "is_home": False,
     "candidate_targets": 28, "other_targets": 0, "targets": 28, "candidates": 1},
)

CANDIDATES = (
    {"season": 2023, "week": 18, "game_id": "g1", "team": "AAA", "candidate_index": 0,
     "player_id": "aaa_wr1", "targets": 20},
    {"season": 2023, "week": 18, "game_id": "g1", "team": "AAA", "candidate_index": 1,
     "player_id": "aaa_wr2", "targets": 8},
    {"season": 2023, "week": 18, "game_id": "g1", "team": "BBB", "candidate_index": 0,
     "player_id": "bbb_wr1", "targets": 30},
    {"season": 2024, "week": 1, "game_id": "g2", "team": "CCC", "candidate_index": 0,
     "player_id": "ccc_wr1", "targets": 18},
    {"season": 2024, "week": 1, "game_id": "g2", "team": "DDD", "candidate_index": 0,
     "player_id": "ddd_wr1", "targets": 25},
    {"season": 2024, "week": 2, "game_id": "g3", "team": "EEE", "candidate_index": 0,
     "player_id": "eee_wr1", "targets": 31},
    {"season": 2024, "week": 2, "game_id": "g3", "team": "FFF", "candidate_index": 0,
     "player_id": "fff_wr1", "targets": 28},
)

CUTOFF = (2023, 18)


def split(panel=None, candidates_frame=None, **kwargs):
    """The fixture split at the 2023 cutoff unless a check asks for another one."""
    frames = (panel if panel is not None else panel_frame(*PANEL),
              candidates_frame if candidates_frame is not None else candidate_frame(*CANDIDATES))
    return team_game_split(*frames, **{**{"train_through": CUTOFF}, **kwargs})


class Bundles(unittest.TestCase):
    def test_training_holds_the_team_games_at_or_before_the_cutoff(self):
        bundle = split()
        self.assertEqual(bundle["train"]["panel"].height, 2)
        self.assertEqual(set(bundle["train"]["panel"]["team"].to_list()), {"AAA", "BBB"})

    def test_prediction_holds_the_team_games_after_the_cutoff(self):
        bundle = split()
        self.assertEqual(bundle["predict"]["panel"].height, 4)
        self.assertEqual(set(bundle["predict"]["panel"]["team"].to_list()), {"CCC", "DDD", "EEE", "FFF"})

    def test_the_prediction_bundle_carries_no_label(self):
        bundle = split()
        self.assertEqual(bundle["predict"]["panel"].columns,
                         [column for column in PANEL_COLUMNS if column not in LABEL_COLUMNS])
        self.assertEqual(bundle["predict"]["candidates"].columns,
                         [column for column in candidate_frame(*CANDIDATES).columns if column != "targets"])

    def test_the_observed_bundle_carries_the_withheld_labels(self):
        bundle = split()
        observed = bundle["observed"]["panel"]
        self.assertEqual(set(observed.columns), {"game_id", "team", "targets", "candidate_targets",
                                                "other_targets", "candidates"})
        self.assertEqual(observed.filter(pl.col("team") == "EEE").row(0, named=True)["targets"], 33)

    def test_the_observed_candidate_labels_are_the_prediction_rows(self):
        bundle = split()
        observed = bundle["observed"]["candidates"]
        self.assertEqual(observed.height, 4)
        self.assertEqual(sorted(observed["player_id"].to_list()),
                         ["ccc_wr1", "ddd_wr1", "eee_wr1", "fff_wr1"])

    def test_training_candidates_keep_their_targets(self):
        bundle = split()
        self.assertEqual(bundle["train"]["candidates"].height, 3)
        self.assertIn("targets", bundle["train"]["candidates"].columns)


class Ordering(unittest.TestCase):
    def test_the_split_orders_by_season_before_week(self):
        """Week 18 of 2023 is older than week 1 of 2024, which a week number alone would reverse."""
        bundle = split()
        self.assertEqual(bundle["train"]["panel"]["season"].max(), 2023)
        self.assertEqual(bundle["predict"]["panel"]["season"].min(), 2024)

    def test_a_cutoff_inside_a_season_leaves_later_weeks_to_predict(self):
        bundle = split(train_through=(2024, 1))
        self.assertEqual(bundle["train"]["panel"].height, 4)
        self.assertEqual(bundle["predict"]["panel"]["week"].unique().to_list(), [2])

    def test_the_manifest_names_the_cutoff_and_the_counts(self):
        manifest = split()["manifest"]
        self.assertEqual(manifest["train_through"], {"season": 2023, "week": 18})
        self.assertEqual(manifest["train"]["team_games"], 2)
        self.assertEqual(manifest["predict"]["team_games"], 4)
        self.assertEqual(manifest["label_columns_withheld"],
                         ["targets", "candidate_targets", "other_targets", "candidates"])


class Togetherness(unittest.TestCase):
    def test_both_sides_of_a_game_land_on_the_same_side_of_the_cutoff(self):
        panel = panel_frame(*PANEL)
        bundle = split()
        sides = {}
        for name in ("train", "predict"):
            for game in bundle[name]["panel"]["game_id"].unique().to_list():
                sides.setdefault(game, set()).add(name)
        self.assertEqual([game for game, where in sides.items() if len(where) != 1], [])

    def test_every_teammate_lands_on_the_same_side_as_his_team_game(self):
        bundle = split()
        for name in ("train", "predict"):
            keys = bundle[name]["panel"].select("game_id", "team")
            joined = bundle[name]["candidates"].join(keys, on=["game_id", "team"], how="anti")
            self.assertEqual(joined.height, 0)


class Refusals(unittest.TestCase):
    def test_a_cutoff_with_no_history_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no team games at or before"):
            split(train_through=(2022, 1))

    def test_a_cutoff_with_nothing_left_to_predict_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nothing to predict after"):
            split(train_through=(2024, 2))

    def test_a_team_game_that_appears_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            split(panel=panel_frame(*PANEL, PANEL[0]))

    def test_a_candidate_outside_the_panel_is_refused(self):
        stray = CANDIDATES + ({"game_id": "g9", "team": "ZZZ", "player_id": "zzz_wr1"},)
        with self.assertRaisesRegex(ValueError, "candidate for a team game the panel does not contain"):
            split(candidates_frame=candidate_frame(*stray), train_through=CUTOFF)

    def test_a_missing_panel_column_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            split(panel=panel_frame(*PANEL).drop("other_targets"))

    def test_a_malformed_cutoff_is_refused(self):
        with self.assertRaisesRegex(ValueError, "train_through"):
            split(train_through=2024)
