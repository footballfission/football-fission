"""Offline checks for standardized contrasts and their support."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.baselines import fit_completion, fit_rushing
from hierarchical_football.inference.design import (
    PASSING_CELLS,
    cell_weights,
    context_cells,
    contrast_sensitivity,
    player_support,
    standardized_contrast,
    standardized_prediction,
    support_report,
)

PASS_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, team="A", defteam="B", qb_id="qb1",
    receiver_id="wr1", complete_pass=1, air_yards=8.0, yardline_100=40.0, ydstogo=10.0, down=1,
    pass_location="left", position="WR",
)
PASS_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, qb_id=pl.String, receiver_id=pl.String, complete_pass=pl.Int8,
    air_yards=pl.Float64, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    pass_location=pl.String, position=pl.String,
)


def passing_frame(*changes):
    rows = [{**PASS_ROW, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=PASS_SCHEMA)


def short_and_deep(rows=60, seed=0):
    """One quarterback who only throws short and one who throws short and deep."""
    rng = np.random.default_rng(seed)
    changes = []
    for index in range(rows):
        changes.append({"qb_id": "qb_short", "receiver_id": "wr1", "air_yards": 3.0,
                        "complete_pass": int(rng.random() < 0.9)})
        changes.append({"qb_id": "qb_deep", "receiver_id": "wr2",
                        "air_yards": 3.0 if index % 2 else 25.0,
                        "complete_pass": int(rng.random() < 0.3)})
    return passing_frame(*changes)


class Cells(unittest.TestCase):
    def test_a_cell_combines_the_bands_and_the_labels(self):
        labels = context_cells(passing_frame({"air_yards": 3.0}, {"air_yards": 22.0}),
                               continuous=(("air_yards", (0.0, 5.0, 20.0, 30.0)),),
                               categorical=("down",))
        self.assertEqual(labels, ["air_yards=0-5|down=1", "air_yards=20-30|down=1"])

    def test_a_missing_value_keeps_its_own_cell(self):
        labels = context_cells(passing_frame({"air_yards": None}, {"down": None}),
                               continuous=(("air_yards", (0.0, 5.0, 30.0)),),
                               categorical=("down",))
        self.assertEqual(labels, ["air_yards=UNKNOWN|down=1", "air_yards=5-30|down=UNKNOWN"])

    def test_a_missing_column_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not in the table"):
            context_cells(passing_frame({}), categorical=("run_gap",))

    def test_weights_are_shares_ordered_by_size(self):
        weights = cell_weights(["a", "a", "a", "b"])
        self.assertEqual(list(weights), ["a", "b"])
        self.assertAlmostEqual(weights["a"], 0.75)
        self.assertAlmostEqual(weights["b"], 0.25)

    def test_thin_cells_are_reported_as_a_share(self):
        table = passing_frame({"down": 1}, {"down": 1}, {"down": 3}, {"down": 3}, {"down": 3})
        report = support_report(table, categorical=("down",), min_rows=3)
        self.assertEqual(report["rows"], 5)
        self.assertEqual(report["cells"], 2)
        self.assertEqual(report["thin_cells"], 1)
        self.assertAlmostEqual(report["thin_share"], 0.4)
        self.assertAlmostEqual(report["largest_cell_share"], 0.6)
        self.assertEqual(report["rows_per_cell"], {"down=3": 3, "down=1": 2})


class PlayerSupport(unittest.TestCase):
    def setUp(self):
        self.table = short_and_deep()
        self.population = self.table

    def test_the_cells_a_player_never_faced_are_reported_as_extrapolation(self):
        report = player_support(self.table, self.population, identity="qb_id", player="qb_short",
                                **PASSING_CELLS)
        self.assertEqual(report["rows"], 60)
        self.assertGreater(report["extrapolated_share"], 0.0)
        self.assertEqual(report["unsupported_cells"], ["air_yards=20-30|down=1|pass_location=left"])
        self.assertAlmostEqual(1.0 - report["extrapolated_share"], report["supported_share"])

    def test_a_player_who_faced_every_cell_extrapolates_nothing(self):
        report = player_support(self.table, self.population, identity="qb_id", player="qb_deep",
                                **PASSING_CELLS)
        self.assertAlmostEqual(report["extrapolated_share"], 0.0)
        self.assertEqual(report["unsupported_cells"], [])

    def test_a_player_with_no_rows_is_refused(self):
        with self.assertRaisesRegex(ValueError, "has no rows"):
            player_support(self.table, self.population, identity="qb_id", player="qb_nobody",
                           **PASSING_CELLS)

    def test_the_identity_must_be_in_both_frames(self):
        with self.assertRaisesRegex(ValueError, "not in the fitting_table"):
            player_support(passing_frame({}), self.population, identity="rusher_id", player="rb1",
                           **PASSING_CELLS)

    def test_a_population_that_is_not_a_slice_of_the_table_is_allowed(self):
        population = self.table.filter(pl.col("air_yards") > 20.0)
        report = player_support(self.table, population, identity="qb_id", player="qb_short",
                                **PASSING_CELLS)
        self.assertAlmostEqual(report["extrapolated_share"], 1.0)
        self.assertEqual(report["population_rows"], population.height)


class StandardizedArms(unittest.TestCase):
    def setUp(self):
        self.table = short_and_deep()
        self.fitted = fit_completion(self.table, identities=("qb_id", "receiver_id"))
        self.population = self.table

    def test_the_same_population_gives_each_arm_its_own_level(self):
        left = standardized_prediction(self.fitted, self.population, levels={"qb_id": "qb_short"})
        right = standardized_prediction(self.fitted, self.population, levels={"qb_id": "qb_deep"})
        self.assertEqual(left["rows"], self.population.height)
        self.assertGreater(left["prediction"], right["prediction"])
        self.assertEqual(left["levels"], {"qb_id": "qb_short"})

    def test_a_player_outside_the_fit_is_reported_rather_than_given_a_term(self):
        arm = standardized_prediction(self.fitted, self.population, levels={"qb_id": "qb_absent"})
        self.assertEqual(arm["unseen_identities"], ["qb_id=qb_absent"])
        left = standardized_prediction(self.fitted, self.population, levels={"qb_id": "qb_short"})
        right = standardized_prediction(self.fitted, self.population, levels={"qb_id": "qb_deep"})
        self.assertLess(arm["prediction"], left["prediction"])
        self.assertGreater(arm["prediction"], right["prediction"])
        self.assertEqual(left["unseen_identities"], [])

    def test_an_identity_outside_the_population_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not in the population"):
            standardized_prediction(self.fitted, self.population, levels={"rusher_id": "rb1"})

    def test_an_empty_population_is_refused(self):
        with self.assertRaisesRegex(ValueError, "holds no rows"):
            standardized_prediction(self.fitted, self.population.filter(pl.col("season") == 1999))

    def test_a_single_forced_level_carries_its_support(self):
        arm = standardized_prediction(self.fitted, self.population, levels={"qb_id": "qb_short"},
                                      fitting_table=self.table, **PASSING_CELLS)
        self.assertEqual(arm["support"]["player"], "qb_short")
        self.assertGreater(arm["support"]["extrapolated_share"], 0.0)


class Contrasts(unittest.TestCase):
    def setUp(self):
        self.table = short_and_deep()
        self.fitted = fit_completion(self.table, identities=("qb_id", "receiver_id"))
        self.population = self.table

    def test_a_contrast_is_the_difference_of_its_two_arms(self):
        contrast = standardized_contrast(self.fitted, self.population, identity="qb_id",
                                         left="qb_short", right="qb_deep", fitting_table=self.table,
                                         **PASSING_CELLS)
        self.assertEqual(contrast["population_rows"], self.population.height)
        self.assertGreater(contrast["difference"], 0.0)
        self.assertAlmostEqual(contrast["difference"],
                               contrast["arms"]["left"]["prediction"]
                               - contrast["arms"]["right"]["prediction"], places=12)
        self.assertEqual(contrast["arms"]["left"]["support"]["player"], "qb_short")
        self.assertEqual(contrast["arms"]["right"]["support"]["player"], "qb_deep")

    def test_a_fixed_partner_is_part_of_the_estimand(self):
        """On the probability scale the average of a sigmoid is not the sigmoid of an average."""
        free = standardized_contrast(self.fitted, self.population, identity="receiver_id",
                                     left="wr1", right="wr2")
        held = standardized_contrast(self.fitted, self.population, identity="receiver_id",
                                     left="wr1", right="wr2", fixed={"qb_id": "qb_short"})
        self.assertNotAlmostEqual(free["difference"], held["difference"], places=3)
        self.assertNotAlmostEqual(free["left_prediction"], held["left_prediction"], places=6)
        self.assertEqual(held["fixed"], {"qb_id": "qb_short"})
        self.assertEqual(held["scale"], "probability")

    def test_the_log_odds_scale_leaves_an_additive_contrast_alone(self):
        free = standardized_contrast(self.fitted, self.population, identity="receiver_id",
                                     left="wr1", right="wr2", scale="log_odds")
        held = standardized_contrast(self.fitted, self.population, identity="receiver_id",
                                     left="wr1", right="wr2", fixed={"qb_id": "qb_short"},
                                     scale="log_odds")
        self.assertAlmostEqual(free["difference"], held["difference"], places=9)
        self.assertNotAlmostEqual(free["left_prediction"], held["left_prediction"], places=6)

    def test_a_log_odds_arm_needs_a_completion_model(self):
        rushing = pl.DataFrame(
            [{**PASS_ROW, "play_id": 1, "rusher_id": "rb1", "rushing_yards": 4.0,
              "run_location": "middle", "run_gap": "guard", "position": "RB"}],
            schema_overrides={**PASS_SCHEMA, "rusher_id": pl.String, "rushing_yards": pl.Float64,
                              "run_location": pl.String, "run_gap": pl.String})
        with self.assertRaisesRegex(ValueError, "needs a completion model"):
            standardized_prediction(fit_rushing(rushing), rushing, scale="log_odds")

    def test_an_unknown_scale_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown scale"):
            standardized_prediction(self.fitted, self.population, scale="points")

    def test_an_identity_cannot_be_both_fixed_and_varied(self):
        with self.assertRaisesRegex(ValueError, "both fixed and varied"):
            standardized_contrast(self.fitted, self.population, identity="qb_id", left="qb_short",
                                  right="qb_deep", fixed={"qb_id": "qb_deep"})

    def test_an_identity_outside_the_population_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not in the population"):
            standardized_contrast(self.fitted, self.population, identity="rusher_id", left="rb1",
                                  right="rb2")

    def test_a_penalty_sweep_reports_a_range_and_names_what_it_is(self):
        report = contrast_sensitivity(
            lambda penalty: fit_completion(self.table, identities=("qb_id", "receiver_id"),
                                           penalty=penalty),
            self.population, identity="qb_id", left="qb_short", right="qb_deep",
            penalties=(0.3, 3.0, 30.0), **PASSING_CELLS)
        self.assertEqual(report["penalties_tried"], 3)
        self.assertEqual(sorted(report["penalties"]), [0.3, 3.0, 30.0])
        self.assertAlmostEqual(report["spread"], report["maximum"] - report["minimum"])
        self.assertGreater(report["spread"], 0.0)
        self.assertEqual(report["note"], "penalty sensitivity range, not a confidence interval")

    def test_the_same_penalty_twice_leaves_no_spread(self):
        report = contrast_sensitivity(
            lambda penalty: fit_completion(self.table, identities=("qb_id", "receiver_id"),
                                           penalty=penalty),
            self.population, identity="qb_id", left="qb_short", right="qb_deep",
            penalties=(1.0, 1.0), **PASSING_CELLS)
        self.assertEqual(report["penalties_tried"], 1)
        self.assertAlmostEqual(report["spread"], 0.0)


if __name__ == "__main__":
    unittest.main()
