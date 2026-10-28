"""Offline checks for the holdout evaluation of the inference baselines."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.evaluation import (
    bootstrap_difference,
    calibration_table,
    chronological_split,
    clustered_comparison,
    clustered_metric,
    completion_metrics,
    row_losses,
    yardage_metrics,
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


class ChronologicalSplit(unittest.TestCase):
    def test_earlier_seasons_fit_and_the_later_season_is_held_out(self):
        table = passing_frame({"season": 2023}, {"season": 2024}, {"season": 2024})
        split = chronological_split(table, train_seasons=[2023], holdout_season=2024)
        self.assertEqual(split["train"].height, 1)
        self.assertEqual(split["holdout"].height, 2)
        self.assertEqual(split["train_seasons"], (2023,))
        self.assertEqual(split["holdout_season"], 2024)

    def test_a_holdout_that_is_not_later_is_refused(self):
        table = passing_frame({"season": 2023}, {"season": 2024})
        with self.assertRaisesRegex(ValueError, "not earlier"):
            chronological_split(table, train_seasons=[2024], holdout_season=2023)
        with self.assertRaisesRegex(ValueError, "not earlier"):
            chronological_split(table, train_seasons=[2024], holdout_season=2024)

    def test_an_empty_training_side_or_holdout_is_refused(self):
        table = passing_frame({"season": 2024})
        with self.assertRaisesRegex(ValueError, "no rows in seasons"):
            chronological_split(table, train_seasons=[2022], holdout_season=2024)
        with self.assertRaisesRegex(ValueError, "no rows in the holdout season"):
            chronological_split(table, train_seasons=[2024], holdout_season=2025)

    def test_naming_no_training_season_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one training season"):
            chronological_split(passing_frame({"season": 2024}), train_seasons=[], holdout_season=2024)

    def test_a_table_that_breaks_the_contract_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            chronological_split(passing_frame({}).drop("air_yards"), train_seasons=[2023],
                                holdout_season=2024)


class Metrics(unittest.TestCase):
    def test_a_confident_mistake_costs_more_than_a_hedged_one(self):
        observed = np.array([1.0, 0.0])
        confident = row_losses(observed, np.array([0.9, 0.1]), task="completion")
        hedged = row_losses(observed, np.array([0.6, 0.4]), task="completion")
        self.assertAlmostEqual(confident[0], -np.log(0.9))
        self.assertLess(confident.sum(), hedged.sum())

    def test_a_probability_outside_the_unit_interval_is_refused(self):
        with self.assertRaisesRegex(ValueError, "inside \\[0, 1\\]"):
            row_losses(np.array([1.0]), np.array([1.4]), task="completion")

    def test_losses_and_predictions_must_line_up(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            row_losses(np.array([1.0, 0.0]), np.array([0.5]), task="completion")
        with self.assertRaisesRegex(ValueError, "unknown task"):
            row_losses(np.array([1.0]), np.array([0.5]), task="points")

    def test_yardage_loss_is_the_absolute_error(self):
        losses = row_losses(np.array([3.0, -2.0]), np.array([4.0, -5.0]), task="yards")
        self.assertEqual(list(losses), [1.0, 3.0])

    def test_completion_metrics_report_losses_calibration_and_rates(self):
        report = completion_metrics(np.array([1.0, 0.0, 1.0, 0.0]), np.array([0.6, 0.4, 0.6, 0.4]))
        self.assertEqual(report["rows"], 4)
        self.assertAlmostEqual(report["log_loss"], -np.log(0.6))
        self.assertAlmostEqual(report["brier"], 0.16)
        self.assertAlmostEqual(report["mean_probability"], 0.5)
        self.assertAlmostEqual(report["observed_rate"], 0.5)
        self.assertEqual(sum(bin_out["rows"] for bin_out in report["calibration"]), 4)

    def test_the_heavily_punished_rows_are_counted_beside_the_mean(self):
        report = completion_metrics(np.array([1.0, 1.0, 0.0]), np.array([0.99, 0.01, 0.5]))
        self.assertAlmostEqual(report["tail_share_at_2"], 1 / 3)
        quiet = completion_metrics(np.array([1.0, 1.0]), np.array([0.9, 0.9]))
        self.assertEqual(quiet["tail_share_at_2"], 0.0)

    def test_calibration_reports_only_the_bins_that_hold_rows(self):
        table = calibration_table(np.array([1.0, 0.0]), np.array([0.05, 0.95]), bins=10)
        self.assertEqual(len(table), 2)
        self.assertEqual([bin_out["lower"] for bin_out in table], [0.0, 0.9])
        self.assertEqual([bin_out["observed"] for bin_out in table], [1.0, 0.0])

    def test_yardage_metrics_report_the_tail_beside_the_mean(self):
        report = yardage_metrics(np.array([3.0, 40.0]), np.array([2.0, 0.0]))
        self.assertEqual(report["rows"], 2)
        self.assertAlmostEqual(report["mae"], 20.5)
        self.assertAlmostEqual(report["rmse"], np.sqrt((1.0 + 1600.0) / 2))
        self.assertAlmostEqual(report["median_absolute_error"], 20.5)
        self.assertAlmostEqual(report["tail_share_at_10"], 0.5)

    def test_calibration_needs_both_sides_and_a_bin(self):
        with self.assertRaisesRegex(ValueError, "one observed value per fitted probability"):
            calibration_table(np.array([1.0]), np.array([0.5, 0.5]))
        with self.assertRaisesRegex(ValueError, "at least one bin"):
            calibration_table(np.array([1.0]), np.array([0.5]), bins=0)


class ClusterResampling(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.games = [f"game{index // 20}" for index in range(400)]
        self.first = rng.normal(0.5, 0.3, 400)
        self.second = self.first + 0.05

    def test_the_estimate_is_the_row_mean_over_clusters(self):
        result = clustered_metric(self.first, clusters=self.games, draws=200)
        self.assertAlmostEqual(result["estimate"], self.first.mean())
        self.assertEqual(result["rows"], 400)
        self.assertEqual(result["clusters"], 20)
        self.assertLess(result["lower"], result["estimate"])
        self.assertLess(result["estimate"], result["upper"])

    def test_the_same_seed_gives_the_same_interval(self):
        first = clustered_metric(self.first, clusters=self.games, draws=300, seed=7)
        second = clustered_metric(self.first, clusters=self.games, draws=300, seed=7)
        self.assertEqual(first, second)

    def test_rows_inside_one_game_are_not_independent(self):
        """A spread shared inside games is far more uncertain than the same spread across rows."""
        rng = np.random.default_rng(5)
        effect = {f"game{index}": rng.normal(0, 1.0) for index in range(20)}
        shared = np.array([effect[game] for game in self.games])
        independent = rng.normal(0, 1.0, 400)
        self.assertAlmostEqual(shared.std(), independent.std(), places=1)
        clustered = clustered_metric(shared, clusters=self.games, draws=2000, seed=3)
        loose = clustered_metric(independent, clusters=self.games, draws=2000, seed=3)
        self.assertGreater(clustered["upper"] - clustered["lower"],
                           3 * (loose["upper"] - loose["lower"]))
        self.assertEqual(clustered["clusters"], 20)

    def test_a_constant_difference_is_certain(self):
        result = bootstrap_difference(self.second, self.first, clusters=self.games, draws=200)
        self.assertAlmostEqual(result["estimate"], 0.05)
        self.assertAlmostEqual(result["lower"], 0.05, places=10)
        self.assertAlmostEqual(result["upper"], 0.05, places=10)

    def test_identical_models_give_a_degenerate_interval(self):
        """Losses that agree on every row leave nothing for resampling to move."""
        result = bootstrap_difference(self.first, self.first.copy(), clusters=self.games, draws=400,
                                      seed=2)
        self.assertEqual((result["estimate"], result["lower"], result["upper"]), (0.0, 0.0, 0.0))

    def test_a_clustered_swing_keeps_the_interval_across_zero(self):
        """Better in the games of one half, worse in the other, so no average difference is proven."""
        swing = np.array([-0.4 if game in {"game0", "game1", "game2", "game3", "game4",
                                           "game5", "game6", "game7", "game8", "game9"} else 0.4
                          for game in self.games])
        result = bootstrap_difference(self.first + swing, self.first, clusters=self.games, draws=400,
                                      seed=4)
        self.assertAlmostEqual(result["estimate"], 0.0)
        self.assertLess(result["lower"], 0.0)
        self.assertGreater(result["upper"], 0.0)

    def test_resampling_needs_at_least_two_clusters(self):
        with self.assertRaisesRegex(ValueError, "at least two clusters"):
            clustered_metric(self.first, clusters=["only"] * 400, draws=10)
        with self.assertRaisesRegex(ValueError, "at least two clusters"):
            bootstrap_difference(self.first, self.second, clusters=["only"] * 400, draws=10)

    def test_resampling_needs_one_loss_per_row(self):
        with self.assertRaisesRegex(ValueError, "one loss per row"):
            clustered_metric(self.first.reshape(-1, 1), clusters=self.games, draws=10)
        with self.assertRaisesRegex(ValueError, "two arrays with one loss per row"):
            bootstrap_difference(self.first, self.first[:10], clusters=self.games, draws=10)

    def test_a_comparison_keeps_the_first_model_as_the_reference(self):
        comparison = clustered_comparison({"context": self.first, "identity": self.second},
                                         clusters=self.games, draws=200, seed=1)
        self.assertEqual(comparison["reference"], "context")
        self.assertEqual(sorted(comparison["metrics"]), ["context", "identity"])
        self.assertEqual(list(comparison["differences"]), ["identity"])
        self.assertAlmostEqual(comparison["differences"]["identity"]["estimate"], 0.05)

    def test_a_named_reference_must_be_one_of_the_models(self):
        with self.assertRaisesRegex(ValueError, "unknown reference model"):
            clustered_comparison({"context": self.first}, clusters=self.games, reference="identity",
                                 draws=10)
        with self.assertRaisesRegex(ValueError, "at least one model"):
            clustered_comparison({}, clusters=self.games, draws=10)


if __name__ == "__main__":
    unittest.main()
