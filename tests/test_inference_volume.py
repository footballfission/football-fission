"""Offline checks for the team-volume baselines."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.volume import (DEFAULT_WINDOW, fit_negative_binomial,
                                                    sample_negative_binomial, team_means,
                                                    volume_comparison, volume_scores)

from hierarchical_football.inference.splits import LABEL_COLUMNS

from tests.inference_fixtures import panel_frame

# Two teams with clearly different volumes and one team that never appears in training at all.
TRAIN = (
    {"season": 2024, "week": 1, "game_id": "t1", "team": "HIGH", "targets": 40},
    {"season": 2024, "week": 2, "game_id": "t2", "team": "HIGH", "targets": 44},
    {"season": 2024, "week": 3, "game_id": "t3", "team": "HIGH", "targets": 36},
    {"season": 2024, "week": 1, "game_id": "t4", "team": "LOW", "targets": 18},
    {"season": 2024, "week": 2, "game_id": "t5", "team": "LOW", "targets": 22},
    {"season": 2024, "week": 3, "game_id": "t6", "team": "LOW", "targets": 20},
    {"season": 2024, "week": 1, "game_id": "t7", "team": "MID", "targets": 30},
    {"season": 2024, "week": 2, "game_id": "t8", "team": "MID", "targets": 32},
)
PREDICT = (
    {"season": 2024, "week": 4, "game_id": "p1", "team": "HIGH", "targets": 42},
    {"season": 2024, "week": 4, "game_id": "p2", "team": "LOW", "targets": 19},
    {"season": 2024, "week": 4, "game_id": "p3", "team": "MID", "targets": 31},
    {"season": 2024, "week": 4, "game_id": "p4", "team": "NEW", "targets": 28},
)


class NegativeBinomial(unittest.TestCase):
    def test_the_fit_recovers_the_mean_of_a_drawn_sample(self):
        rng = np.random.default_rng(0)
        drawn = rng.negative_binomial(8, 8 / (8 + 30.0), size=4000)
        fitted = fit_negative_binomial(drawn)
        self.assertAlmostEqual(fitted["mean"], drawn.mean(), delta=1.5)

    def test_the_fit_recovers_a_dispersion_the_sample_supports(self):
        rng = np.random.default_rng(1)
        drawn = rng.negative_binomial(12, 12 / (12 + 30.0), size=8000)
        fitted = fit_negative_binomial(drawn)
        self.assertGreater(fitted["dispersion"], 6.0)
        self.assertLess(fitted["dispersion"], 30.0)

    def test_the_fit_is_deterministic(self):
        drawn = np.random.default_rng(2).negative_binomial(5, 5 / (5 + 20.0), size=500)
        self.assertEqual(fit_negative_binomial(drawn), fit_negative_binomial(drawn))

    def test_a_sample_with_no_spread_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least two distinct"):
            fit_negative_binomial(np.full(20, 25))

    def test_an_empty_or_negative_sample_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least two counts"):
            fit_negative_binomial(np.array([], dtype=int))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            fit_negative_binomial(np.array([1, -1, 5]))


class Sampling(unittest.TestCase):
    def test_samples_have_the_asked_shape_and_are_counts(self):
        samples = sample_negative_binomial(np.array([30.0, 20.0]), 8.0, draws=50,
                                           rng=np.random.default_rng(3))
        self.assertEqual(samples.shape, (2, 50))
        self.assertTrue(np.all(samples >= 0))
        self.assertTrue(np.all(samples == np.round(samples)))

    def test_the_same_seed_gives_the_same_draws(self):
        means = np.array([30.0, 20.0, 40.0])
        first = sample_negative_binomial(means, 8.0, draws=20, rng=np.random.default_rng(4))
        second = sample_negative_binomial(means, 8.0, draws=20, rng=np.random.default_rng(4))
        np.testing.assert_array_equal(first, second)

    def test_the_drawn_mean_follows_the_asked_mean(self):
        means = np.array([40.0, 18.0])
        samples = sample_negative_binomial(means, 20.0, draws=4000, rng=np.random.default_rng(5))
        np.testing.assert_allclose(samples.mean(axis=1), means, rtol=0.05)

    def test_a_non_positive_mean_is_refused(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            sample_negative_binomial(np.array([0.0]), 8.0, draws=5, rng=np.random.default_rng(0))


class TeamMeans(unittest.TestCase):
    def test_a_team_mean_uses_its_most_recent_training_games(self):
        means = team_means(panel_frame(*TRAIN), window=2)
        high = means.filter(pl.col("team") == "HIGH").row(0, named=True)
        self.assertAlmostEqual(high["mean"], 40.0)

    def test_the_window_is_a_bound_not_a_target(self):
        means = team_means(panel_frame(*TRAIN), window=DEFAULT_WINDOW)
        low = means.filter(pl.col("team") == "LOW").row(0, named=True)
        self.assertAlmostEqual(low["mean"], 20.0)

    def test_every_team_with_history_is_reported(self):
        means = team_means(panel_frame(*TRAIN), window=17)
        self.assertEqual(sorted(means["team"].to_list()), ["HIGH", "LOW", "MID"])

    def test_a_window_below_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "window"):
            team_means(panel_frame(*TRAIN), window=0)


class Scores(unittest.TestCase):
    def test_a_sample_that_centres_on_the_observation_scores_well(self):
        observed = np.array([30.0, 20.0])
        samples = sample_negative_binomial(np.array([30.0, 20.0]), 12.0, draws=2000,
                                           rng=np.random.default_rng(6))
        scores = volume_scores(samples, observed)
        self.assertEqual(scores["rows"], 2)
        self.assertLess(scores["mae"], 1.5)
        self.assertLess(scores["crps"], 4.0)
        self.assertGreater(scores["coverage_90"], 0.9)

    def test_a_sample_far_from_the_observation_scores_worse(self):
        observed = np.array([30.0, 20.0])
        close = sample_negative_binomial(np.array([30.0, 20.0]), 12.0, draws=2000,
                                         rng=np.random.default_rng(7))
        far = sample_negative_binomial(np.array([60.0, 5.0]), 12.0, draws=2000,
                                       rng=np.random.default_rng(7))
        self.assertLess(volume_scores(close, observed)["crps"], volume_scores(far, observed)["crps"])

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            volume_scores(np.zeros((2, 10)), np.array([1.0]))


class Comparison(unittest.TestCase):
    def setUp(self):
        self.train = panel_frame(*TRAIN)
        self.predict = panel_frame(*PREDICT).drop(*LABEL_COLUMNS)
        self.observed = panel_frame(*PREDICT).select("game_id", "team", "targets")

    def test_both_baselines_are_scored_on_the_same_rows(self):
        report = volume_comparison(self.train, self.predict, self.observed, draws=2000, seed=8)
        self.assertEqual(report["rows"], 4)
        self.assertEqual(sorted(report["models"]), ["pooled", "team_history"])
        for name in report["models"]:
            self.assertEqual(report["models"][name]["rows"], 4)

    def test_the_team_history_reference_beats_the_league_wide_law_when_teams_differ(self):
        report = volume_comparison(self.train, self.predict, self.observed, draws=4000, seed=9)
        self.assertLess(report["models"]["team_history"]["mae"], report["models"]["pooled"]["mae"])

    def test_a_team_without_training_history_falls_back_to_the_league_mean(self):
        report = volume_comparison(self.train, self.predict, self.observed, draws=500, seed=10)
        means = report["predictions"]["team_history_mean"]
        self.assertEqual(means.shape, (4,))
        self.assertAlmostEqual(means[3], report["league_mean"], delta=0.5)
        self.assertGreater(means[0], means[1])

    def test_the_report_carries_the_fitted_parameters(self):
        report = volume_comparison(self.train, self.predict, self.observed, draws=100, seed=12)
        self.assertAlmostEqual(report["league_mean"], 30.25)
        self.assertAlmostEqual(report["pooled"]["mean"], 30.25, delta=0.5)
        self.assertGreater(report["pooled"]["dispersion"], 0)

    def test_a_prediction_panel_that_carries_the_label_is_refused(self):
        with self.assertRaisesRegex(ValueError, "prediction panel carries"):
            volume_comparison(self.train, panel_frame(*PREDICT), self.observed, draws=100, seed=13)

    def test_an_observed_panel_about_other_team_games_is_refused(self):
        other = panel_frame({"season": 2024, "week": 4, "game_id": "zz", "team": "ZZZ", "targets": 30})
        with self.assertRaisesRegex(ValueError, "observed panel"):
            volume_comparison(self.train, self.predict, other, draws=100, seed=14)

    def test_the_window_reaches_the_report(self):
        report = volume_comparison(self.train, self.predict, self.observed, draws=100, seed=15, window=2)
        self.assertEqual(report["window"], 2)
