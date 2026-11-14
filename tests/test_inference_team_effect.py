"""Offline checks for the team-effect volume model."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.splits import LABEL_COLUMNS
from hierarchical_football.inference.team_effect import (count_log_score, fit_team_effect,
                                                         select_penalty, team_design,
                                                         team_effect_comparison, team_effect_means)

from tests.inference_fixtures import panel_frame

TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")
TRUE_EFFECTS = {"AAA": 0.30, "BBB": 0.20, "CCC": 0.10, "DDD": 0.03,
                "EEE": -0.03, "FFF": -0.10, "GGG": -0.20, "HHH": -0.30}


def synthetic_panel(*, weeks=24, season=2024, dispersion=12.0, effects=TRUE_EFFECTS, seed=0):
    """A panel whose team volumes follow known effects, so recovery can be checked against them."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, weeks + 1):
        for team in TEAMS:
            mean = float(np.exp(np.log(32.0) + effects.get(team, 0.0)))
            rows.append({"season": season, "week": week, "game_id": f"{season}_{week:02d}_{team}",
                         "team": team, "targets": int(rng.negative_binomial(dispersion,
                                                                            dispersion / (dispersion + mean)))})
    return panel_frame(*rows)


def fitted_effects(fit):
    """The fitted effects in the team order the design uses."""
    return np.array([fit["effects"][team] for team in fit["teams"]])


def true_effects(fit):
    return np.array([TRUE_EFFECTS[team] for team in fit["teams"]])


class Fitting(unittest.TestCase):
    def test_the_fit_recovers_the_effects_it_was_given(self):
        fit = fit_team_effect(synthetic_panel(), penalty=1.0)
        self.assertEqual(fit["converged"], True)
        self.assertGreater(np.corrcoef(fitted_effects(fit), true_effects(fit))[0, 1], 0.8)

    def test_a_stronger_penalty_shrinks_the_effects(self):
        panel = synthetic_panel()
        spreads = [fitted_effects(fit_team_effect(panel, penalty=penalty)).std()
                   for penalty in (0.0, 100.0, 10000.0)]
        self.assertGreater(spreads[0], spreads[1])
        self.assertGreater(spreads[1], spreads[2])

    def test_the_fit_is_deterministic(self):
        panel = synthetic_panel()
        self.assertEqual(fit_team_effect(panel, penalty=10.0), fit_team_effect(panel, penalty=10.0))

    def test_the_design_names_every_team_in_one_order(self):
        teams, design = team_design(synthetic_panel(weeks=2))
        self.assertEqual(teams, tuple(sorted(TEAMS)))
        self.assertEqual(design.shape, (2 * len(TEAMS), len(TEAMS)))
        self.assertTrue(np.all(design.sum(axis=1) == 1.0))

    def test_an_unseen_team_falls_back_to_the_intercept(self):
        fit = fit_team_effect(synthetic_panel(), penalty=1.0)
        unseen = panel_frame({"season": 2025, "week": 1, "game_id": "x", "team": "ZZZ"})
        means = team_effect_means(fit, unseen)
        self.assertAlmostEqual(means[0], float(np.exp(fit["intercept"])), places=9)

    def test_a_negative_penalty_is_refused(self):
        with self.assertRaisesRegex(ValueError, "penalty must be non-negative"):
            fit_team_effect(synthetic_panel(weeks=2), penalty=-1.0)

    def test_a_panel_without_spread_is_refused(self):
        flat = panel_frame(*[{"season": 2024, "week": week, "game_id": f"g{week}", "team": team,
                              "targets": 30}
                             for week in (1, 2) for team in TEAMS])
        with self.assertRaisesRegex(ValueError, "no spread"):
            fit_team_effect(flat, penalty=1.0)


class LogScore(unittest.TestCase):
    def test_the_score_prefers_means_closer_to_the_truth(self):
        observed = np.array([30.0, 20.0, 40.0])
        close = count_log_score(np.array([30.0, 20.0, 40.0]), 12.0, observed)
        far = count_log_score(np.array([45.0, 8.0, 55.0]), 12.0, observed)
        self.assertLess(close, far)

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaisesRegex(ValueError, "differ in length"):
            count_log_score(np.array([1.0, 2.0]), 12.0, np.array([1.0]))

    def test_a_wider_dispersion_moves_the_score(self):
        observed = np.array([30.0, 31.0, 29.0])
        means = np.array([30.0, 30.0, 30.0])
        self.assertNotAlmostEqual(count_log_score(means, 5.0, observed),
                                  count_log_score(means, 500.0, observed), places=3)


class Selection(unittest.TestCase):
    def test_the_penalty_is_chosen_on_the_inner_split(self):
        report = select_penalty(synthetic_panel(), penalties=(0.0, 1.0, 100.0), inner_through=(2024, 18))
        self.assertIn(report["penalty"], (0.0, 1.0, 100.0))
        self.assertEqual(len(report["scores"]), 3)
        self.assertEqual(report["inner_rows"], {"fit": 8 * 18, "score": 8 * 6})
        self.assertEqual(report["inner_through"], {"season": 2024, "week": 18})
        best = min(report["scores"], key=lambda row: row["log_score"])
        self.assertEqual(report["log_score"], best["log_score"])

    def test_an_inner_cutoff_with_no_history_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nothing to fit or nothing to score"):
            select_penalty(synthetic_panel(), inner_through=(2023, 1))

    def test_a_cutoff_beyond_the_training_weeks_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nothing to fit or nothing to score"):
            select_penalty(synthetic_panel(), inner_through=(2025, 18))


class Comparison(unittest.TestCase):
    def setUp(self):
        self.train = synthetic_panel(weeks=20)
        self.predict = synthetic_panel(weeks=24).filter(pl.col("week") > 20).drop(*LABEL_COLUMNS)
        self.observed = synthetic_panel(weeks=24).filter(pl.col("week") > 20).select(
            "game_id", "team", "targets")

    def report(self, **kwargs):
        options = {"inner_through": (2024, 16), "penalties": (0.0, 10.0, 1000.0), "draws": 2000,
                   "seed": 3}
        return team_effect_comparison(self.train, self.predict, self.observed, **{**options, **kwargs})

    def test_three_models_are_scored_on_the_same_rows(self):
        report = self.report()
        self.assertEqual(sorted(report["models"]), ["pooled", "team_effect", "team_history"])
        for name in report["models"]:
            self.assertEqual(report["models"][name]["rows"], self.predict.height)

    def test_the_selected_penalty_is_the_one_the_model_uses(self):
        report = self.report()
        self.assertEqual(report["team_effect"]["penalty"], report["team_effect"]["selection"]["penalty"])

    def test_the_team_effect_beats_the_pooled_law_when_teams_differ(self):
        report = self.report()
        self.assertLess(report["models"]["team_effect"]["crps"], report["models"]["pooled"]["crps"])

    def test_the_prediction_frame_gets_a_mean_per_team_game(self):
        report = self.report()
        self.assertEqual(report["predictions"]["team_effect_mean"].shape, (self.predict.height,))

    def test_a_prediction_panel_that_carries_the_label_is_refused(self):
        with self.assertRaisesRegex(ValueError, "prediction panel carries"):
            team_effect_comparison(self.train, synthetic_panel(weeks=24).filter(pl.col("week") > 20),
                                   self.observed, inner_through=(2024, 16), draws=100)
