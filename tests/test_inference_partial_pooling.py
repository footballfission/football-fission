"""Offline checks for partial pooling of the identified players."""
import math
import unittest

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from hierarchical_football.inference.baselines import build_design, fit_completion, fit_rushing
from hierarchical_football.inference.partial_pooling import (
    IDENTITY_PENALTIES,
    fit_pooled_completion,
    fit_pooled_rushing,
    identity_columns,
    select_identity_penalty,
)

PASS_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, team="A", defteam="B", qb_id="qb1",
    receiver_id="wr1", complete_pass=1, air_yards=8.0, yardline_100=40.0, ydstogo=10.0, down=1,
    pass_location="left", position="WR",
)
RUSH_ROW = dict(
    season=2024, week=1, game_id="2024_01_A_B", play_id=1, team="A", defteam="B", rusher_id="rb1",
    rushing_yards=4.0, run_location="middle", run_gap="guard", yardline_100=40.0, ydstogo=10.0,
    down=1, position="RB",
)
PASS_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, qb_id=pl.String, receiver_id=pl.String, complete_pass=pl.Int8,
    air_yards=pl.Float64, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    pass_location=pl.String, position=pl.String,
)
RUSH_SCHEMA = dict(
    season=pl.Int32, week=pl.Int32, game_id=pl.String, play_id=pl.Int64, team=pl.String,
    defteam=pl.String, rusher_id=pl.String, rushing_yards=pl.Float64, run_location=pl.String,
    run_gap=pl.String, yardline_100=pl.Float64, ydstogo=pl.Float64, down=pl.Int32,
    position=pl.String,
)


def passing_frame(*changes):
    rows = [{**PASS_ROW, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=PASS_SCHEMA)


def rushing_frame(*changes):
    rows = [{**RUSH_ROW, "play_id": index + 1, **change} for index, change in enumerate(changes)]
    return pl.DataFrame(rows, schema_overrides=RUSH_SCHEMA)


def season_of(quarterbacks, season, rows=200, seed=0):
    """A synthetic season: one completion per drawn play, at its quarterback's own rate."""
    rng = np.random.default_rng(seed)
    changes = []
    for quarterback, rate in quarterbacks:
        for index in range(rows):
            changes.append({"season": season, "game_id": f"{season}_game{index // 20}",
                            "qb_id": quarterback, "receiver_id": "wr%d" % (1 + index % 2),
                            "complete_pass": int(rng.random() < rate)})
    return passing_frame(*changes)


def rushing_season(runners, season, rows=200, seed=0):
    rng = np.random.default_rng(seed)
    changes = []
    for index in range(rows):
        for runner, effect in runners:
            changes.append({"season": season, "game_id": f"{season}_game{index // 20}",
                            "rusher_id": runner,
                            "rushing_yards": float(round(4.0 + effect + rng.normal(0, 1), 1))})
    return rushing_frame(*changes)


class BlockScaling(unittest.TestCase):
    def setUp(self):
        self.table = season_of((("qb1", 0.8), ("qb2", 0.2)), 2024)

    def test_only_the_identity_columns_are_marked(self):
        design = build_design(self.table, continuous=("air_yards",), categorical=("down",),
                              identities=("qb_id",))
        mask = identity_columns(design, ("qb_id",))
        self.assertEqual([column for column, marked in zip(design.columns, mask) if marked],
                         ["qb_id=qb1", "qb_id=qb2"])

    def test_the_fitted_coefficients_reproduce_the_scaled_fit(self):
        fitted = fit_pooled_completion(self.table, identities=("qb_id",), identity_penalty=10.0,
                                       context_penalty=1.0)
        factor = math.sqrt(1.0 / 10.0)
        mask = identity_columns(fitted.design, ("qb_id",))
        matrix = np.array(fitted.design.matrix, copy=True)
        matrix[:, mask] *= factor
        target = self.table["complete_pass"].to_numpy().astype(float)
        scaled = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-8).fit(matrix, target)
        np.testing.assert_allclose(fitted.predict(self.table),
                                   scaled.predict_proba(matrix)[:, 1], rtol=1e-9, atol=1e-9)

    def test_equal_block_penalties_are_the_ordinary_fit(self):
        pooled = fit_pooled_completion(self.table, identities=("qb_id", "receiver_id"))
        plain = fit_completion(self.table, identities=("qb_id", "receiver_id"))
        shared = set(pooled.coefficients()) & set(plain.coefficients())
        for column in sorted(shared):
            self.assertAlmostEqual(pooled.coefficients()[column], plain.coefficients()[column], places=9)

    def test_a_larger_player_penalty_shrinks_the_player_terms(self):
        light = fit_pooled_completion(self.table, identities=("qb_id",), identity_penalty=0.1)
        heavy = fit_pooled_completion(self.table, identities=("qb_id",), identity_penalty=50.0)
        light_terms = {term["player"]: term["term"] for term in light.identity_terms()}
        heavy_terms = {term["player"]: term["term"] for term in heavy.identity_terms()}
        self.assertEqual(sorted(light_terms), sorted(heavy_terms))
        for player in light_terms:
            self.assertLess(abs(heavy_terms[player]), abs(light_terms[player]))

    def test_an_enormous_player_penalty_reaches_the_context_model(self):
        pooled = fit_pooled_completion(self.table, identities=("qb_id",), identity_penalty=1e6)
        context = fit_completion(self.table)
        np.testing.assert_allclose(pooled.predict(self.table), context.predict(self.table), atol=0.02)
        self.assertLess(max(abs(term["term"]) for term in pooled.identity_terms()), 0.05)

    def test_the_report_carries_the_two_penalties_and_the_scale(self):
        fitted = fit_pooled_completion(self.table, identities=("qb_id",), identity_penalty=4.0,
                                       context_penalty=1.0)
        report = fitted.report()
        self.assertEqual(report["block_penalties"]["identity"], 4.0)
        self.assertEqual(report["block_penalties"]["context"], 1.0)
        self.assertAlmostEqual(report["block_penalties"]["identity_scale"], 0.5)
        self.assertIn("identity_penalty=4,context_penalty=1", report["name"])

    def test_the_rushing_runner_block_pools_the_same_way(self):
        table = rushing_season((("rb1", 3.0), ("rb2", -3.0)), 2024)
        pooled = fit_pooled_rushing(table, identities=("rusher_id",), support=True,
                                    identity_penalty=25.0)
        plain = fit_rushing(table, identities=("rusher_id",), support=True)
        heavy = max(abs(term["term"]) for term in pooled.identity_terms())
        light = max(abs(term["term"]) for term in plain.identity_terms())
        self.assertLess(heavy, light)
        self.assertAlmostEqual(pooled.report()["block_penalties"]["identity_scale"], 0.2)

    def test_a_pooled_fit_that_stops_at_the_cap_says_so(self):
        with self.assertWarns(RuntimeWarning):
            fitted = fit_pooled_completion(self.table, identities=("qb_id",), max_iter=1)
        self.assertFalse(fitted.report()["in_sample"]["converged"])

    def test_a_penalty_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "penalty must be positive"):
            fit_pooled_completion(self.table, identities=("qb_id",), identity_penalty=0.0)

    def test_an_identity_column_that_is_not_there_is_refused(self):
        with self.assertRaisesRegex(ValueError, "identity columns"):
            fit_pooled_rushing(rushing_frame({}), identities=("qb_id",))


class PenaltySelection(unittest.TestCase):
    def setUp(self):
        self.table = pl.concat([season_of((("qb1", 0.8), ("qb2", 0.2)), 2023, seed=1),
                                season_of((("qb1", 0.8), ("qb2", 0.2)), 2024, seed=2)])

    def test_a_real_identity_effect_keeps_little_shrinkage(self):
        report = select_identity_penalty(self.table, identities=("qb_id",), task="completion",
                                         train_seasons=[2023], validation_season=2024)
        self.assertEqual(report["selected"], min(IDENTITY_PENALTIES))
        self.assertEqual(sorted(report["scores"]), sorted(float(p) for p in IDENTITY_PENALTIES))
        self.assertEqual(report["metric"], "log_loss")
        self.assertEqual(report["rows"], {"train": 400, "validation": 400})
        self.assertFalse(report["flat"])

    def test_a_grid_with_no_identity_columns_is_flat_and_keeps_the_most_shrinkage(self):
        report = select_identity_penalty(self.table, identities=(), task="completion",
                                         train_seasons=[2023], validation_season=2024,
                                         penalties=(0.1, 10.0))
        self.assertTrue(report["flat"])
        self.assertEqual(report["selected"], 10.0)

    def test_the_rushing_selection_reports_the_yardage_metric(self):
        table = pl.concat([rushing_season((("rb1", 3.0), ("rb2", -3.0)), 2023, seed=1),
                           rushing_season((("rb1", 3.0), ("rb2", -3.0)), 2024, seed=2)])
        report = select_identity_penalty(table, identities=("rusher_id",), task="yards",
                                         train_seasons=[2023], validation_season=2024,
                                         penalties=(0.1, 10.0))
        self.assertEqual(report["metric"], "mae")
        self.assertIn(report["selected"], (0.1, 10.0))
        self.assertEqual(report["rows"], {"train": 400, "validation": 400})

    def test_the_selection_refuses_an_unknown_task(self):
        with self.assertRaisesRegex(ValueError, "unknown task"):
            select_identity_penalty(self.table, identities=("qb_id",), task="points",
                                    train_seasons=[2023], validation_season=2024)

    def test_the_selection_never_scores_the_training_season_as_its_holdout(self):
        with self.assertRaisesRegex(ValueError, "not earlier"):
            select_identity_penalty(self.table, identities=("qb_id",), task="completion",
                                    train_seasons=[2024], validation_season=2023, penalties=(1.0,))


if __name__ == "__main__":
    unittest.main()
