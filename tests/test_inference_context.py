"""Passing context must explain difficulty before player identities are credited."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.baselines import fit_completion, passing_design
from hierarchical_football.inference.evaluation import completion_metrics
from tests.inference_fixtures import passing_frame


def curved_season(season, seed, rows=1800):
    rng = np.random.default_rng(seed)
    depth = rng.choice([-5.0, 0.0, 5.0, 10.0, 20.0, 30.0], rows)
    # A deliberately nonlinear conversion mechanism, with no player effects.
    rates = np.where(depth <= 5, 0.9, np.where(depth <= 10, 0.25, 0.65))
    return passing_frame(*[
        dict(season=season, game_id=f"{season}_{i // 30}", air_yards=float(d),
             complete_pass=int(rng.random() < p)) for i, (d, p) in enumerate(zip(depth, rates))])


class PassingContext(unittest.TestCase):
    def test_nonlinear_context_predicts_a_curved_mechanism_out_of_season(self):
        train, holdout = curved_season(2022, 1), curved_season(2023, 2)
        linear = fit_completion(train)
        expanded = fit_completion(train, context="expanded")
        y = holdout["complete_pass"].to_numpy()
        self.assertLess(completion_metrics(y, expanded.predict(holdout))["log_loss"],
                        completion_metrics(y, linear.predict(holdout))["log_loss"] - 0.05)

    def test_later_context_reuses_training_levels_and_depth_scaling(self):
        train = curved_season(2022, 3)
        fit = fit_completion(train, context="expanded")
        later = passing_frame(dict(defteam="NEW", position="TE", air_yards=80.0))
        design = fit.holdout_design(later)
        self.assertEqual(design.columns, fit.design.columns)
        self.assertEqual(design.notes["unseen_categorical_rows"]["defteam"], 1)
        self.assertEqual(design.notes["unseen_categorical_rows"]["position"], 1)
        self.assertTrue(np.all(np.isfinite(fit.predict(later))))
        self.assertEqual(design.continuous, fit.design.continuous)

    def test_outcome_columns_cannot_change_the_context_prediction(self):
        fit = fit_completion(curved_season(2022, 4), context="expanded")
        holdout = curved_season(2023, 5)
        changed = holdout.with_columns((1 - pl.col("complete_pass")).alias("complete_pass"),
                                       pl.lit(999).alias("yards_after_catch"))
        np.testing.assert_array_equal(fit.predict(holdout), fit.predict(changed))

    def test_default_design_remains_linear(self):
        table = curved_season(2022, 1)
        self.assertEqual(passing_design(table).columns,
                         passing_design(table, context="linear").columns)
        self.assertNotIn("defteam=B", passing_design(table).columns)

    def test_unknown_context_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown passing context"):
            fit_completion(curved_season(2022, 1), context="automatic")


if __name__ == "__main__":
    unittest.main()
