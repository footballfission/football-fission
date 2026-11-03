"""The study protocol freezes selection before scoring expanding later seasons."""
import importlib.util
import unittest

import numpy as np
import polars as pl

from tests.test_inference_partial_pooling import season_of


def study_table():
    return pl.concat([season_of((("qb1", 0.8), ("qb2", 0.2)), year, rows=50, seed=year)
                      for year in (2022, 2023, 2024, 2025)])


class PassingStudy(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("hierarchical_football.inference.study"),
                             "passing research needs a reproducible study composer")
        from hierarchical_football.inference.study import PassingStudySpec, run_passing_study
        return PassingStudySpec, run_passing_study

    def spec(self, **changes):
        cls, _ = self.api()
        return cls(**dict(dict(train_seasons=(2022,), selection_season=2023,
                               evaluation_seasons=(2024, 2025), contexts=("linear",),
                               penalties=(10.0,), draws=20), **changes))

    def test_expanding_folds_record_models_transfer_and_row_aligned_predictions(self):
        _, run = self.api()
        result = run(study_table(), spec=self.spec())
        folds = result["report"]["folds"]
        self.assertEqual([fold["train_seasons"] for fold in folds], [[2022, 2023], [2022, 2023, 2024]])
        self.assertEqual(result["predictions"].height, 200)
        for fold in folds:
            self.assertEqual(set(fold["models"]), {"linear_context", "context", "qb", "receiver", "both"})
            self.assertEqual(sum(c["rows"] for c in fold["transfer"]["cohorts"].values()), 100)
            self.assertIn("both", fold["comparison"]["differences"])
            self.assertTrue(fold["models"]["both"]["in_sample"]["converged"])
            self.assertEqual(set(fold["models"]["qb"]["identity_levels"]), {"qb_id"})
            self.assertEqual(set(fold["models"]["receiver"]["identity_levels"]), {"receiver_id"})
        self.assertFalse(result["report"]["untouched_holdout_verified"])
        self.assertEqual(result["report"]["purpose"], "development")

    def test_outer_labels_do_not_change_selection_or_same_fold_predictions(self):
        _, run = self.api()
        table = study_table()
        a = run(table, spec=self.spec(evaluation_seasons=(2024,)))
        table = table.with_columns(pl.when(pl.col("season") == 2024)
                                  .then(1 - pl.col("complete_pass"))
                                  .otherwise(pl.col("complete_pass")).alias("complete_pass"))
        b = run(table, spec=self.spec(evaluation_seasons=(2024,)))
        self.assertEqual(a["report"]["selection"], b["report"]["selection"])
        for column in ("p_context", "p_both"):
            np.testing.assert_array_equal(a["predictions"][column], b["predictions"][column])

    def test_invalid_or_overlapping_seasons_and_reserved_seasons_fail(self):
        for changes in (dict(selection_season=2022), dict(evaluation_seasons=(2023,)),
                        dict(evaluation_seasons=(2025, 2024)), dict(train_seasons=(2022, 2022)),
                        dict(reserved_seasons=(2024,)), dict(purpose="confirmatory"),
                        dict(penalties=(float("nan"),)), dict(draws=0)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.spec(**changes)

    def test_reserved_and_unrequested_rows_never_enter_the_study(self):
        _, run = self.api()
        table = study_table()
        spec = self.spec(evaluation_seasons=(2024,), reserved_seasons=(2025,))
        # Even corrupt outcomes outside requested seasons cannot influence model fitting.
        changed = table.with_columns(pl.when(pl.col("season") == 2025).then(pl.lit(8))
                                    .otherwise(pl.col("complete_pass")).alias("complete_pass"))
        a, b = run(table, spec=spec), run(changed, spec=spec)
        self.assertEqual(a["report"], b["report"])
        self.assertTrue(a["predictions"].equals(b["predictions"]))

    def test_missing_evaluation_season_is_an_error(self):
        _, run = self.api()
        with self.assertRaisesRegex(ValueError, "missing.*2025"):
            run(study_table().filter(pl.col("season") < 2025), spec=self.spec())

    def test_selected_contrasts_use_only_training_contexts(self):
        _, run = self.api()
        result = run(study_table(), spec=self.spec(evaluation_seasons=(2024,)), contrast_draws=20,
                     contrasts=({"identity": "receiver_id", "left": "wr1", "right": "wr2",
                                 "fixed": {"qb_id": "qb1"}, "population_position": "WR"},))
        contrast = result["report"]["folds"][0]["contrasts"][0]
        self.assertEqual(contrast["reference_population"]["seasons"], [2022, 2023])
        self.assertEqual(contrast["reference_population"]["rows"], 200)
        self.assertIn(contrast["status"], ("ok", "incomplete_resamples"))


if __name__ == "__main__":
    unittest.main()
