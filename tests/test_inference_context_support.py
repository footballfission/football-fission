"""Marginal model support detects contexts hidden by coarse player cells."""
import importlib.util
import unittest

import polars as pl

from tests.inference_fixtures import passing_frame


class ContextSupport(unittest.TestCase):
    def audit(self, windows, population, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec("hierarchical_football.inference.context_support"),
                             "reference comparisons need modeled context coverage")
        from hierarchical_football.inference.context_support import model_context_population
        return model_context_population(windows, population, **kwargs)

    def test_unknown_opponents_are_removed_only_when_the_model_uses_opponents(self):
        first = passing_frame(dict(defteam="D1"), dict(defteam="D2"))
        later = passing_frame(dict(defteam="D1"))
        strict = self.audit({"early": first, "late": later}, first, context="expanded")
        self.assertEqual(strict["population"].height, 1)
        self.assertEqual(strict["report"]["windows"]["late"]["unseen_categorical_rows"]["defteam"], 1)
        linear = self.audit({"early": first, "late": later}, first, context="linear")
        self.assertEqual(linear["population"].height, 2)

    def test_reference_values_must_be_inside_each_windows_observed_ranges(self):
        first = passing_frame(dict(air_yards=0.0), dict(air_yards=4.0), dict(air_yards=10.0))
        later = passing_frame(dict(air_yards=2.0), dict(air_yards=8.0))
        result = self.audit({"a": first, "b": later}, first)
        self.assertEqual(result["population"]["air_yards"].to_list(), [4.0])
        self.assertEqual(result["report"]["windows"]["b"]["outside_continuous_rows"]["air_yards"], 2)
        self.assertEqual(result["report"]["retained_share"], 1 / 3)

    def test_missing_numeric_context_is_exposed_instead_of_silently_imputed(self):
        train = passing_frame(dict(air_yards=1.0), dict(air_yards=10.0))
        population = passing_frame(dict(air_yards=None), dict(air_yards=4.0))
        result = self.audit({"a": train}, population)
        self.assertEqual(result["population"].height, 1)
        self.assertEqual(result["report"]["windows"]["a"]["nonfinite_continuous_rows"]["air_yards"], 1)

    def test_outcome_changes_leave_coverage_unchanged(self):
        train = passing_frame(dict(complete_pass=0), dict(complete_pass=1))
        a = self.audit({"a": train}, train)
        changed = train.with_columns((1 - pl.col("complete_pass")).alias("complete_pass"))
        b = self.audit({"a": changed}, train)
        self.assertEqual(a["report"], b["report"])
        self.assertTrue(a["population"].equals(b["population"]))

    def test_empty_windows_and_unknown_model_context_are_refused(self):
        for windows, context in (({}, "linear"), ({"a": passing_frame({})}, "automatic")):
            with self.subTest(context=context), self.assertRaises(ValueError):
                self.audit(windows, passing_frame({}), context=context)


if __name__ == "__main__":
    unittest.main()
