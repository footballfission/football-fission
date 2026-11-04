"""Shared reference contexts must be supported by both arms in every period."""
import unittest

import polars as pl

from hierarchical_football.inference import design
from tests.inference_fixtures import passing_frame


def support_window(season, missing_deep=False):
    rows = []
    for receiver in ("r1", "r2"):
        for depth in (2.0, 12.0):
            count = 1 if missing_deep and receiver == "r2" and depth == 12 else 4
            for i in range(count):
                rows.append(dict(season=season, game_id=f"{season}_{i}", qb_id="q1",
                                 receiver_id=receiver, air_yards=depth, complete_pass=i % 2))
    return passing_frame(*rows)


class SharedPopulation(unittest.TestCase):
    def select(self, windows, population, **kwargs):
        function = getattr(design, "shared_passing_population", None)
        self.assertIsNotNone(function, "comparisons need an observed shared reference population")
        return function(windows, population, identity="receiver_id", left="r1", right="r2",
                        fixed={"qb_id": "q1"}, min_rows=3, **kwargs)

    def test_reference_restricts_to_cells_supported_in_every_window(self):
        first, second = support_window(2022), support_window(2024, missing_deep=True)
        result = self.select({"early": first, "late": second}, first)
        self.assertEqual(result["population"].height, 8)
        self.assertEqual(set(result["population"]["air_yards"]), {2.0})
        self.assertEqual(result["report"]["retained_share"], 0.5)
        self.assertEqual(result["report"]["shared_cells"], 1)
        self.assertEqual(result["report"]["status"], "ok")

    def test_fixed_partner_requires_joint_support_not_just_player_exposure(self):
        first = support_window(2022)
        second = support_window(2024).with_columns(pl.lit("q2").alias("qb_id"))
        result = self.select({"early": first, "late": second}, first)
        self.assertEqual(result["report"]["status"], "no_shared_support")
        self.assertEqual(result["population"].height, 0)
        self.assertEqual(result["population"].schema, first.schema)

    def test_outcomes_do_not_determine_the_shared_population(self):
        first, second = support_window(2022), support_window(2024, missing_deep=True)
        before = self.select({"a": first, "b": second}, first)
        changed = second.with_columns((1 - pl.col("complete_pass")).alias("complete_pass"))
        after = self.select({"a": first, "b": changed}, first)
        self.assertEqual(before["report"], after["report"])
        self.assertTrue(before["population"].equals(after["population"]))

    def test_empty_windows_or_invalid_thresholds_are_refused(self):
        function = getattr(design, "shared_passing_population", None)
        self.assertIsNotNone(function)
        for windows, threshold in (({}, 3), ({"a": support_window(2022)}, 0)):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                function(windows, support_window(2022), identity="receiver_id", left="r1", right="r2",
                         min_rows=threshold)


if __name__ == "__main__":
    unittest.main()
