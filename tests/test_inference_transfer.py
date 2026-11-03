"""A new partnership is different from an unseen player."""
import unittest

import numpy as np

from hierarchical_football.inference import evaluation
from tests.inference_fixtures import passing_frame


class PassingTransfer(unittest.TestCase):
    def setUp(self):
        self.train = passing_frame(dict(season=2022, qb_id="q1", receiver_id="r1"),
                                   dict(season=2022, qb_id="q2", receiver_id="r2"))
        self.later = passing_frame(*[
            dict(season=2023, game_id=f"g{i}", qb_id=q, receiver_id=r, complete_pass=i % 2)
            for i, (q, r) in enumerate([("q1", "r1"), ("q1", "r2"), ("new", "r1"),
                                        ("q2", "new"), ("new", "new")])])

    def transfer(self, **kwargs):
        function = getattr(evaluation, "passing_transfer", None)
        self.assertIsNotNone(function, "evaluation must separate new partnerships from unseen players")
        return function(self.train, self.later, **kwargs)

    def test_five_disjoint_cohorts_partition_the_scored_rows(self):
        report = self.transfer(predictions={"context": np.full(5, 0.5),
                                             "identity": np.full(5, 0.75)}, draws=30)
        self.assertEqual(report["assignments"], ["known_pair", "new_pair", "unseen_qb",
                                                 "unseen_receiver", "unseen_both"])
        self.assertEqual(sum(c["rows"] for c in report["cohorts"].values()), 5)
        for entry in report["cohorts"].values():
            self.assertEqual(entry["rows"], 1)
            self.assertEqual(entry["games"], 1)
            self.assertTrue(entry["few_games"])
            self.assertIsNone(entry["comparison"])
            self.assertIn("identity", entry["metrics"])

    def test_comparison_resamples_games_inside_a_cohort(self):
        extra = passing_frame(dict(season=2023, game_id="extra", qb_id="q1", receiver_id="r2"))
        import polars as pl
        self.later = pl.concat([self.later, extra])
        report = self.transfer(predictions={"context": [0.5] * 6, "identity": [0.7] * 6}, draws=20)
        entry = report["cohorts"]["new_pair"]
        self.assertEqual(entry["games"], 2)
        self.assertIn("identity", entry["comparison"]["differences"])

    def test_no_rows_means_no_score_rather_than_a_fabricated_zero(self):
        self.later = self.later.head(1)
        report = self.transfer(predictions={"context": [0.5]}, draws=10)
        self.assertEqual(report["cohorts"]["new_pair"]["rows"], 0)
        self.assertIsNone(report["cohorts"]["new_pair"]["comparison"])

    def test_classification_does_not_use_outcomes(self):
        before = self.transfer(predictions={"context": [0.5] * 5}, draws=10)["assignments"]
        self.later = self.later.with_columns((1 - self.later["complete_pass"]).alias("complete_pass"))
        self.assertEqual(before, self.transfer(predictions={"context": [0.5] * 5},
                                               draws=10)["assignments"])

    def test_nonchronological_data_and_invalid_predictions_are_refused(self):
        with self.assertRaisesRegex(ValueError, "earlier"):
            self.train = self.train.with_columns(self.train["season"] + 1)
            self.transfer(predictions={"context": [0.5] * 5})
        self.train = self.train.with_columns(self.train["season"] - 1)
        for values in ([0.5], [float("nan")] * 5, [2.0] * 5):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.transfer(predictions={"context": values})


if __name__ == "__main__":
    unittest.main()
