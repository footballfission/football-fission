"""Contrast recovery and failures under crossed versus confounded partnerships."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference import design
from hierarchical_football.inference.partial_pooling import fit_block_completion
from tests.inference_fixtures import passing_frame


def crossed_targets(seed=0, games=16):
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(games):
        for qb in ("q1", "q2"):
            for receiver, effect in (("r1", 1.2), ("r2", -1.2)):
                for _ in range(15):
                    rate = 1 / (1 + np.exp(-(effect + (0.2 if qb == "q1" else -0.2))))
                    rows.append(dict(season=2022, game_id=f"g{game}", qb_id=qb,
                                     receiver_id=receiver, complete_pass=int(rng.random() < rate)))
    return passing_frame(*rows)


class ContrastUncertainty(unittest.TestCase):
    def run_contrast(self, train=None, **kwargs):
        function = getattr(design, "bootstrap_contrast", None)
        self.assertIsNotNone(function, "player contrasts need refitted game-bootstrap uncertainty")
        train = crossed_targets() if train is None else train
        return function(train, train.head(40), identity="receiver_id", left="r1", right="r2",
                        fixed={"qb_id": "q1"}, identity_penalties={"qb_id": 1, "receiver_id": 1},
                        draws=20, seed=9, **kwargs)

    def test_crossed_data_recovers_the_direction_with_reproducible_uncertainty(self):
        report = self.run_contrast()
        self.assertEqual(report, self.run_contrast())
        self.assertEqual(report["status"], "ok")
        self.assertGreater(report["interval"]["lower"], 0)
        self.assertGreater(report["contrast"]["difference"], 0.4)
        self.assertEqual(report["successful_draws"], 20)
        self.assertEqual(report["contrast"]["arms"]["left"]["support"]["supported_share"], 1)

    def test_disconnected_partnerships_do_not_get_a_misleading_interval(self):
        train = crossed_targets().filter(((pl.col("qb_id") == "q1") & (pl.col("receiver_id") == "r1")) |
                                         ((pl.col("qb_id") == "q2") & (pl.col("receiver_id") == "r2")))
        report = self.run_contrast(train)
        self.assertEqual(report["status"], "disconnected_players")
        self.assertIsNone(report["interval"])
        # The same successful partnerships admit incompatible credit assignments.
        terms = []
        for qb, receiver in ((0.1, 100), (100, 0.1)):
            fit = fit_block_completion(train, identity_penalties={"qb_id": qb, "receiver_id": receiver})
            terms.append(next(t["term"] for t in fit.identity_terms(identity="receiver_id")
                              if t["player"] == "r1"))
        self.assertGreater(abs(terms[0] - terms[1]), 0.5)

    def test_one_game_cannot_support_a_game_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "two games"):
            self.run_contrast(crossed_targets(games=1))

    def test_reference_context_outside_player_support_has_no_interval(self):
        train = crossed_targets().with_columns(
            pl.when(pl.col("receiver_id") == "r1").then(pl.lit(2.0))
            .otherwise(pl.lit(25.0)).alias("air_yards"))
        report = self.run_contrast(train)
        self.assertEqual(report["status"], "unsupported_context")
        self.assertIsNone(report["interval"])

    def test_unobserved_player_is_refused_instead_of_becoming_an_average_player(self):
        train = crossed_targets().with_columns(pl.lit("other").alias("receiver_id"))
        with self.assertRaisesRegex(ValueError, "observed"):
            self.run_contrast(train)

    def test_missing_players_in_resamples_are_reported_without_a_partial_interval(self):
        train = crossed_targets(games=2).with_columns(
            pl.when(pl.col("game_id") == "g0").then(pl.lit("r1"))
            .otherwise(pl.lit("r2")).alias("receiver_id"))
        report = self.run_contrast(train)
        self.assertEqual(report["status"], "incomplete_resamples")
        self.assertIsNone(report["interval"])
        self.assertGreater(report["failures"]["missing_player"], 0)


if __name__ == "__main__":
    unittest.main()
