"""Independent passing priors must not leak outer-season outcomes into selection."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference import partial_pooling as pooling
from tests.test_inference_partial_pooling import season_of


class PassingBlockPooling(unittest.TestCase):
    def test_opponent_penalty_can_remove_team_terms_without_removing_player_terms(self):
        table = self.train.with_columns(pl.when(pl.col("qb_id") == "qb1")
                                       .then(pl.lit("D1")).otherwise(pl.lit("D2")).alias("defteam"))
        fit = pooling.fit_block_completion(table, context="expanded", opponent_penalty=1e8,
                                          identity_penalties={"qb_id": 1, "receiver_id": 1})
        opponent_terms = [abs(value) for key, value in fit.coefficients().items() if key.startswith("defteam=")]
        self.assertLess(max(opponent_terms), 1e-5)
        self.assertGreater(max(abs(term["term"]) for term in fit.identity_terms(identity="qb_id")), 0.5)
        self.assertEqual(fit.report()["block_penalties"]["opponent"], 1e8)

    def test_context_only_fit_uses_the_same_opponent_prior(self):
        from hierarchical_football.inference.baselines import fit_completion
        fit = pooling.fit_block_completion(self.train, identities=(), identity_penalties={},
                                          context="expanded", opponent_penalty=1)
        ordinary = fit_completion(self.train, context="expanded")
        np.testing.assert_allclose(fit.predict(self.train), ordinary.predict(self.train), atol=1e-9)
        self.assertEqual(fit.report()["identity_levels"], {})

    def test_opponent_penalties_are_finite_positive_and_require_opponent_context(self):
        for penalty, context in ((0, "expanded"), (float("inf"), "expanded"),
                                 (float("nan"), "expanded"), (10, "depth")):
            with self.subTest(penalty=penalty, context=context), self.assertRaisesRegex(ValueError, "opponent"):
                self.fit(identity_penalties={"qb_id": 1, "receiver_id": 1},
                         context=context, opponent_penalty=penalty)

    def setUp(self):
        self.train = season_of((("qb1", 0.85), ("qb2", 0.15)), 2022, rows=100)

    def fit(self, **kwargs):
        fitter = getattr(pooling, "fit_block_completion", None)
        self.assertIsNotNone(fitter, "completion fitting needs independent identity-block penalties")
        return fitter(self.train, **kwargs)

    def test_equal_penalties_reproduce_the_existing_pooled_fit(self):
        result = self.fit(identity_penalties={"qb_id": 10, "receiver_id": 10})
        old = pooling.fit_pooled_completion(self.train, identity_penalty=10)
        np.testing.assert_allclose(result.predict(self.train), old.predict(self.train), atol=1e-8)

    def test_shrinking_only_quarterbacks_preserves_receiver_penalty(self):
        fit = self.fit(identity_penalties={"qb_id": 1e8, "receiver_id": 2})
        self.assertLess(max(abs(t["term"]) for t in fit.identity_terms(identity="qb_id")), 1e-5)
        self.assertEqual(fit.report()["block_penalties"]["identities"],
                         {"qb_id": 1e8, "receiver_id": 2})

    def test_expanded_context_is_shared_with_the_context_only_baseline(self):
        fit = self.fit(identity_penalties={"qb_id": 100, "receiver_id": 30}, context="expanded")
        self.assertIn("defteam=B", fit.design.columns)
        self.assertIn("air_yards_above_10", fit.design.columns)

    def test_invalid_or_misspelled_blocks_are_refused(self):
        for penalties in ({"qb_id": 2}, {"qb_id": 1, "receiver_id": float("inf")},
                          {"qb_id": -1, "receiver_id": 2}, {"qb": 1, "receiver_id": 2}):
            with self.subTest(penalties=penalties):
                with self.assertRaisesRegex(ValueError, "penalt"):
                    self.fit(identity_penalties=penalties)

    def test_selection_never_uses_outer_outcomes(self):
        selector = getattr(pooling, "select_passing_penalties", None)
        self.assertIsNotNone(selector, "passing selection needs an explicit inner season")
        later = season_of((("qb1", 0.85), ("qb2", 0.15)), 2023, rows=100, seed=1)
        outer = season_of((("qb1", 0.1), ("qb2", 0.9)), 2024, rows=100, seed=2)
        table = pl.concat([self.train, later, outer])
        args = dict(train_seasons=[2022], holdout_season=2023,
                    penalties=(1.0, 100.0), contexts=("linear", "expanded"))
        selected = selector(table, **args)
        changed = table.with_columns(pl.when(pl.col("season") == 2024)
                                     .then(1 - pl.col("complete_pass"))
                                     .otherwise(pl.col("complete_pass")).alias("complete_pass"))
        self.assertEqual(selected, selector(changed, **args))
        self.assertEqual(len(selected["scores"]), 8)
        self.assertIn(selected["selected"]["qb_id"], (1, 100))
        self.assertTrue(all(row["converged"] for row in selected["scores"]))

    def test_opponent_selection_does_not_repeat_contexts_without_opponents(self):
        later = season_of((("qb1", 0.85), ("qb2", 0.15)), 2023, rows=100, seed=1)
        table = pl.concat([self.train, later])
        selected = pooling.select_passing_penalties(
            table, train_seasons=[2022], holdout_season=2023, penalties=(10,),
            contexts=("depth", "depth_opponent"), opponent_penalties=(1, 100))
        self.assertEqual(len(selected["scores"]), 3)
        depth = [row for row in selected["scores"] if row["context"] == "depth"]
        self.assertEqual(len(depth), 1)
        self.assertIsNone(depth[0]["opponent_penalty"])
        self.assertIn("opponent_penalty", selected["selected"])

    def test_opponent_selection_ignores_future_outcomes(self):
        table = pl.concat([self.train, season_of((("qb1", 0.7), ("qb2", 0.3)), 2023, rows=60),
                           season_of((("qb1", 0.1), ("qb2", 0.9)), 2024, rows=60)])
        args = dict(train_seasons=[2022], holdout_season=2023, penalties=(10,),
                    contexts=("depth", "expanded"), opponent_penalties=(1, 100))
        selected = pooling.select_passing_penalties(table, **args)
        changed = table.with_columns(pl.when(pl.col("season") == 2024).then(1 - pl.col("complete_pass"))
                                    .otherwise(pl.col("complete_pass")).alias("complete_pass"))
        self.assertEqual(selected, pooling.select_passing_penalties(changed, **args))

    def test_invalid_opponent_grids_are_refused(self):
        table = pl.concat([self.train, self.train.with_columns(pl.lit(2023).cast(pl.Int32).alias("season"))])
        # Game keys include their season in real data; make this fixture unambiguous too.
        table = table.with_columns((pl.col("season").cast(pl.String) + pl.col("game_id")).alias("game_id"))
        for grid in ((), (0,), (float("nan"),), (10, 10)):
            with self.subTest(grid=grid), self.assertRaisesRegex(ValueError, "opponent"):
                pooling.select_passing_penalties(table, train_seasons=[2022], holdout_season=2023,
                                                 opponent_penalties=grid)


if __name__ == "__main__":
    unittest.main()
