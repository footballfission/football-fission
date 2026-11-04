"""Situation enters only the declared model and must be audited on the same inputs."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.baselines import fit_completion, passing_design, passing_context_sources
from hierarchical_football.inference.context_support import model_context_population
from hierarchical_football.inference.evaluation import completion_metrics
from tests.inference_fixtures import passing_frame


def situation_targets(seed=0, rows=600):
    rng = np.random.default_rng(seed)
    return passing_frame(*[
        dict(shotgun=i % 2, no_huddle=(i // 2) % 2, score_differential=float((i % 7) * 7 - 21),
             game_seconds_remaining=float((i % 60) * 60),
             complete_pass=int(rng.random() < (0.9 if i % 2 else 0.3))) for i in range(rows)])


class SituationCompletionModels(unittest.TestCase):
    def test_new_variant_has_declared_inputs_and_preserves_existing_design(self):
        table = situation_targets()
        plain = passing_design(table, context='expanded')
        new = passing_design(table, context='expanded_situation')
        self.assertNotIn('score_differential', plain.columns)
        self.assertIn('score_differential', new.columns)
        self.assertIn('game_seconds_remaining', new.columns)
        self.assertIn('shotgun=1', new.columns)
        self.assertIn('no_huddle=0', new.columns)
        self.assertIn('air_yards_above_10', new.columns)
        sources = passing_context_sources('expanded_situation')
        audit = model_context_population({'fit': table}, table, context='expanded_situation')['report']
        self.assertEqual(set(audit['windows']['fit']['bounds']), set(sources['continuous']))
        self.assertEqual(set(audit['windows']['fit']['levels']), set(sources['categorical']))

    def test_situation_signal_improves_new_season_predictions_without_player_effects(self):
        train, later = situation_targets(1), situation_targets(2)
        plain = fit_completion(train, context='expanded')
        new = fit_completion(train, context='expanded_situation')
        loss = lambda fitted: completion_metrics(later['complete_pass'].to_numpy(), fitted.predict(later))['log_loss']
        self.assertLess(loss(new), loss(plain) - 0.1)
        changed = later.with_columns((1 - pl.col('complete_pass')).alias('complete_pass'),
                                      pl.lit(999).alias('score_differential_post'))
        np.testing.assert_array_equal(new.predict(later), new.predict(changed))

    def test_missing_or_invalid_inputs_fail_in_training_and_holdout(self):
        table = situation_targets()
        fit = None
        for changed in (table.drop('shotgun'), table.with_columns(pl.lit(2).alias('no_huddle')),
                        table.with_columns(pl.lit(float('inf')).alias('score_differential'))):
            with self.assertRaises(ValueError):
                fit_completion(changed, context='expanded_situation')
        fit = fit_completion(table, context='expanded_situation')
        with self.assertRaises(ValueError):
            fit.predict(table.with_columns(pl.lit(0.5).alias('shotgun')))
        with self.assertRaises(ValueError):
            fit.predict(table.drop('game_seconds_remaining'))

    def test_new_context_domains_remove_unseen_flags_and_clock_score_extrapolation(self):
        train = situation_targets()
        q = train.head(3).with_columns(pl.Series('shotgun', [0, 1, 2]),
                                      pl.Series('score_differential', [0.0, 100.0, 0.0]),
                                      pl.Series('game_seconds_remaining', [float('nan'), 120.0, 120.0]))
        audit = model_context_population({'fit': train}, q, context='expanded_situation')['report']
        self.assertEqual(audit['retained_rows'], 0)
        detail = audit['windows']['fit']
        self.assertEqual(detail['unseen_categorical_rows']['shotgun'], 1)
        self.assertEqual(detail['outside_continuous_rows']['score_differential'], 1)
        self.assertEqual(detail['nonfinite_continuous_rows']['game_seconds_remaining'], 1)


if __name__ == '__main__':
    unittest.main()
