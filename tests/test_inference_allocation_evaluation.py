"""Conditional allocation scores include exact count mass and the OTHER category."""
import json
import math
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.allocation_evaluation import allocation_log_scores
from tests.inference_fixtures import panel_frame, candidate_frame


def example():
    panel = panel_frame(dict(game_id='g', targets=4, candidates=2, candidate_targets=3, other_targets=1))
    candidates = candidate_frame(dict(game_id='g', player_id='r1', targets=2),
                                  dict(game_id='g', player_id='r2', candidate_index=1, targets=1))
    prediction = {'team_game_keys': panel.select('game_id','team'),
                  'candidate_keys': candidates.select('game_id','team','candidate_index','player_id'),
                  'shares': np.array([0.5,0.25]), 'other_shares': np.array([0.25])}
    return prediction, panel, candidates


class AllocationLogScores(unittest.TestCase):
    def test_known_multinomial_mass_includes_counts_and_other(self):
        prediction, panel, candidates = example()
        result = allocation_log_scores(prediction, panel, candidates)
        # P(2,1,1 | N=4, p=.5,.25,.25) = 12 * .5**2 * .25**2.
        self.assertAlmostEqual(result['scores']['negative_log_likelihood'][0], -math.log(0.1875))
        self.assertAlmostEqual(result['report']['mean_negative_log_likelihood'], -math.log(0.1875))
        self.assertAlmostEqual(result['report']['target_weighted_log_loss'],
                                -(2*math.log(.5)+2*math.log(.25))/4)
        self.assertEqual(result['report']['status'], 'ok')
        self.assertEqual(result['report']['targets'], 4)
        json.dumps(result['report'], allow_nan=False)

    def test_predictions_and_observations_align_by_identity_not_row_position(self):
        prediction, panel, candidates = example()
        original = allocation_log_scores(prediction, panel, candidates)
        reordered = allocation_log_scores(prediction, panel, candidates.reverse())
        self.assertTrue(original['scores'].equals(reordered['scores']))
        prediction['candidate_keys'] = prediction['candidate_keys'].reverse()
        prediction['shares'] = prediction['shares'][::-1]
        self.assertTrue(original['scores'].equals(allocation_log_scores(prediction,panel,candidates)['scores']))

    def test_empty_roster_and_zero_target_pools_have_valid_zero_log_loss(self):
        prediction, panel, candidates = example()
        prediction.update(candidate_keys=prediction['candidate_keys'].head(0), shares=np.array([]),
                          other_shares=np.array([1.0]))
        panel = panel.with_columns(pl.lit(0).alias('candidates'), pl.lit(0).alias('candidate_targets'),
                                   pl.col('targets').alias('other_targets'))
        result = allocation_log_scores(prediction,panel,candidates.head(0))
        self.assertEqual(result['report']['mean_negative_log_likelihood'], 0)
        self.assertEqual(result['report']['target_weighted_log_loss'], 0)
        panel = panel.with_columns(pl.lit(0).alias('targets'), pl.lit(0).alias('other_targets'))
        result = allocation_log_scores(prediction,panel,candidates.head(0))
        self.assertEqual(result['report']['mean_negative_log_likelihood'], 0)
        self.assertIsNone(result['report']['target_weighted_log_loss'])
        prediction,panel,candidates=example()
        prediction.update(shares=np.array([0.0,0.5]),other_shares=np.array([0.5]))
        panel=panel.with_columns(pl.lit(0).alias('targets'),pl.lit(0).alias('candidate_targets'),
                                 pl.lit(0).alias('other_targets'))
        candidates=candidates.with_columns(pl.lit(0).alias('targets'))
        self.assertEqual(allocation_log_scores(prediction,panel,candidates)['report']['status'],'ok')

    def test_impossible_events_remain_infinite_and_do_not_get_dropped_from_summary(self):
        prediction, panel, candidates = example()
        panel = pl.concat([panel, panel.with_columns(pl.lit('g2').alias('game_id'))])
        candidates = pl.concat([candidates, candidates.with_columns(pl.lit('g2').alias('game_id'))])
        prediction = {'team_game_keys':panel.select('game_id','team'),
                      'candidate_keys':candidates.select('game_id','team','candidate_index','player_id'),
                      'shares':np.array([0.0,0.5,0.5,0.25]),'other_shares':np.array([0.5,0.25])}
        result = allocation_log_scores(prediction,panel,candidates)
        self.assertEqual(result['report']['status'], 'infinite_loss')
        self.assertEqual(result['report']['zero_probability_games'], 1)
        self.assertEqual(result['scores'].height, 2)
        self.assertTrue(math.isinf(result['scores']['negative_log_likelihood'][0]))
        self.assertIsNone(result['report']['mean_negative_log_likelihood'])
        self.assertIsNone(result['report']['target_weighted_log_loss'])
        json.dumps(result['report'], allow_nan=False)

    def test_missing_mismatched_or_duplicate_prediction_keys_are_refused(self):
        for change in ('missing','identity','duplicate','label'):
            prediction,panel,candidates=example()
            if change == 'missing':
                prediction['candidate_keys']=prediction['candidate_keys'].head(1)
                prediction['shares']=np.array([.75])
            elif change == 'identity':
                prediction['candidate_keys']=prediction['candidate_keys'].with_columns(
                    pl.when(pl.col('player_id')=='r1').then(pl.lit('wrong')).otherwise(pl.col('player_id')).alias('player_id'))
            elif change == 'duplicate':
                prediction['candidate_keys']=pl.concat([prediction['candidate_keys'],prediction['candidate_keys'].head(1)])
                prediction['shares']=np.array([.25,.25,.25])
            else:
                prediction['team_game_keys']=prediction['team_game_keys'].with_columns(pl.lit(4).alias('targets'))
            with self.subTest(change=change),self.assertRaises(ValueError):
                allocation_log_scores(prediction,panel,candidates)

    def test_invalid_probabilities_and_observed_counts_are_refused(self):
        for shares in ([.4,.25],[-.1,.85],[float('nan'),.75]):
            prediction,panel,candidates=example();prediction['shares']=np.array(shares)
            with self.subTest(shares=shares),self.assertRaises(ValueError):
                allocation_log_scores(prediction,panel,candidates)
        for value in (-1, .5, None):
            prediction,panel,candidates=example()
            with self.subTest(count=value),self.assertRaises(ValueError):
                allocation_log_scores(prediction,panel,candidates.with_columns(pl.lit(value).alias('targets')))


if __name__ == '__main__':
    unittest.main()
