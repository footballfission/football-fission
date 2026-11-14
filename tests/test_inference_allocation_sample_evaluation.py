"""Marginal calibration checks keep zeros and verify conditional pool conservation."""
import json
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.allocation_evaluation import conditional_allocation_sample_scores
from tests.test_inference_allocation_evaluation import example


def sampled_example():
    prediction,panel,candidates=example()
    sampled = {'candidate_keys':prediction['candidate_keys'],'team_game_keys':prediction['team_game_keys'],
               'candidate_samples':np.array([[0,1,2,3],[1,1,1,1]]),
               'other_samples':np.array([[3,2,1,0]])}
    return sampled,panel,candidates


class ConditionalAllocationCalibration(unittest.TestCase):
    def test_known_sample_errors_crps_and_intervals(self):
        sampled,panel,candidates=sampled_example()
        result=conditional_allocation_sample_scores(sampled,panel,candidates)
        metrics=result['report']['candidates']
        self.assertAlmostEqual(metrics['mae'],.25)
        self.assertAlmostEqual(metrics['crps'],.1875)
        self.assertEqual(metrics['coverage_50'],1.0)
        self.assertEqual(metrics['coverage_90'],1.0)
        self.assertAlmostEqual(metrics['mean_width_90'],1.35)
        self.assertAlmostEqual(result['report']['other']['crps'],.375)
        self.assertEqual(result['candidate_scores']['crps'].to_list(),[.375,0.0])
        self.assertTrue(result['report']['conditional_on_observed_totals'])
        self.assertTrue(result['report']['conservation_verified'])
        json.dumps(result['report'],allow_nan=False)

    def test_observation_row_order_does_not_change_identity_aligned_scores(self):
        sampled,panel,candidates=sampled_example()
        original=conditional_allocation_sample_scores(sampled,panel,candidates)
        changed=conditional_allocation_sample_scores(sampled,panel,candidates.reverse())
        self.assertTrue(original['candidate_scores'].equals(changed['candidate_scores']))
        sampled['candidate_keys']=sampled['candidate_keys'].reverse()
        sampled['candidate_samples']=sampled['candidate_samples'][::-1]
        changed=conditional_allocation_sample_scores(sampled,panel,candidates)
        self.assertTrue(original['candidate_scores'].sort('player_id').equals(
            changed['candidate_scores'].sort('player_id')))

    def test_empty_rosters_have_null_candidate_summary_and_score_other(self):
        sampled,panel,candidates=sampled_example()
        sampled.update(candidate_keys=sampled['candidate_keys'].head(0),
                       candidate_samples=np.zeros((0,4),dtype=int),other_samples=np.full((1,4),4))
        panel=panel.with_columns(pl.lit(0).alias('candidates'),pl.lit(0).alias('candidate_targets'),
                                 pl.lit(4).alias('other_targets'))
        result=conditional_allocation_sample_scores(sampled,panel,candidates.head(0))
        self.assertIsNone(result['report']['candidates'])
        self.assertEqual(result['candidate_scores'].height,0)
        self.assertEqual(result['report']['other']['mae'],0)
        self.assertEqual(result['report']['other']['crps'],0)
        json.dumps(result['report'],allow_nan=False)

    def test_zero_target_candidates_remain_in_error_and_coverage_denominators(self):
        sampled,panel,candidates=sampled_example()
        candidates=candidates.with_columns(pl.when(pl.col('player_id')=='r2').then(0)
                                            .otherwise(pl.col('targets')).alias('targets'))
        panel=panel.with_columns(pl.lit(2).alias('candidate_targets'),pl.lit(2).alias('other_targets'))
        result=conditional_allocation_sample_scores(sampled,panel,candidates)
        self.assertEqual(result['report']['candidates']['rows'],2)
        self.assertEqual(result['report']['candidates']['coverage_50'],.5)
        self.assertAlmostEqual(result['report']['candidates']['mae'],.75)

    def test_nonconserving_or_invalid_sample_matrices_fail(self):
        for change in ('conservation','negative','fractional','draw_count','zero_draws','keys'):
            sampled,panel,candidates=sampled_example()
            if change=='conservation': sampled['other_samples'][0,0]+=1
            elif change=='negative': sampled['candidate_samples'][0,0]=-1
            elif change=='fractional': sampled['candidate_samples']=sampled['candidate_samples'].astype(float)+.5
            elif change=='draw_count': sampled['other_samples']=sampled['other_samples'][:,:-1]
            elif change=='zero_draws':
                sampled['candidate_samples']=sampled['candidate_samples'][:,:0]
                sampled['other_samples']=sampled['other_samples'][:,:0]
            else: sampled['candidate_keys']=sampled['candidate_keys'].with_columns(pl.lit('wrong').alias('game_id'))
            with self.subTest(change=change),self.assertRaises(ValueError):
                conditional_allocation_sample_scores(sampled,panel,candidates)


if __name__=='__main__':
    unittest.main()
