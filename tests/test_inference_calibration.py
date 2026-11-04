"""Repeated known-truth trials distinguish bias, coverage and unavailable intervals."""
import json
import unittest
from unittest.mock import patch

import polars as pl

from hierarchical_football.inference.calibration import CalibrationSpec, run_calibration
from hierarchical_football.inference.simulation import simulate_passing


class ContrastCalibrationTests(unittest.TestCase):
    def test_real_crossed_trials_reveal_strong_shrinkage_bias(self):
        spec = CalibrationSpec(trials=2, draws=20, games=8, repeats=3,
                               penalties=(1.0, 1000.0), scenarios=('crossed',), seed=17)
        result = run_calibration(spec)
        self.assertTrue(result['report']['synthetic'])
        self.assertFalse(result['report']['real_player_coverage_verified'])
        summaries = result['report']['summaries']
        self.assertEqual([s['available_intervals'] for s in summaries], [2, 2])
        self.assertGreater(abs(summaries[1]['bias']), abs(summaries[0]['bias']) + 0.1)
        self.assertEqual(summaries[1]['covered_intervals'], 0)
        self.assertGreater(summaries[1]['coverage_wilson_95']['upper'], 0)
        rows = result['trials'].to_dicts()
        self.assertEqual(rows[0]['dataset_seed'], rows[1]['dataset_seed'])
        self.assertEqual(rows[0]['bootstrap_seed'], rows[1]['bootstrap_seed'])
        self.assertNotEqual(rows[0]['dataset_seed'], rows[2]['dataset_seed'])
        self.assertEqual(rows, run_calibration(spec)['trials'].to_dicts())
        json.dumps(result['report'], allow_nan=False)

    def test_nested_design_records_unavailable_intervals_and_estimator_bias(self):
        result = run_calibration(CalibrationSpec(trials=2, draws=20, games=3, repeats=2,
                                                penalties=(1.0,), scenarios=('nested',)))
        summary = result['report']['summaries'][0]
        self.assertEqual(summary['statuses'], {'disconnected_players': 2})
        self.assertEqual(summary['available_intervals'], 0)
        self.assertIsNone(summary['coverage_given_available'])
        self.assertIsNone(summary['coverage_wilson_95'])
        self.assertIsNone(summary['mean_interval_width'])
        self.assertEqual(summary['covered_over_all_trials'], 0)
        self.assertIsNotNone(summary['bias'])
        self.assertTrue(all(row['covered'] is None for row in result['trials'].to_dicts()))

    def test_partial_availability_does_not_drop_failed_trials_from_denominator(self):
        replies = [dict(status='ok', contrast={'difference': 0.3},
                        interval={'lower': 0.0, 'upper': 1.0}),
                   dict(status='incomplete_resamples', contrast={'difference': 0.1}, interval=None)]
        with patch('hierarchical_football.inference.calibration.bootstrap_contrast', side_effect=replies):
            result = run_calibration(CalibrationSpec(trials=2, draws=20, penalties=(1.0,),
                                                    scenarios=('crossed',)))
        s = result['report']['summaries'][0]
        self.assertEqual(s['trials'], 2)
        self.assertEqual(s['interval_availability'], 0.5)
        self.assertEqual(s['coverage_given_available'], 1.0)
        self.assertEqual(s['covered_over_all_trials'], 0.5)
        self.assertAlmostEqual(s['bias'], 0.2 - result['report']['truth_by_scenario']['crossed']['difference'])
        self.assertAlmostEqual(s['mean_interval_width'], 1.0)

    def test_bad_specs_fail_before_simulation(self):
        for kwargs in ({'trials': 1}, {'trials': True}, {'draws': 19}, {'games': 1},
                       {'repeats': 0}, {'seed': -1}, {'seed': True}, {'level': 1.0},
                       {'penalties': ()}, {'penalties': (1, 1)}, {'penalties': (float('nan'),)},
                       {'penalties': (True,)}, {'scenarios': ()}, {'scenarios': ('other',)},
                       {'scenarios': ('crossed', 'crossed')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CalibrationSpec(**kwargs)

    def test_single_outcome_trials_remain_visible_without_a_point_estimate(self):
        data = simulate_passing(games=2, repeats=1)
        data['train'] = data['train'].with_columns(pl.lit(1).alias('complete_pass'))
        with patch('hierarchical_football.inference.calibration.simulate_passing', return_value=data):
            result = run_calibration(CalibrationSpec(trials=2, draws=20, penalties=(1.0,),
                                                    scenarios=('crossed',)))
        s = result['report']['summaries'][0]
        self.assertEqual(s['statuses'], {'single_outcome': 2})
        self.assertEqual(s['point_estimates'], 0)
        self.assertIsNone(s['bias'])
        self.assertIsNone(s['rmse'])
        self.assertEqual(result['trials'].height, 2)
        json.dumps(result['report'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
