"""Known player effects must remain explicit in synthetic identification checks."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.contracts import PASSING_CONTRACT, validate
from hierarchical_football.inference.simulation import simulate_passing


class PassingSimulationTests(unittest.TestCase):
    def test_crossed_design_is_canonical_balanced_and_seeded(self):
        result = simulate_passing(games=3, repeats=2, seed=4)
        table = result['train']
        validate(table, PASSING_CONTRACT)
        self.assertEqual(table.height, 3 * 2 * 2 * 2 * 2)
        self.assertEqual(table['game_id'].n_unique(), 3)
        self.assertTrue(table.equals(simulate_passing(games=3, repeats=2, seed=4)['train']))
        self.assertFalse(table.equals(simulate_passing(games=3, repeats=2, seed=5)['train']))
        self.assertEqual(table.group_by('qb_id', 'receiver_id', 'air_yards').len()['len'].unique().to_list(), [6])
        self.assertTrue(all(str(value).startswith('synthetic_') for value in table['qb_id']))
        self.assertTrue(result['synthetic'])

    def test_truth_averages_probabilities_under_fixed_partner_and_reference(self):
        result = simulate_passing(seed=2)
        q = result['population']
        validate(q, PASSING_CONTRACT)
        self.assertEqual(q['air_yards'].to_list(), [5.0, 15.0])
        base = 0.4 + 0.25 - 0.04 * (q['air_yards'].to_numpy() - 10)
        sigmoid = lambda value: 1 / (1 + np.exp(-value))
        expected = float(np.mean(sigmoid(base + 0.6) - sigmoid(base - 0.6)))
        self.assertAlmostEqual(result['truth']['difference'], expected)
        self.assertEqual(result['truth']['fixed'], {'qb_id': 'synthetic_q1'})
        self.assertEqual(result['truth']['scale'], 'probability')

    def test_scenarios_change_support_without_changing_effects(self):
        nested = simulate_passing(scenario='nested', games=2, repeats=1)
        self.assertEqual(nested['train'].select('qb_id', 'receiver_id').unique().height, 2)
        self.assertEqual(nested['truth'], simulate_passing(games=2, repeats=1)['truth'])
        confounded = simulate_passing(scenario='depth_confounded', games=2, repeats=1)
        pairs = confounded['train'].select('receiver_id', 'air_yards').unique()
        self.assertEqual(pairs.height, 2)
        self.assertEqual(set(pairs['air_yards']), {6.0, 9.0})
        self.assertEqual(confounded['train'].select('qb_id', 'receiver_id').unique().height, 4)
        self.assertEqual(confounded['population']['air_yards'].to_list(), [6.0, 9.0])

    def test_invalid_generation_parameters_fail(self):
        for kwargs in ({'games': 1}, {'games': True}, {'repeats': 0}, {'repeats': 1.5},
                       {'seed': -1}, {'seed': True}, {'scenario': 'other'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_passing(**kwargs)


if __name__ == '__main__':
    unittest.main()
