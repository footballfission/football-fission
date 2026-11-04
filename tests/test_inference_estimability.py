"""A connected partner graph cannot establish separation from continuous context."""
import unittest

import polars as pl

from hierarchical_football.inference.estimability import audit_contrast_estimability
from hierarchical_football.inference.partial_pooling import fit_block_completion
from hierarchical_football.inference.simulation import simulate_passing


class ContrastEstimabilityTests(unittest.TestCase):
    def audit(self, scenario='crossed', context='linear', population=None, penalty=1.0, **kwargs):
        data = simulate_passing(games=4, repeats=2, scenario=scenario, seed=13)
        fitted = fit_block_completion(data['train'], context=context,
                                      identity_penalties={'qb_id': penalty, 'receiver_id': penalty})
        return audit_contrast_estimability(
            fitted, data['population'] if population is None else population,
            identity='receiver_id', left='synthetic_r1', right='synthetic_r2',
            fixed={'qb_id': 'synthetic_q1'}, **kwargs)

    def test_dummy_gauges_do_not_invalidate_crossed_contrasts(self):
        audit = self.audit()
        self.assertEqual(audit['status'], 'estimable')
        self.assertLess(audit['rank'], audit['augmented_columns'])
        self.assertTrue(audit['log_odds_contrast']['estimable'])
        self.assertTrue(audit['probability_arms_estimable'])
        self.assertEqual(audit['arms']['left']['unestimable_rows'], 0)
        self.assertEqual(audit, self.audit(penalty=1000.0))

    def test_nested_partnerships_need_prior_information_for_counterfactual_arm(self):
        audit = self.audit('nested')
        self.assertEqual(audit['status'], 'prior_dependent')
        self.assertFalse(audit['log_odds_contrast']['estimable'])
        self.assertEqual(audit['arms']['left']['unestimable_rows'], 0)
        self.assertEqual(audit['arms']['right']['unestimable_rows'], 2)

    def test_depth_confounded_players_fail_despite_crossed_partners(self):
        audit = self.audit('depth_confounded')
        self.assertEqual(audit['status'], 'prior_dependent')
        self.assertFalse(audit['log_odds_contrast']['estimable'])
        self.assertEqual(audit['arms']['left']['unestimable_rows'], 1)
        self.assertEqual(audit['arms']['right']['unestimable_rows'], 1)

    def test_unseen_context_can_preserve_log_odds_difference_but_not_probability_arms(self):
        q = simulate_passing()['population'].with_columns(pl.lit('unseen_defense').alias('defteam'))
        audit = self.audit(context='expanded', population=q)
        self.assertTrue(audit['log_odds_contrast']['estimable'])
        self.assertFalse(audit['probability_arms_estimable'])
        self.assertEqual(audit['status'], 'prior_dependent')
        self.assertEqual(audit['arms']['left']['unestimable_rows'], 2)

    def test_invalid_tolerance_or_empty_reference_fails(self):
        for rtol in (0, -1, 1, float('nan'), True):
            with self.subTest(rtol=rtol), self.assertRaises(ValueError):
                self.audit(rtol=rtol)
        with self.assertRaisesRegex(ValueError, 'population'):
            self.audit(population=simulate_passing()['population'].head(0))

    def test_unknown_or_conflicting_player_requests_are_refused(self):
        data = simulate_passing(games=3)
        fitted = fit_block_completion(data['train'], identity_penalties={'qb_id': 1, 'receiver_id': 1})
        for request in ({'left': 'unobserved'}, {'right': 'synthetic_r1'},
                        {'fixed': {'receiver_id': 'synthetic_r1'}}, {'fixed': {'team': 'x'}},
                        {'identity': 'team'}):
            args = dict(identity='receiver_id', left='synthetic_r1', right='synthetic_r2',
                        fixed={'qb_id': 'synthetic_q1'})
            args.update(request)
            with self.subTest(request=request), self.assertRaises(ValueError):
                audit_contrast_estimability(fitted, data['population'], **args)

    def test_reference_outcomes_do_not_enter_the_audit(self):
        q = simulate_passing()['population']
        self.assertEqual(self.audit(population=q),
                         self.audit(population=q.with_columns((1 - pl.col('complete_pass')).alias('complete_pass'))))


if __name__ == '__main__':
    unittest.main()
