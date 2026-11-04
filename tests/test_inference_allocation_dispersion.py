"""Extra share variation must preserve means, zeros, keys and exact target pools."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.allocation import sample_target_allocation
from tests import test_inference_allocation as allocation_fixture
from tests.test_inference_allocation import roster


class AllocationDispersion(unittest.TestCase):
    def test_default_multinomial_seeded_draws_are_unchanged(self):
        original=allocation_fixture.JointTargetAllocation().run_allocation()
        explicit=allocation_fixture.JointTargetAllocation().run_allocation(concentration=None)
        np.testing.assert_array_equal(original['candidate_samples'],[[5,0,10],[0,4,9],[2,0,9]])
        np.testing.assert_array_equal(original['other_samples'],[[0,1,2],[1,0,1]])
        np.testing.assert_array_equal(original['candidate_samples'],explicit['candidate_samples'])
        self.assertEqual(explicit['allocation_law'],'multinomial')

    def test_dirichlet_multinomial_preserves_mean_and_increases_variance(self):
        panel,candidates=roster()
        panel=panel.filter(pl.col('team')=='A')
        candidates=candidates.filter(pl.col('team')=='A')
        result=sample_target_allocation(panel,candidates,np.full((1,20000),50),
            shares=np.array([.3,.2]),other_shares=np.array([.5]),seed=4,concentration=10)
        values=result['candidate_samples']
        np.testing.assert_allclose(values.mean(axis=1),[15,10],atol=.25)
        self.assertAlmostEqual(values[0].var(),50*.3*.7*(50+10)/(1+10),delta=4)
        self.assertLess(np.cov(values)[0,1],-8)
        np.testing.assert_array_equal(values.sum(axis=0)+result['other_samples'][0],np.full(20000,50))
        self.assertEqual(result['allocation_law'],'dirichlet_multinomial')
        self.assertEqual(result['concentration'],10)

    def test_structural_zeros_and_empty_rosters_remain_exact(self):
        result=allocation_fixture.JointTargetAllocation().run_allocation(shares=np.array([0.,1.,1.]),
            other=np.array([0.,0.]),concentration=3)
        np.testing.assert_array_equal(result['candidate_samples'][0],[0,0,0])
        np.testing.assert_array_equal(result['other_samples'],np.zeros((2,3),dtype=int))
        _,candidates=roster()
        result=allocation_fixture.JointTargetAllocation().run_allocation(candidates=candidates.head(0),shares=np.array([]),
            other=np.ones(2),concentration=3)
        np.testing.assert_array_equal(result['other_samples'],[[0,5,11],[8,0,20]])

    def test_keyed_results_are_invariant_to_input_order(self):
        original=allocation_fixture.JointTargetAllocation().run_allocation(concentration=10)
        panel,candidates=roster()
        changed=allocation_fixture.JointTargetAllocation().run_allocation(panel=panel.reverse(),candidates=candidates.reverse(),
            draws=np.array([[8,0,20],[0,5,11]]),shares=np.array([.5,.75,.3]),other=np.array([.2,.25]),
            concentration=10)
        np.testing.assert_array_equal(original['candidate_samples'],changed['candidate_samples'][::-1])
        np.testing.assert_array_equal(original['other_samples'],changed['other_samples'][::-1])

    def test_invalid_concentrations_are_refused(self):
        for value in (0,-1,True,float('nan'),float('inf'),'10'):
            with self.subTest(concentration=value),self.assertRaises(ValueError):
                allocation_fixture.JointTargetAllocation().run_allocation(concentration=value)


if __name__=='__main__':
    unittest.main()
