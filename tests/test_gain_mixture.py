"""Offline checks for the gain mixture the position chains share.

The four position modules carry their own copy of the class, so the same checks run against all four.
"""
import unittest

import numpy as np

from hierarchical_football import chains_qb, chains_rb, chains_te, hierarchical

MIXTURES = {
    "hierarchical": hierarchical.GainMixture,
    "chains_qb": chains_qb.GainMixture,
    "chains_rb": chains_rb.GainMixture,
    "chains_te": chains_te.GainMixture,
}


def mixture(cls):
    return cls(np.array([0.5, 0.5]), np.array([2.0, 8.0]), np.array([1.0, 3.0]))


def deterministic(cls, gain=0.0):
    """A mixture with no spread, so a total is exactly `gain * size` plus the shifts."""
    return cls(np.array([1.0]), np.array([gain]), np.array([0.0]))


class SampleTotals(unittest.TestCase):
    def test_a_group_of_size_zero_totals_zero(self):
        for name, cls in MIXTURES.items():
            with self.subTest(module=name):
                totals = mixture(cls).sample_totals(np.array([0, 0, 4]), np.zeros(3),
                                                    np.random.default_rng(0))
                self.assertEqual(totals.shape, (3,))
                self.assertEqual(totals[:2].tolist(), [0.0, 0.0])

    def test_a_trailing_empty_group_does_not_read_past_the_draws(self):
        """`reduceat` reads one element past the array when its last offset is the array length."""
        for name, cls in MIXTURES.items():
            with self.subTest(module=name):
                totals = mixture(cls).sample_totals(np.array([3, 0, 2]), np.zeros(3),
                                                    np.random.default_rng(1))
                self.assertEqual(totals.shape, (3,))
                self.assertEqual(totals[1], 0.0)

    def test_an_empty_group_does_not_borrow_from_its_neighbour(self):
        cls = MIXTURES["chains_qb"]
        law = deterministic(cls, gain=0.0)
        totals = law.sample_totals(np.array([2, 0, 3]), np.array([1.0, 100.0, 2.0]),
                                   np.random.default_rng(4))
        np.testing.assert_allclose(totals, [2.0, 0.0, 6.0])

    def test_every_group_empty_stays_empty(self):
        cls = MIXTURES["chains_qb"]
        totals = deterministic(cls).sample_totals(np.zeros(3, dtype=int), np.zeros(3),
                                                  np.random.default_rng(2))
        self.assertEqual(totals.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(mixture(cls).sample_totals(np.zeros(0, dtype=int), np.zeros(0),
                                                    np.random.default_rng(2)).shape, (0,))

    def test_the_total_is_the_sum_of_the_drawn_gains(self):
        cls = MIXTURES["chains_te"]
        law = mixture(cls)
        sizes = np.array([2, 3])
        shifts = np.array([1.5, -0.5])
        totals = law.sample_totals(sizes, shifts, np.random.default_rng(7))
        draws, offsets = law.sample_gains(sizes, shifts, np.random.default_rng(7))
        self.assertEqual(offsets.tolist(), [0, 2])
        np.testing.assert_allclose(totals, [draws[:2].sum(), draws[2:].sum()])

    def test_the_shift_is_paid_per_opportunity(self):
        cls = MIXTURES["chains_rb"]
        totals = deterministic(cls).sample_totals(np.array([4, 0]), np.array([2.5, 9.0]),
                                                  np.random.default_rng(3))
        np.testing.assert_allclose(totals, [10.0, 0.0])

    def test_a_group_total_is_repeatable_for_one_seed(self):
        cls = MIXTURES["hierarchical"]
        law = mixture(cls)
        first = law.sample_totals(np.array([3, 0, 3]), np.zeros(3), np.random.default_rng(11))
        second = law.sample_totals(np.array([3, 0, 3]), np.zeros(3), np.random.default_rng(11))
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
