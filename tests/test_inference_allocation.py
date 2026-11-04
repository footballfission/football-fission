"""Joint allocations must preserve each supplied team/draw opportunity total."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.allocation import sample_target_allocation


def roster():
    panel = pl.DataFrame({'game_id': ['g2', 'g1'], 'team': ['B', 'A']})
    candidates = pl.DataFrame({'game_id': ['g1', 'g2', 'g1'], 'team': ['A', 'B', 'A'],
                               'candidate_index': [0, 0, 1], 'player_id': ['r1', 'r3', 'r2']})
    return panel, candidates


class JointTargetAllocation(unittest.TestCase):
    def run_allocation(self, panel=None, candidates=None, draws=None, shares=None, other=None, **kwargs):
        p, c = roster()
        return sample_target_allocation(
            p if panel is None else panel, c if candidates is None else candidates,
            np.array([[0, 5, 11], [8, 0, 20]]) if draws is None else draws,
            shares=np.array([0.3, 0.75, 0.5]) if shares is None else shares,
            other_shares=np.array([0.25, 0.2]) if other is None else other, seed=4, **kwargs)

    def test_allocations_conserve_every_game_draw_and_preserve_keys(self):
        panel, candidates = roster()
        result = self.run_allocation()
        self.assertEqual(result['candidate_samples'].shape, (3, 3))
        self.assertEqual(result['other_samples'].shape, (2, 3))
        self.assertTrue(np.issubdtype(result['candidate_samples'].dtype, np.integer))
        self.assertTrue(result['candidate_keys'].equals(candidates))
        self.assertTrue(result['team_game_keys'].equals(panel))
        totals = np.array([[0, 5, 11], [8, 0, 20]])
        for i, game in enumerate(panel['game_id']):
            selected = np.array(candidates['game_id'] == game)
            np.testing.assert_array_equal(result['candidate_samples'][selected].sum(axis=0)
                                          + result['other_samples'][i], totals[i])
        np.testing.assert_array_equal(result['candidate_samples'], self.run_allocation()['candidate_samples'])
        self.assertEqual(result['condition'], 'supplied_roster_probabilities_and_team_target_draws')

    def test_order_does_not_change_keyed_samples_or_draw_identity(self):
        panel, candidates = roster()
        original = self.run_allocation()
        reverse = self.run_allocation(panel=panel.reverse(), candidates=candidates.reverse(),
                                      draws=np.array([[8, 0, 20], [0, 5, 11]]),
                                      shares=np.array([0.5, 0.75, 0.3]), other=np.array([0.2, 0.25]))
        np.testing.assert_array_equal(original['candidate_samples'], reverse['candidate_samples'][::-1])
        np.testing.assert_array_equal(original['other_samples'], reverse['other_samples'][::-1])

    def test_zero_share_players_get_zero_and_empty_rosters_go_to_other(self):
        panel, candidates = roster()
        result = self.run_allocation(shares=np.array([0.0, 1.0, 1.0]), other=np.array([0.0, 0.0]))
        np.testing.assert_array_equal(result['candidate_samples'][0], [0, 0, 0])
        result = self.run_allocation(candidates=candidates.head(0), shares=np.array([]), other=np.ones(2))
        self.assertEqual(result['candidate_samples'].shape, (0, 3))
        np.testing.assert_array_equal(result['other_samples'], [[0, 5, 11], [8, 0, 20]])

    def test_multinomial_means_and_negative_competition_match_declared_shares(self):
        panel = pl.DataFrame({'game_id': ['g1'], 'team': ['A']})
        _, candidates = roster()
        candidates = candidates.filter(pl.col('team') == 'A')
        result = self.run_allocation(panel=panel, candidates=candidates,
                                      draws=np.full((1, 20000), 50), shares=np.array([0.3, 0.2]),
                                      other=np.array([0.5]))
        np.testing.assert_allclose(result['candidate_samples'].mean(axis=1), [15, 10], atol=0.1)
        self.assertLess(np.cov(result['candidate_samples'])[0, 1], -2.0)

    def test_supplied_shared_volume_can_generate_positive_teammate_covariance(self):
        panel, candidates = roster()
        panel = panel.filter(pl.col('team') == 'A')
        candidates = candidates.filter(pl.col('team') == 'A')
        result = self.run_allocation(panel=panel, candidates=candidates,
                                      draws=np.tile([0, 100], 1000).reshape(1, -1),
                                      shares=np.array([0.4, 0.4]), other=np.array([0.2]))
        self.assertGreater(np.cov(result['candidate_samples'])[0, 1], 300)
        self.assertTrue(np.all(result['candidate_samples'][:, ::2] == 0))

    def test_both_offenses_in_one_game_keep_separate_pools(self):
        panel, candidates = roster()
        panel = panel.with_columns(pl.lit('g').alias('game_id'))
        candidates = candidates.with_columns(pl.lit('g').alias('game_id'))
        result = self.run_allocation(panel=panel, candidates=candidates)
        totals = np.array([[0, 5, 11], [8, 0, 20]])
        for index, team in enumerate(panel['team']):
            mask = np.array(candidates['team'] == team)
            np.testing.assert_array_equal(result['candidate_samples'][mask].sum(axis=0)
                                          + result['other_samples'][index], totals[index])

    def test_invalid_draws_probabilities_and_labels_are_refused(self):
        panel, candidates = roster()
        invalid = [dict(draws=np.array([[1.5], [1.0]])), dict(draws=np.array([[-1], [2]])),
                   dict(draws=np.ones((1, 3), dtype=int)), dict(draws=np.zeros((2, 0), dtype=int)),
                   dict(shares=np.array([0.4, 0.75, 0.5])), dict(other=np.array([0.25, -0.2])),
                   dict(shares=np.array([0.3, float('nan'), 0.5])),
                   dict(candidates=candidates.with_columns(pl.lit(5).alias('targets'))),
                   dict(panel=panel.with_columns(pl.lit(20).alias('targets'))),
                   dict(candidates=candidates.with_columns(pl.lit('unknown_game').alias('game_id')))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_allocation(**kwargs)
        with self.assertRaises(ValueError):
            self.run_allocation(candidates=pl.concat([candidates, candidates.head(1)]),
                                shares=np.array([0.3, 0.75, 0.5, 0.1]))


if __name__ == '__main__':
    unittest.main()
