"""Historical share baselines use eligible exposure and strictly earlier training weeks."""
import unittest

import numpy as np
import polars as pl

from hierarchical_football.inference.allocation import sample_target_allocation
from hierarchical_football.inference.allocation_baseline import fit_proportional_allocation
from hierarchical_football.inference.splits import LABEL_COLUMNS
from tests.inference_fixtures import panel_frame, candidate_frame


def training():
    panel = panel_frame(dict(game_id='g1', targets=10, candidates=2, candidate_targets=8, other_targets=2),
                        dict(game_id='g2', week=2, targets=20, candidates=1,
                             candidate_targets=10, other_targets=10))
    players = candidate_frame(dict(game_id='g1', player_id='r1', position='WR', targets=6),
                              dict(game_id='g1', player_id='r2', position='RB', candidate_index=1, targets=2),
                              dict(game_id='g2', week=2, player_id='r1', position='WR', targets=10))
    return panel, players


def prediction():
    panel = panel_frame(dict(game_id='g3', week=3)).drop(*LABEL_COLUMNS)
    players = candidate_frame(dict(game_id='g3', week=3, player_id='r1', position='WR'),
                              dict(game_id='g3', week=3, player_id='r2', position='RB', candidate_index=1),
                              dict(game_id='g3', week=3, player_id='new', position='TE', candidate_index=2)).drop('targets')
    return panel, players


class ProportionalAllocation(unittest.TestCase):
    def fit(self, **kwargs):
        return fit_proportional_allocation(*training(), **kwargs)

    def test_rates_use_only_team_targets_in_eligible_roster_games(self):
        fit = self.fit(prior_exposure=10)
        # r2 was eligible only in g1, so its exposure is 10, not 30 or two games.
        global_rate = 18.4 / 41
        wr_prior, rb_prior = (16 + 10 * global_rate) / 40, (2 + 10 * global_rate) / 20
        rates = np.array([(16 + 10 * wr_prior) / 40,
                          (2 + 10 * rb_prior) / 20, global_rate])
        self.assertAlmostEqual(fit.global_share, global_rate)
        self.assertAlmostEqual(fit.other_rate, 0.4)
        result = fit.predict(*prediction())
        denominator = rates.sum() + 0.4
        np.testing.assert_allclose(result['shares'], rates / denominator)
        np.testing.assert_allclose(result['other_shares'], [0.4 / denominator])
        self.assertEqual(result['unseen_players'], 1)
        self.assertEqual(result['unseen_player_rows'], 1)
        self.assertEqual(result['unseen_position_rows'], 1)
        self.assertEqual(fit.report()['train_through'], {'season': 2024, 'week': 2})

    def test_removing_a_supplied_candidate_is_proportional_redistribution(self):
        fit = self.fit()
        panel, players = prediction()
        full = fit.predict(panel, players)
        remaining = fit.predict(panel, players.filter(pl.col('player_id') != 'r1'))
        np.testing.assert_allclose(remaining['shares'], full['shares'][1:] / (1 - full['shares'][0]))
        self.assertGreater(remaining['shares'][0], full['shares'][1])
        empty = fit.predict(panel, players.head(0))
        np.testing.assert_array_equal(empty['other_shares'], [1.0])

    def test_position_reference_uses_same_training_priors_and_roster_without_player_history(self):
        fit = self.fit(prior_exposure=10)
        panel, players = prediction()
        players = players.with_columns(pl.lit('WR').alias('position'))
        history = fit.predict(panel, players)
        reference = fit.predict(panel, players, include_player_history=False)
        self.assertGreater(history['shares'][0], history['shares'][2])
        np.testing.assert_array_equal(reference['shares'], np.repeat(reference['shares'][0], 3))
        self.assertFalse(reference['include_player_history'])
        self.assertTrue(reference['candidate_keys'].equals(history['candidate_keys']))
        self.assertAlmostEqual(reference['other_shares'][0] + reference['shares'].sum(), 1)
        with self.assertRaises(ValueError):
            fit.predict(panel, players, include_player_history=0)

    def test_repeated_unseen_player_rows_do_not_inflate_identity_count(self):
        panel, players = prediction()
        panel = pl.concat([panel, panel.with_columns(pl.lit('g4').alias('game_id'), pl.lit(4).alias('week'))])
        players = pl.concat([players, players.with_columns(pl.lit('g4').alias('game_id'), pl.lit(4).alias('week'))])
        result = self.fit().predict(panel, players)
        self.assertEqual(result['unseen_players'], 1)
        self.assertEqual(result['unseen_player_rows'], 2)
        self.assertEqual(result['unseen_position_rows'], 2)

    def test_predictions_are_keyed_order_invariant_and_feed_joint_sampler(self):
        panel, players = training()
        fit = fit_proportional_allocation(panel.reverse(), players.reverse())
        p, c = prediction()
        original = self.fit().predict(p, c)
        reordered = fit.predict(p, c.reverse())
        np.testing.assert_array_equal(original['shares'], reordered['shares'][::-1])
        self.assertTrue(original['candidate_keys'].equals(c.select('game_id','team','candidate_index','player_id')))
        samples = sample_target_allocation(p, c, np.full((1, 20), 30),
                                           shares=original['shares'], other_shares=original['other_shares'])
        np.testing.assert_array_equal(samples['candidate_samples'].sum(axis=0)+samples['other_samples'][0],
                                      np.full(20, 30))

    def test_prediction_labels_clock_overlap_and_metadata_disagreement_fail(self):
        fit = self.fit()
        panel, players = prediction()
        cases = [(panel.with_columns(pl.lit(30).alias('targets')), players),
                 (panel, players.with_columns(pl.lit(10).alias('targets'))),
                 (panel.with_columns(pl.lit(2).alias('week')), players.with_columns(pl.lit(2).alias('week'))),
                 (panel.with_columns(pl.lit('g1').alias('game_id')), players.with_columns(pl.lit('g1').alias('game_id'))),
                 (panel, players.with_columns(pl.lit(4).alias('week')))]
        for p, c in cases:
            with self.subTest(panel=p.to_dicts()), self.assertRaises(ValueError):
                fit.predict(p, c)

    def test_zero_history_receivers_and_other_have_positive_mass(self):
        panel, players = training()
        players = players.with_columns(pl.when(pl.col('player_id') == 'r2').then(0)
                                        .otherwise(pl.col('targets')).alias('targets'))
        panel = panel.with_columns(pl.when(pl.col('game_id') == 'g1').then(6)
                                   .otherwise(pl.col('candidate_targets')).alias('candidate_targets'),
                                   pl.when(pl.col('game_id') == 'g1').then(4)
                                   .otherwise(pl.col('other_targets')).alias('other_targets'))
        result = fit_proportional_allocation(panel, players).predict(*prediction())
        self.assertTrue(np.all(result['shares'] > 0))
        self.assertGreater(result['other_shares'][0], 0)
        no_other = panel.with_columns(pl.col('candidate_targets').alias('targets'), pl.lit(0).alias('other_targets'))
        fitted = fit_proportional_allocation(no_other, players)
        self.assertGreater(fitted.predict(*prediction())['other_shares'][0], 0)

    def test_invalid_training_counts_and_priors_fail(self):
        panel, players = training()
        for value in (-1, 0.5, float('nan'), None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'counts'):
                fit_proportional_allocation(panel, players.with_columns(pl.lit(value).alias('targets')))
        for value in (0, -1, True, float('inf')):
            with self.subTest(prior=value), self.assertRaises(ValueError):
                self.fit(prior_exposure=value)
        with self.assertRaises(ValueError):
            fit_proportional_allocation(panel, players.with_columns(pl.lit(3).alias('week')))
        with self.assertRaisesRegex(ValueError, 'exposure'):
            fit_proportional_allocation(panel.with_columns(pl.lit(0).alias('targets'),
                                        pl.lit(0).alias('candidate_targets'), pl.lit(0).alias('other_targets')),
                                        players.with_columns(pl.lit(0).alias('targets')))


if __name__ == '__main__':
    unittest.main()
