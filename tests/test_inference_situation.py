"""Pre-play context joins preserve the canonical target population and outcomes."""
import unittest

import polars as pl

from hierarchical_football.inference.situation import attach_passing_situation, SITUATION_COLUMNS
from tests.inference_fixtures import passing_frame


def situation_source():
    return pl.DataFrame({
        'game_id': ['g', 'g', 'unused'], 'play_id': [2.0, 1.0, 1.0], 'season': [2024, 2024, 2024],
        'posteam': ['A', 'A', 'B'], 'shotgun': [1.0, 0.0, 1.0], 'no_huddle': [0.0, 1.0, 0.0],
        'score_differential': [-7.0, 3.0, 0.0], 'game_seconds_remaining': [120.0, 1800.0, 20.0],
        'complete_pass': [0, 0, 0], 'score_differential_post': [50, 50, 50]})


class PassingSituationTests(unittest.TestCase):
    def targets(self):
        return passing_frame(dict(game_id='g', play_id=1), dict(game_id='g', play_id=2))

    def test_join_preserves_rows_order_and_outcomes_and_only_attaches_declared_context(self):
        table = self.targets()
        result, report = attach_passing_situation(table, situation_source())
        self.assertTrue(result.select(table.columns).equals(table))
        self.assertEqual(result['shotgun'].to_list(), [0, 1])
        self.assertEqual(result.schema['shotgun'], pl.Int8)
        self.assertEqual(result['score_differential'].to_list(), [3.0, -7.0])
        self.assertEqual(result.columns, table.columns + list(SITUATION_COLUMNS))
        self.assertEqual(report['matched_rows'], table.height)
        self.assertEqual(report['missing_context'], {c: 0 for c in SITUATION_COLUMNS})

    def test_null_context_is_retained_and_counted_without_imputation(self):
        source = situation_source().with_columns(pl.lit(None).alias('shotgun'))
        result, report = attach_passing_situation(self.targets(), source)
        self.assertEqual(result['shotgun'].null_count(), 2)
        self.assertEqual(report['missing_context']['shotgun'], 2)

    def test_unmatched_or_mismatched_metadata_is_refused(self):
        source = situation_source()
        for changed in (source.filter(pl.col('play_id') != 1),
                        source.with_columns(pl.lit('wrong').alias('posteam')),
                        source.with_columns(pl.lit(2023).alias('season'))):
            with self.subTest(source=changed.to_dicts()), self.assertRaisesRegex(ValueError, 'match'):
                attach_passing_situation(self.targets(), changed)

    def test_duplicate_or_fractional_source_keys_are_refused(self):
        source = situation_source()
        for changed in (pl.concat([source, source.head(1)]),
                        source.with_columns((pl.col('play_id') + 0.5).alias('play_id')),
                        source.with_columns(pl.lit(float('nan')).alias('play_id'))):
            with self.subTest(source=changed.to_dicts()), self.assertRaises(ValueError):
                attach_passing_situation(self.targets(), changed)

    def test_invalid_values_or_overwriting_existing_context_are_refused(self):
        source = situation_source()
        for column, value in (('shotgun', 0.5), ('no_huddle', 2),
                              ('game_seconds_remaining', -1), ('score_differential', float('inf'))):
            with self.subTest(column=column), self.assertRaises(ValueError):
                attach_passing_situation(self.targets(), source.with_columns(pl.lit(value).alias(column)))
        with self.assertRaisesRegex(ValueError, 'existing'):
            attach_passing_situation(self.targets().with_columns(pl.lit(1).alias('shotgun')), source)
        with self.assertRaisesRegex(ValueError, 'missing'):
            attach_passing_situation(self.targets(), source.drop('no_huddle'))

    def test_empty_population_preserves_typed_context(self):
        result, report = attach_passing_situation(self.targets().head(0), situation_source())
        self.assertEqual(result.height, 0)
        self.assertEqual(result.schema['shotgun'], pl.Int8)
        self.assertEqual(result.schema['score_differential'], pl.Float64)
        self.assertEqual(report['matched_rows'], 0)


if __name__ == '__main__':
    unittest.main()
