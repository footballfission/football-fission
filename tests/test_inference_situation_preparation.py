"""Situation fields enter canonical artifacts only through an explicit option."""
import io
import json
from contextlib import redirect_stdout
import unittest

import polars as pl

from hierarchical_football.data.store import write_frame
from hierarchical_football.inference.__main__ import main
from hierarchical_football.inference.prepare import build, passing_targets
from hierarchical_football.inference.situation import SITUATION_COLUMNS
from tests.test_inference_build import season_plays, PBP_SCHEMA, workspace, write_raw
from tests.test_prepare_script import run_script


def with_situation(table):
    return table.with_columns(pl.lit(1.0).alias('shotgun'), pl.lit(0.0).alias('no_huddle'),
                              pl.lit(-3.0).alias('score_differential'),
                              pl.lit(120.0).alias('game_seconds_remaining'))


class SituationPreparation(unittest.TestCase):
    def test_opt_in_preserves_existing_extraction_and_reports_join(self):
        raw = with_situation(pl.DataFrame(season_plays(2024), schema_overrides=PBP_SCHEMA))
        plain, original = passing_targets(raw)
        result, report = passing_targets(raw, include_situation=True)
        self.assertTrue(result.select(plain.columns).equals(plain))
        self.assertFalse(set(SITUATION_COLUMNS) & set(plain.columns))
        self.assertEqual(report['situation']['matched_rows'], 2)
        self.assertEqual(report['exclusions'], original['exclusions'])
        self.assertEqual(result['score_differential'].to_list(), [-3.0, -3.0])

    def write_situation_raw(self, root):
        write_raw(root, seasons=(2024,))
        path = root/'pbp'/'season=2024'/'part.parquet'
        write_frame(path, with_situation(pl.read_parquet(path)))

    def test_build_saves_option_metadata_and_only_enriches_passing(self):
        with workspace() as root:
            self.write_situation_raw(root/'raw')
            report = build(root/'raw', root/'output', [2024], include_situation=True)
            self.assertTrue(report['include_situation'])
            result = pl.read_parquet(root/'output'/'passing_targets.parquet')
            rushing = pl.read_parquet(root/'output'/'rushing_plays.parquet')
            self.assertTrue(set(SITUATION_COLUMNS) <= set(result.columns))
            self.assertFalse(set(SITUATION_COLUMNS) & set(rushing.columns))
            sidecar = json.loads((root/'output'/'passing_targets.json').read_text())
            self.assertTrue(sidecar['include_situation'])

    def test_cli_option_persists_the_enriched_table(self):
        with workspace() as root:
            self.write_situation_raw(root/'raw')
            with redirect_stdout(io.StringIO()):
                main(['prepare', '--seasons', '2024', '--raw-root', str(root/'raw'),
                      '--output-root', str(root/'output'), '--include-passing-situation'])
            result = pl.read_parquet(root/'output'/'passing_targets.parquet')
            self.assertTrue(set(SITUATION_COLUMNS) <= set(result.columns))

    def test_missing_source_fields_and_nonboolean_option_are_refused(self):
        raw = pl.DataFrame(season_plays(2024), schema_overrides=PBP_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'missing'):
            passing_targets(raw, include_situation=True)
        with self.assertRaises(ValueError):
            passing_targets(raw, include_situation=1)

    def test_root_script_option_reaches_the_enriched_table(self):
        with workspace() as root:
            self.write_situation_raw(root/'raw')
            run_script(['--seasons', '2024', '--raw-root', str(root/'raw'),
                        '--output-root', str(root/'output'), '--include-passing-situation'])
            result = pl.read_parquet(root/'output'/'passing_targets.parquet')
            self.assertTrue(set(SITUATION_COLUMNS) <= set(result.columns))


if __name__ == '__main__':
    unittest.main()
