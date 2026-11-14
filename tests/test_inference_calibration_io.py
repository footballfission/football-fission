"""Synthetic calibration bundles must never imply an external NFL data input."""
import hashlib
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from hierarchical_football.inference import study_io
from hierarchical_football.inference.__main__ import main, parser


class CalibrationFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='passing_calibration_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec, self.output = self.root/'protocol.json', self.root/'run'
        self.config = {'schema_version': 1, 'calibration': {
            'trials': 2, 'draws': 20, 'games': 3, 'repeats': 2,
            'penalties': [1], 'scenarios': ['nested'], 'seed': 12}}
        self.write_spec()

    def write_spec(self):
        self.spec.write_text(json.dumps(self.config, indent=2)+'\n', encoding='utf-8')

    def run_calibration(self):
        function = getattr(study_io, 'run_calibration_file', None)
        self.assertIsNotNone(function, 'known-truth studies need a frozen synthetic bundle entry point')
        return function(self.spec, self.output)

    def test_complete_bundle_has_exact_protocol_trials_and_synthetic_provenance(self):
        manifest = self.run_calibration()
        self.assertEqual(manifest['input'], {'kind': 'synthetic'})
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual((self.output/'specification.json').read_bytes(), self.spec.read_bytes())
        self.assertEqual(manifest['specification_sha256'], hashlib.sha256(self.spec.read_bytes()).hexdigest())
        self.assertIn('inference/simulation.py', manifest['source_hashes'])
        for name, digest in manifest['outputs'].items():
            self.assertEqual(hashlib.sha256((self.output/name).read_bytes()).hexdigest(), digest)
        report = json.loads((self.output/'report.json').read_text())
        self.assertTrue(report['synthetic'])
        self.assertFalse(report['real_player_coverage_verified'])
        self.assertEqual(report['summaries'][0]['available_intervals'], 0)
        self.assertEqual(pl.read_parquet(self.output/'trials.parquet').height, 2)
        self.assertFalse((self.output/'predictions.parquet').exists())
        self.assertFalse((self.output/'failure.json').exists())

    def test_existing_output_is_not_overwritten(self):
        self.run_calibration()
        before = (self.output/'manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_calibration()
        self.assertEqual((self.output/'manifest.json').read_bytes(), before)

    def test_cli_runs_synthetic_protocol_without_external_input_argument(self):
        main(['calibrate-passing', '--spec', str(self.spec), '--output-root', str(self.output)])
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(manifest['input'], {'kind': 'synthetic'})
        self.assertTrue((self.output/'trials.parquet').exists())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['calibrate-passing', '--input', 'NFL.parquet',
                                 '--spec', str(self.spec), '--output-root', str(self.output)])

    def test_invalid_or_duplicate_protocol_fails_before_output_creation(self):
        invalid = [dict(self.config, unknown=True), {'schema_version': True, 'calibration': {}},
                   {'schema_version': 1, 'calibration': []},
                   {'schema_version': 1, 'calibration': {'unknown': 1}},
                   {'schema_version': 1, 'calibration': {'trials': 1}}]
        for config in invalid:
            self.spec.write_text(json.dumps(config))
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_calibration()
            self.assertFalse(self.output.exists())
        for raw in ('{"schema_version":1,"calibration":{},"calibration":{}}',
                    '{"schema_version":1,"calibration":{"penalties":[NaN]}}'):
            self.spec.write_text(raw)
            with self.assertRaises(ValueError):
                self.run_calibration()
            self.assertFalse(self.output.exists())

    def test_protocol_changes_cannot_produce_complete_manifest(self):
        from hierarchical_football.inference.calibration import run_calibration

        def change_protocol(spec):
            result = run_calibration(spec)
            self.spec.write_text(self.spec.read_text()+'\n')
            return result

        with patch('hierarchical_football.inference.calibration.run_calibration', change_protocol):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                self.run_calibration()
        self.assertTrue((self.output/'failure.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())

    def test_source_changes_cannot_produce_complete_manifest(self):
        with patch.object(study_io, '_source_hashes', side_effect=[{'simulation.py': 'before'},
                                                                 {'simulation.py': 'after'}]):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                self.run_calibration()
        self.assertTrue((self.output/'failure.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())

    def test_nonfinite_results_leave_explicit_failure_without_complete_manifest(self):
        with patch('hierarchical_football.inference.calibration.run_calibration',
                   return_value={'report': {'bad': float('nan')}, 'trials': pl.DataFrame({'trial': [0]})}):
            with self.assertRaises(ValueError):
                self.run_calibration()
        self.assertTrue((self.output/'failure.json').exists())
        self.assertFalse((self.output/'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
