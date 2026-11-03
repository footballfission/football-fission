"""Offline studies preserve exact specifications and refuse ambiguous or overwritten runs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from hierarchical_football.inference.__main__ import main
from tests.test_inference_study import study_table


class StudyFiles(unittest.TestCase):
    def test_manifest_records_only_declared_numerical_thread_environment(self):
        with patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "3",
                                     "UNRELATED_STUDY_SETTING": "unrecorded"}):
            manifest = self.run_study()
        settings = manifest["environment"]["thread_environment"]
        self.assertEqual(settings["OPENBLAS_NUM_THREADS"], "2")
        self.assertEqual(settings["OMP_NUM_THREADS"], "3")
        self.assertNotIn("UNRELATED_STUDY_SETTING", settings)

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="passing_study_")
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.input = self.root / "targets.parquet"
        self.spec = self.root / "protocol.json"
        self.output = self.root / "result"
        study_table().write_parquet(self.input)
        self.config = {"schema_version": 1, "study": {
            "train_seasons": [2022], "selection_season": 2023, "evaluation_seasons": [2024],
            "penalties": [10], "contexts": ["linear"], "draws": 20}}
        self.write_spec()

    def write_spec(self):
        self.spec.write_text(json.dumps(self.config), encoding="utf-8")

    def run_study(self):
        self.assertIsNotNone(importlib.util.find_spec("hierarchical_football.inference.study_io"),
                             "the passing study needs traceable file output")
        from hierarchical_football.inference.study_io import run_study_file
        return run_study_file(self.input, self.spec, self.output)

    def test_bundle_has_exact_specification_hashes_versions_and_keyed_predictions(self):
        manifest = self.run_study()
        self.assertEqual((self.output / "specification.json").read_bytes(), self.spec.read_bytes())
        self.assertEqual(manifest["input"]["sha256"], hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertEqual(manifest["specification_sha256"], hashlib.sha256(self.spec.read_bytes()).hexdigest())
        self.assertIn("scikit-learn", manifest["environment"]["packages"])
        self.assertIn("inference/study.py", manifest["source_hashes"])
        for name, digest in manifest["outputs"].items():
            self.assertEqual(digest, hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        frame = pl.read_parquet(self.output / "predictions.parquet")
        self.assertEqual(frame.height, 100)
        self.assertEqual(frame.select("game_id", "play_id").unique().height, 100)
        self.assertIn("transfer_cohort", frame.columns)
        report = json.loads((self.output / "report.json").read_text())
        self.assertFalse(report["untouched_holdout_verified"])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "report.json"
        sentinel.write_text("preserve me")
        with self.assertRaises(FileExistsError):
            self.run_study()
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_unknown_fields_nonfinite_and_duplicate_keys_are_refused_before_output(self):
        for config in ({**self.config, "typo": 1},
                       {**self.config, "study": {**self.config["study"], "future": 2027}},
                       {**self.config, "study": {**self.config["study"], "penalties": [float("nan")]}},
                       {**self.config, "contrasts": [{"identity": "receiver_id", "typo": "r1"}]}):
            self.config = config
            self.write_spec()
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_study()
            self.assertFalse(self.output.exists())
        self.spec.write_text('{"schema_version": 1, "schema_version": 1, "study": {}}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_study()

    def test_failed_study_keeps_failure_record_without_a_complete_manifest(self):
        self.config["study"]["evaluation_seasons"] = [2026]
        self.write_spec()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_study()
        self.assertTrue((self.output / "failure.json").exists())
        self.assertFalse((self.output / "manifest.json").exists())

    def test_cli_reaches_the_same_workflow(self):
        main(["study-passing", "--input", str(self.input), "--spec", str(self.spec),
              "--output-root", str(self.output)])
        self.assertTrue((self.output / "manifest.json").is_file())

    def test_specification_change_during_fitting_cannot_produce_a_complete_run(self):
        from hierarchical_football.inference.study import run_passing_study

        def changed_protocol(*args, **kwargs):
            result = run_passing_study(*args, **kwargs)
            self.spec.write_text(self.spec.read_text() + "\n")
            return result

        with patch("hierarchical_football.inference.study_io.run_passing_study", changed_protocol):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.run_study()
        self.assertTrue((self.output / "failure.json").is_file())
        self.assertFalse((self.output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
