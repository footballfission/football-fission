"""Frozen comparison protocols preserve their exact supported reference population."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from hierarchical_football.inference import study_io
from hierarchical_football.inference.__main__ import main
from tests.test_inference_comparisons import comparison_targets


class ComparisonFiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="passing_comparison_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input, self.spec, self.output = (self.root / name for name in ("input.parquet", "protocol.json", "run"))
        comparison_targets().write_parquet(self.input)
        self.config = {"schema_version": 1, "comparison": {
            "team": "A", "identity": "receiver_id", "left": "r1", "right": "r2", "partner": "q1",
            "windows": [[2022], [2024]], "context": "linear", "opponent_penalty": None,
            "qb_penalty": 1, "receiver_penalty": 1, "sensitivity_penalties": [1], "draws": 20}}
        self.write_spec()

    def write_spec(self):
        self.spec.write_text(json.dumps(self.config), encoding="utf-8")

    def run_comparison(self):
        function = getattr(study_io, "run_comparison_file", None)
        self.assertIsNotNone(function, "comparison protocols need an offline artifact entry point")
        return function(self.input, self.spec, self.output)

    def test_complete_bundle_saves_reference_rows_instead_of_forecast_predictions(self):
        manifest = self.run_comparison()
        self.assertTrue((self.output / "reference_population.parquet").is_file())
        self.assertFalse((self.output / "predictions.parquet").exists())
        self.assertEqual((self.output / "specification.json").read_bytes(), self.spec.read_bytes())
        for name, digest in manifest["outputs"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["purpose"], "retrospective_comparison")
        self.assertFalse(report["forecast_evaluation"])

    def test_nonestimable_comparison_is_a_complete_report_with_no_estimates(self):
        self.config["comparison"]["partner"] = "never_observed"
        self.write_spec()
        self.run_comparison()
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "no_shared_support")
        self.assertEqual(report["windows"], [])
        self.assertEqual(pl.read_parquet(self.output / "reference_population.parquet").height, 0)

    def test_existing_paths_and_unknown_protocol_fields_are_refused(self):
        self.config["comparison"]["unknown"] = True
        self.write_spec()
        with self.assertRaises(ValueError):
            self.run_comparison()
        self.assertFalse(self.output.exists())
        del self.config["comparison"]["unknown"]
        self.write_spec()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_comparison()

    def test_cli_runs_the_frozen_comparison(self):
        main(["compare-passing", "--input", str(self.input), "--spec", str(self.spec),
              "--output-root", str(self.output)])
        self.assertTrue((self.output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
