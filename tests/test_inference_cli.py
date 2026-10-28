"""Offline checks for the inference extractor command line.

The cached-raw fixtures come from the build checks, so both files exercise one raw tree shape.
"""
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import polars as pl

from hierarchical_football.inference.__main__ import main, prepare_summary
from tests.test_inference_build import workspace, write_raw


class PrepareCommand(unittest.TestCase):
    def run_command(self, argv):
        output = io.StringIO()
        with patch.object(sys, "argv", ["hierarchical_football.inference", *argv]):
            with redirect_stdout(output):
                main(argv)
        return json.loads(output.getvalue())

    def test_prepare_writes_both_tables_and_prints_a_summary(self):
        with workspace() as root:
            write_raw(root / "raw")
            summary = self.run_command(["prepare", "--seasons", "2024", "2025",
                                        "--raw-root", str(root / "raw"),
                                        "--output-root", str(root / "inference")])
            self.assertEqual(summary, {"passing_targets": 4, "rushing_plays": 2, "inputs": 4,
                                       "excluded_plays": 10})
            self.assertTrue((root / "inference" / "passing_targets.parquet").exists())
            self.assertTrue((root / "inference" / "rushing_plays.parquet").exists())
            self.assertEqual(pl.read_parquet(root / "inference" / "passing_targets.parquet").height, 4)

    def test_scope_option_reaches_the_rushing_table(self):
        with workspace() as root:
            write_raw(root / "raw", with_rosters=False)
            summary = self.run_command(["prepare", "--seasons", "2024", "--raw-root", str(root / "raw"),
                                        "--output-root", str(root / "inference"),
                                        "--rusher-scope", "all"])
            self.assertEqual(summary["rushing_plays"], 1)

    def test_an_unknown_scope_is_refused(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                main(["prepare", "--seasons", "2024", "--rusher-scope", "skilled"])

    def test_missing_inputs_fail_the_command(self):
        with workspace() as root:
            with self.assertRaises(FileNotFoundError):
                with redirect_stderr(io.StringIO()):
                    main(["prepare", "--seasons", "2024", "--raw-root", str(root / "raw"),
                          "--output-root", str(root / "inference")])

    def test_the_summary_counts_only_what_was_written(self):
        report = {"passing": {"preparation": {"target_rows": 3, "exclusions": {"sack": 2}}},
                  "rushing": {"preparation": {"target_rows": 5, "exclusions": {"kneel": 1}}},
                  "inputs": [{"table_partition": "pbp/season=2024/part.parquet"}]}
        self.assertEqual(prepare_summary(report),
                         {"passing_targets": 3, "rushing_plays": 5, "inputs": 1, "excluded_plays": 3})


if __name__ == "__main__":
    unittest.main()
