"""Offline checks for the repository-root extractor script."""
import io
import json
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import polars as pl

from tests.test_inference_build import workspace, write_raw

SCRIPT = Path(__file__).resolve().parent.parent / "prepare_inference_tables.py"


def run_script(argument_list):
    output = io.StringIO()
    with patch.object(sys, "argv", ["prepare_inference_tables.py", *argument_list]):
        with redirect_stdout(output):
            runpy.run_path(str(SCRIPT), run_name="__main__")
    return json.loads(output.getvalue())


class ExtractorScript(unittest.TestCase):
    def test_script_writes_the_tables_it_reports(self):
        with workspace() as root:
            write_raw(root / "raw")
            summary = run_script(["--seasons", "2024", "2025", "--raw-root", str(root / "raw"),
                                  "--output-root", str(root / "inference")])
            self.assertEqual(summary["passing_targets"], 4)
            self.assertEqual(summary["rushing_plays"], 2)
            self.assertEqual(pl.read_parquet(root / "inference" / "rushing_plays.parquet").height,
                             summary["rushing_plays"])

    def test_script_uses_the_requested_scope(self):
        with workspace() as root:
            write_raw(root / "raw", with_rosters=False)
            summary = run_script(["--seasons", "2024", "--raw-root", str(root / "raw"),
                                  "--output-root", str(root / "inference"),
                                  "--rusher-scope", "all"])
            self.assertEqual(summary["rushing_plays"], 1)

    def test_script_fails_when_a_cached_season_is_missing(self):
        with workspace() as root:
            with self.assertRaises(FileNotFoundError):
                run_script(["--seasons", "2019", "--raw-root", str(root / "raw"),
                            "--output-root", str(root / "inference")])


if __name__ == "__main__":
    unittest.main()
