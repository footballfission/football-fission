"""Repository hygiene: generated and scratch paths stay out of the index."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ignored(relative_path):
    """True when git would ignore the path, whatever the working tree contains."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", relative_path],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed for {relative_path}: {result.stderr.strip()}")
    return result.returncode == 0


class IgnoredPaths(unittest.TestCase):
    def test_generated_dataset_partitions_are_ignored(self):
        self.assertTrue(ignored("data/raw/player_stats_week/season=2026/part.parquet"))
        self.assertTrue(ignored("data/raw/rosters_weekly/season=2026/part.json"))
        self.assertTrue(ignored("data/processed/WR/history.parquet"))
        self.assertTrue(ignored("data/processed/WR/features.json"))
        self.assertTrue(ignored("data/splits/wr_2025_10/train.parquet"))
        self.assertTrue(ignored("data/predict_2026w8/forecast.csv"))

    def test_ingest_and_build_manifests_are_ignored(self):
        self.assertTrue(ignored("data/raw/ingest_report.json"))
        self.assertTrue(ignored("data/processed/build_report.json"))

    def test_scratch_notes_are_ignored(self):
        self.assertTrue(ignored("notes/INCREMENTAL_COMMIT_GUIDE.md"))
        self.assertTrue(ignored("notes/worksheet.md"))

    def test_package_source_named_data_stays_tracked(self):
        for path in (
            "src/hierarchical_football/data/panels.py",
            "src/hierarchical_football/data/__init__.py",
            "src/hierarchical_football/data/store.py",
        ):
            self.assertFalse(ignored(path), f"{path} must stay visible to git")

    def test_tests_and_docs_stay_tracked(self):
        for path in (
            "tests/test_data_pipeline.py",
            "docs/DATA_CONTRACT.md",
            "docs/PIPELINE_VALIDATION.md",
            "pyproject.toml",
        ):
            self.assertFalse(ignored(path), f"{path} must stay visible to git")


if __name__ == "__main__":
    unittest.main()
