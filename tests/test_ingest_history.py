"""Ingest report history survives later runs over the same cache."""
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from hierarchical_football.data.ingest import ingest


@contextmanager
def workspace():
    parent = Path(__file__).resolve().parent
    folder = parent / ("ingest_history_" + uuid.uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError("unexpected cleanup path")
        shutil.rmtree(folder)


def loader(table, season):
    return pl.DataFrame({"season": [season], "value": [float(season)]})


def broken(table, season):
    raise OSError("source unavailable")


class IngestReportHistory(unittest.TestCase):
    def report(self, root):
        return json.loads((root / "ingest_report.json").read_text(encoding="utf-8"))

    def test_later_season_keeps_earlier_partition(self):
        with workspace() as root:
            ingest(root, [2024], ["schedules"], loader=loader)
            ingest(root, [2025], ["schedules"], loader=loader)
            report = self.report(root)
            self.assertEqual([(p["table"], p["season"]) for p in report["partitions"]],
                             [("schedules", 2024), ("schedules", 2025)])
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(report["runs"][0]["seasons"], [2024])
            self.assertEqual(report["runs"][1]["seasons"], [2025])

    def test_refresh_updates_partition_without_duplicating_it(self):
        with workspace() as root:
            ingest(root, [2024], ["schedules"], loader=loader)
            ingest(root, [2024], ["schedules"], refresh=True, loader=loader)
            report = self.report(root)
            self.assertEqual(len(report["partitions"]), 1)
            self.assertEqual(report["partitions"][0]["status"], "downloaded")
            self.assertEqual([run["downloaded"] for run in report["runs"]], [1, 1])
            self.assertEqual([run["cached"] for run in report["runs"]], [0, 0])

    def test_cached_run_is_recorded_without_fetching(self):
        with workspace() as root:
            calls = []

            def counting(table, season):
                calls.append(season)
                return loader(table, season)

            ingest(root, [2024], ["schedules"], loader=counting)
            ingest(root, [2024], ["schedules"], loader=counting)
            self.assertEqual(calls, [2024])
            report = self.report(root)
            self.assertEqual([run["cached"] for run in report["runs"]], [0, 1])

    def test_failed_run_preserves_partitions_and_records_the_error(self):
        with workspace() as root:
            ingest(root, [2024], ["schedules"], loader=loader)
            report_before = self.report(root)
            with self.assertRaises(RuntimeError):
                ingest(root, [2025], ["schedules"], loader=broken)
            report = self.report(root)
            self.assertEqual([(p["table"], p["season"]) for p in report["partitions"]],
                             [("schedules", 2024)])
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual([run["errors"] for run in report["runs"]],
                             [report_before["runs"][0]["errors"], report["errors"]])

    def test_first_start_survives_and_update_time_advances(self):
        with workspace() as root:
            ingest(root, [2024], ["schedules"], loader=loader)
            first = self.report(root)
            ingest(root, [2025], ["schedules"], loader=loader)
            second = self.report(root)
            self.assertEqual(second["started_at"], first["started_at"])
            self.assertGreaterEqual(second["updated_at"], first["updated_at"])


if __name__ == "__main__":
    unittest.main()
