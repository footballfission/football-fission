"""Offline checks for the inference table audits."""
import unittest

import polars as pl

from hierarchical_football.inference.audits import pairing_support, table_coverage
from hierarchical_football.inference.contracts import PASSING_CONTRACT

from tests.inference_fixtures import empty_passing_frame, passing_frame


def empty_frame(**extra):
    """A canonical passing frame with no rows, headlined by the charted columns under test."""
    return empty_passing_frame(**extra)


BRIDGED = (
    {"qb_id": "qb1", "receiver_id": "wr1"},
    {"qb_id": "qb1", "receiver_id": "wr2"},
    {"qb_id": "qb2", "receiver_id": "wr2"},
)


class TableCoverage(unittest.TestCase):
    def test_rows_are_reported_per_season(self):
        report = table_coverage(passing_frame({"season": 2024}, {"season": 2024}, {"season": 2025}),
                                PASSING_CONTRACT)
        self.assertEqual(report["table"], "passing_targets")
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["seasons"], {"2024": {"rows": 2}, "2025": {"rows": 1}})

    def test_missing_values_are_reported_as_counts_and_shares(self):
        report = table_coverage(passing_frame({}, {"air_yards": None}, {"air_yards": None, "down": None}),
                                PASSING_CONTRACT)
        self.assertEqual(report["columns"]["air_yards"], {"nulls": 2, "share": 0.6667})
        self.assertEqual(report["columns"]["down"], {"nulls": 1, "share": 0.3333})
        self.assertEqual(report["columns"]["qb_id"], {"nulls": 0, "share": 0.0})

    def test_charted_columns_are_reported_only_when_present(self):
        plain = table_coverage(passing_frame({}), PASSING_CONTRACT)
        self.assertEqual(plain["present_optional_columns"], [])
        charted = table_coverage(passing_frame({}).with_columns(pl.lit(1).alias("is_drop")),
                                 PASSING_CONTRACT)
        self.assertEqual(charted["present_optional_columns"], ["is_drop"])

    def test_an_empty_table_is_reported_without_a_crash(self):
        report = table_coverage(empty_frame(), PASSING_CONTRACT)
        self.assertEqual(report["rows"], 0)
        self.assertIsNone(report["columns"]["air_yards"]["share"])
        self.assertEqual(report["seasons"], {})

    def test_a_table_that_breaks_the_contract_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            table_coverage(passing_frame({}).drop("air_yards"), PASSING_CONTRACT)


class PairingSupport(unittest.TestCase):
    def test_players_who_never_cross_are_separate_components(self):
        report = pairing_support(passing_frame({"qb_id": "qb1", "receiver_id": "wr1"},
                                               {"qb_id": "qb1", "receiver_id": "wr1"},
                                               {"qb_id": "qb2", "receiver_id": "wr2"}))
        self.assertEqual(report["components"], 2)
        self.assertEqual(report["single_pairing_components"], 2)
        self.assertEqual(report["largest_component"], {
            "players": 2, "quarterbacks": 1, "receivers": 1, "pairings": 1, "rows": 2})
        self.assertEqual(report["largest_component_row_share"], 0.6667)

    def test_a_shared_receiver_bridges_passing_units(self):
        report = pairing_support(passing_frame(*BRIDGED))
        self.assertEqual(report["components"], 1)
        self.assertEqual(report["largest_component"], {
            "players": 4, "quarterbacks": 2, "receivers": 2, "pairings": 3, "rows": 3})
        self.assertEqual(report["largest_component_row_share"], 1.0)

    def test_exposure_is_counted_per_player_and_pairing(self):
        report = pairing_support(passing_frame(*BRIDGED))
        self.assertEqual((report["rows"], report["quarterbacks"], report["receivers"]), (3, 2, 2))
        self.assertEqual(report["pairings"], 3)
        self.assertEqual(report["pairings_per_receiver"], {"min": 1, "median": 1.5, "max": 2})
        self.assertEqual(report["receivers_with_one_quarterback"], 1)
        self.assertEqual(report["quarterbacks_with_one_receiver"], 1)

    def test_a_single_pairing_is_reported_as_one_unknown_quantity(self):
        report = pairing_support(passing_frame({"qb_id": "qb1", "receiver_id": "wr1"}))
        self.assertEqual(report["components"], 1)
        self.assertEqual(report["pairings_per_receiver"], {"min": 1, "median": 1, "max": 1})

    def test_component_detail_is_ordered_by_rows(self):
        report = pairing_support(passing_frame({"qb_id": "qb1", "receiver_id": "wr1"},
                                               {"qb_id": "qb1", "receiver_id": "wr1"},
                                               {"qb_id": "qb2", "receiver_id": "wr2"}))
        self.assertEqual([entry["rows"] for entry in report["components_detail"]], [2, 1])

    def test_an_empty_table_has_no_components(self):
        report = pairing_support(empty_frame())
        self.assertEqual(report["components"], 0)
        self.assertIsNone(report["largest_component"])
        self.assertEqual(report["pairings_per_receiver"], {"min": 0, "median": None, "max": 0})

    def test_a_table_that_breaks_the_contract_is_refused(self):
        with self.assertRaisesRegex(ValueError, "share a duplicate play key"):
            pairing_support(passing_frame({"play_id": 1}, {"play_id": 1}))


if __name__ == "__main__":
    unittest.main()
