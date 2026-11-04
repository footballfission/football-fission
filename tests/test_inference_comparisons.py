"""Retrospective temporal contrasts use one shared Q and fixed model specifications."""
import importlib.util
import unittest

import polars as pl

from tests.test_inference_contrast_uncertainty import crossed_targets


def comparison_targets():
    return pl.concat([crossed_targets(seed=year, games=8).with_columns(
        pl.lit(year).cast(pl.Int32).alias("season"),
        (pl.lit(str(year)) + pl.col("game_id")).alias("game_id")) for year in (2022, 2024)])


class TemporalComparisons(unittest.TestCase):
    def test_model_context_policy_excludes_reference_opponents_unseen_in_later_window(self):
        _, run = self.api()
        table = comparison_targets().with_columns(
            pl.when((pl.col("season") == 2022) & (pl.col("play_id") % 2 == 0))
            .then(pl.lit("D2")).otherwise(pl.lit("D1")).alias("defteam"))
        result = run(table, spec=self.spec(context="expanded", opponent_penalty=100,
                                           reference_policy="model_context"))
        self.assertEqual(result["reference_population"].height, 240)
        self.assertEqual(result["report"]["reference_rows"], 240)
        self.assertEqual(result["report"]["support"]["retained_rows"], 480)
        self.assertEqual(result["report"]["context_support"]["retained_rows"], 240)
        self.assertEqual(result["report"]["windows"][1]["reference_context"]["unseen_categorical_rows"], {})

    def test_no_common_modeled_context_declines_estimation_even_with_coarse_support(self):
        _, run = self.api()
        table = comparison_targets().with_columns(pl.col("season").cast(pl.String).alias("defteam"))
        result = run(table, spec=self.spec(context="expanded", opponent_penalty=100,
                                           reference_policy="model_context"))
        self.assertEqual(result["report"]["status"], "no_shared_model_context")
        self.assertEqual(result["report"]["windows"], [])
        self.assertEqual(result["reference_population"].height, 0)

    def test_unknown_reference_policy_is_refused(self):
        with self.assertRaisesRegex(ValueError, "reference_policy"):
            self.spec(reference_policy="best_results")

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("hierarchical_football.inference.comparisons"),
                             "player explanations need supported temporal comparisons")
        from hierarchical_football.inference.comparisons import PassingComparisonSpec, run_passing_comparison
        return PassingComparisonSpec, run_passing_comparison

    def spec(self, **changes):
        cls, _ = self.api()
        return cls(**{**dict(team="A", identity="receiver_id", left="r1", right="r2", partner="q1",
                             windows=((2022,), (2024,)), position="WR", context="linear", opponent_penalty=None,
                             qb_penalty=1, receiver_penalty=1, sensitivity_penalties=(1, 10), draws=20), **changes})

    def test_crossed_fixture_has_a_recoverable_contrast_in_both_periods(self):
        _, run = self.api()
        result = run(comparison_targets(), spec=self.spec())
        report = result["report"]
        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["forecast_evaluation"])
        self.assertEqual(result["reference_population"].height, 480)
        self.assertTrue(report["summary"]["point_direction_consistent"])
        for window in report["windows"]:
            self.assertGreater(window["bootstrap"]["contrast"]["difference"], 0.4)
            self.assertEqual(len(window["sensitivity"]["fits"]), 4)
            self.assertTrue(window["sensitivity"]["complete"])

    def test_reversing_the_players_reverses_the_contrast_on_identical_Q(self):
        _, run = self.api()
        a = run(comparison_targets(), spec=self.spec())
        b = run(comparison_targets(), spec=self.spec(left="r2", right="r1"))
        self.assertTrue(a["reference_population"].equals(b["reference_population"]))
        for left, right in zip(a["report"]["windows"], b["report"]["windows"]):
            self.assertAlmostEqual(left["bootstrap"]["contrast"]["difference"],
                                   -right["bootstrap"]["contrast"]["difference"], places=10)

    def test_no_joint_support_produces_no_player_estimates(self):
        _, run = self.api()
        table = comparison_targets().with_columns(pl.when(pl.col("season") == 2024)
            .then(pl.lit("unseen_partner")).otherwise(pl.col("qb_id")).alias("qb_id"))
        result = run(table, spec=self.spec())
        self.assertEqual(result["report"]["status"], "no_shared_support")
        self.assertEqual(result["report"]["windows"], [])
        self.assertEqual(result["reference_population"].height, 0)

    def test_declared_support_threshold_is_used_in_bootstrap_arm_reports(self):
        _, run = self.api()
        result = run(comparison_targets(), spec=self.spec(min_rows=3))
        for window in result["report"]["windows"]:
            self.assertEqual(window["bootstrap"]["contrast"]["arms"]["left"]["support"]["min_rows"], 3)

    def test_overlapping_windows_and_invalid_settings_are_refused(self):
        for changes in (dict(windows=((2022,), (2022, 2024))), dict(windows=((2024,), (2022,))),
                        dict(min_rows=0), dict(left="r2"), dict(draws=1), dict(sensitivity_penalties=()),
                        dict(opponent_penalty=10), dict(scale="talent")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.spec(**changes)

    def test_missing_declared_years_fail_instead_of_shortening_a_window(self):
        _, run = self.api()
        with self.assertRaisesRegex(ValueError, "missing"):
            run(comparison_targets().filter(pl.col("season") == 2022), spec=self.spec())


if __name__ == "__main__":
    unittest.main()
