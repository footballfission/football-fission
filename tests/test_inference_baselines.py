"""Offline checks for the regularized baseline estimators."""
import unittest

import numpy as np
from sklearn.metrics import log_loss

from hierarchical_football.inference.baselines import (
    baseline_name,
    convergence,
    fit_completion,
    fit_rushing,
    passing_design,
    penalty_value,
    rushing_design,
    warn_if_not_converged,
)

from tests.inference_fixtures import passing_frame, rushing_frame


def completions(quarterbacks, rows=200, seed=0):
    """One completion per drawn play, at the rate its quarterback is given.

    An entry is `(quarterback, rate)` or `(quarterback, rate, rows)` when the exposures differ.
    """
    rng = np.random.default_rng(seed)
    changes = []
    for entry in quarterbacks:
        quarterback, rate = entry[0], entry[1]
        for index in range(entry[2] if len(entry) > 2 else rows):
            changes.append({
                "qb_id": quarterback,
                "receiver_id": "wr%d" % (1 + index % 3),
                "complete_pass": int(rng.random() < rate),
            })
    return passing_frame(*changes)


def carries(runners, rows=200, seed=0):
    """Signed yards around a baseline, shifted by the runner's own effect."""
    rng = np.random.default_rng(seed)
    changes = []
    for index in range(rows):
        for runner, effect in runners:
            changes.append({
                "rusher_id": runner,
                "rushing_yards": float(round(4.0 + effect + rng.normal(0, 1), 1)),
            })
    return rushing_frame(*changes)


class Designs(unittest.TestCase):
    def test_context_design_carries_no_player_column(self):
        design = passing_design(passing_frame({}))
        self.assertEqual(design.columns,
                         ("air_yards", "yardline_100", "ydstogo", "down=1", "pass_location=left"))
        self.assertEqual(design.identities, {})

    def test_identity_columns_are_one_block_per_player(self):
        design = passing_design(passing_frame({"qb_id": "qb1"}, {"qb_id": "qb2"}),
                                identities=("qb_id",))
        self.assertEqual(design.columns[-2:], ("qb_id=qb1", "qb_id=qb2"))
        self.assertEqual(design.identities["qb_id"]["rows"], {"qb1": 1, "qb2": 1})

    def test_a_composite_column_carries_the_combination_of_its_sources(self):
        design = passing_design(passing_frame({"position": "TE"}), composites={"position": ("position",)})
        self.assertIn("position=TE", design.columns)
        self.assertEqual(design.composites["position"]["sources"], ("position",))

    def test_a_reference_keeps_the_training_levels_and_reports_the_unseen_ones(self):
        training = passing_frame({"pass_location": "left"}, {"pass_location": "middle"})
        reference = passing_design(training)
        holdout = passing_design(passing_frame({"pass_location": "right"}), reference=reference)
        self.assertEqual(holdout.columns, reference.columns)
        self.assertEqual(holdout.notes["unseen_categorical_rows"], {"pass_location": 1})
        block = [index for index, column in enumerate(holdout.columns) if column.startswith("pass_location=")]
        self.assertEqual(holdout.matrix[0][block].sum(), 0.0)
        self.assertEqual(holdout.matrix[0][block[0]], 0.0)

    def test_a_null_feature_is_filled_with_the_reference_median_and_counted(self):
        reference = passing_design(passing_frame({"air_yards": 5.0}, {"air_yards": 15.0}))
        self.assertEqual(reference.continuous["air_yards"]["median"], 10.0)
        holdout = passing_design(passing_frame({"air_yards": None}), reference=reference)
        self.assertEqual(holdout.notes["filled_continuous"], {"air_yards": 1})
        self.assertEqual(holdout.matrix[0][0], 0.0)

    def test_a_missing_feature_keeps_its_own_indicator_column(self):
        design = passing_design(passing_frame({"air_yards": None}, {"air_yards": 5.0}))
        self.assertIn("air_yards_missing", design.columns)
        self.assertEqual(design.matrix[0][design.columns.index("air_yards_missing")], 1.0)

    def test_an_identity_column_that_is_not_there_is_refused(self):
        with self.assertRaisesRegex(ValueError, "identity columns"):
            rushing_design(rushing_frame({}), identities=("qb_id",))

    def test_a_frame_that_breaks_the_contract_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            passing_design(passing_frame({}).drop("air_yards"))


class CompletionBaseline(unittest.TestCase):
    def test_player_columns_beat_context_on_a_held_out_season(self):
        training = completions((("qb1", 0.8), ("qb2", 0.2)))
        holdout = completions((("qb1", 0.8), ("qb2", 0.2)), seed=1)
        observed = holdout["complete_pass"].to_numpy().astype(float)
        context = fit_completion(training)
        identity = fit_completion(training, identities=("qb_id",))
        context_loss = log_loss(observed, context.predict(holdout), labels=[0, 1])
        identity_loss = log_loss(observed, identity.predict(holdout), labels=[0, 1])
        self.assertLess(identity_loss, context_loss - 0.1)

    def test_a_player_the_training_season_never_saw_carries_no_term(self):
        fitted = fit_completion(completions((("qb1", 0.8), ("qb2", 0.2))), identities=("qb_id",))
        holdout = passing_frame({"qb_id": "qb3"})
        design = fitted.holdout_design(holdout)
        self.assertNotIn("qb_id=qb3", design.columns)
        self.assertEqual(design.notes["unseen_identity_rows"], {"qb_id": 1})
        self.assertEqual(design.matrix[0][[index for index, column in enumerate(design.columns)
                                             if column.startswith("qb_id=")]].sum(), 0.0)
        self.assertTrue(np.isfinite(fitted.predict(holdout)).all())

    def test_probabilities_are_fitted_per_row(self):
        fitted = fit_completion(completions((("qb1", 0.8), ("qb2", 0.2))), identities=("qb_id",))
        probability = fitted.predict(passing_frame({"qb_id": "qb1"}, {"qb_id": "qb2"}))
        self.assertEqual(probability.shape, (2,))
        self.assertLess(probability[1], probability[0])

    def test_terms_are_ordered_by_name_and_carry_their_exposure(self):
        fitted = fit_completion(completions((("qb1", 0.4, 40), ("qb2", 0.95, 400))),
                               identities=("qb_id",))
        terms = fitted.identity_terms()
        self.assertEqual([term["player"] for term in terms], ["qb1", "qb2"])
        self.assertEqual([term["rows"] for term in terms], [40, 400])
        self.assertEqual(terms[0]["share"], round(40 / 440, 6))
        self.assertIsNone(terms[0]["nested"])

    def test_the_terms_of_one_player_block_are_centred_on_zero(self):
        fitted = fit_completion(completions((("qb1", 0.8), ("qb2", 0.2), ("qb3", 0.5))),
                               identities=("qb_id",))
        self.assertAlmostEqual(sum(term["term"] for term in fitted.identity_terms()), 0.0, places=5)

    def test_thin_evidence_keeps_a_smaller_term_than_plenty_of_evidence(self):
        """Three completions from three attempts are not the same term as sixty from sixty."""
        fitted = fit_completion(
            passing_frame(*[{"qb_id": "qb_plenty", "complete_pass": int(index % 2 == 0)}
                            for index in range(400)]
                          + [{"qb_id": "qb_thin", "complete_pass": 1} for _ in range(3)]
                          + [{"qb_id": "qb_mid", "complete_pass": 1} for _ in range(60)]),
            identities=("qb_id",))
        terms = {term["player"]: term for term in fitted.identity_terms()}
        self.assertLess(abs(terms["qb_thin"]["term"]), abs(terms["qb_mid"]["term"]))
        probability = fitted.predict(passing_frame({"qb_id": "qb_thin"}))[0]
        self.assertLess(probability, 0.95)

    def test_one_identity_can_be_selected(self):
        fitted = fit_completion(completions((("qb1", 0.8), ("qb2", 0.2))),
                                identities=("qb_id", "receiver_id"))
        self.assertEqual({term["identity"] for term in fitted.identity_terms(identity="qb_id")}, {"qb_id"})
        self.assertEqual(len(fitted.identity_terms(identity="qb_id")), 2)
        self.assertEqual(len(fitted.identity_terms(limit=3)), 3)

    def test_a_stronger_penalty_shrinks_a_player_term(self):
        training = completions((("qb1", 0.8), ("qb2", 0.2)))
        light = fit_completion(training, identities=("qb_id",), penalty=1.0)
        heavy = fit_completion(training, identities=("qb_id",), penalty=100.0)
        self.assertLess(abs(heavy.identity_terms()[0]["term"]), abs(light.identity_terms()[0]["term"]))

    def test_a_fit_that_stops_at_the_iteration_cap_says_so(self):
        """A capped fit is still returned, so the fact has to travel with the coefficients."""
        with self.assertWarns(RuntimeWarning):
            fitted = fit_completion(completions((("qb1", 0.8), ("qb2", 0.2))),
                                    identities=("qb_id",), max_iter=1)
        self.assertFalse(fitted.report()["in_sample"]["converged"])
        self.assertEqual(fitted.report()["in_sample"]["iterations"], 1)

    def test_a_penalty_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "penalty must be positive"):
            fit_completion(completions((("qb1", 0.5), ("qb2", 0.5))), penalty=0.0)

    def test_one_outcome_alone_cannot_be_fitted(self):
        with self.assertRaisesRegex(ValueError, "one outcome only"):
            fit_completion(passing_frame({"complete_pass": 1}, {"complete_pass": 1}))

    def test_the_report_marks_its_error_as_in_sample(self):
        fitted = fit_completion(completions((("qb1", 0.8), ("qb2", 0.2))), identities=("qb_id",))
        report = fitted.report()
        self.assertEqual(report["name"], "passing:context+identity(qb_id)")
        self.assertEqual(report["rows"], 400)
        self.assertEqual(report["features"], 7)
        self.assertEqual(sorted(report["in_sample"]),
                         ["brier", "converged", "iterations", "log_loss"])
        self.assertTrue(report["in_sample"]["converged"])
        self.assertEqual(report["identity_levels"], {"qb_id": 2})

    def test_two_fits_of_the_same_rows_agree(self):
        training = completions((("qb1", 0.8), ("qb2", 0.2)))
        first = fit_completion(training, identities=("qb_id",)).coefficients()
        second = fit_completion(training, identities=("qb_id",)).coefficients()
        self.assertEqual(first, second)


class RushingBaseline(unittest.TestCase):
    def test_a_runner_column_beats_context_on_a_held_out_season(self):
        training = carries((("rb1", 3.0), ("rb2", -3.0)))
        holdout = carries((("rb1", 3.0), ("rb2", -3.0)), seed=1)
        observed = holdout["rushing_yards"].to_numpy()
        context = fit_rushing(training)
        runner = fit_rushing(training, identities=("rusher_id",))
        context_mae = np.mean(np.abs(observed - context.predict(holdout)))
        runner_mae = np.mean(np.abs(observed - runner.predict(holdout)))
        self.assertLess(runner_mae, context_mae)

    def test_a_runner_inside_one_support_environment_is_marked_as_nested(self):
        training = rushing_frame(
            {"rusher_id": "rb1", "team": "A"}, {"rusher_id": "rb1", "team": "A"},
            {"rusher_id": "rb2", "team": "A"}, {"rusher_id": "rb2", "team": "B"},
        )
        fitted = fit_rushing(training, identities=("rusher_id",), support=True)
        terms = {term["player"]: term for term in fitted.identity_terms()}
        self.assertEqual(terms["rb1"]["distinct_contexts"], 1)
        self.assertTrue(terms["rb1"]["nested"])
        self.assertEqual(terms["rb2"]["distinct_contexts"], 2)
        self.assertFalse(terms["rb2"]["nested"])
        self.assertEqual(fitted.report()["composites"]["support"]["sources"],
                         ("team", "run_location", "run_gap"))

    def test_support_adds_the_environment_composition_beside_the_context(self):
        plain = rushing_design(rushing_frame({}))
        composed = rushing_design(rushing_frame({}), support=True)
        self.assertNotIn("support=A|middle|guard", plain.columns)
        self.assertIn("support=A|middle|guard", composed.columns)
        self.assertIn("run_gap=guard", composed.columns)

    def test_the_report_carries_the_in_sample_error(self):
        fitted = fit_rushing(carries((("rb1", 3.0), ("rb2", -3.0))))
        report = fitted.report()
        self.assertEqual(report["name"], "rushing:context")
        self.assertEqual(sorted(report["in_sample"]), ["converged", "iterations", "mae", "rmse"])
        self.assertEqual(report["features"], 5)

    def test_a_null_yardline_is_filled_and_counted(self):
        fitted = fit_rushing(carries((("rb1", 0.0),)))
        holdout = rushing_frame({"yardline_100": None}, {"rushing_yards": 3.0})
        design = fitted.holdout_design(holdout)
        self.assertEqual(design.notes["filled_continuous"], {"yardline_100": 1})
        self.assertEqual(fitted.predict(holdout).shape, (2,))

    def test_a_held_out_frame_that_breaks_the_contract_is_refused(self):
        fitted = fit_rushing(carries((("rb1", 0.0),)))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            fitted.predict(rushing_frame({}).drop("run_gap"))


class Helpers(unittest.TestCase):
    def test_a_penalty_comes_back_as_a_positive_float_or_not_at_all(self):
        self.assertEqual(penalty_value(2), 2.0)
        self.assertIsInstance(penalty_value(2), float)
        with self.assertRaisesRegex(ValueError, "baselines.penalty_value: penalty must be positive"):
            penalty_value(-1.0)

    def test_a_name_states_the_design_it_grew_from(self):
        self.assertEqual(baseline_name("passing", (), passing_design(passing_frame({}))),
                         "passing:context")
        sized = passing_design(passing_frame({}), composites={"position": ("position",)},
                               identities=("qb_id",))
        self.assertEqual(baseline_name("passing", ("qb_id",), sized),
                         "passing:context+position+identity(qb_id)")

    def test_a_closed_form_fit_reports_no_iteration_count(self):
        in_sample = fit_rushing(rushing_frame({}, {})).report()["in_sample"]
        self.assertIsNone(in_sample["iterations"])
        self.assertTrue(in_sample["converged"])

    def test_a_capped_fit_is_reported_and_its_state_passes_through(self):
        class Capped:
            n_iter_ = [7]

        self.assertTrue(convergence(Capped(), 8)["converged"])
        self.assertTrue(convergence(Capped(), None)["converged"])
        with self.assertWarns(RuntimeWarning):
            state = warn_if_not_converged(convergence(Capped(), 7), "passing_targets:test")
        self.assertEqual(state, {"converged": False, "iterations": 7})


if __name__ == "__main__":
    unittest.main()
