"""Data-only linear-design estimability, separate from prior precision or causality."""
from __future__ import annotations

import numpy as np
import polars as pl


def audit_contrast_estimability(fitted, population, *, identity, left, right, fixed=None, rtol=1e-10):
    """Check whether requested predictors lie in the augmented training row space.

    Includes the intercept and tolerates redundant dummy gauges. The average
    log-odds difference is a linear functional. For probability contrasts, each
    arm's rowwise predictor being estimable is a sufficient condition, stronger
    than checking only the difference. Rank is numerical at declared tolerance;
    passing is neither causal identification nor evidence of useful precision.
    """
    if (isinstance(rtol, bool) or not isinstance(rtol, (int, float)) or not 0 < rtol < 1):
        raise ValueError('estimability needs 0 < rtol < 1')
    if not population.height:
        raise ValueError('estimability needs a nonempty population')
    if fitted.task != 'completion' or identity not in fitted.design.identities or left == right:
        raise ValueError('estimability needs a completion model and two distinct modeled identities')
    fixed = dict(fixed or {})
    if identity in fixed or set(fixed) - set(fitted.design.identities):
        raise ValueError('fixed levels must name other modeled identities')
    for column, level in [(identity, left), (identity, right), *fixed.items()]:
        if level not in fitted.design.identities[column]['levels']:
            raise ValueError('estimability requires observed player levels')
    matrix = np.column_stack((np.ones(fitted.design.rows), fitted.design.matrix))
    # QR avoids materializing the tall left-singular-vector matrix. R has the
    # same singular values and right singular vectors as the training matrix.
    reduced = np.linalg.qr(matrix, mode='r')
    _, singular, right_vectors = np.linalg.svd(reduced, full_matrices=False)
    cutoff = float(singular[0] * rtol)
    rank = int(np.sum(singular > cutoff))
    basis = right_vectors[:rank]

    def residuals(vectors):
        remainder = vectors - (vectors @ basis.T) @ basis
        return np.linalg.norm(remainder, axis=-1) / np.maximum(1.0, np.linalg.norm(vectors, axis=-1))

    arms, predictors = {}, []
    for side, player in (('left', left), ('right', right)):
        frame = population.with_columns([pl.lit(level).alias(column)
                                         for column, level in {**fixed, identity: player}.items()])
        vectors = np.column_stack((np.ones(frame.height), fitted.holdout_design(frame).matrix))
        relative = residuals(vectors)
        unestimable = int(np.sum(relative > rtol))
        arms[side] = {'rows': frame.height, 'estimable_rows': frame.height - unestimable,
                      'unestimable_rows': unestimable, 'max_relative_residual': float(relative.max())}
        predictors.append(vectors)
    contrast_residual = float(residuals((predictors[0] - predictors[1]).mean(axis=0)))
    log_odds_estimable = contrast_residual <= rtol
    probability_estimable = all(arm['unestimable_rows'] == 0 for arm in arms.values())
    return {'status': 'estimable' if log_odds_estimable and probability_estimable else 'prior_dependent',
            'training_rows': fitted.design.rows, 'augmented_columns': matrix.shape[1],
            'rank': rank, 'nullity': matrix.shape[1] - rank, 'rtol': rtol,
            'singular_value_cutoff': cutoff, 'arms': arms,
            'log_odds_contrast': {'estimable': log_odds_estimable, 'relative_residual': contrast_residual},
            'probability_arms_estimable': probability_estimable,
            'note': 'Training design row-space audit including intercept; probability arm checks are '
                    'sufficient, not necessary. Ignores regularization and outcome precision. '
                    'Marginal extrapolation may remain estimable under the model. '
                    'Passing does not establish causal identification or calibrated uncertainty.'}
