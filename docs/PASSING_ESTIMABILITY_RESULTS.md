# Original-design estimability of the Philadelphia comparisons

## Refit audit of frozen reference populations

At code `850c01e081bb621851ca48836b28ffcb32b527e0`, the original
Philadelphia Brown-minus-Smith comparisons with Hurts were refitted using
their exact saved coarse and model-context reference populations. All four
point contrasts reproduce their frozen values within `1e-10`. Penalties,
training rows and the reference populations were unchanged. The earlier
bootstrap was not repeated, and its intervals were not recalibrated.

The ignored bundle `data/experiments/passing_estimability_20261103` records
the frozen runner, source hashes, original bundle and canonical input hashes,
and four original-design audits. All input, source and output hashes were
verified. Its manifest SHA-256 is
`c2aa752ebedbd863f2cba8545577e56e3248ba4bfecce0d786bb007a884257cc`;
report SHA-256 is
`700cac5268ef45da80568c41c49100a182ec1b3b9b66b718d8fdea1416cb0a80`.
Both numerical thread variables were set to 1 before Python startup.

## Log-odds contrasts and probability arms differ

| Reference policy | Window | Q rows | Training rank / augmented columns | Log-odds contrast estimable | Unestimable probability rows, each arm |
| --- | --- | ---: | ---: | --- | ---: |
| Coarse | 2022-2023 | 228 | 56 / 64 | Yes | 0 |
| Coarse | 2024-2025 | 228 | 58 / 67 | Yes | 76 |
| Model context | 2022-2023 | 152 | 56 / 64 | Yes | 0 |
| Model context | 2024-2025 | 152 | 58 / 67 | Yes | 0 |

The audit uses an augmented intercept plus the fitted design, with `rtol=1e-10`.
Relative row-space residuals for passing arms are below `2.2e-15`. The later
coarse arms have maximum residual approximately 0.0830. Redundant dummy gauges
cause rank deficiency even in the passing cases; full column rank is not the
criterion for a particular contrast.

The 76 later failing rows match the reference opponents absent from that
fitting window. A receiver log-odds difference can be estimable because its
common baseline cancels. Probability differences also depend on that baseline
through the sigmoid, so a linear contrast check alone cannot certify their
arm predictions. The stricter reference removes those rows and passes both
original-window arm checks. The probability check is a sufficient condition,
not an exhaustive proof for every possible cancellation.

## What passing does and does not establish

The [context sensitivity experiment](PASSING_CONTEXT_SUPPORT_RESULTS.md)
already showed tiny changes in this contrast after restricting Q, and
frequent context loss during game resampling. This audit addresses only the
original fitting designs, not each resample. It does not invalidate the
regularized estimator's defined fallback predictions or certify their
interpretation as skill.

The [known-truth calibration study](PASSING_CALIBRATION_RESULTS.md) also
shows severe undercoverage under strong shrinkage in fully crossed data.
That design passes structural estimability. Removing exact aliasing therefore
does not remove shrinkage bias, missing-variable bias, finite-sample uncertainty
or the need to validate parameter intervals. No player ranking or between-window
talent change follows from these diagnostics.

## Public workflow verification

A parallel CLI smoke run at the same code and numerical environment used two
datasets per scenario, 20 refits, six games and two repetitions, with all three
penalties. All 18 trial rows and their hashes were verified in
`data/experiments/passing_calibration_cli_smoke_20261103`. It returned six
estimable crossed trials, six prior-dependent nested trials without intervals,
and six prior-dependent depth-confounded trials with completed estimator
intervals. The common manifest records package versions and synthetic input
provenance. This tiny run verifies the artifact path, not coverage rates.
Its manifest SHA-256 is
`f742a3bdee3f885ba477341ad46b8056e9bed1a3af0827b6d47a03dda935aa3e`.
