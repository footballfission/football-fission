# Passing comparison context sensitivity

## Frozen retrospective check

Two Philadelphia Brown-minus-Smith comparisons with Hurts repeat the protocol in
[the first player comparison](PASSING_PLAYER_COMPARISON.md). Both use the same
2022-2023 and 2024-2025 team-local fits, penalties of 100, exposure threshold of
five joint targets per cell, and 200 conditional whole-game bootstrap draws.
Only the reference policy differs: `coarse` versus `model_context`.

Both bundles completed at code `922094854fb4ac67dde91358f3519978de7b2e58`:

| Policy | Frozen specification | Local output directory |
| --- | --- | --- |
| Coarse | `examples/passing_player_comparison.json` | `data/experiments/passing_context_coarse_audit_20261103` |
| Model context | `examples/passing_player_context_comparison.json` | `data/experiments/passing_context_strict_audit_20261103` |

The canonical input SHA-256 is
`ebe6f593139df4cadcbb6d0b07e7b797f9603ad350f34c1d0bb2f70ffa54c98c`.
Manifest SHA-256 values are
`e6c9cd9cb9186c1263e70c06ceb8ad0c42f710e51905772df1a07633e1a820be`
(coarse) and
`4cc72fd7044762a7e2753aad281915990ff3327525256869dff041cd641ab7f2`
(model context). All output, input and package-source hashes were verified;
saved reference keys are unique and row counts match the reports. Artifacts
remain ignored local research outputs. Neither run evaluates forecasts.

## Restricting the reference population

The coarse screen retains 228 of 687 original WR targets (33.19%). Of those,
76 have opponent categories absent from the later fitting window. One also
falls outside its observed depth range; these failures overlap. The stricter
policy retains 152 targets (22.13% of the original reference), with categories
and numeric source ranges represented in both original fitting windows.
Training data and fitted coefficients are identical between policies.

| Window | Coarse difference, percentage points | Model-context difference, percentage points | Model-context conditional 95% interval |
| --- | ---: | ---: | ---: |
| 2022-2023 | -1.8434 | -1.8471 | [-4.4966, 0.6988] |
| 2024-2025 | -0.5297 | -0.5287 | [-2.3475, 1.2955] |

The diagnostic-only coarse run reproduces the previous point estimates and
intervals exactly. Restricting Q barely changes this particular contrast;
both intervals still include zero. This is evidence of stability under this
specific population restriction, not evidence that the missing contexts are
harmless in general or that either receiver is better. The additive player
intercepts also limit how much contextual heterogeneity the model can express.

## Resampling still loses context

All 200 refits per window converge without identity failures. Nevertheless,
sampling games with replacement often omits opponents represented in fixed Q.

| Policy | Window | Refits with context failures / 200 | Mean failed share of Q |
| --- | --- | ---: | ---: |
| Coarse | 2022-2023 | 200 | 21.60% |
| Coarse | 2024-2025 | 200 | 42.57% |
| Model context | 2022-2023 | 199 | 16.00% |
| Model context | 2024-2025 | 197 | 13.85% |

Every affected refit loses at least one opponent category. Numeric range
failures also occur and overlap with category failures. Under the stricter Q,
field-position ranges fail in 85 early and 41 later refits; distance fails in
two early and nine later refits. Depth ranges no longer fail. No nonfinite
reference values occur. Per-draw failed rows reach 52 early and 66 later under
the stricter policy.

These are diagnostics of the fixed regularized estimator. The workflow does
not discard refits based on their context coverage: doing so would silently
condition the bootstrap distribution on favorable samples. A completed
interval therefore does not certify full covariate coverage. Marginal
category/range screening also does not establish joint positivity, causal
identification, or calibrated confidence coverage.

The next research priority is simulation with known player contrasts: measure
shrinkage bias and conditional interval coverage before interpreting these
intervals as uncertainty about player skill. Retrospective real-player
stability alone cannot establish that interpretation.
