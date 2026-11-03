# Passing context components and conditional identity transfer

Observed November 3, 2026. This continues the [first passing study](PASSING_STUDY_RESULTS.md)
with declared context diagnostics and direct identity increments. All seasons remain
previously inspected development data; no final model, skill grade or causal claim is promoted.

## Frozen run

```powershell
python -m hierarchical_football.inference study-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_context_study.json --output-root data/experiments/passing_context_20261103
```

The local output directory is ignored and cannot be overwritten by a replay. Its manifest
records source hashes and exact runtime versions; generated data are not committed.

| Provenance | Value |
| --- | --- |
| Code revision | `3c03e46047679cb29a08a21a34241e00aded01b2` |
| Input SHA-256 | `ebe6f593139df4cadcbb6d0b07e7b797f9603ad350f34c1d0bb2f70ffa54c98c` |
| Specification SHA-256 | `acae3ea36c1f9e0d97f1c49d17f745530598419b032ea91859a224c4a4a8d706` |
| Report SHA-256 | `9b43419b8f183cf1280ae1d450240edbb19e62dc02d18e9384c5710e879bec9a` |
| Predictions SHA-256 | `d7fbed116a50e763e94b13962a9e29901dd6da46a76bed979c04c25121f9cc5a` |

The protocol preserves the first study's inner split and full 18-candidate grid. Selection
fits 2022 and scores 2023, choosing between linear/expanded context and independent QB
and receiver penalties of 10, 100 and 1000. It again selects expanded context with both
identity penalties at 100. All candidates and outer fits converge.

Outer folds train 2022-2023 and score 17,011 targets in 2024, then train 2022-2024 and
score 16,609 targets in 2025, with 272 games in each scored season. Eight context-only
diagnostics use the same training folds and fixed context penalty 1.0. They do not enter
selection. Intervals use paired game-clustered percentile bootstrap, 2,000 draws, seed 0,
95% level. These overlapping diagnostic intervals have no multiplicity adjustment.

## What supplied the context improvement?

Every combination retains linear target depth, field position, distance, down and pass
direction. `depth` adds fixed nonlinear depth hinges; `position` adds receiver position;
`opponent` adds one static defending-team intercept.

| Context-only model | 2024 log loss | 2025 log loss |
| --- | ---: | ---: |
| Linear reference | 0.571067 | 0.579700 |
| Depth | 0.568014 | 0.576125 |
| Position | 0.570925 | 0.579698 |
| Opponent | 0.572084 | 0.579688 |
| Depth + position | 0.567980 | 0.576268 |
| Depth + opponent | 0.569027 | 0.576140 |
| Position + opponent | 0.571935 | 0.579661 |
| Expanded: all three | 0.568988 | 0.576260 |

Nonlinear depth improves loss against the linear reference by -0.003053
[-0.004141, -0.001979] in 2024 and -0.003574 [-0.004708, -0.002416] in 2025.
The twelve matched component increments per season tell the same story: each depth
addition improves loss with its interval below zero, regardless of which other components
are already included.

Adding a component to the other two gives these richer-minus-simpler loss differences:

| Added component | Reference | 2024 difference [95% interval] | 2025 difference [95% interval] |
| --- | --- | --- | --- |
| Nonlinear depth | Position + opponent | -0.002947 [-0.004016, -0.001880] | -0.003401 [-0.004551, -0.002270] |
| Position | Depth + opponent | -0.000040 [-0.000254, +0.000173] | +0.000120 [-0.000125, +0.000382] |
| Opponent | Depth + position | +0.001008 [+0.000227, +0.001804] | -0.000008 [-0.000591, +0.000560] |

Thus the nonlinear difficulty curve supplies the consistent improvement. Position is
not a reliable extra gain here. Opponent intercepts cost loss in 2024 and are roughly
neutral in 2025. This does **not** establish that defense or position does not matter:
these are static additive terms under a shared context penalty, without defensive
personnel, coverage, season effects, or separately tuned opponent shrinkage.

Depth-only outperforms the selected expanded context on these scored seasons, but it
was a diagnostic rather than a selection candidate in this protocol. It is not promoted
from an inspected outer result. Component gains depend on their reference models and
must not be turned into additive percentages of explanation or causal credit.

## What do receivers add beyond quarterbacks?

The runner now reports joint-minus-QB loss directly as `receiver_given_qb`, alongside
joint-minus-receiver as `qb_given_receiver`, overall and within each transfer cohort.
The identity specification still uses the selected expanded context. Ablations keep the
jointly selected penalties rather than tuning their own priors.

| Receiver increment beyond QB | 2024 difference [95% interval] | 2025 difference [95% interval] |
| --- | --- | --- |
| All scored targets | -0.000402 [-0.000700, -0.000116] | -0.000294 [-0.000643, +0.000045] |
| Known partnerships | -0.000630 [-0.001064, -0.000199] | -0.000229 [-0.000767, +0.000294] |
| New partnerships, both players known | -0.000327 [-0.000944, +0.000269] | -0.000602 [-0.001292, +0.000090] |

The receiver block contributes beyond the QB model overall in 2024. The 2025 interval
crosses zero, and new-partnership intervals cross zero in both seasons. The small combined
identity gain against context therefore remains stronger evidence than receiver-specific
transfer. This reproduces the previous secondary diagnostic as a standard report field.
Unseen-player cohort increments also reflect refitted context and other coefficients;
they cannot establish information about a player who has no fitted identity term.

## Verification and direction

All output and package-source hashes were verified. The 33,620 scored play keys are unique;
all saved probabilities are finite and between zero and one. The original five model
prediction columns match the previous study within absolute tolerance 1e-10, and identity,
outcome and cohort columns match exactly. Enabling diagnostics leaves the main study intact.
All 373 offline tests and Python syntax checks passed. Ruff remains unavailable.

The next study should admit the simpler depth combinations to an inner selection grid,
then test whether receiver increments survive that context specification. Static opponent
terms warrant separate shrinkage or season-aware modeling before being used as attribution
controls. Any resulting choice must be validated prospectively; no untouched period is
certified here. Player-season effects, supported preselected NFL contrasts, richer charting,
route exposure, uncertainty calibration and a full Bayesian posterior remain future work.
