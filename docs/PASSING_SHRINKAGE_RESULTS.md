# Passing context selection with independent opponent shrinkage

Observed November 3, 2026. This extends the [context-component study](PASSING_CONTEXT_RESULTS.md)
by admitting simpler contexts during inner selection and tuning opponent shrinkage separately.
All scored seasons remain development data. No final model, player ranking or causal claim
is promoted by these results.

## Frozen protocol and provenance

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m hierarchical_football.inference study-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_shrinkage_study.json --output-root data/experiments/passing_shrinkage_20261103
```

Replays require a fresh output path. The exact configuration, runtime versions, source
hashes and output hashes are in the local ignored bundle's complete manifest.

| Item | Value |
| --- | --- |
| Code revision | `6f2683190967692d57abf4b720ff12d4da00701e` |
| Input SHA-256 | `ebe6f593139df4cadcbb6d0b07e7b797f9603ad350f34c1d0bb2f70ffa54c98c` |
| Specification SHA-256 | `d72ab9d5e405d7c72822867b16ee07e448a0505d86c67d674d4cdae94820d738` |
| Report SHA-256 | `9aafee1771588b4d32586d1c3a331a8a3e368ad69ea1ed67083bd99ca5b7d873` |
| Predictions SHA-256 | `e0786b15d1a295cea4126794eefc61cb5de450ab450d45e8d134a6b957e7b23a` |
| Numerical environment requests | OpenBLAS and OMP thread counts 1; other recorded thread settings unset |
| Runtime | Python 3.13.12; NumPy 2.5.4; SciPy 1.18.1; Polars 2.0.0; scikit-learn 1.9.1 |

Selection fits 2022 and scores 2023 using completion log loss. The context grid is
linear, depth, depth-plus-position, depth-plus-opponent, and expanded. QB and receiver
penalties independently range over 10, 100 and 1000. Opponent contexts also search
penalties 1, 100 and 10000; contexts without opponent terms are not redundantly repeated.
There are 81 distinct candidates, all converged. The situational context penalty is 1.

The selected specification is **expanded context, QB penalty 100, receiver penalty 100,
and opponent penalty 100**. All three searched penalties are interior grid values.
Context-only and player ablations retain the same opponent prior. Selection is frozen
before fitting 2022-2023 to score 2024 and 2022-2024 to score 2025.

## Earlier-season evidence

Every row below includes both player identity blocks and reports its best inner loss.
These are not the context-only diagnostic scores from the previous report.

| Candidate context | Best 2023 log loss | Opponent penalty |
| --- | ---: | ---: |
| Linear | 0.581282 | absent |
| Depth | 0.578697 | absent |
| Depth + position | 0.578583 | absent |
| Depth + opponent | 0.578461 | 100 |
| Expanded | 0.578348 | 100 |

Within expanded context, the best player-grid score at opponent penalty 1 is 0.578624;
at 100 it is 0.578348; at 10000 it is 0.578573. The preference is for substantial
shrinkage, not removing the opponent block entirely. The margin is small and comes from
one validation season on a coarse grid, so this is not a universal best-prior claim.

The simpler depth contexts remain meaningful alternatives even though they did not win
the inner criterion. This selection does not reverse the previous finding that nonlinear
depth supplies the consistent context-only improvement.

## Later-season completion performance

All outer fits converge. The scored samples are unchanged: 17,011 targets in 2024 and
16,609 in 2025, with 272 games each. Intervals use paired game-clustered percentile
bootstrap, 2,000 draws, seed 0 and level 95%. Negative differences favor the added block.

| Model | 2024 log loss | 2025 log loss |
| --- | ---: | ---: |
| Historical linear context | 0.571067 | 0.579700 |
| Selected context, opponent penalty 100 | 0.568440 | 0.576144 |
| Selected context + QB | 0.567527 | 0.575609 |
| Selected context + receiver | 0.567849 | 0.575669 |
| Selected context + both | 0.567116 | 0.575310 |

| Identity comparison | 2024 difference [95% interval] | 2025 difference [95% interval] |
| --- | --- | --- |
| Both versus selected context | -0.001323 [-0.001835, -0.000815] | -0.000834 [-0.001454, -0.000191] |
| Receiver beyond QB, all targets | -0.000410 [-0.000706, -0.000124] | -0.000299 [-0.000649, +0.000039] |
| Both versus context, new partnerships | -0.001006 [-0.001853, -0.000188] | -0.001235 [-0.002357, -0.000227] |
| Receiver beyond QB, new partnerships | -0.000348 [-0.000963, +0.000241] | -0.000603 [-0.001293, +0.000088] |

The combined identity block still carries transferable information beyond this difficulty
model. Receiver-specific increments on new partnerships remain uncertain in both seasons,
and the overall receiver increment remains uncertain in 2025. Changing the environment
prior does not establish portable receiver skill or an individual player's attribution.

## Comparison with the previous specification

The earlier joint model used expanded context, player penalties 100 and opponent penalty 1.
A secondary paired comparison of saved predictions gives new-minus-previous loss:

| Population | 2024 difference [95% interval] | 2025 difference [95% interval] |
| --- | --- | --- |
| All targets | -0.000549 [-0.000810, -0.000295] | -0.000170 [-0.000318, -0.000019] |
| New partnerships | -0.000637 [-0.001096, -0.000202] | -0.000222 [-0.000479, +0.000048] |

Thus independently selected opponent shrinkage improves overall completion loss slightly
in both scored seasons. This comparison was computed after the primary run and did not
choose the prior. It remains development evidence on inspected seasons. The 2024 share
of joint-model targets with individual log loss at least 2 falls from 0.182% to 0.135%;
the 2025 share is unchanged at 0.181%. Average and tail behavior are both retained in reports.

For reproduction, align the current and previous `predictions.parquet` files on their
identical ordered game/play keys. Compute `row_losses(y, p_both, task="completion")`
for each file, then `bootstrap_difference(new_losses, old_losses, clusters=game_id,
draws=2000, seed=0)`, overall and within `transfer_cohort == "new_pair"` per scored season.

## Verification and limits

All source and output hashes were checked. The 33,620 scored play keys are unique, all
probabilities are finite and bounded, and identity, outcome and cohort rows match the
previous study exactly. The historical linear reference differs by at most 5.1e-12.
All 384 offline tests, compilation and whitespace checks passed. Ruff remains unavailable.

The protocol was motivated by inspected development results; an earlier selection split
does not turn the outer seasons into untouched evidence. No individual NFL contrast was
selected in this run. Conditional bootstrap comparisons retain fixed priors and omit
selection uncertainty. Static opponent intercepts still lack season-specific defensive
personnel, pressure and coverage information. Target earning, routes, player-season
variation, calibrated player uncertainty and a full Bayesian posterior remain unresolved.
The next useful work is temporal stability and supported player-comparison analysis,
with prospective evaluation reserved before any model promotion.
