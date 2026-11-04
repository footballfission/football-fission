# Supported historical player comparisons

Observed November 3, 2026. This is the first real-player demonstration of the retrospective
comparison workflow. It asks about target conversion under a narrow shared context mix,
not total receiver value, portable talent, or an effect of changing players.

## Protocol and provenance

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m hierarchical_football.inference compare-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_player_comparison.json --output-root data/experiments/passing_players_phi_20261103
python -m hierarchical_football.inference compare-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_unsupported_comparison.json --output-root data/experiments/passing_players_bal_20261103
```

Replays require fresh output paths. Each local ignored bundle contains an exact protocol,
a report, the retained reference frame and a complete hashed manifest. Code revision is
`899ce065796a4dac4ea84eee6044db3235bdae13`; canonical input SHA-256 is
`ebe6f593139df4cadcbb6d0b07e7b797f9603ad350f34c1d0bb2f70ffa54c98c`.
Runtime versions are Python 3.13.12, NumPy 2.5.4, SciPy 1.18.1, Polars 2.0.0 and
scikit-learn 1.9.1; OpenBLAS and OMP thread requests are 1.

| Artifact | Philadelphia | Baltimore |
| --- | --- | --- |
| Protocol SHA-256 | `da83e93899c22e0d58eb06f7f7aac02f7a0e0c6d11a6b105093de2adf30ae659` | `17cbb18ff27531e6e7397f2dbdf67aa41cf0dfbdd014b3ef064355da051ec8e9` |
| Report SHA-256 | `1c3cc42830bfa91aaf477428e5a4579aa8918c4689e3665dffa996ef9ab814fb` | `8838ba12bd7dec673e83beb325d95b311d49713f4b72e0e9d0e1a09c6c66af91` |
| Reference frame SHA-256 | `bb4f80c35b6a8178ab6264c657f1ba3bd8dcc99fe18b2b245d09bd59390c3df7` | `bee34b4af09ff5196cfd647da7ac3f13d15bf81bef9a7ee0d660249150079e4b` |

Pairs were chosen from teammate roles and exposure, before inspecting fitted effects.
Names and GSIS IDs were checked against cached 2024 weekly rosters. Philadelphia compares
A.J. Brown (`00-0035676`) with DeVonta Smith (`00-0036912`), fixing Jalen Hurts
(`00-0036389`). Baltimore compares Mark Andrews (`00-0034753`) with Isaiah Likely
(`00-0037838`), fixing Lamar Jackson (`00-0034796`). Neither example was chosen from a
coefficient leaderboard.

Every fit uses only that team's targeted passes, separately in 2022-2023 and 2024-2025.
Both windows use expanded context and fixed QB, receiver and opponent penalties of 100,
with situational penalty 1. These priors come from the earlier development specification;
they were not reselected for a team-local study. The sensitivity grid independently varies
QB and receiver penalties over 10, 100 and 1000, keeping the opponent prior fixed.

Q begins with the earliest window's observed targets to the declared position. A retained
cell must contain at least five targets for each player with the fixed QB in each window.
Cells use target-depth bands, down and pass direction. Later exposure can restrict Q,
so the workflow explicitly reports `forecast_evaluation=false`.

## Philadelphia: a narrow estimable comparison

The initial WR reference has 687 targets in 56 cells. Only 228 targets, 33.2%, survive in
seven shared cells. They describe short/intermediate throws: second-down left/right
0-5-yard bands, first/second-down left/right 5-10-yard bands, and third-down right
5-10-yard targets. The existing bin scheme clamps negative depths into its first band;
these are coarse context cells, not exact route or tracking matches.

| Fitting period | All PHI targeted passes | Games | Brown targets with Hurts | Smith targets with Hurts |
| --- | ---: | ---: | ---: | ---: |
| 2022-2023 | 1,030 | 34 | 286 | 221 |
| 2024-2025 | 882 | 34 | 200 | 184 |

Both fitted windows evaluate the same 228-row Q with Hurts fixed in both arms. The
predictions below are model-standardized rates, not either player's raw catch percentage.
Differences are Brown minus Smith, in percentage points.

| Fitting period | Brown prediction | Smith prediction | Difference | Conditional 95% interval | Prior sensitivity range |
| --- | ---: | ---: | ---: | --- | --- |
| 2022-2023 | 74.11% | 75.96% | -1.84 pp | [-4.49, +0.70] pp | [-4.57, -0.27] pp |
| 2024-2025 | 77.58% | 78.11% | -0.53 pp | [-2.34, +1.29] pp | [-1.54, -0.07] pp |

Each conditional interval uses 200 whole-game refits with fixed priors and fixed Q.
Both windows complete all 200 replicates without missing players, single-outcome samples,
disconnection or convergence failures. The windows use independent seeds derived from
protocol seed 0. All nine sensitivity fits per period converge.

Point directions agree across windows and the declared prior grid, but both sampling
intervals include zero. Effect magnitude changes substantially with receiver shrinkage.
The sensitivity range measures a modeling choice and is not a confidence interval. There
is no interval for the change between windows, so the smaller later point estimate does
not establish that a receiver's ability changed or that a gap narrowed.

One-third of the original reference population is already a narrow estimand. Furthermore,
76 of its 228 rows have defending-team categories absent from the later fitting window;
those context terms fall back to zero under the reference design. The report exposes this
in `reference_context.unseen_categorical_rows`. No continuous context values are filled.
Shared coarse cells therefore do not certify full covariate overlap, and this comparison
cannot support a general claim that one receiver is better.

## Baltimore: no shared population under the declared rule

The initial TE reference contains 314 targets in 44 cells. Andrews has 140 targets with
Jackson in the early window and 120 later; Likely has 68 and 86. Likely has only one cell
with five or more joint targets early and four later. No cell meets the joint threshold
for both receivers in both windows.

The report returns `status="no_shared_support"`, an empty typed reference frame, and no
player estimates or intervals. The workflow's manifest is complete because reporting
non-estimability is a valid completed analysis. The threshold was not relaxed after seeing
this result, and a lack of support is not evidence that the receivers are equally good.

## What this moves forward

These examples demonstrate a working explanation report that can expose insufficient
support and distinguish sampling uncertainty from prior sensitivity. They do not validate
individual skill grades. Completion conditional on a target omits target earning, route
quality, catchability, pressure and other mechanisms; team-local coefficients cannot be
read as league-wide effects. Player terms are additive intercepts, so this model does
not identify receiver-specific strengths at different depths or on different routes.
The intervals do not include tuning uncertainty, prior bias,
or a full identification analysis, and 200 bootstrap draws are a demonstration setting,
not a claim of calibrated real-player interval coverage.

All output and package-source hashes were verified, reference keys are unique, retained
counts match reports, and all 398 offline tests pass. Compilation and whitespace checks
pass; Ruff remains unavailable. Generated data stay ignored and all commits stay local.

The next useful step is to investigate coverage beyond the coarse cells and compare
supported findings under alternative context specifications. Any temporal talent claim
also needs a stronger identification design and a prospectively reserved evaluation.
