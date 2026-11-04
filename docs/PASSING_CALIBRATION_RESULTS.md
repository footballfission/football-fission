# Synthetic passing contrast calibration

## Frozen protocol and provenance

This is a known-truth stress test of the current regularized estimator and
conditional game bootstrap, not a calibration claim about NFL players.
Code revision `53e267419ba80c5a706ca3d7249c94288c8ca14e` ran:

```python
CalibrationSpec(trials=30, draws=50, games=20, repeats=4,
                penalties=(1.0, 100.0, 1000.0),
                scenarios=('crossed', 'nested', 'depth_confounded'),
                seed=20261103, level=0.95)
```

The ignored bundle `data/experiments/passing_calibration_20261103` contains
the exact specification, frozen runner, report, 270 trial rows and a complete
manifest with package-source hashes, versions and thread environment. Every
output and source hash was verified, with unique scenario/trial/penalty keys.
Manifest SHA-256:
`acf9e23bf585dbeb5096886952788e1a773384d82ea34eb178792b9dfc3eb5ac`.
Report SHA-256:
`11c5c13b2852ee778af5acbeb1853995959c8962ad5b5ab91c79fdcd7dfac7ea`.
Trial parquet SHA-256:
`7c30a2d98b1b1903ffc137d699deddbfefdd275463d2db9d355d439874cc31d6`.
Both `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` were 1 before Python startup.

The DGP is independent Bernoulli completion with logit
`0.4 + QB_effect + receiver_effect - 0.04 * (depth - 10)`.
QB effects are +/-0.25; receiver effects are +/-0.6. There are no latent game
effects or missing variables. The fitted linear logistic specification contains
the generating terms. Both identity penalties equal the declared value;
context penalty is 1. Each dataset and game-resampling seed is paired across
penalties. Independent datasets are generated for the 30 trials per scenario.
Q and priors are fixed within each interval. No tuning or holdout is involved.

## Fully crossed design: bias is absent from the interval

Every game includes all four QB/receiver pairs at depths 5 and 15, with four
targets per combination: 640 targets per dataset. Q gives equal weight to
these depths with synthetic_q1 fixed. The exact receiver contrast is
26.3013 percentage points. All 90 trial intervals complete; resampling loses
no context because every game contains the same design support.

| Both player penalties | Mean bias, percentage points | RMSE, percentage points | Mean interval width, percentage points | Truth covered / 30 |
| --- | ---: | ---: | ---: | ---: |
| 1 | -0.7594 | 4.2498 | 12.7997 | 27 |
| 100 | -14.8311 | 14.9495 | 5.6698 | 0 |
| 1000 | -24.3841 | 24.3861 | 0.9525 | 0 |

The nominal 95% intervals measure resampling variation around a regularized
estimate. Strong shrinkage substantially attenuates the known contrast, while
also narrowing that resampling distribution. It does not account for the
bias relative to the DGP parameter. Successful convergence, complete support
and a correctly specified model therefore do not establish parameter coverage.

With only 30 datasets and 50 draws per interval, these are coarse Monte Carlo
checks. Coverage of 27/30 has Wilson 95% bounds [74.38%, 96.54%], which include
95%; 0/30 has bounds [0%, 11.35%]. The weak-penalty result does not establish
calibration. Nor does this stress test select a better NFL penalty: sample
sizes, effect sizes, missing covariates and real design differ, and predictive
penalty selection optimizes a different objective.

## Disconnected partners: the existing guard works

The nested scenario includes only q1/r1 and q2/r2, each at both depths, for
320 targets per dataset. Its declared Q and exact contrast match the crossed
scenario. All 90 trials report `disconnected_players` and return no interval.
Regularized point estimates remain computable and biased, but conditional
coverage is null because no intervals are available. Zero returned intervals
containing truth is not a zero conditional coverage estimate.

## Connected partners can still be structurally confounded

The depth-confounded design contains all four QB/receiver pairs, but r1 only
has depth 6 and r2 only depth 9. It has 320 targets per dataset. Both depths
occupy the same coarse depth/down/direction cell; categories and numeric ranges
are represented in every original fit and resample. Q weights depth 6 and 9
equally with q1 fixed. Its exact contrast is 25.6560 percentage points.

| Both player penalties | Mean bias, percentage points | Mean interval width, percentage points | Truth covered / 30 |
| --- | ---: | ---: | ---: |
| 1 | -16.8297 | 5.6174 | 0 |
| 100 | -25.5247 | 0.0729 | 0 |
| 1000 | -25.6427 | 0.0073 | 0 |

All intervals complete, despite the data being unable to separate receiver
and depth effects. For example, adding 0.2 to the depth slope while adding
0.8 to r1's intercept and 0.2 to r2's leaves every observed logit unchanged,
but changes the standardized receiver contrast. A prior chooses among such
allocations; graph connectivity and coarse cells cannot rule them out.

The next implementation priority is a data-only design estimability audit,
followed by clear reporting of prior dependence alongside the conditional
bootstrap. Bias-aware uncertainty needs a separately justified method and
larger predeclared calibration studies. These results do not turn the existing
intervals into skill intervals, or establish causal identification.
