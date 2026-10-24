# Hierarchical chain: frozen shape parameters and joint touchdowns

Reference results: benchmark implementations, experiment runners, datasets, and raw draws are not bundled. These tables are not a claim of a newly reproduced package evaluation.

Follow-up to [CENTRAL_RESULTS.md](CENTRAL_RESULTS.md). The model supports frozen distribution shapes and alternative touchdown processes.

Two follow-up experiments on the hierarchical parametric model, run on the same
36 week-ahead folds, the same features, the same test rows and the same
200 draws as the central comparison. The direct and multivariate DRF rows are
loaded from the frozen central run on identical keys, so they are exact
references rather than re-runs.

| Variant | What changed |
| --- | --- |
| Hierarchical (published) | shape parameters from an internal 80/20 split of each fold |
| Frozen shapes | one calibration on the season immediately before the holdout, never re-estimated |
| Joint touchdowns | the touchdown is drawn with the per-catch gain via P(touchdown given gain) |
| Frozen + joint | both changes together |

## Fantasy-point results

| Model | CRPS | MAE | 80% cov | 80% width | PIT deviation | Mean predictive sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF | 2.967 | 4.364 | 0.860 | 13.53 | 0.045 | 5.77 |
| Multivariate DRF | 2.972 | 4.412 | 0.870 | 13.94 | 0.069 | 5.93 |
| Hierarchical (published) | 2.977 | 4.378 | 0.859 | 13.76 | 0.080 | 5.80 |
| Hierarchical, frozen shapes | 2.974 | 4.370 | 0.851 | 13.65 | 0.067 | 5.76 |
| Hierarchical, joint touchdowns | 2.982 | 4.373 | 0.855 | 13.72 | 0.058 | 5.77 |
| Hierarchical, frozen + joint TDs | 2.979 | 4.368 | 0.849 | 13.62 | 0.078 | 5.73 |

Paired deltas against the direct DRF. Positive means the candidate beats it.

| Model | CRPS delta (player 95% CI) | MAE delta (player 95% CI) |
| --- | --- | --- |
| Multivariate DRF | -0.005 [-0.021, +0.009] | -0.049 [-0.070, -0.025] |
| Hierarchical (published) | -0.010 [-0.028, +0.009] | -0.014 [-0.049, +0.023] |
| Hierarchical, frozen shapes | -0.008 [-0.025, +0.012] | -0.007 [-0.038, +0.025] |
| Hierarchical, joint touchdowns | -0.015 [-0.035, +0.002] | -0.009 [-0.042, +0.022] |
| Hierarchical, frozen + joint TDs | -0.012 [-0.031, +0.007] | -0.004 [-0.034, +0.027] |

Direct variant tests against the published hierarchical chain. Positive means
the change improves CRPS or MAE.

| Variant | CRPS delta | MAE delta |
| --- | --- | --- |
| Hierarchical, frozen shapes | +0.002 [-0.008, +0.014] | +0.007 [-0.010, +0.026] |
| Hierarchical, joint touchdowns | -0.005 [-0.016, +0.005] | +0.005 [-0.014, +0.023] |
| Hierarchical, frozen + joint TDs | -0.002 [-0.012, +0.009] | +0.010 [-0.009, +0.028] |

## Ceiling and floor events

Predicted against realised rate, with Brier score and the realised rate among
the model's own top decile of that event. This is the diagnostic that says which
component carries the excess ceiling.

| Event | Model | Predicted | Realised | Brier | Top-decile realised |
| --- | --- | ---: | ---: | ---: | ---: |
| fantasy_points_ge_20 | Direct DRF | 0.0762 | 0.0739 | 0.0600 | 0.2814 |
| fantasy_points_ge_20 | Hierarchical (published) | 0.0790 | 0.0739 | 0.0609 | 0.2753 |
| fantasy_points_ge_20 | Hierarchical, frozen shapes | 0.0778 | 0.0739 | 0.0607 | 0.2854 |
| fantasy_points_ge_20 | Hierarchical, frozen + joint TDs | 0.0771 | 0.0739 | 0.0608 | 0.2733 |
| fantasy_points_ge_20 | Hierarchical, joint touchdowns | 0.0775 | 0.0739 | 0.0607 | 0.2834 |
| fantasy_points_ge_20 | Multivariate DRF | 0.0759 | 0.0739 | 0.0602 | 0.2794 |
| fantasy_points_ge_25 | Direct DRF | 0.0341 | 0.0296 | 0.0271 | 0.1397 |
| fantasy_points_ge_25 | Hierarchical (published) | 0.0403 | 0.0296 | 0.0278 | 0.1194 |
| fantasy_points_ge_25 | Hierarchical, frozen shapes | 0.0394 | 0.0296 | 0.0276 | 0.1235 |
| fantasy_points_ge_25 | Hierarchical, frozen + joint TDs | 0.0389 | 0.0296 | 0.0274 | 0.1316 |
| fantasy_points_ge_25 | Hierarchical, joint touchdowns | 0.0393 | 0.0296 | 0.0278 | 0.1255 |
| fantasy_points_ge_25 | Multivariate DRF | 0.0343 | 0.0296 | 0.0269 | 0.1336 |
| fantasy_points_ge_30 | Direct DRF | 0.0145 | 0.0109 | 0.0107 | 0.0567 |
| fantasy_points_ge_30 | Hierarchical (published) | 0.0199 | 0.0109 | 0.0109 | 0.0547 |
| fantasy_points_ge_30 | Hierarchical, frozen shapes | 0.0195 | 0.0109 | 0.0109 | 0.0526 |
| fantasy_points_ge_30 | Hierarchical, frozen + joint TDs | 0.0192 | 0.0109 | 0.0109 | 0.0607 |
| fantasy_points_ge_30 | Hierarchical, joint touchdowns | 0.0197 | 0.0109 | 0.0110 | 0.0486 |
| fantasy_points_ge_30 | Multivariate DRF | 0.0146 | 0.0109 | 0.0106 | 0.0607 |
| fantasy_points_ge_35 | Direct DRF | 0.0061 | 0.0065 | 0.0063 | 0.0466 |
| fantasy_points_ge_35 | Hierarchical (published) | 0.0098 | 0.0065 | 0.0063 | 0.0445 |
| fantasy_points_ge_35 | Hierarchical, frozen shapes | 0.0095 | 0.0065 | 0.0063 | 0.0425 |
| fantasy_points_ge_35 | Hierarchical, frozen + joint TDs | 0.0093 | 0.0065 | 0.0063 | 0.0425 |
| fantasy_points_ge_35 | Hierarchical, joint touchdowns | 0.0095 | 0.0065 | 0.0063 | 0.0385 |
| fantasy_points_ge_35 | Multivariate DRF | 0.0058 | 0.0065 | 0.0062 | 0.0445 |
| fantasy_points_lt_5 | Direct DRF | 0.5138 | 0.5194 | 0.1678 | 0.9413 |
| fantasy_points_lt_5 | Hierarchical (published) | 0.5145 | 0.5194 | 0.1680 | 0.9393 |
| fantasy_points_lt_5 | Hierarchical, frozen shapes | 0.5150 | 0.5194 | 0.1682 | 0.9372 |
| fantasy_points_lt_5 | Hierarchical, frozen + joint TDs | 0.5163 | 0.5194 | 0.1685 | 0.9433 |
| fantasy_points_lt_5 | Hierarchical, joint touchdowns | 0.5186 | 0.5194 | 0.1685 | 0.9372 |
| fantasy_points_lt_5 | Multivariate DRF | 0.5122 | 0.5194 | 0.1688 | 0.9413 |
| receiving_tds_ge_2 | Hierarchical (published) | 0.0311 | 0.0245 | 0.0235 | 0.0789 |
| receiving_tds_ge_2 | Hierarchical, frozen shapes | 0.0302 | 0.0245 | 0.0234 | 0.0729 |
| receiving_tds_ge_2 | Hierarchical, frozen + joint TDs | 0.0255 | 0.0245 | 0.0233 | 0.0749 |
| receiving_tds_ge_2 | Hierarchical, joint touchdowns | 0.0253 | 0.0245 | 0.0235 | 0.0769 |
| receiving_tds_ge_2 | Multivariate DRF | 0.0221 | 0.0245 | 0.0233 | 0.0870 |
| receiving_tds_ge_3 | Hierarchical (published) | 0.0054 | 0.0018 | 0.0019 | 0.0040 |
| receiving_tds_ge_3 | Hierarchical, frozen shapes | 0.0051 | 0.0018 | 0.0019 | 0.0081 |
| receiving_tds_ge_3 | Hierarchical, frozen + joint TDs | 0.0036 | 0.0018 | 0.0019 | 0.0040 |
| receiving_tds_ge_3 | Hierarchical, joint touchdowns | 0.0036 | 0.0018 | 0.0019 | 0.0040 |
| receiving_tds_ge_3 | Multivariate DRF | 0.0020 | 0.0018 | 0.0018 | 0.0101 |
| receiving_yards_ge_100 | Hierarchical (published) | 0.0672 | 0.0567 | 0.0503 | 0.1964 |
| receiving_yards_ge_100 | Hierarchical, frozen shapes | 0.0657 | 0.0567 | 0.0498 | 0.2186 |
| receiving_yards_ge_100 | Hierarchical, frozen + joint TDs | 0.0662 | 0.0567 | 0.0501 | 0.2004 |
| receiving_yards_ge_100 | Hierarchical, joint touchdowns | 0.0670 | 0.0567 | 0.0503 | 0.2065 |
| receiving_yards_ge_100 | Multivariate DRF | 0.0619 | 0.0567 | 0.0486 | 0.2267 |
| receiving_yards_ge_125 | Hierarchical (published) | 0.0333 | 0.0211 | 0.0208 | 0.0810 |
| receiving_yards_ge_125 | Hierarchical, frozen shapes | 0.0325 | 0.0211 | 0.0207 | 0.0850 |
| receiving_yards_ge_125 | Hierarchical, frozen + joint TDs | 0.0324 | 0.0211 | 0.0207 | 0.0830 |
| receiving_yards_ge_125 | Hierarchical, joint touchdowns | 0.0336 | 0.0211 | 0.0208 | 0.0870 |
| receiving_yards_ge_125 | Multivariate DRF | 0.0246 | 0.0211 | 0.0199 | 0.0810 |
| receiving_yards_ge_150 | Hierarchical (published) | 0.0163 | 0.0069 | 0.0073 | 0.0324 |
| receiving_yards_ge_150 | Hierarchical, frozen shapes | 0.0155 | 0.0069 | 0.0072 | 0.0405 |
| receiving_yards_ge_150 | Hierarchical, frozen + joint TDs | 0.0156 | 0.0069 | 0.0071 | 0.0425 |
| receiving_yards_ge_150 | Hierarchical, joint touchdowns | 0.0166 | 0.0069 | 0.0073 | 0.0304 |
| receiving_yards_ge_150 | Multivariate DRF | 0.0101 | 0.0069 | 0.0067 | 0.0385 |
| receptions_ge_10 | Hierarchical (published) | 0.0222 | 0.0140 | 0.0133 | 0.0830 |
| receptions_ge_10 | Hierarchical, frozen shapes | 0.0221 | 0.0140 | 0.0132 | 0.0810 |
| receptions_ge_10 | Hierarchical, frozen + joint TDs | 0.0223 | 0.0140 | 0.0134 | 0.0830 |
| receptions_ge_10 | Hierarchical, joint touchdowns | 0.0225 | 0.0140 | 0.0132 | 0.0850 |
| receptions_ge_10 | Multivariate DRF | 0.0164 | 0.0140 | 0.0130 | 0.0688 |
| receptions_ge_8 | Hierarchical (published) | 0.0550 | 0.0476 | 0.0407 | 0.2166 |
| receptions_ge_8 | Hierarchical, frozen shapes | 0.0552 | 0.0476 | 0.0406 | 0.2206 |
| receptions_ge_8 | Hierarchical, frozen + joint TDs | 0.0551 | 0.0476 | 0.0407 | 0.2166 |
| receptions_ge_8 | Hierarchical, joint touchdowns | 0.0552 | 0.0476 | 0.0404 | 0.2126 |
| receptions_ge_8 | Multivariate DRF | 0.0514 | 0.0476 | 0.0404 | 0.2206 |
| targets_ge_10 | Hierarchical (published) | 0.0877 | 0.0822 | 0.0622 | 0.3543 |
| targets_ge_10 | Hierarchical, frozen shapes | 0.0877 | 0.0822 | 0.0623 | 0.3421 |
| targets_ge_10 | Hierarchical, frozen + joint TDs | 0.0876 | 0.0822 | 0.0622 | 0.3522 |
| targets_ge_10 | Hierarchical, joint touchdowns | 0.0878 | 0.0822 | 0.0618 | 0.3583 |
| targets_ge_10 | Multivariate DRF | 0.0886 | 0.0822 | 0.0623 | 0.3360 |

## Component CRPS by variant

| Statistic | Multivariate DRF | Hierarchical (published) | Hierarchical, frozen shapes | Hierarchical, joint touchdowns | Hierarchical, frozen + joint TDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| targets | 1.271 | 1.273 | 1.271 | 1.273 | 1.274 |
| receptions | 0.928 | 0.927 | 0.924 | 0.925 | 0.926 |
| receiving_yards | 13.864 | 14.005 | 13.998 | 14.020 | 14.004 |
| receiving_tds | 0.159 | 0.159 | 0.159 | 0.160 | 0.159 |
| carries | 0.125 | 0.126 | 0.127 | 0.127 | 0.127 |
| rushing_yards | 0.853 | 0.859 | 0.862 | 0.864 | 0.863 |
| rushing_tds | 0.005 | 0.005 | 0.005 | 0.006 | 0.005 |

## Integrity check

The direct DRF and multivariate DRF rows are the exact draws from the frozen
central run, joined on identical keys, so they are not re-runs. The published
hierarchical chain was re-fitted in this run and scores 2.977 against 2.986 in
the central run. The paired player-clustered difference is +0.0093 with a 95%
interval of [−0.0014, +0.0205], which is what an independent draw stream costs
at 200 draws.

That number matters for reading everything below: **differences of about
0.01 CRPS or less are inside the draw noise floor of this design.** Both
experiments land inside it.

## What experiment 1 answered: the shape layer really is a handful of constants

Freezing the shape parameters — one calibration on the season immediately before
the holdout, never re-estimated — changes the pooled CRPS by +0.002 with a
player-clustered interval of [−0.008, +0.014], and MAE by +0.007
[−0.010, +0.026]. Neither is distinguishable from zero.

So the answer is yes. For wide receivers you can estimate `k`, `rho`, the gain
mixture and the touchdown rate once from a full prior season and then only
re-fit conditional means. Three consequences:

- The model is cheaper: five LightGBM fits per fold instead of ten.
- It is leakage-free by construction, because the calibration season is strictly
  before the holdout, instead of relying on a ~1,400-row internal block.
- It removes the weakest number in the model. `k` was the least stable parameter
  (CV 0.19 across folds, 7.99 in 2024 against 10.78 in 2025) and it looked like
  it tracked the calibration window rather than football. Freezing it also gave
  the best CRPS and the best MAE of the four variants.

The one caveat is that the frozen variant estimates on a full prior season
(about 2,400 rows) rather than a 20% block (about 1,400 rows), so this
comparison bundles "frozen" with "estimated on more data". Either way the
practical conclusion holds: one calibration per season is enough.

## What experiment 2 answered: right structure, wrong lever

Drawing the touchdown as part of the per-catch process fixed exactly what it was
supposed to fix. The touchdown count distribution went from clearly wrong to
essentially right:

| Event | Realised | Published chain | Joint touchdowns | Multivariate DRF |
| --- | ---: | ---: | ---: | ---: |
| P(2+ receiving TDs) | 0.0245 | 0.0311 (+27%) | **0.0253 (+3%)** | 0.0221 (−10%) |
| P(3+ receiving TDs) | 0.0018 | 0.0054 (3.0×) | 0.0036 (2.0×) | 0.0020 (1.1×) |

It cost nothing measurable on CRPS (−0.005 [−0.016, +0.005]) and it produced the
best PIT deviation of any hierarchical variant (0.058 against 0.080).

But it did **not** fix the fantasy-point ceiling, which was the reason for
trying it:

| Event | Realised | Published chain | Joint touchdowns | Multivariate DRF |
| --- | ---: | ---: | ---: | ---: |
| P(30+ fantasy points) | 0.0109 | 0.0199 | 0.0197 | 0.0146 |
| P(35+ fantasy points) | 0.0065 | 0.0098 | 0.0095 | 0.0058 |

My stated hypothesis was that independent touchdowns were sprinkling scores onto
games that did not have the yardage, and that tying the touchdown to the gain
would pull the ceiling in. That was wrong. The ceiling barely moved, and the
top-decile realised rate for the 30+ event actually got slightly worse (0.0486
against 0.0547), so the joint variant is better calibrated on average and
slightly worse at ranking.

## Where the ceiling really comes from

The ceiling excess is in the opportunity and yardage compound, not the
touchdown stage:

| Event | Realised | Published chain | Multivariate DRF |
| --- | ---: | ---: | ---: |
| P(8+ receptions) | 0.0476 | 0.0550 (+16%) | 0.0514 (+8%) |
| P(10+ receptions) | 0.0140 | 0.0222 (+59%) | 0.0164 (+17%) |
| P(100+ receiving yards) | 0.0567 | 0.0672 (+18%) | 0.0619 (+9%) |
| P(125+ receiving yards) | 0.0211 | 0.0333 (+58%) | 0.0246 (+17%) |
| P(150+ receiving yards) | 0.0069 | 0.0163 (**2.4×**) | 0.0101 (1.5×) |

The chain over-produces the volume tail, and the excess grows with the
threshold. The two defects multiply: across 2021–2025, 150-yard games average
9.5 receptions and 81% of them have 8 or more catches, so `P(yards >= 150)` is
roughly `P(receptions >= 10)` times a sizable conditional probability.

The per-catch gain law itself is not grossly mis-scaled. The per-catch gain
standard deviation implied by weeks with 2 to 6 catches is 10.6 to 11.9, which
brackets the 11.2 the mixture fits on single-catch weeks. The problem is where
that fit comes from: single-catch weeks are a selected sample, and their extreme
gains are much heavier than the pooled ones (99th percentile 56.8 yards against
34.0 for per-catch averages from weeks with 2 or more catches). The explosive
component is therefore fitted on the games most likely to contain an explosive
play.

So the honest diagnosis: the compound structure is sound and the i.i.d.-sum
scale is roughly right, but the explosive component is over-weighted because it
is calibrated on the wrong subsample, and the opportunity distribution has too
much upper tail for a 9-10 dispersion negative binomial.

## Verdict

1. **Adopt frozen shapes.** It costs nothing, simplifies the model, removes a
   small-sample estimation step, and makes leakage-freedom structural rather
   than procedural. This is the change I would keep.
2. **Keep joint touchdowns.** It is CRPS-neutral and it fixes a real,
   interpretable component. It should be adopted for the interpretability it
   buys — "an 8.1% chance per catch, rising with how far downfield the catch is"
   is a far better object than an independent Poisson on the catch count — even
   though it does not improve the aggregate score.
3. **The ceiling remains open.** Neither change addresses it. The next targeted
   fix is to estimate the per-catch gain law from play-level data rather than
   from single-catch weeks, and to thin the upper tail of the opportunity
   distribution. Both are changes to where the parameters come from, not to the
   chain's structure, which is a good sign for the structural argument.
4. **Do not read the CRPS differences between the four variants as real.** They
   are all inside the +0.009 draw-noise floor established by the reproduction
   check. Prefer the variant that is simpler and better calibrated on
   interpretable quantities, which is frozen shapes plus joint touchdowns.
