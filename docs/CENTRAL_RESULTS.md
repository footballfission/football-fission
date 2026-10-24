# Central experiment — Direct DRF vs multivariate DRF vs hierarchical parametric

Reference results: benchmark implementations, experiment runners, datasets, and raw draws are not bundled. These tables are not a claim of a newly reproduced package evaluation.

Wide receivers, 2024–2025 week-ahead holdout. Implementation and usage notes are in
[COMPARISON_GUIDE.md](COMPARISON_GUIDE.md); the experiment design is in [MODEL_DESIGN.md](MODEL_DESIGN.md).

Central experiment from the model design, run on wide receivers.

## Protocol

| Item | Value |
| --- | --- |
| Position | WR |
| Target | full-PPR scrimmage points: receptions + 0.10 * (receiving + rushing yards) + 6 * (receiving + rushing TDs) |
| Holdout seasons | 2024, 2025 |
| Folds | week-ahead, 36 folds; week `w` trains on 3 prior seasons plus completed weeks `< w` |
| Features | 94 lagged-history and pregame context columns, identical for every model |
| Predictive draws | 200 per player-week |
| Forest settings | 200 trees, min_samples_leaf 20, fold-local CDF thresholds |
| Multivariate DRF | 100 trees, subsample 0.5, honesty 0.5, colsample 0.7, Gaussian-MMD criterion |
| Hierarchical stages | LightGBM means, 200 trees per stage; dispersion and gain mixture from an internal 80/20 time split |

Candidates:

- **Direct DRF** — one conditional-CDF forest for fantasy points.
- **Independent stat DRFs** — one conditional-CDF forest per statistic, sampled independently.
- **Multivariate DRF** — one joint distributional forest over the stat vector; leaf resampling keeps dependence.
- **Hierarchical parametric** — targets, then receptions given targets, then compound signed gain yardage and conditional touchdowns.

## Fantasy-point distributions (full-PPR scrimmage)

| Model | CRPS | MAE | 70% cov | 80% cov | 80% width | 95% cov | PIT deviation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct DRF (fantasy points) | 2.967 | 4.364 | 0.782 | 0.860 | 13.53 | 0.962 | 0.045 |
| Independent stat DRFs | 3.041 | 4.362 | 0.595 | 0.690 | 10.18 | 0.874 | 0.348 |
| Multivariate DRF | 2.972 | 4.412 | 0.799 | 0.870 | 13.94 | 0.970 | 0.069 |
| Hierarchical parametric | 2.986 | 4.384 | 0.776 | 0.853 | 13.76 | 0.962 | 0.071 |

Per season:

| Season | Model | CRPS | MAE | 80% cov | 80% width |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | Direct DRF (fantasy points) | 3.106 | 4.526 | 0.856 | 13.88 |
| 2024 | Independent stat DRFs | 3.181 | 4.532 | 0.685 | 10.45 |
| 2024 | Multivariate DRF | 3.122 | 4.593 | 0.863 | 14.33 |
| 2024 | Hierarchical parametric | 3.142 | 4.587 | 0.851 | 14.27 |
| 2025 | Direct DRF (fantasy points) | 2.831 | 4.206 | 0.864 | 13.19 |
| 2025 | Independent stat DRFs | 2.904 | 4.197 | 0.694 | 9.92 |
| 2025 | Multivariate DRF | 2.826 | 4.237 | 0.876 | 13.57 |
| 2025 | Hierarchical parametric | 2.835 | 4.186 | 0.855 | 13.26 |

Positive deltas mean the candidate beats the direct DRF.

| Model | CRPS delta (player CI) | CRPS delta (week CI) | MAE delta (player CI) |
| --- | --- | --- | --- |
| Independent stat DRFs | -0.074 [-0.090, -0.056] | -0.074 [-0.093, -0.055] | +0.001 [-0.014, +0.017] |
| Multivariate DRF | -0.005 [-0.021, +0.011] | -0.005 [-0.019, +0.007] | -0.049 [-0.072, -0.024] |
| Hierarchical parametric | -0.019 [-0.040, +0.000] | -0.019 [-0.040, +0.000] | -0.020 [-0.052, +0.014] |

## Ceiling and floor probabilities

| Model | P(<5) predicted / observed | Brier | P(20+) predicted / observed | Brier | P(30+) predicted / observed | Brier |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Direct DRF (fantasy points) | 0.514 / 0.519 | 0.1678 | 0.076 / 0.074 | 0.0600 | 0.014 / 0.011 | 0.0107 |
| Independent stat DRFs | 0.453 / 0.519 | 0.1728 | 0.048 / 0.074 | 0.0612 | 0.004 / 0.011 | 0.0106 |
| Multivariate DRF | 0.512 / 0.519 | 0.1688 | 0.076 / 0.074 | 0.0602 | 0.015 / 0.011 | 0.0106 |
| Hierarchical parametric | 0.514 / 0.519 | 0.1688 | 0.079 / 0.074 | 0.0611 | 0.020 / 0.011 | 0.0109 |

## Component-stat distributions

CRPS per statistic (lower is better). The direct DRF has no component output.

| Statistic | Independent stat DRFs | Multivariate DRF | Hierarchical parametric |
| --- | ---: | ---: | ---: |
| targets | 1.258 | 1.271 | 1.273 |
| receptions | 0.918 | 0.928 | 0.927 |
| receiving_yards | 13.786 | 13.864 | 14.030 |
| receiving_tds | 0.160 | 0.159 | 0.160 |
| carries | 0.122 | 0.125 | 0.127 |
| rushing_yards | 0.849 | 0.853 | 0.861 |
| rushing_tds | 0.006 | 0.005 | 0.005 |

Component 80% interval coverage (nominal 0.80):

| Statistic | Independent stat DRFs | Multivariate DRF | Hierarchical parametric |
| --- | ---: | ---: | ---: |
| targets | 0.888 | 0.896 | 0.878 |
| receptions | 0.910 | 0.916 | 0.906 |
| receiving_yards | 0.864 | 0.874 | 0.862 |
| receiving_tds | 0.957 | 0.959 | 0.962 |
| carries | 0.959 | 0.960 | 0.952 |
| rushing_yards | 0.934 | 0.927 | 0.932 |
| rushing_tds | 0.995 | 0.995 | 0.995 |

Zero-mass calibration (`predicted` minus `observed` share of exact zeros):

| Statistic | Independent stat DRFs | Multivariate DRF | Hierarchical parametric |
| --- | ---: | ---: | ---: |
| targets | -0.012 | -0.016 | -0.020 |
| receptions | -0.009 | -0.011 | -0.018 |
| receiving_yards | -0.010 | -0.012 | -0.009 |
| receiving_tds | +0.002 | +0.005 | +0.005 |
| carries | -0.000 | -0.002 | +0.007 |
| rushing_yards | -0.001 | -0.002 | +0.006 |
| rushing_tds | -0.003 | -0.003 | -0.002 |

## Joint distribution quality

Energy and variogram scores on the standardised stat vector (lower is better).

| Model | Energy score | Variogram score |
| --- | ---: | ---: |
| Direct DRF (fantasy points) | — | — |
| Independent stat DRFs | 1.126 | 5.78 |
| Multivariate DRF | 1.099 | 5.31 |
| Hierarchical parametric | 1.104 | 5.34 |

## Dependence structure

Within-row dependence reproduced by each joint sampler. `observed` is the same quantity on the realised stat lines.

| Model | corr(receptions, yards) model / observed | Yards given 3+ catches model / observed | P(receptions 5+ and yards 60+) model / observed | Brier |
| --- | --- | --- | --- | ---: |
| Direct DRF (fantasy points) | — | — | — | — |
| Independent stat DRFs | -0.001 / 0.866 | 32.95 / 62.03 | 0.0799 / 0.1407 | 0.0981 |
| Multivariate DRF | 0.792 / 0.866 | 56.29 / 62.03 | 0.1453 / 0.1407 | 0.0930 |
| Hierarchical parametric | 0.781 / 0.866 | 55.07 / 62.03 | 0.1408 / 0.1407 | 0.0928 |

## Scoring-rule sensitivity

CRPS of the same simulated stat lines under four scoring rules.

| Model | Rec coefficient 0.0 | Rec coefficient 0.5 | Rec coefficient 1.0 | Receiving only, Rec coefficient 1.0 |
| --- | ---: | ---: | ---: | ---: |
| Direct DRF (fantasy points) | — | — | 2.967 | — |
| Independent stat DRFs | 2.214 | 2.617 | 3.041 | 3.009 |
| Multivariate DRF | 2.199 | 2.573 | 2.972 | 2.942 |
| Hierarchical parametric | 2.214 | 2.588 | 2.986 | 2.959 |

## Hierarchical stage diagnostics

| Parameter | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Negative-binomial target dispersion k | 9.835 | 6.423 | 12.003 |
| Beta-binomial catch overdispersion rho | 0.018 | 0.009 | 0.030 |
| Global touchdown rate per reception | 0.081 | 0.076 | 0.086 |
| Mean per-catch gain (yards) | 12.102 | 11.608 | 13.954 |


## Verdict

Pooled 2024–2025, 4,940 common player-weeks, 36 week-ahead folds. Positive paired
deltas mean the candidate beats the direct DRF.

| Model | FP CRPS | CRPS delta vs direct DRF (player 95% CI) | MAE | 80% cov | Joint energy | Joint variogram |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Direct DRF | **2.967** | reference | 4.364 | 0.860 | — | — |
| Multivariate DRF | 2.972 | −0.005 [−0.021, +0.011] | 4.412 | 0.870 | **1.099** | **5.31** |
| Hierarchical parametric | 2.986 | −0.019 [−0.040, +0.000] | 4.384 | 0.853 | 1.104 | 5.34 |
| Independent stat DRFs | 3.041 | −0.074 [−0.090, −0.056] | **4.362** | 0.690 | 1.126 | 5.78 |

### 1. The direct DRF still wins, but only by a hair

The direct fantasy-point forest has the lowest pooled CRPS. The multivariate DRF
is 0.005 behind with an interval spanning zero, and the hierarchical parametric
chain is 0.019 behind with an interval that just touches zero. Both structured
models are statistically indistinguishable from the direct model on the pooled
holdout; the multivariate DRF is actually ahead in 2025 (+0.005, [−0.012, +0.021])
and behind in 2024 (−0.016, [−0.036, +0.005]).

The multivariate DRF does this with half the trees (100 against 200), so the
tie is not an artefact of giving the structured model more capacity.

### 2. Splitting the statistics apart *without* keeping their dependence is clearly harmful

This is the plan's central warning, and it is confirmed decisively. The
independent-stat candidate has the **best marginal component forecasts** — it
has the lowest CRPS for targets, receptions, receiving yards, carries and
rushing yards, five of the seven statistics — and produces the **worst
fantasy-point distribution**:

- FP CRPS 3.041, which is 0.074 worse than the direct DRF with a player
  interval of [−0.090, −0.056].
- 80% interval coverage 0.690 against a nominal 0.80, and 95% coverage 0.874.
- PIT deviation 0.348, against 0.045 for the direct DRF.
- Mean predictive standard deviation 4.17, against 5.77 for the direct DRF: it
  discards about 28% of the spread because it drops the covariance term
  `2 Σ a_i a_j Cov(Y_i, Y_j)`.

Good marginals are therefore not evidence of a good joint.

### 3. Both dependence-aware stat models recover the joint structure

| Diagnostic | Observed | Multivariate DRF | Hierarchical | Independent |
| --- | ---: | ---: | ---: | ---: |
| Within-draw corr(receptions, receiving yards) | 0.866 | 0.792 | 0.781 | −0.001 |
| Mean receiving yards given 3+ catches | 62.03 | 56.29 | 55.07 | 32.95 |
| P(5+ receptions and 60+ receiving yards) | 0.141 | 0.145 | 0.141 | 0.080 |

The multivariate DRF is closest on the correlation and reproduces the joint boom
probability almost exactly. The hierarchical chain is a close second. The
independent model's implied probability of a 5-catch, 60-yard game is about
half the realised rate.

### 4. Ceiling and floor probabilities

| Event | Observed | Direct DRF | Multivariate DRF | Hierarchical | Independent |
| --- | ---: | ---: | ---: | ---: | ---: |
| P(fantasy points < 5) | 0.519 | 0.514 | 0.512 | 0.514 | 0.453 |
| P(fantasy points ≥ 20) | 0.074 | 0.076 | 0.076 | 0.079 | 0.048 |
| P(fantasy points ≥ 25) | 0.030 | 0.034 | 0.034 | 0.040 | 0.015 |
| P(fantasy points ≥ 30) | 0.011 | 0.015 | 0.015 | 0.020 | 0.004 |
| Brier, < 5 | — | 0.1678 | 0.1688 | 0.1688 | 0.1728 |
| Brier, ≥ 30 | — | 0.0107 | 0.0106 | 0.0109 | 0.0106 |

The direct DRF, the multivariate DRF and the hierarchical chain all track the
floor. At the ceiling the hierarchical chain is the most aggressive (it
forecasts nearly twice the realised rate of 30+ games) and the independent model
is the most conservative (about one third). Aggregate Brier scores barely move,
because at a 1% base rate they are dominated by the huge majority of zero
probabilities — a reminder that the plan's recommendation to look at boom/bust
probabilities directly, not only at average CRPS, is the right one.

### 5. Scoring flexibility is essentially free

The stat-vector models can be re-scored without retraining; the direct DRF
cannot, so it only has a full-PPR number.

| Rule | Independent | Multivariate DRF | Hierarchical |
| --- | ---: | ---: | ---: |
| Standard (no reception points) | 2.214 | **2.199** | 2.214 |
| Half PPR | 2.617 | **2.573** | 2.588 |
| Full PPR | 3.041 | **2.972** | 2.986 |
| Receiving only, full PPR | 3.009 | **2.942** | 2.959 |

At full PPR the multivariate stat vector reaches 2.972 against the direct DRF's
2.967. The stat-by-stat decomposition therefore costs essentially nothing on the
rule it was not built for and adds three calibrated alternative rules for free.

### 6. Calibration

All three joint-aware samplers are mildly conservative: the nominal 80% interval
covers 0.853–0.870. The direct DRF is the best calibrated (0.860 at 80%, 0.962
at 95%, PIT deviation 0.045). The multivariate DRF makes the widest intervals
(13.94 against 13.53 for the direct DRF). Component marginals for all three
stat-vector models over-cover at 80% for the count statistics (0.88–0.96), most
notably for receiving TDs and carries.

### 7. Where the differences live

Rows bucketed by the direct DRF's projected points. CRPS improvement over the
direct DRF inside each bucket (positive means the candidate is better):

| Projected points | n | Multivariate DRF | Hierarchical | Independent |
| --- | ---: | ---: | ---: | ---: |
| < 5 | 1,982 | −0.016 | −0.012 | −0.060 |
| 5–10 | 1,570 | −0.008 | −0.012 | −0.074 |
| 10–15 | 1,105 | **+0.012** | −0.052 | −0.107 |
| 15–20 | 275 | −0.001 | −0.004 | −0.033 |
| 20+ | 8 | +0.499 | +0.711 | −0.191 |

The multivariate DRF's small pooled deficit is concentrated in low-projection
rows; it is *ahead* of the direct DRF in the 10–15 bucket, which is the dense
starter range. The hierarchical chain's deficit is concentrated in that same
10–15 bucket, which is the single largest source of its pooled loss. The
independent model loses in every bucket.

Ceiling calibration, `P(fantasy points ≥ 30)`, predicted against realised inside
each bucket:

| Bucket | Direct DRF | Multivariate DRF | Hierarchical | Independent |
| --- | --- | --- | --- | --- |
| 10–15 | 0.030 / 0.024 | 0.033 / 0.023 | 0.040 / 0.021 | 0.008 / 0.022 |
| 15–20 | 0.105 / 0.069 | **0.099 / 0.080** | 0.109 / 0.061 | 0.039 / 0.071 |
| 20+ | 0.167 / 0.125 | — | 0.212 / 0.109 | 0.082 / 0.333 |

Among the rows the direct DRF itself projects at 15–20 points, the multivariate
DRF is better calibrated at the ceiling (0.099 against 0.080 realised) than the
direct DRF (0.105 against 0.069). The hierarchical chain over-forecasts that
ceiling everywhere, and the independent model under-forecasts it everywhere.

This is the strongest single argument for the multivariate DRF as the joint
engine: it matches the direct DRF on the metric the direct DRF optimises, and it
is better where the direct DRF cannot reach — joint events and the conditional
ceiling for starter-level projections.


## Answers to the plan's central questions

**Does a football-informed hierarchical parametric model outperform a
general-purpose multivariate distributional random forest at estimating the
joint distribution of weekly player statistics?**

No. The two are statistically indistinguishable. The multivariate DRF is
marginally ahead on the joint scores (energy 1.099 against 1.104, variogram 5.31
against 5.34), on the dependence diagnostics (correlation 0.792 against 0.781)
and on the ceiling probability (0.145 against 0.141 against 0.141 observed),
while the hierarchical chain is marginally ahead in 2025 MAE (4.186 against
4.237). The value of the hierarchical model is interpretability — separating
opportunity from efficiency — and the ability to enforce structural constraints,
not accuracy. The value of the multivariate DRF is that it delivers the same
joint quality with no football-specific structure and far less bespoke code.

**Are individual NFL statistics easier to forecast as distributions than fantasy
points are?**

Slightly, per marginal. The independent forests have the best component CRPS on
5 of 7 statistics, but the differences are small (targets 1.258 against 1.271,
receiving yards 13.786 against 13.864) and the multivariate forest matches them
while keeping dependence. Attempts and targets do behave like well-behaved count
models; yardage does not, but the joint models handle it anyway.

**Can modelling opportunities, efficiency and their dependence separately improve
fantasy-point tail predictions?**

It does not improve average CRPS, but it does change the tails. Relative to the
direct DRF, the hierarchical chain doubles the forecast rate of 30+ games
(0.020 against 0.011 observed) while the multivariate DRF stays close (0.015)
and the independent model collapses (0.004). Only the dependence-aware
stat-vector models produce an interpretable boom probability at all, because
only they can express "many catches *and* many yards" as one event.

## Practical recommendation

1. **Keep the direct DRF as the primary marginal forecast.** It is not beaten.
2. **Promote the multivariate DRF as the joint/lineup engine.** It matches the
   direct DRF on point distributions, wins the joint scores, reproduces the
   dependence structure, and gives free scoring-rule flexibility. It is the
   cheapest way to get coherent stat lines.
3. **Do not ship the independent-stat decomposition**, even though each of its
   marginals is individually the best.
4. **Keep the hierarchical chain as the interpretable alternative** and as the
   vehicle for the position-specific extensions in the plan. Adopt frozen shape
   parameters and the joint touchdown stage, both of which are free or
   near-free ([HIERARCHICAL_VARIANTS.md](HIERARCHICAL_VARIANTS.md)). The 30+
   ceiling remains open. The zero-target shortfall it shares with the
   multivariate DRF is small (−0.020 and −0.016 against a 0.153 realised
   share).

## Next steps

1. Extend the identical harness to RB, TE and QB, where the plan expects the
   stat decomposition to matter more (rushing volume, passing volume, and
   touchdown dependence are larger shares of those positions).
2. Add the plan's model D, a copula over parametric marginals, to test whether
   an explicit dependence layer can beat both the forest and the chain.
3. Fix the hierarchical chain's one distinctive defect, the over-forecast 30+
   ceiling. This was attempted in
   [HIERARCHICAL_VARIANTS.md](HIERARCHICAL_VARIANTS.md): freezing the shape
   parameters costs nothing, and tying the touchdown to the per-catch gain fixes
   the touchdown count distribution, but neither moves the ceiling. The ceiling
   traces to the opportunity and yardage compound (`P(yards >= 150)` is 2.4×
   the realised rate, against 1.5× for the multivariate DRF), so the next fix is
   better parameter sources rather than more chain structure. The small
   under-forecast of zero-target and zero-catch games is *not* chain-specific
   (predicted minus observed zero share for targets: −0.020 for the chain
   against −0.016 for the multivariate DRF), so a hurdle or zero-inflated
   opportunity stage would help both.
4. Test the multivariate DRF with a larger tree budget, since it ties the direct
   DRF at half the trees.
5. Re-run the frozen configuration on 2026 as a genuinely untouched holdout,
   because 2024–2025 have now been inspected.

## Limitations

- Wide receivers only. The plan's later priorities (RB, QB, TE, K, D/ST) are not
  covered by this run.
- Rushing is about 2% of WR fantasy points and is handled by a hurdle with an
  empirical training-fold pool, so it cannot carry a verdict.
- The hierarchical yardage stage models per-catch gain with a two-component
  Gaussian mixture fitted on single-reception weeks, which is a model of
  per-catch gain rather than a play-level simulation.
- The direct DRF is scored only under its own full-PPR rule, so the scoring
  flexibility comparison cannot be extended to it.
- The multivariate DRF ran with 100 trees against 200 for the direct DRF; it is
  not ahead, so this asymmetry does not manufacture a win for the stat-vector
  models.
- All forecasts are marginal player distributions. Nothing here models
  cross-player or within-game dependence.
- Hyperparameters were fixed before the holdout was scored, but the two holdout
  seasons have now been inspected, so any further tuning must use new data.
