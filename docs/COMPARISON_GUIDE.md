# Reading the model comparisons

The result documents retain historical research tables to guide work on the hierarchical models. Benchmark implementations, evaluation runners, source datasets, and raw prediction artifacts are not included. Use [the prepared-panel example](../examples/predict_wr.py) to fit the package models; the reports are not executable benchmark instructions.

## Candidates

| Candidate | Forecast target | Purpose in the reports |
| --- | --- | --- |
| Direct DRF | Fantasy-point distribution | Reference for scoring-specific accuracy |
| Independent stat DRFs | Separately sampled statistic distributions | Diagnostic for lost dependence |
| Multivariate DRF | Joint stat distribution | Dependence-aware benchmark |
| Hierarchical parametric | Conditional football production stages | Model family implemented in this package |

Only the hierarchical model family is implemented here. Mention of a benchmark in a table does not mean it is available as a package class.

## Reports

- [CENTRAL_RESULTS.md](CENTRAL_RESULTS.md): WR comparison on 2024?2025 week-ahead folds, including stat forecasts, fantasy points, and ceiling/floor events.
- [HIERARCHICAL_VARIANTS.md](HIERARCHICAL_VARIANTS.md): separate WR follow-up runs examining frozen shapes and joint touchdown sampling.
- [CROSS_POSITION_RESULTS.md](CROSS_POSITION_RESULTS.md): extensions across positions, including their different calibration weaknesses.

Do not merge numerical results from separate runs as though they were one identical experiment. Preserve each report's target, training protocol, model configuration, and sample restrictions.

## Units and interpretation

MAE is the average absolute difference between a predictive mean and the observed value. For fantasy points, MAE is in fantasy points. CRPS also has the outcome's units but evaluates the whole distribution rather than only its mean. Lower values are better for both.

Coverage measures how often observations fall in nominal predictive intervals; width describes their spread. Brier scores and reliability assess event probabilities. Joint energy and variogram scores assess multivariate forecasts. Tiny score differences require uncertainty intervals and attention to Monte Carlo noise.

## Limits on conclusions

Population averages can hide poor replacement-player or injury-scenario behavior. The hierarchy's small average accuracy gap does not establish that its injury response works. The [team opportunity plan](TEAM_OPPORTUNITY_PLAN.md) defines the new evaluations needed.

Reference data from 2024?2025 have already informed model choices. New tuning needs an untouched chronological evaluation period. No result here establishes team-wide coherence: player-level joint statistics and cross-player joint simulation are different requirements.
