# Injury and opportunity research boundaries

## Data and labels

- Treat weekly reports as reports, not complete injury episodes with known start and end dates.
- Build the eligible population from roster-complete panels; report-only data omit many healthy players.
- Distinguish missing observations, nonparticipation, zero touches, and bye weeks.
- Match every input to the forecast cutoff and retain the timestamp used to establish availability.
- Define labels before model selection. Do not relabel a scenario after seeing its fantasy outcome.
- Do not equate consecutive Out reports with true recovery duration.

## Experiments

Keep availability, opportunity allocation, and downstream efficiency changes as separate stages so their value can be measured. Use chronological folds, common information cutoffs, fixed reference configurations, and paired uncertainty estimates.

Conditional what-if forecasts and operational weekly forecasts must have separate evaluation labels. Realized active status can support an oracle diagnostic but cannot be used as an early-week predictor.

Do not silently fill unknown status with healthy or multiply all conditional outputs by a season-level risk estimate. Represent uncertain participation through explicitly defined weekly scenarios.

## Model behavior

Unavailable players receive zero opportunity in a fixed absence scenario. Available teammates compete within defined team pools. Include residual participants so incomplete roster coverage cannot inflate modeled players' workloads.

Do not claim causal injury effects or coherent team outcomes solely because a model exposes interpretable stages. Validate opportunity conservation, replacement behavior, efficiency assumptions, and shared QB/receiver statistics separately.

## Promotion

Scenario analysis is research until calibration, data freshness, fallback behavior, and untouched-data performance are demonstrated. Integration into rankings or lineup decisions requires a separate review against the intended use and practical accuracy tolerances.

See [TEAM_OPPORTUNITY_PLAN.md](TEAM_OPPORTUNITY_PLAN.md) for milestones and [DATA_CONTRACT.md](DATA_CONTRACT.md) for model inputs.
