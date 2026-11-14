# Conditional receiving allocation development

## Question and frozen design

Does a player's historical target usage predict allocation beyond broad position
rates, on the same supplied roster and team target pool? The
[predeclared protocol](../examples/target_allocation_development.json) compares a
position-only reference with the position-pooled player-history baseline. Both use
the same training position/OTHER priors and exactly the same eligible candidates.
The exposure prior stays fixed at 100, with no parameter search.

Training 2022-2023 scores 2024; expanded training through 2024 scores 2025.
The primary loss is mean team-game multinomial negative log likelihood, including
the full count coefficient and OTHER. Player-history minus position-only loss is
paired by whole NFL game, with both offenses resampled together: 2,000 draws,
seed 0 through recorded child seeds, 95% intervals. Count metrics are secondary,
using 2,000 draws/model/fold and a common fold sampling seed.

These are conditional diagnostics. Team target totals are realized, and rosters
are the supplied weekly ACT scenario. Roster publication before kickoff has not
been verified. The evaluated seasons were previously inspected during development.
This study evaluates historical-usage prediction given those inputs; it does not
evaluate team volume, operational availability or fantasy-point forecasts.

## Data and conservation

The original canonical targets and cached play-by-play/weekly rosters produce
2,174 team-games and 25,637 candidate-game rows over 2022-2025. Candidates account
for 68,259 of 68,408 targets; OTHER retains the remaining 149 in 104 team-games.
No target or zero-target eligible candidate is dropped. Candidate indices are
ordered by player ID, independently of targets.

The 2024 fold contains 544 team-games, 272 games and 6,410 candidate rows; 2025
contains 544, 272 and 6,401 respectively. Every generated team/draw conserves
its realized total exactly. Sample score checks independently verify that
closure. Keys, observed counts and stored candidate summaries reconcile.

## Primary allocation likelihood

| Development year | Position-only mean NLL | Player-history mean NLL | Paired difference [95% interval] |
| --- | ---: | ---: | --- |
| 2024 | 27.775592 | 21.910064 | -5.865529 [-6.294464, -5.447513] |
| 2025 | 27.279297 | 21.025586 | -6.253711 [-6.698107, -5.819226] |

Both intervals are below zero. Target-weighted log loss also falls from 2.414427
to 2.226851 in 2024 and from 2.430916 to 2.226086 in 2025. There are no
zero-probability observed events in either model/fold.

Historical individual usage therefore adds conditional predictive value beyond
the matched WR/RB/TE rates in these development folds. It can encode coaching,
participation and role persistence as well as player characteristics. This
increment does not identify causal target-earning ability or an injury response.
These count-distribution losses have different units from the completion and
weekly fantasy-point comparisons elsewhere in the repository.

## Marginal count accuracy and calibration

| Year | Model | Candidate MAE | Candidate CRPS | 50% coverage | 90% coverage | Mean 90% width |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | Position only | 2.283755 | 1.667713 | 36.58% | 75.73% | 4.876412 |
| 2024 | Player history | 1.745775 | 1.228195 | 54.13% | 85.30% | 4.550530 |
| 2025 | Position only | 2.210267 | 1.605929 | 38.21% | 76.85% | 4.808905 |
| 2025 | Player history | 1.637817 | 1.138447 | 56.76% | 87.39% | 4.467693 |

Observed zeros remain in every denominator. Player-history point accuracy and
CRPS improve, but nominal 90% intervals under-cover in both years. Central 50%
coverage is above nominal in the history model. This pattern motivates testing
extra share variation and more responsive role means; it does not isolate which
source causes the tail errors. Fixed-share multinomial variation includes no
share-parameter or availability uncertainty.

Cold-start diagnostics are particularly limited. In 2024, 123 distinct players
absent from training contribute 1,127 candidate rows; in 2025, 117 contribute
1,162. Under player history:

| Year | Training-seen 90% coverage | Training-unseen 90% coverage | Unseen candidate MAE / CRPS |
| --- | ---: | ---: | --- |
| 2024 | 87.11% | 76.84% | 2.187177 / 1.540513 |
| 2025 | 88.72% | 81.41% | 1.954531 / 1.372722 |

Unseen players receive positive position-prior mass, but this is insufficient for
well-calibrated cold-start predictions. Shared roster normalization means a cold
player's predictions can change when known teammates' history is included; that
does not infer a new player's individual skill. OTHER is scored separately:
history-model CRPS is 0.037582/0.093679 and 90% coverage 98.71%/95.59% in
2024/2025. Its many observed zeros and discrete intervals limit a nominal-level
reading of those high coverage rates.

## Reproducibility and next direction

The ignored bundle `data/experiments/target_allocation_development_20261104`
completed at code `a3810377becaa7f6f0b94ef1828289583ff7a033`. It retains exact
protocol and runner bytes, constructed panel/candidate tables, keyed candidate
and team predictions/scores, report, and a complete-last manifest. All nine
input files plus the protocol, package sources and output hashes were verified;
prediction keys are unique. Both numerical thread settings were 1 before Python
startup. The manifest also records runtime versions.

Manifest SHA-256:
`76dfcd214bb26ee15a4aa827e94e6f7e10a4a3f616671fd16c54f635f0f784a8`.
Report SHA-256:
`d5848f2a31a82d9d74921709ec739206a842b7995d8b1942d004f921a56a2a32`.
Original canonical target SHA-256:
`ebe6f593139df4cadcbb6d0b07e7b797f9603ad350f34c1d0bb2f70ffa54c98c`.
All artifacts remain local and ignored; original data and earlier experiments
were not overwritten.

The next useful comparison holds mean shares fixed while testing a coherent
overdispersed allocation law, selected only on an earlier development split.
The [team opportunity plan](TEAM_OPPORTUNITY_PLAN.md) already proposes this
Dirichlet-multinomial reference. Better tail coverage must be measured with proper
likelihood and width, and cannot be purchased by silently broadening every
interval. Temporal role adaptation and pregame roster provenance remain separate
requirements before an operational forecasting claim.
