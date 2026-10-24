# Team opportunity and injury scenarios for the hierarchical model

Status: proposed research plan. No implementation or production promotion is implied.

## 1. Objective and hypothesis

Build a team-level opportunity model that supplies coherent player opportunities to the existing hierarchical parametric chains. Given an injury or availability scenario, it should forecast how team volume and player roles change, then propagate that uncertainty into receptions, yards, touchdowns, and fantasy points.

Example: WR1 is unavailable; simulate team passing volume, redistribute routes and targets across eligible teammates, simulate their downstream production, and score the resulting stat lines under standard, half PPR, or full PPR.

The hypothesis is that explicitly modeling team allocation improves forecasts around absences and role changes while enabling useful scenario analysis. Improved average fantasy-point accuracy is a question to test, not an assumed benefit.

Success has three dimensions:

- Competitive player-level distributions, including injury-affected teammates.
- Coherent team totals and cross-player dependence.
- Clear explanations of how availability, team volume, role, and efficiency contribute to a forecast.

## 2. Current foundation and limits

- [hierarchical.py](../src/hierarchical_football/hierarchical.py) models WR targets, receptions conditional on targets, compound yardage, touchdowns, and rare rushing. It currently draws player opportunities internally.
- [chains_rb.py](../src/hierarchical_football/chains_rb.py) couples an RB's rushing and receiving through a latent volume factor. This is within-player coupling, not team allocation.
- [chains_qb.py](../src/hierarchical_football/chains_qb.py) and [chains_te.py](../src/hierarchical_football/chains_te.py) provide additional downstream pieces. They are not yet one reconciled team simulator.
- [scoring.py](../src/hierarchical_football/scoring.py) separates stat vectors from scoring, including standard, half PPR, and full PPR for WR/RB/TE. A learned scoring-conversion model is unnecessary.
- [CENTRAL_RESULTS.md](CENTRAL_RESULTS.md) reports original WR MAE of 4.364 for direct DRF, 4.384 for the hierarchy, and 4.412 for multivariate DRF. These are population averages, not evidence about injury scenarios. [HIERARCHICAL_VARIANTS.md](HIERARCHICAL_VARIANTS.md) reports separate follow-up runs; freeze the exact version and artifact before making comparisons.
- [CROSS_POSITION_RESULTS.md](CROSS_POSITION_RESULTS.md) identifies TE touchdown dependence, QB volume bias, and D/ST tier calibration as unresolved. Team allocation alone will not fix them. Current comparison forecasts do not model cross-player or within-game dependence.
- [The injury modeling notes](INJURY_RESEARCH_SNAPSHOT.md) distinguish season-level risk from weekly availability, workload limits, early exits, and recovery duration. No injury prediction model is currently implemented in this package.
- Timestamp-aware injury-report selection still needs to be implemented. Audit historical report coverage and the meaning of publication/update timestamps before modeling weekly availability.
- A future dynamic role state could capture persistent changes in player opportunity. It would need to be coupled to team pools rather than fitted as an unconstrained player count model.

## 3. Scope and sequence

Start with weekly offensive passing opportunity across **all WRs, TEs, and RBs on a team**, conditional on a supplied availability scenario. Evaluate WR fantasy output first, but do not allocate targets only among WRs.

Initially keep the current downstream efficiency assumptions and expose their limitations. Add rushing allocation, QB effects, efficiency responses, and shared outcomes in separate experiments. Keep kickers, D/ST, season-long injury duration, and full drive/play simulation outside the first version.

Separate three products:

| Product | Availability input | Meaning |
| --- | --- | --- |
| Conditional scenario | User or availability model specifies a player out, active, or limited | Forecast given that assumption |
| Operational weekly forecast | Calibrated probabilities based on information available at the cutoff | Mixture over plausible scenarios |
| Diagnostic oracle | Realized availability supplied after the game | Measures allocation quality conditional on knowledge; cannot demonstrate deployable forecast accuracy |

Scenario differences are model-based conditional estimates. Observational injury data do not automatically identify causal effects.

## 4. Proposed generative structure

For every simulated team-game:

1. **Availability:** sample a joint roster scenario, or accept a fixed scenario.
2. **Team environment:** sample offensive volume and pass/run mix conditional on that scenario and pregame context.
3. **Participation and role:** sample eligible players' deployment and opportunity shares.
4. **Allocation:** allocate integer opportunities from finite team pools.
5. **Efficiency and outcomes:** run player conditional production stages using the allocated opportunities.
6. **Reconciliation:** ensure any shared QB/receiver outcomes agree before reporting a coherent team simulation.
7. **Scoring:** apply deterministic scoring rules to completed stat draws.

Preserve the same scenario and draw identity for every teammate. Never independently resample players after allocation when constructing a lineup or team outcome.

### Team volume

- Establish distributions for offensive plays, dropbacks, pass attempts, sacks, scrambles, designed runs, and kneels with explicit definitions. Include overtime and special-case plays consistently.
- For the first passing experiment, a direct model of targetable team opportunities is acceptable. Document its relationship to recorded attempts and account for throwaways, spikes, and other untargeted attempts; do not equate attempts with player targets.
- Compare a simple overdispersed count baseline with a richer plays-plus-pass-rate model. Avoid choosing a family solely for convenience; inspect zero mass and dispersion.
- Absences can change total volume and play selection. Removing a receiver need not leave a fixed pool of vacated targets, and losing a QB or offensive lineman can affect every skill player.
- Market context may already reflect injury news. Use only timestamp-valid prices and avoid applying a second manual injury penalty to an already adjusted volume forecast.
- A shared game environment can create positive teammate volume dependence even when allocation creates competition conditional on a fixed total.

### Availability and deployment

- Represent inactive/out, active with ordinary usage, active with limited usage, and uncertain participation separately. Active status does not guarantee a snap or touch.
- A known-out player receives exactly zero opportunity. A questionable player enters through scenario probabilities, rather than multiplying every output by a single availability probability.
- Include rookies, backups, practice-squad elevations, trades, IR/PUP, suspensions, healthy scratches, and an explicit residual player bucket.
- Unknown status must remain unknown; absence from an injury report is not proof of full health.
- Multiple absences and shared events can require joint scenarios. Independent injury flags are a baseline assumption to evaluate, not a guaranteed description.
- In-game injury is a separate mechanism. A full-week absence and an early exit expose different redistribution windows. Initially treat early exits as unmodeled outcome uncertainty and label the limitation; add exposure-time states later if reliable data support them.
- Weekly uncertainty and season-long injury risk require different labels and calibration. Recovery duration needs a defensible longitudinal source, not consecutive Out-report counts.

### Shares and integer allocation

Start with roster-masked, role-informed logits followed by a multinomial allocation; compare a Dirichlet-multinomial version for extra share uncertainty. Consider logistic-normal shares only if richer substitution patterns earn the added complexity.

For a target pool T and eligible share vector p:

```text
p_i >= 0; sum_i p_i = 1
(targets_1, ..., targets_n, targets_other) ~ Multinomial(T, p)
sum_i targets_i + targets_other = T
```

- Use separate logits and pools for targets and carries. A carry replacement and a receiving replacement need not be the same teammate or position.
- Shares should depend on lagged role, recent snaps/routes, alignment or role similarity where available, teammate availability, coaching, and depth-chart information known at the cutoff.
- Missing players redistribute opportunity through the learned allocation. Simple proportional renormalization is an essential baseline, not the presumed answer.
- Shrink new or rarely used players toward role/position/team priors. Give eligible replacements a defensible nonzero share; a structural zero cannot grow after an injury.
- Fit roster composition symmetrically so arbitrary player ordering cannot change predictions.
- Do not force individual snap or route shares to sum to one: several eligible receivers participate on the same play. Route counts need eligible-slot/personnel accounting if modeled explicitly. Target shares do sum to one within the defined target pool.
- Red-zone, goal-line, and third-down usage can differ from ordinary usage. If modeled as subpools, partition opportunities without counting them twice.
- Allocation bounds and route exposure should be compatible. If a simplified aggregate model cannot guarantee targets do not exceed routes, document that limitation rather than claiming play-level coherence.
- A basic Dirichlet-multinomial imposes limited conditional dependence. Audit teammate covariance and substitution patterns instead of assuming conservation alone makes the joint forecast correct.

## 5. Integration with the hierarchical chains

Refactor sampling conceptually into an opportunity stage and a conditional production stage. The future interface should accept externally drawn targets/carries with shape `(draw, player)` and team-game metadata, while retaining the original internal sampler as a frozen baseline.

Important implementation decisions:

- **Replace opportunity draws; do not add an injury adjustment on top.** Existing lagged role features and injury-aware inputs may already contain some response to an absence.
- Replace or re-estimate RB latent volume coupling when the team model supplies shared volume and allocation uncertainty. Retaining both unchanged could double-count variance and dependence.
- Preserve dependence between workload and efficiency where supported. Determine whether existing efficiency models condition on sampled opportunity, expected opportunity, or only pregame features; injecting larger counts alone does not learn a new role's efficiency.
- Initially freeze efficiency to isolate allocation value. Then test QB changes, defensive attention, route mix, offensive-line changes, and expanded-role efficiency as separate additions with shrinkage and uncertainty.
- Do not assume every replacement maintains the starter's efficiency, or that every incumbent maintains efficiency at higher usage.
- Keep shape calibration strictly within the training window, including any frozen calibration season.
- Preserve named stat columns and scoring checks from the existing harness.

### Shared QB and receiver outcomes

An opportunity-only prototype can improve player marginals without being a fully coherent team stat simulator. Label it accordingly.

Before claiming lineup-level joint outcomes, define one owner for completions, passing yards, and passing TDs. Sampling QB and receiver outcomes independently can make team receptions exceed completions or produce contradictory passing TD totals.

A later shared pass-outcome layer should derive QB passing totals and receiving totals from the same outcomes, with explicit handling of sacks, interceptions, laterals, multiple passers, and statistical attribution exceptions. Tie rushing totals to the same carry allocation. If team TD opportunities are modeled, allocate them consistently with touches and opportunity locations rather than independently rescaling TD counts afterward.

## 6. Data and labels to audit first

Build a roster-complete player-team-game panel, including eligible players with zero opportunity. Keep DNP, inactive, no offensive snaps, no touches, missing data, and team bye distinct. Verify team identity at the time of each game and handle trades and multiple-team weeks correctly.

Candidate inputs, subject to coverage and timestamp audit:

- Weekly rosters, transactions, active lists, depth charts, and injury/practice reports.
- Player and team opportunities, snaps, routes, personnel, red-zone usage, and play-by-play reconciliation.
- Pregame opponent, venue, weather, team/coaching context, rest, and market information.
- Lagged opportunity shares and role histories, with explicit treatment of first starts and new teams.

Define forecasting cutoffs before choosing features: for example, an early-week forecast and a separate post-inactive-list forecast. Store event time, source publication/update time, ingestion time, and prediction time where available. A date-modified field must be verified to represent historical knowledge rather than a later revision.

Never use same-game snaps, routes, final depth charts, realized playing time, future injury duration, or post-cutoff market prices as operational predictors. They may be evaluation labels or clearly separated oracle inputs. Fit imputation, scaling, priors, and share calibration on training data only.

Check historical feed completeness, changing report conventions, missing healthy-player rows, corrections, roster ID mappings, and selection bias from requiring a player to record a stat. Do not introduce a new vendor or production ingestion change as part of the first modeling experiment.

## 7. Learning injury responses responsibly

- Train on all eligible team-games, not only memorable injury replacements. Evaluate absence events separately.
- Distinguish injury from benching, trade, suspension, and coaching changes where labels permit; each can shift roles differently.
- Use prior role and roster state, not the injured player's realized previous-game targets, as the sole definition of vacated workload.
- Include team, position/role, coaching, season, and opponent context with partial pooling. A player's missingness is not randomized; teammates may be hurt too, and game plans may change simultaneously.
- Report support for each scenario: analogous training examples, new-role uncertainty, and whether the request lies outside observed experience.
- Separate outcome randomness, uncertainty about shares/parameters, and uncertainty about availability. A mixture over materially different roster states can be multimodal; reducing it to one mean loses useful information.
- Use matched or within-team absence comparisons as diagnostics, not automatic causal proof. Check pre-absence trends and time since role change.
- For multiweek forecasts, model role persistence and recovery jointly over time rather than independently redrawing each future week. Defer this until the weekly model is established.

## 8. Evaluation design

Use expanding, week-ahead folds with a common roster universe and common information cutoff. Keep every team-game together when splitting and sampling uncertainty. Avoid training on teammates from a test game.

The 2024–2025 comparison holdouts have already been examined. Use them for reproduction/development only; predeclare an untouched later chronological evaluation period. Do not call all of 2026 unseen merely because the season is later—exclude any weeks already used to select features or models.

### Baselines and ablations

1. Frozen direct DRF and multivariate DRF with the same available injury context and features where feasible.
2. Current independent-player hierarchical model.
3. Fixed team-volume model with proportional redistribution among available players.
4. Learned team allocation with unchanged efficiency.
5. Learned allocation plus injury-dependent team volume.
6. Availability mixtures instead of supplied known-state scenarios.
7. Efficiency responses and shared outcome simulation, each tested separately.

Direct DRFs need separately trained targets for each scoring system; the stat models reuse the same draws. Evaluate every model on common rows and target definitions, and separately report any expanded-roster population the old harness excluded.

### Metrics

| Layer | Required checks |
| --- | --- |
| Availability | Brier/log score, reliability, limited-usage calibration, by report status and cutoff |
| Team volume | MAE, CRPS, interval coverage, count dispersion, pass/run mix |
| Allocation | Player opportunity MAE/CRPS, zero probability, interval coverage, share/count predictive scores |
| Fantasy production | Mean-prediction MAE in fantasy points; CRPS; floor/ceiling Brier scores and reliability; interval width/coverage |
| Joint behavior | Exact pool conservation, teammate/opponent dependence where modeled, aggregate team totals, energy/variogram scores, QB/receiver reconciliation |
| Scenario usefulness | Teammate opportunity shifts, performance on first absences/returns, uncertainty under limited historical support |

Report full-population and predeclared segments: stable rosters, known absences, uncertain/limited players, multiple absences, QB changes, replacement roles, returning starters, and projected starters versus low-volume players. Define affected teammates using pregame role information, not whether their realized output improved.

Use paired score differences with player-clustered and team-game or week-block uncertainty; report sensitivity to persistent team effects and repeated absence episodes. State sample and independent-event counts. Rare scenarios and ceiling events need uncertainty intervals, not just percentages. Predeclare primary metrics to limit selection across many segments.

Measure Monte Carlo error on a small fixed comparison before interpreting tiny gaps. Preserve seeds, draw identities, source snapshots, training windows, model versions, and runtime budgets. Use equal-information comparisons and document capacity differences.

## 9. Milestones and decision gates

| Stage | Deliverable | Decision gate |
| --- | --- | --- |
| 0. Audit | Target definitions, timestamp/coverage report, roster-complete sample, frozen baselines | Reliable historical knowledge and reconciled team pools |
| 1. Passing allocation | Known-state scenarios, target pool, proportional baseline, learned shares, WR downstream adapter | Exact target conservation and credible uncertainty; improvement over simple redistribution on affected teammates |
| 2. Weekly availability | Calibrated weekly states and operational mixture forecasts | Gains survive removal of oracle availability; missing reports handled explicitly |
| 3. Rushing and roles | Carry allocation, RB dual roles, QB participation, role persistence | No duplicate volume factors; realistic rushing/receiving dependence |
| 4. Efficiency response | Shrunk role/QB/coverage adjustments | Incremental untouched-data benefit beyond opportunity changes |
| 5. Joint outcomes | Shared pass outcomes and reconciled team production | Valid QB/receiver identities and useful cross-player distributions |
| 6. Promotion review | Frozen evaluation, runtime/data-quality report, scenario explanations | Agreed practical accuracy tolerance, calibration, and operational reliability |

Before tuning, set an acceptable overall MAE/CRPS degradation budget and a minimum useful gain for injury-affected players, based on intended use. Do not choose a tolerance after seeing results. A scenario tool can be useful without beating the primary forecast overall, but its role and limitations must be explicit.

If learned allocation cannot beat proportional redistribution, retain the simple allocator. If richer efficiency or availability stages hurt calibration, keep them out. Preserve the direct and multivariate DRFs as references throughout.

## 10. Outputs and operational considerations

Save per-draw scenario IDs, team totals, shares, player opportunities, downstream stat lines, and scoring outputs, plus source/cutoff/model metadata. Retain a residual bucket so incomplete rosters cannot silently donate all workload to modeled players.

For each scenario, explain the change in team volume, player share, efficiency assumptions, and resulting fantasy-point range. Use paired random draws for scenario sensitivity comparisons to reduce simulation noise; those paired differences remain assumption-dependent, not observed causal effects.

Keep refresh logic distinct from model fitting. Handle late scratches, stale feeds, postponed games, duplicate updates, missing depth charts, and unsupported teams with explicit fallback behavior. Cache inputs and batch by team-game. Monitor probability calibration, role drift, conservation failures, source freshness, inference cost, and simulated tail inflation.

No automatic lineup, draft-ranking, VORP, or production integration belongs to this research plan. Promotion is a separate decision after validation.

## 11. Questions to settle before implementation

- Is the first product conditional what-if analysis, a weekly forecast, or both with clearly separate outputs?
- Which prediction cutoff has reliable historical report and roster snapshots?
- Are routes/personnel available historically as-of, or should the first allocator use only snaps and target history?
- What exactly belongs in each team opportunity pool, and how are no-target attempts and low-usage players represented?
- Which weekly availability and workload labels can be supported without reconstructing injury episodes from incomplete reports?
- How much adaptation should happen through explicit role state versus lagged features already in the hierarchy?
- Can existing efficiency stages accept external draws without silently retaining independent opportunity variance?
- What is the simplest defensible treatment of QB changes, early exits, and multiple absences?
- What accuracy/calibration tradeoff is acceptable for the intended scenario-analysis use?
- Which untouched dates and minimum event counts will decide whether the added complexity is worthwhile?

Recommended first experiment: supplied pregame availability scenarios → team target pool → simple versus learned target allocation across the full receiving roster → unchanged WR conditional production. Establish allocation value before adding injury-duration prediction, elaborate efficiency responses, or full game simulation.
