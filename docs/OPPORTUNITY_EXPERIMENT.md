# Offensive opportunity share and total experiments

Status: research proposal / experimentation plan. No model fitting, pipeline change, or production promotion is implied by this document.

This proposal extends [TEAM_OPPORTUNITY_PLAN.md](TEAM_OPPORTUNITY_PLAN.md); it does not replace it. Availability, allocation, and efficiency remain separate stages per [INJURY_RESEARCH_BOUNDARIES.md](INJURY_RESEARCH_BOUNDARIES.md). Injury work here is nested after a data-audit gate and is never promoted as a clinical or causal injury model.

## 1. Objective and question

Given a pregame roster scenario, predict two quantities for every offensive skill player:

- **Share** of each team pool: dropbacks, carries, and targets.
- **Total** opportunity: the pool size itself, then the player's count, so a missing quarterback or a cluster of injured linemen can shrink the pool instead of only renormalizing shares.

Two allocators see the same scenario:

| Allocator | What it receives | What identity means |
| --- | --- | --- |
| **Latent-only** | Lagged skill vector, position, availability state; no player intercept | Two players with the same vector, position, and teammate multiset get the same share. `player_id` only attaches the lagged vector. |
| **Identity-aware** | The same score plus a partially pooled player intercept; team intercept on the volume equation only | Sticky coaching roles can live in the intercept after the vector is accounted for. |

The hypothesis is that teammate skill geometry and vacated-role features improve share forecasts beyond proportional renormalization of sticky rates, and that roster-conditional pool size improves team-total forecasts beyond the current volume baselines. Both claims are to be tested, not assumed.

## 2. What exists, and the gap

The inference layer already does three pieces. None of them is the model this experiment asks for.

| Piece | Module | What it does today | Gap for this experiment |
| --- | --- | --- | --- |
| Target panel | [panels.py](../src/hierarchical_football/inference/panels.py) | Team-game target totals for active `WR`/`RB`/`TE`, with an `OTHER` residual so candidate targets plus residual equal the team total | Fullbacks and quarterbacks who are targeted fall into `OTHER`. Carries and dropbacks are not panels. |
| Proportional shares | [allocation_baseline.py](../src/hierarchical_football/inference/allocation_baseline.py) | Renormalizes shrunk historical target rates. `include_player_history=False` is position-only; `True` is a sticky player rate | Neither score depends on teammates' skills except through renormalization. Neither moves when a listed active player is merely banged up. |
| Team volume | [volume.py](../src/hierarchical_football/inference/volume.py) | Pooled negative binomial or a team's own recent mean for target totals | Roster composition is not an input. Opponent history is not yet in the baseline. |
| Integer allocation | [allocation.py](../src/hierarchical_football/inference/allocation.py) | Turns a supplied total into conserved integer counts; optional Dirichlet-multinomial concentration | Shares, availability, and volume remain supplied. |
| Share scoring | [allocation_evaluation.py](../src/hierarchical_football/inference/allocation_evaluation.py) | Exact conditional multinomial / Dirichlet-multinomial NLL and conserved-draw calibration | This remains the share-scoring path; the experiment should not invent a parallel one. |

Weekly panels in [panels.py](../src/hierarchical_football/data/panels.py) already lag `target_share`, `carry_share`, and offensive snaps. [reports.py](../src/hierarchical_football/data/reports.py) attaches the latest injury row with `date_modified` strictly before the Wednesday 04:00 UTC cutoff. Those columns are excluded from `model_features`. Depth charts are cached under `facts/` and never joined. `historical_asof_verified` is false on every partition.

[TARGET_ALLOCATION_RESULTS.md](TARGET_ALLOCATION_RESULTS.md) already showed that player-history rates beat position rates on 2024 and 2025 (about 6 nats of multinomial NLL per team-game) while 90% intervals still under-cover, especially for players absent from training. Those seasons are inspected. Model selection for this experiment uses earlier seasons. **2024–2025 stay a confirmation window, not a tuning window.**

## 3. Population and pools

**Population:** offensive skill players — `QB`, `RB`, `FB`, `WR`, `TE`. Offensive linemen are volume covariates (how many are out or limited), not fantasy-opportunity candidates. Kickers and defense stay out.

Pools are defined from play-by-play and reconciled to weekly stats:

| Pool | Definition | Candidates | Residual |
| --- | --- | --- | --- |
| **Dropbacks** | Pass attempts, sacks, and scrambles. Designed runs are not dropbacks. | Quarterbacks on the active roster | `OTHER` for a non-QB or a QB the roster missed |
| **Carries** | Rush attempts excluding kneels | Active `QB`/`RB`/`FB`/`WR`/`TE` | `OTHER` |
| **Targets** | Existing canonical target definition, extended so an active QB or FB is a candidate instead of automatic residual | Active skill players in scope | `OTHER`. Throwaways stay out of the target pool, matching the current passing-table rule. Document the attempt-versus-target gap rather than equating them. |

Conservation rules:

- A known-out player has share exactly zero in that scenario.
- `OTHER` stays a real category.
- Shares inside a pool sum to one.
- Counts are integer and conserve the pool.
- Candidate indices order by `player_id`, never by the outcome (same rule as the current target panel).
- The Dirichlet-multinomial sampler already implements the overdispersed version; concentration is chosen on the development split, not on 2024–2025.

## 4. Generative structure

```text
Injury/practice reports before cutoff ──► Roster scenario
                                              │
Lagged skill vector per player ───────────────┼──► Deployment: out, limited, or ordinary
                                              │
Opponent, home, week, OL-out count ───────────┼──► Three pools: dropbacks, carries, targets
                                              │
                                              ├──► Shares over the active set plus OTHER
                                              └──► Player counts = share × pool size
```

For every simulated team-game:

1. Build or accept a roster scenario from reports and roster feeds known before the cutoff.
2. Attach each skill player's lagged skill vector and deployment state.
3. Sample or predict three pool sizes conditional on who is expected to play and pregame context.
4. Allocate shares over the active set plus `OTHER` under one of the two allocators.
5. Form player counts as share times pool size with one draw identity per team-game so teammates compete.
6. Stop before efficiency. Downstream hierarchical chains remain frozen until Stage F.

## 5. Skill vector

The latent arm is only as honest as the vector. Build it from prior games only, using the existing lag rule in [history.py](../src/hierarchical_football/data/history.py): a game counts only when kickoff plus four hours is before the prediction timestamp.

Core coordinates, each shrunk toward a position mean with a declared prior exposure (start from the existing 100 team-opportunity prior in [allocation_baseline.py](../src/hierarchical_football/inference/allocation_baseline.py); do not search it on the confirmation window):

| Coordinate | Source / note |
| --- | --- |
| Target share and carry share when active | Already lagged on weekly panels |
| Offensive snap rate | Already lagged; coverage is incomplete and must be reported |
| Dropback share (QBs) | From play-by-play |
| On-field rate on pass plays (skill players) | From `load_participation` `offense_players`. Route proxy only: `route` is the primary receiver's route, and it is absent for the current season until the season is over. Never use same-game participation as a predictor. |
| Depth alignment before cutoff | `depth_position` and `depth_team` from `load_depth_charts`, joined on the chart timestamp if one exists |
| Shrinkage weight | Number of prior active games, so a rookie is mostly the position prior |

Stage D (not Stage B) may add a recent-injury coordinate: practice participation this week and weeks since the last `Out` or `DNP`, among players who are still in the scenario.

**Do not put into the vector:** completion rate, yards, fantasy points, Next Gen separation, or `load_ff_opportunity` expected points. Those mix efficiency and post-hoc opportunity models into a usage question. `load_ff_rankings` and closing spreads on `load_schedules` stay out for the same reason: they are not timestamp-certified, and a market line can already contain the injury.

### Identity-free score

Permutation-invariant, linear first:

```text
logit_i = position intercept + W * z_i + vacated-role features
```

Vacated-role features are sums of `z` over teammates ruled out of the scenario, split by position and by depth alignment (another outside receiver versus a back). That lets a missing high-target receiver move the ball toward similar vectors rather than scaling everyone by the same constant. The proportional renormalization already in `predict(include_player_history=True)` is the baseline this has to beat, including on games where the top target-share teammate is absent.

Add a set-attention scorer only if the linear vacated-role residuals are still structured by position. Do not start there.

### Identity-aware score

Adds `u_player` with the same partial-pooling idea as [partial_pooling.py](../src/hierarchical_football/inference/partial_pooling.py): one penalty, fit on the development split, shared across players. No team intercept inside the share equation in the first identity arm. A team intercept on shares can absorb scheme and then masquerade as a player effect.

## 6. Totals, not just shares

Fit the pool size on training team-games with a negative binomial log mean that can include:

- team history and opponent (the current volume baseline does not yet include the opponent)
- who is the expected quarterback — that player's latent vector in one arm, a player intercept in the other
- counts of offensive linemen and skill players listed `Out` at the cutoff
- home, week, and a short-rest indicator from the schedule (`days_since_last_game` already exists at player level; build the team version)

Baselines remain the current pooled and team-history models in [volume.py](../src/hierarchical_football/inference/volume.py). A roster-conditional total is useful only if it improves team-total CRPS without destroying share likelihood.

Player total for a draw is pool size times share, using one draw identity per team-game. Report both:

| Score type | Conditioning | Purpose |
| --- | --- | --- |
| Conditional share scores | Realized pool size | Comparable to [TARGET_ALLOCATION_RESULTS.md](TARGET_ALLOCATION_RESULTS.md) |
| Marginal count scores | Predicted pool size | Joint usefulness of volume plus allocation |

A model can win one and lose the other; both numbers are required. Keep share parameters from Stage B fixed when scoring the Stage C joint so volume is not credited for allocation.

## 7. Injury design

Treat reports as reports. `load_injuries` (2009 onward) has `report_status`, `report_primary_injury`, `report_secondary_injury`, `practice_status`, `practice_primary_injury`, and `date_modified`. The join in `latest_known_injury_status` already refuses a report at or after the cutoff and refuses two statuses at the same timestamp.

### Audit gate (blocking before Stage D)

Before any coefficient is fit, audit the cache:

- Do historical files keep Wednesday, Thursday, Friday, and game-status updates, or only the last row per player-week? If only the last row survives, `date_modified` cannot reconstruct the pregame report and the column stays a candidate label, not a feature.
- How often is a player absent from the report while on IR, suspended, or a healthy scratch? Absence from the report is not health.
- Coverage by season of `practice_status` versus `report_status`.

### Nested comparisons (only if the audit shows real pre-cutoff updates)

| Nest | Behavior | Hypothesis / product |
| --- | --- | --- |
| **Mask only** | `Out` / inactive / IR / PUP / suspension are structural zeros. Questionable players stay in the active pool at their ordinary share. | Current supplied-roster behavior, made explicit. |
| **Recent-injury haircut** | Among players still in the pool, practice status (`DNP`, `Limited`, `Full`) and weeks since return shift that player's logit. | An active but recently out/limited player draws a smaller share; vacated mass flows through the same substitution features as a full absence. Body-part group (lower limb / upper limb / illness) is optional with a strong position-level prior — a usage association, not a clinical effect. |
| **Scenario mixture** | Questionable or doubtful → probability of the zero scenario versus the haircut scenario | Operational weekly forecast. Not a single availability multiplier on fantasy points. Oracle evaluation may use realized active status and must be labeled oracle. Conditional what-if (user sets a player out) stays a separate product. |

In-game exits stay unmodeled outcome noise. Consecutive `Out` labels are not recovery durations. Season-long injury risk is a different task and is not multiplied into weekly counts.

## 8. Data to add, and what to leave out

Already ingested and usable after the audits: schedules, weekly player and team stats, weekly rosters, snap counts (2012+, PFR id crosswalk via `load_ff_playerids`), injuries, depth charts, and play-by-play. Default `ingest` still skips snaps, injuries, depth charts, and play-by-play; the experiment's ingest list should include them.

### Add to ingest `TABLES`

Still as explicit partitions with `historical_asof_verified: false` in [ingest.py](../src/hierarchical_football/data/ingest.py):

| Table | Loader | Role |
| --- | --- | --- |
| `participation` | `load_participation` | 2016 through last completed season only. Derive lagged on-field rates. Do not treat `route` as every player's route. |
| `trades` | `load_trades` | A team change is a team change rather than a stale roster extension. Roster extension in [DECISIONS.md](DECISIONS.md) already copies an earlier snapshot and can miss a trade; flag `roster_carried_forward` rows and prefer not to train on them. |
| `players` | `load_players` | Stable position, draft year, and entry season. Draft year is a cold-start prior, not a weekly feature. Skip combine measurables first; `load_combine` remains available if the rookie prior is too weak. |

`depth_charts` is already ingested. The new work is an as-of join, not another download.

### Leave out of the feature set

| Source | Why |
| --- | --- |
| `load_ff_opportunity`, `load_ff_rankings` | Useful later as external baselines with a leakage label, not as inputs |
| `load_nextgen_stats`, `load_ftn_charting` | FTN (2022+) is post-snap play concept; Next Gen separation is efficiency. Either can be a later lagged efficiency covariate after opportunity is settled |
| `load_contracts`, closing moneylines | Not as-of safe |
| `load_pfr_advstats` | Overlaps snaps and usage; easy to double-count |

### Pipeline improvements this proposal specifies (not implemented here)

- One offensive player-team-week spine for `QB`/`RB`/`FB`/`WR`/`TE`, including explicit zeros for active players with no stat line. Today's default leaves those outcomes `unknown` unless `--zero-missing-player-stats` is set. The opportunity panel needs the zero when the player was active and the team stat feed exists, and needs `unknown` when the game is incomplete or the roster was carried forward.
- Separate labels for bye (no team-game), inactive, out, injured reserve, healthy scratch, and no stat row. `roster_status` from `load_rosters_weekly` is the start; it is not a full inactive list.
- Three conserved pools written beside the existing target panel, with the same candidate-index rule.
- Injury features computed only from rows before the cutoff, stored beside the panel and kept out of generic `features.json` until a study opts in — preserving the leakage contract in [schema.py](../src/hierarchical_football/data/schema.py).
- A small audit report: fraction of injury rows with a usable timestamp, fraction of player-weeks with more than one report, snap-join miss rate, participation season limit, and residual `OTHER` mass after QBs and fullbacks are promoted to candidates.

## 9. Evaluation protocol

Each stage has a frozen protocol in `examples/`, chronological fits, and paired uncertainty by whole game, matching the 2,000-draw game bootstrap already used for allocation.

| Window | Role |
| --- | --- |
| Development | Seasons chosen from **2016–2023** (participation and snaps both exist) |
| Confirmation | **2024 and 2025 only** — already inspected; never used for model selection |

### Metrics

| Layer | Primary | Secondary |
| --- | --- | --- |
| Shares | Mean team-game multinomial or Dirichlet-multinomial NLL, including `OTHER` | Candidate MAE, CRPS, 50% and 90% interval coverage and width; training-seen versus unseen players |
| Totals | Team-pool CRPS | Player-count CRPS when the pool is predicted |

### Predeclared slices

- Games in which the prior week's top target-share or carry-share teammate is out
- A player's first week back
- Active players with `Limited` or `DNP` practice status
- Rookies

## 10. Experiment sequence

| Stage | Deliverable | Decision gate |
| --- | --- | --- |
| **A — Audits and three-pool panel** | Blocking. Record `OTHER` mass, snap coverage, injury timestamp structure, and depth-chart timestamp structure. | No model selection until the panel and audits exist. |
| **B — Shares given realized pool and supplied active roster** | Arms: position-only proportional; player-history proportional; latent-only linear score; identity-aware linear score. Targets first, then carries, then dropbacks. No injury haircut. | Does skill geometry beat a sticky rate? Does identity still matter after the vector? |
| **C — Pool size given who is expected to play, then player totals** | Baselines: pooled and team-history NBs. Arms: latent QB + out-counts; same plus team and QB intercepts. Share parameters from B stay fixed for joint scoring. | Improves team-pool CRPS without destroying share likelihood? |
| **D — Injury (only if A says timestamps are real)** | Nested: out-mask → practice-status / weeks-since-return haircut → questionable mixture (operational). Oracle active status reported beside it and labeled oracle. | Gains beyond mask-only on confirmation slices? |
| **E — Scenario notes, not a new fit** | For a handful of held-out absences, publish shift in share and total for the absent player and teammates, with interval width. | Describe as model conditionals. Do not call them causal injury effects. |
| **F — Later, only if C and D are better on confirmation** | Feed sampled targets and carries into hierarchical chains in place of internal opportunity draws. Leave efficiency frozen. Revisit the within-player latent volume factor in [chains_rb.py](../src/hierarchical_football/chains_rb.py) so team allocation and that factor are not both adding the same dependence. | Separate promotion decision; not part of the first experiment go-ahead. |

## 11. What would change a conclusion

| Observation on confirmation | Reading |
| --- | --- |
| Latent-only ties or beats identity on NLL, especially for unseen players and games with a starter out | Skill vector carries the useful signal; identity is not required for those cases. |
| Identity wins overall but latent-only wins on first absences and trades | The vector transfers; the intercept is mostly sticky role. |
| Neither beats proportional player-history on absence slices | Teammate skill is not yet doing substitution work; Stage D should not be interpreted as an injury model. |
| Volume CRPS does not move when stars or linemen are out | Total opportunity stays a team-history problem; only shares are worth the extra model. |

## 12. Out of scope for the first experiments

Red-zone and third-down subpools, play-level route trees, efficiency changes after a role expands, quarterback-receiver stat reconciliation, in-game exit redistribution, season-long injury risk, contracts, markets, and injecting opportunity draws into fantasy scoring. Those remain follow-ons listed in [TEAM_OPPORTUNITY_PLAN.md](TEAM_OPPORTUNITY_PLAN.md).

## 13. Related documents

| Document | Relationship |
| --- | --- |
| [TEAM_OPPORTUNITY_PLAN.md](TEAM_OPPORTUNITY_PLAN.md) | Broader product plan this experiment extends |
| [INJURY_RESEARCH_BOUNDARIES.md](INJURY_RESEARCH_BOUNDARIES.md) | Availability / allocation / efficiency separation |
| [INFERENCE_LAYER.md](INFERENCE_LAYER.md) | Existing panel, volume, allocation, and scoring contracts |
| [TARGET_ALLOCATION_RESULTS.md](TARGET_ALLOCATION_RESULTS.md) | Inspected 2024–2025 proportional baseline evidence (confirmation only here) |
| [DECISIONS.md](DECISIONS.md) | Roster extension and as-of policies |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | Model input contract |
