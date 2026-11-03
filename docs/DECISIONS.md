# Policy decisions

Records the policies this package commits to, and the limits of each. It is not
a changelog: each section describes why the current behavior is what it is.

## Contents

- [Tracked and generated paths](#tracked-and-generated-paths)
- [Ingest report history](#ingest-report-history)
- [Roster extension](#roster-extension)
- [Prediction bundle metadata](#prediction-bundle-metadata)
- [Inference table contracts](#inference-table-contracts)
- [Passing target selection](#passing-target-selection)
- [Rushing attempt selection](#rushing-attempt-selection)
- [Pairing support reporting](#pairing-support-reporting)
- [Inference build inputs and manifest](#inference-build-inputs-and-manifest)
- [Extractor command surface](#extractor-command-surface)
- [First-stage extractor script](#first-stage-extractor-script)
- [Baseline estimator design](#baseline-estimator-design)
- [Holdout evaluation and clustered intervals](#holdout-evaluation-and-clustered-intervals)
- [Per-block penalties for the identity effects](#per-block-penalties-for-the-identity-effects)
- [Standardized contrasts and their support](#standardized-contrasts-and-their-support)
- [Reporting a fit that stopped at its iteration cap](#reporting-a-fit-that-stopped-at-its-iteration-cap)
- [A declared rusher scope is not a promise of carries](#a-declared-rusher-scope-is-not-a-promise-of-carries)
- [Summing a group of gains without borrowing or reading past the array](#summing-a-group-of-gains-without-borrowing-or-reading-past-the-array)
- [Giving a weak penalty enough iterations to converge](#giving-a-weak-penalty-enough-iterations-to-converge)
- [Team-game panels: candidates, the OTHER residual, and a label-free index](#team-game-panels-candidates-the-other-residual-and-a-label-free-index)
- [Chronological splits by week, and what a prediction bundle may carry](#chronological-splits-by-week-and-what-a-prediction-bundle-may-carry)
- [Team volume baselines: a pooled count law and a team-history reference](#team-volume-baselines-a-pooled-count-law-and-a-team-history-reference)
- [Team volume with a team effect, and the penalty that shrinks it](#team-volume-with-a-team-effect-and-the-penalty-that-shrinks-it)
- [One word per concept across the inference layer](#one-word-per-concept-across-the-inference-layer)
- [The charted pass location is the one context column recorded at the throw](#the-charted-pass-location-is-the-one-context-column-recorded-at-the-throw)

## Tracked and generated paths

Generated datasets, run manifests, and scratch notes are never tracked. The
`/data/` ignore rule is anchored to the repository root, so it does not hide
`src/hierarchical_football/data/`, which is ordinary source.

Every frame stored by `write_frame` writes a `.json` manifest beside its
Parquet file, and `ingest` and `build` write their reports beside the trees they
describe. All of those paths live under `data/`, so all of them are ignored.

The rule does not stop a caller from writing somewhere tracked. Pointing
`--raw-root` or `--output` at a tracked directory still produces files that git
will see.

## Ingest report history

`raw/ingest_report.json` records each invocation rather than replacing the previous one: a run appends a summary to `runs` and merges its partition results into `partitions`, so a cache assembled over several invocations still lists every partition it holds.

The merge is keyed by `(table, season)` and the newest record for a key wins. A partition downloaded in one run and served from cache in a later one reports `cached`, because the entry describes the last touch rather than the first fetch.

This is not a complete audit trail. It keeps one entry per partition and per run, not the intermediate state of a failed download, and it does not hash the report itself. A deleted report restarts the history; earlier runs are not recoverable.

## Roster extension

A team-week with a scheduled game and no roster snapshot inherits the most recent earlier snapshot from the same season when extension is requested. The copied row keeps its original `roster_status` and is marked `roster_carried_forward`, so a consumer can still separate observed rosters from inferred ones.

Extension is opt-in. With it off, a week without a snapshot produces no player rows, because the panel is assembled from rosters and the schedule; inventing participation would make an unsnapshotted week look modeled.

It does not model roster change. A trade, signing, injury designation, or practice-squad move after the copied week is invisible, and a season's first week is never inferred from a later one.

| Situation | Result |
| --- | --- |
| Week has its own snapshot | used unchanged, `roster_carried_forward` false |
| Week has none, a same-season earlier snapshot exists | copied from the newest such snapshot, flagged true |
| Week has none, no earlier snapshot exists | no player rows |
| Extension not requested | no player rows for unsnapshotted weeks |

## Prediction bundle metadata

A prediction bundle carries what identifies a row and what its cutoff was; it never carries the outcome. Roster status and the carried-forward flag travel with the bundle because eligibility is a caller decision, not a model output.

Roster columns stay out of `features.json`. They are context a consumer may filter on, not signals a model may fit.

The bundle does not say who will play. `roster_status` is the last observed designation, and for a week built by roster extension it is inherited from an earlier snapshot; an inactive list published after the cutoff never appears.

## Inference table contracts

A canonical inference table is validated, never repaired. A consumer that receives a
`passing_targets` or `rushing_plays` frame knows that the play key is unique, that both
identities and the outcome are present, and that every label is either declared in the
contract or `UNKNOWN`. Preparation owns those guarantees; validation only proves them.

`UNKNOWN` is a label, not a missing value. A pass whose charted location is absent, and a
player the weekly roster feed does not tag with a known position, carry `UNKNOWN` so an
analysis can count them instead of dropping or guessing them. A null in those columns is a
contract violation, because it would be indistinguishable from a failed join.

The contract covers required columns only. The optional charted and defensive fields -
catchability, drop, contested, screen, pressure, yards after catch, box count, personnel,
formation - vary in coverage by season, so a table without them is valid and their absence
is a data-availability question rather than a schema error.

| Situation | Result |
| --- | --- |
| Required column absent | `ValueError` naming the table and the missing columns |
| Column of the wrong kind | `ValueError` naming the column and the expected kind |
| Null key, season, week, team, identity, outcome, or position | `ValueError` naming the column and its null count |
| Label outside the declared set | `ValueError` listing the unexpected labels |
| Duplicate `(game_id, play_id)` | `ValueError` with the duplicate count; no row is dropped |
| Trick-play reception by a player tagged `QB` | kept; analyses choose their own position subset |

It does not certify that play-by-play classified a play correctly, that a snap belongs to
the player credited with it, or that the optional charted fields are trustworthy in the
seasons where they exist.

## Passing target selection

A targeted pass enters the canonical table when it is a regular-season pass attempt, play-by-play
names the passer, the receiver, and the possession team, and the completion flag is `0` or `1`.
Everything else is removed and counted, never repaired. Reason order decides which reason is
credited: postseason, two-point attempt, aborted play, sack, scramble, not a pass play, no pass
attempt, missing quarterback, missing receiver, missing team, invalid outcome.

Sacks and scrambles are removed because the play has no target a receiver could have converted;
a scramble is a run in the record but a failed pass for this question, and counting it as a
failed completion would penalise a passer for leaving the pocket. Two-point attempts and aborted
plays are removed because their situation is not comparable to a normal down. A completion flag
that is missing or outside `0` and `1` is an invalid outcome rather than an incompletion, so a
corrupt source row never becomes evidence about a passer.

| Source play | Kept | Reason when removed |
| --- | --- | --- |
| Regular-season pass attempt, flag `0` or `1`, identities and team present | yes | - |
| Sack, scramble, throwaway with no receiver | no | `sack`, `scramble`, `missing_receiver` |
| Two-point attempt, aborted play, non-pass play, no pass attempt | no | `two_point_attempt`, `aborted_play`, `not_a_pass_play`, `no_pass_attempt` |
| Postseason play | no | `postseason` |
| Passer or possession team unidentified | no | `missing_quarterback`, `missing_team` |
| Completion flag missing or not `0` or `1` | no | `invalid_outcome` |

The table measures conversion once a pass is targeted. It cannot measure whether the target should
have existed: play-by-play records the receiver of the attempt, not the routes run by every other
eligible receiver. Missing `air_yards`, `yardline_100`, `ydstogo`, and `down` are retained and
counted in the report rather than filled, so a later model can decide whether to use them.

Charted fields are absent from this stage. Their coverage varies by season, and joining them
before that coverage is measured would put a season-dependent definition inside the canonical
table.

## Rushing attempt selection

A carry enters the canonical table when it is a designed regular-season rush attempt by a runner
inside the named scope, with a known team and a recorded signed yardage. Reason order decides
which reason is credited: postseason, two-point attempt, aborted play, kneel, spike, scramble,
not a run play, no rush attempt, missing rusher, missing team, missing yardage, outside the
named rusher scope.

Kneels and spikes are removed because they are recorded as runs without being attempts to gain
yardage, and a scramble is removed because it is a pass that failed rather than a designed run;
leaving either in would mix two different decisions into one average. Signed yardage is kept
whole, including negative carries, because the distribution of rushing yards is the object of
study and truncating it would remove exactly the plays a runner or a blocking unit most affects.

The scope decides whose carries are counted. `backfield` is the default because the weekly roster
feed identifies running backs reliably, `non_qb` adds receiver and tight-end handoffs as a
sensitivity case, and `all` keeps every identified rusher including a quarterback. A scope that
depends on position refuses to run without a roster: returning an empty table would look like a
finding.

| Carry | Kept | Reason when removed |
| --- | --- | --- |
| Regular-season designed carry, in scope, yardage recorded | yes | - |
| Kneel, spike, scramble | no | `kneel`, `spike`, `scramble` |
| Two-point attempt, aborted play, non-run play, no rush attempt | no | `two_point_attempt`, `aborted_play`, `not_a_run_play`, `no_rush_attempt` |
| Rusher, possession team, or yardage unidentified | no | `missing_rusher`, `missing_team`, `missing_yardage` |
| Carry by a position outside the scope, or by an untagged player | no | `outside_rusher_scope` |

This table supports an environment association, not a blocker grade. A runner, the linemen who
blocked for him, the scheme, the quarterback's threat, and the opponent all move together, and
most linemen appear on the same snaps as each other. Missing `run_location`, `run_gap`,
`yardline_100`, `ydstogo`, and `down` are retained and counted rather than filled, and defensive
box count is not joined at this stage because it is plausibly a mediator rather than a context.

## Pairing support reporting

An audit reports the evidence a table holds rather than a claim about it. Row counts per season,
null shares per column, and the connected components of the passer-receiver graph exist so that a
later contrast can say "this pairing rests on two games" or "these two passers never share a
receiver" instead of presenting an interval that looks precise.

The graph is built from distinct pairings, not from plays: a receiver with 400 targets from one
passer is still one edge. Components are reported with their player counts, pairing counts, and
rows, ordered by rows, and the largest component carries its share of all rows. `single_pairing_components`
counts the components where every player appears only with each other, which is the case where a
split between passer and receiver is not identified by the data at all.

Validation runs first, so an audit refuses a table with a duplicate play key or a missing required
column rather than reporting on a frame that is not canonical.

It does not test separation, estimate anything, or decide whether a contrast is legitimate. It
reports exposure and connectedness; the analysis decides what that evidence supports.
It also does not audit charted fields that are absent: `present_optional_columns` names the ones
the table carries, and their coverage is a question for the season they come from.

## Inference build inputs and manifest

The inference tables are built from the local cache and never from the network. `build` reads the
partitions that are already there, and a season without a cached `pbp` partition raises
`FileNotFoundError`. A preparation step that silently downloaded would make a published table
irreproducible, because the source state it was built from would no longer be pinned by the
manifest.

`prepare_report.json` records the `sha256` of every input partition, both preparation reports with
their per-reason exclusions, and the coverage and pairing audits. The tables themselves go to one
Parquet file per table with a `.json` sidecar, matching the file names the research plan declares:
the table is cross-season, so partitioning it by season would suggest a boundary the data does not
have.

Roster positions are optional. Without cached `rosters_weekly` partitions the passing table carries
`UNKNOWN` positions and a position-scoped rushing table refuses to build, because an empty rushing
table would look like a season with no carries. `rusher_scope="all"` is the documented way to build
without rosters.

The manifest does not make the build reproducible from nothing: it records what was read, not where
it could be fetched again. It also does not verify that a cached partition was the source state at
some earlier date; the ingest sidecars already record `historical_asof_verified: false`.

## Extractor command surface

The extractor command prepares tables from the local cache and nothing else. It has no refresh or
download option, because fetching belongs to `hierarchical_football.data ingest`, which writes the
partitions and reports that the inference manifest then hashes. A command that both downloaded and
prepared would hide which source state a published table came from.

It prints one line of JSON rather than a progress log, and it fails with the exception it hit: a
missing `pbp` partition, an absent cached roster under a position-scoped build, or a duplicate play
key all exit non-zero with the reason on stderr. Silence would be worse than failure here, because a
table built from fewer seasons looks exactly like a table that was supposed to have fewer seasons.

The command does not validate a previously written table, compare it with an earlier build, or
publish anything. It writes to `--output-root` and stops.

## First-stage extractor script

The repository-root script and the package command are two entry points to one implementation, and
that duplication is deliberate. The research plan publishes a root-level command line for the first
stage, while the maintained surface is the package command that `uv run python -m
hierarchical_football.inference` exposes. The script holds no selection policy of its own: it parses
the documented options and calls `build`.

It does not vendor the preparation code, so it needs the package installed in the environment it
runs in. If the two entry points ever disagree, the package command is the one to trust and the
script is the one to fix.

## Baseline estimator design

The two baselines per table are nested by construction, not by convention. Context alone and context
plus identified players share one design builder, one standardization, one set of level sets, and one
penalty schedule, so the error difference between them is the identity increment the research plan
asks for. Two models built separately would differ in more than the question.

The context is only pre-play. Target depth, field position, distance to go, down, and charted pass
location are all known before the throw, while a completion-derived column is partly the outcome the
model is meant to explain. The same reasoning keeps defensive box count out of the rushing context at
this stage, because the plan treats it as a likely mediator of the carry rather than as a setting.

An identity is a penalized one-hot block, and the penalty is what keeps a passer with 40 attempts and
a receiver with 4 targets from producing the extremes of an unpenalized fit. The cost is stated, not
hidden: L2 centres the terms of one block on zero, so a term is a deviation from the average
identified player, and its size follows the outcome rate inside a player's own rows as much as the
number of them. Measured on a two-player fixture, the two terms come out symmetric at the same
magnitude however lopsided the exposure is, which is why the term list is ordered by name, never by
size, and why `nested` sits next to every term.

The logistic tolerance is 1e-8 rather than the solver's default. The gap between an identity block and
the unpenalized intercept is flat in the loss, and the default tolerance stops along it: on a
three-passer fixture the default left the block sum at -0.046 and the intercept at -0.049, while the
tight solve reached a block sum of 1e-6 and the intercept its actual value. A term offset by an
arbitrary constant would be reported as an effect.

A player the training season never saw gets no column. A term for a player with no training rows is
not a coefficient but a zero row, and the design counts the rows that took it. The same rule covers a
category the reference never saw, except that those columns exist and contribute a zero row, while a
null continuous value is filled with the reference median and carries its own indicator column when
the reference frame had nulls at all. Filling with the holdout's own median would leak the later
season's distribution into a model fitted on an earlier one.

The support environment is a composition, not a talent. `support=True` folds team, location, and gap
into one column, and a runner who only ever carried behind one of those combinations is marked
`nested` because the term and the environment cannot be told apart. Baseline 1 is that column and
Baseline 2 is the runner block on top of it, so they are separate switches: a caller can report the
environment model without the runner model and see what each one adds.

The module computes no holdout metric, no interval, and no cross-validated choice. A baseline that
picked its own penalty in-sample would report the choice as evidence, and the chronological split
belongs to the evaluation protocol, where the clustered resampling lives too. `report()` carries
`in_sample` error for one purpose: to catch a fit that did not converge. A model with a column per
player fits its own rows by construction and that number is never evidence about identity.

## Holdout evaluation and clustered intervals

The unit of resampling is the game, not the row. Plays inside one game share a defense, a score, a
weather, a passer, and a game plan, so a standard error computed across rows treats those repeats as
independent evidence and comes out too narrow. Resampling whole games keeps that dependence inside the
unit that is drawn. The cluster column is a parameter, so a caller can use a team-week or a runner-week
instead when that is the dependence of interest.

The comparison is paired and the pooling is by row. Both models are scored on the same drawn clusters, so
a play that is hard for one model is hard for the other in every replicate, and the interval is about the
difference between the models rather than about the season they were scored on. Within a replicate the
mean is taken over the drawn rows rather than over the drawn clusters' means, because games do not carry
the same number of plays and the estimand is a row-level average loss.

Sign is stated once and used everywhere: a positive difference means the first model carries the larger
loss, so a difference interval that stays above zero is evidence against "no difference". The same
convention applies to `clustered_comparison`, where every model is measured against the reference.

The reported metrics follow the plan's primary list: log loss and Brier for completion, MAE for rushing,
with RMSE, the median absolute error, the share of residuals at ten yards or more, and a reliability table
as supporting output. The tail share sits beside the mean because a squared penalty on a skewed outcome is
driven by extreme carries, and two models can share an MAE while disagreeing about who is badly wrong.

An interval here is not a ranking. Nothing in this module tunes a penalty, searches a model, or declares a
winner: a baseline that picked its own penalty on the holdout would report the choice as evidence, and the
protocol decides what a difference means before the number is read.

Two limits are stated rather than hidden. A percentile bootstrap over very few clusters is narrow, because
a handful of distinct resamples cannot reproduce the sampling spread of a handful of games; the cluster
count is therefore reported with every interval. And these intervals cover the difference in a row-level
loss over a season, not the uncertainty of an individual player's term, which belongs to the posterior
layer the plan schedules after the baselines.

## Per-block penalties for the identity effects

One penalty over the whole coefficient vector makes two different modelling choices into one: how much
to shrink 800 player columns, and how much to shrink target depth and field position. The first needs a
strong prior to stay stable out of season; the second needs almost none, because those columns are the
pre-play context the model exists to condition on. `partial_pooling` splits the penalty by block, which
is the ridge form of a random-effect prior on players.

The split is implemented by rescaling the identity columns and folding the scale back into their fitted
coefficients. The fit stays an ordinary scikit-learn solve, the fitted object stays an ordinary
`FittedBaseline`, and every downstream function - prediction, `identity_terms`, the clustered comparison -
keeps working. Equal block penalties reproduce the ordinary fit exactly, which the tests check rather
than assume.

The player penalty is chosen on an inner chronological split: earlier seasons fit, the next season
scores, every grid point is measured the same way, and ties go to the larger penalty. A prior chosen on
the season that is later reported would make the choice part of the result. The whole grid is returned
rather than the winner alone, because a flat grid says the choice does not matter and a peaked one says
it does.

The penalties are reported in the fit's `block_penalties` and are never turned into a probability, a
share of credit, or a ranking. A variance component lives on the scale it was estimated on.

## Standardized contrasts and their support

A conditional model answers a question about the contexts a player actually saw. A contrast asks about
one shared population Q, and the plan asks for the observed-support mix rather than a synthetic grid.
The population is therefore a slice of a canonical table: that is what the design builder can score, and
it keeps the contrast inside contexts the league actually produced.

Overlap is reported, not assumed. Cell labels turn "is this player supported over Q?" into numbers: the
share of Q his own rows cover, the share they cover well, and the share he never faced. A contrast whose
extrapolated share is large is the model speaking outside the data, and the report says so before the
number is read. A player with no fitted column gets no term - his arm is the average identified player,
and `unseen_identities` names him so the arm is not mistaken for a coefficient.

Two scales, one estimand each. The plan asks for both a catch rate and an adjusted log-odds deviation,
and the two disagree in a way worth stating: the mean probability under Q depends on which partner is
fixed even in an additive model, because the average of a sigmoid is not the sigmoid of an average, while
the mean linear predictor does not. A reader comparing two receivers while free to choose the passer is
making a different statement from one who holds the passer fixed, and both reports carry the fixed levels.

No interval is attached to a contrast. The panel bootstrap in `evaluation` measures a model's loss over a
season, not the uncertainty of a player term, and resampling the population would measure the spread of
contexts rather than the uncertainty of the estimate. What is offered instead is a penalty sensitivity
range, labelled as such, which is honest about being a study of one modelling choice.

Nothing here is causal. Holding the context fixed removes the observed context difference between two
players. Coach, coverage, route assignment, health, and game script are not in the model and are not
removed by it, and a player swap is never described as an intervention.

## Reporting a fit that stopped at its iteration cap

A logistic fit that reaches the iteration limit is still a fit: scikit-learn returns coefficients, the
sampler and the metric functions use them, and nothing in the call stack says the solve never finished.
That happened here in practice - every selection fit at the weakest penalties (0.03 and 0.1) on the
identity designs hit the cap of 1000 iterations. The module now reports the iteration count and a
`converged` flag inside `in_sample`, and warns with a `RuntimeWarning` when the cap was reached.

The fit is returned rather than refused, because a caller searching a penalty grid needs the weak end of
the grid to be measurable, and because the fact is what matters: a coefficient that is not at the optimum
is still useful evidence, as long as the reader knows. A closed-form solver reports `converged` true with
no iteration count, which is why the flag is a pair and not a boolean.

## A declared rusher scope is not a promise of carries

`rusher_scope` declares which canonical positions belong to the table, and the preparation report now
carries `kept_positions` beside `scope_positions`, so the declared labels and the labels that produced
rows are both visible. The weekly roster feed does not tag fullbacks in the cached seasons - its
`position` column holds `RB` but never `FB` or `HB`, while `depth_chart_position` does - so `backfield`
keeps running backs only, and a `FB` count of zero is the honest report of that rather than a defect in
the filter.

The alternative, resolving fullbacks from the depth chart, is a real decision about which feed owns a
player's position and would change the population and every downstream number. Until that decision is
made, the report states the gap instead of a footnote claiming a scope the source cannot fill.

## Summing a group of gains without borrowing or reading past the array

`GainMixture.sample_totals` summed each group with `np.add.reduceat(draws, offsets)`, which has two
documented quirks that a caller with an empty group meets immediately: a zero-size group returns the
value of the draw at its offset rather than zero, and an offset that equals the length of the array -
exactly what a trailing empty group produces - raises `IndexError`. Measured on a deterministic mixture
with sizes `[0, 0, 4]`, the old path returned `[2, 2, 8]` instead of `[0, 0, 8]`, and sizes `[2, 1, 0]`
raised `IndexError: index 3 out-of-bounds in add.reduceat`.

The fix reduces only the non-empty groups and leaves the rest at zero. It is written into all four copies
of the class, because each position module carries its own, and `tests/test_gain_mixture.py` runs the
same checks against all four. The production paths mask their groups to a positive size before calling
`reduceat` directly, so the defect was latent there; the public helper was the part a caller would hit.

That test module is also the first check in the suite that imports the model layer at all. The data and
inference layers have per-module tests; the position chains are covered only through this helper, which
is a gap rather than a design.

## Giving a weak penalty enough iterations to converge

The logistic solver was capped at 1000 iterations, and the identity designs at the weak end of the
selection grid needed more: on the 2022 passing table a penalty of 0.03 converges after 2021 iterations
and a penalty of 0.1 after 1163, so four fits in the first penalty study stopped at the cap and reported a
point that was not the optimum. The cap is now 5000, and every fit reports its iteration count and whether
it reached its own stopping rule.

The cost is stated rather than hidden: the weakest grid point takes about a minute instead of half of one.
The alternative is a number that looks like an optimum because a counter reached a limit. Raising the cap
cannot move a fit that already converged, so the outer comparisons recorded in the audit documents stand;
the weak-penalty rows of the 2023 selection grid in `docs/INFERENCE_MODEL_DEVELOPMENT.md` were re-measured
and are marked there.

## Team-game panels: candidates, the OTHER residual, and a label-free index

The two opportunity models need one object: a team game, the targets its offense recorded, and the
receivers who could have drawn them. `inference.panels` builds it from what the layer already has - the
canonical passing table, the cached play-by-play for game context, and the weekly rosters for
availability - rather than from the processed panels the plan names, because this checkout's cache
holds play-by-play and rosters and not the weekly stat or schedule partitions. The unit of observation
is unchanged; only the source is.

A candidate is a player the weekly roster lists as **active** for that team and week with a position of
`WR`, `RB`, or `TE`. Practice-squad, reserve, and inactive rows are not candidates, because a player who
cannot take the field cannot draw a target and counting him would put a zero in a pool he was never part
of. Other positions are not candidates either: a lineman who reports eligible is a real receiver for one
play, not a member of the allocation pool.

Whatever the candidates do not explain lands in `OTHER`, which the allocation model carries as a
category. In the four cached seasons that is 149 targets of 68,408 (0.22%), spread over 104 team games.
Small, but it is the difference between a pool that closes and a pool that quietly loses a target.

Conservation is checked rather than assumed. `team_game_panel` returns the candidate targets and the
`OTHER` targets beside the team total, and raises if a candidate explains more targets than the team
recorded or if the three numbers do not add up. A build that cannot satisfy the equality is a bug in the
join, not a row to clip.

The candidate index orders players by `player_id` within a team game. Sorting by targets would put the
label in the row key and would change the key whenever a week changed, so permutation tests and
position-stable joins would both depend on the outcome.

Zero targets are recorded as zeros when the cache contains the game, because the totals and the plays
come from the same feed: a team game with no in-scope target is a measured zero, not an unknown. A team
game the cache does not contain produces no row, and the panel never invents one. Coverage in the cached
seasons is complete: all 2,174 team games have 8 to 14 candidates.

What the panel does not promise: the weekly roster is a weekly snapshot and not a confirmed game-day
active list, so a candidate may have been inactive on Sunday; roster carry-forward is not used here;
snap counts and target depth are not attached; and the split that will gate these rows is a separate
step.

## Chronological splits by week, and what a prediction bundle may carry

A cutoff names a season and a week, and the split orders team games by that pair rather than by the week
number: week 18 of 2023 is older than week 1 of 2024, and a model that learned from the previous season
must not be handed the following one by accident. Everything at or before the cutoff is training data;
everything after it is prediction data.

The prediction bundle carries no label at all. `targets`, `candidate_targets`, `other_targets`, and
`candidates` are dropped from the predicted panel, and `targets` is dropped from its candidate rows; the
observed bundle returns exactly those columns for exactly those team games. Dropping the columns is
mechanical and therefore checkable, which is the point: an as-of gate that depends on a caller not
reading a column is not a gate.

A team game is the unit, so the split refuses a panel whose game has team games on both sides of the
cutoff. Both teams of a game and every teammate of a team game share one week, so a week cutoff keeps
them together without a further rule; the refusal covers the case where the panel itself is inconsistent
about when a game was played. The split also refuses a cutoff with no history, a cutoff with nothing left
to predict, predicted team games with no candidates (nothing could be allocated), a team game that
appears twice, and a candidate whose team game the panel does not contain.

What the split does not do: it does not build features, and it does not promise that a feature computed
outside it is safe. A lag or a rolling average must still be computed from training weeks only. It also
does not walk forward by itself; a caller that wants expanding-history folds takes one cutoff per fold
from the same function, and the plan's early-season floor is the caller's rule to state.

## Team volume baselines: a pooled count law and a team-history reference

The plan asks for a pooled count baseline and a rolling team baseline before a hierarchical volume model
is fitted, and for both to be scored on the same team games. `inference.volume` fits a negative binomial
by maximum likelihood with the variance written `mu + mu**2 / k`, and a team-history reference that gives
each team its mean over its most recent 17 training team games, falling back to the training league mean
for a team that never appeared. The reference reuses the pooled dispersion, which is the empirical-noise
form of a rolling baseline.

The comparison consumes a split's three bundles rather than three frames of the caller's choosing. It
refuses a prediction panel that still carries a label column, because that means the wrong bundle was
passed, and an observed panel whose team games differ from the predicted ones, because two models scored
on different rows prove nothing. Both baselines see the same rows, the same target definition, and the
same cutoff.

Measured on the cached seasons. Cutting at the end of 2023: pooled CRPS 4.30, MAE 6.09, coverage 0.549 at
50% and 0.915 at 90%; team history CRPS 4.43, MAE 6.34, coverage 0.534 and 0.909. Cutting at the end of
2024: pooled CRPS 4.307, MAE 6.135; team history CRPS 4.285, MAE 6.095. The team's own volume history did
not beat the league's count law in 2024 and beat it by 0.02 CRPS in 2025, so neither baseline is a
forecast; they are the floor the hierarchical model must clear.

What these baselines do not do: they carry no team, opponent, or season effect, so the pooled law cannot
separate a good offense from a lucky one; the dispersion is shared across teams; the reference reads only
the training weeks and never updates inside the predicted weeks, so it cannot follow a mid-season change;
the intervals are central intervals rather than a PIT curve; and a single cutoff is not walk-forward
evaluation. Each of those is the hierarchical model's job, and each has to be measured on the same rows.

## Team volume with a team effect, and the penalty that shrinks it

The plan's Model A gives each team a deviation from the league mean, partially pooled, with one shared
count dispersion. `inference.team_effect` fits that model by penalized maximum likelihood: a negative
binomial whose log mean is an intercept plus one deviation per team, with a ridge penalty on the
deviations. It is the frequentist cousin of the plan's Bayesian hierarchy, not its posterior, and every
number it reports is a point estimate.

The penalty is a variance component, so it must not be chosen on the rows the model is scored on. The
selector takes an inner cutoff inside the training history, fits on the weeks at or before it, scores the
weeks after it with the count log score, and returns the whole table so the margin is visible. The grid
starts at zero shrinkage, which makes "no team effect" a candidate answer rather than a grid edge, and
ends at 10,000, which is far enough out that a boundary answer means the search needs another order of
magnitude. The reported model is refitted on all the training weeks at the penalty the inner split chose.

Measured on the cached seasons. Training through the end of 2023 and predicting 2024: the inner split
chooses a penalty of 100, the effect deviations have a standard deviation of 0.078 on the log scale, and
the model scores CRPS 4.42 and MAE 6.28 against CRPS 4.30 and MAE 6.10 for the pooled law, so it loses by
0.12. Training through the end of 2024 and predicting 2025: the split chooses 1,000, the deviations shrink
to 0.025, and the model scores CRPS 4.27 and MAE 6.11 against 4.29 and 6.12, so it wins by 0.02. Its 90%
coverage is 0.88 in the first season and 0.92 in the second.

The verdict this record takes is that the team effect is not yet worth promoting. It is small, it changes
sign between the two cuts, and the model asks for tighter shrinkage the more history it has, which is the
signature of effects that are barely there. A more complicated model is allowed to lose; it is not allowed
to lose quietly, which is why the losses are in this record and in the guide.

What the model does not carry: no opponent effect, no season trend, and no team-by-season interaction,
which the plan introduces one at a time and only when the simpler model miscalibrates; the deviations are
a single number per franchise, so a team that changes coach or quarterback mid-sample is one deviation; the
dispersion is shared across teams; the dispersion search is bounded rather than given a prior; and a single
cutoff is not walk-forward evaluation.

## One word per concept across the inference layer

Two names for one thing and one name for two things cost a reader more than they cost the code. Four
words were doing both, and each now means one thing everywhere in the package, in the result keys, and in
the tests.

- `holdout` is the later season that is scored. `chronological_split` and `select_identity_penalty` both
  say `holdout_season` and both report `{"train", "holdout"}`; the selection calls it a holdout too,
  because that is what it is, even though nothing downstream is chosen with it.
- `reference` is the earlier design whose means and level sets a later frame is scored on. The model every
  other model is measured against in a comparison is `reference_model`, so `clustered_comparison` no
  longer reuses the word for a second job.
- `identity_mask` is a boolean mask over design columns, which is what the per-block penalty scales;
  `identities` stays the tuple of column names a caller names.
- `label` is one declared value of a categorical column, in `contracts`. The scalar helper lives beside
  `UNKNOWN` and is imported by the modules that need it, and the Polars expression in `prepare` is
  `_declared_label`, because it returns an expression over a column rather than one label.
- `require_task` decides what metric family a loss is measured in, so the vocabulary lives in one place
  and the same message is not written twice.

`player` remains the word for the label of an identity column inside `identity_terms`, because that entry
is about a person rather than about a level of a column; the docstring says so. The change is a naming
policy, not a behaviour change: every recorded number in `docs/` was produced by the same arithmetic
before and after it.

## The charted pass location is the one context column recorded at the throw

`pass_location` is charted rather than pre-play: it says where the ball went, not what the defense showed
before the snap. It is admitted to the completion baselines because the plan's context list names the
direction of the pass beside target depth, field position, distance, and down, and because a conversion
model without it would be conditioning on less than the plan declares. What is not admitted is anything
the outcome itself produces: the completion, yards after the catch, catchability, a drop, or a contested
catch.

The distinction matters for how a fitted term is read: a passer's term is an association conditional on
where he threw the ball, so it is a smaller question than "how well did he throw" and a larger one than
"how well did he choose". `docs/INFERENCE_DATA_AUDIT.md` and the layer guide state the same thing, and all
three say it in the same words now: an earlier draft of the data audit claimed the column stays out of the
model, which the code, the guide, and this record contradict.

## Passing difficulty before player attribution

The expanded completion context uses fixed depth hinges rather than selecting knots on
scored seasons. Position and opponent intercepts are included in the same context block
for both models in an identity comparison. The reference design carries transformations,
training scaling, and observed category levels into later predictions. Derived hinge inputs
are recomputed from depth, so caller-supplied columns cannot change the basis.

The linear default remains available for historical reproduction. A defending-team
intercept is an association, not a defense grade; neither specification sees coverage,
routes, pressure, or catchability. The expanded context must earn its predictive value
on earlier validation rows before any final evaluation.

## Independent passing identity priors

Quarterback and receiver intercepts may carry different shrinkage strengths. Both blocks
must be named explicitly with finite positive penalties; misspelled or omitted blocks fail.
The existing shared-identity pooling API is unchanged. Selection scores the Cartesian
product on one earlier season and records convergence for every candidate. Capped fits
cannot win. Exact ties prefer larger total penalty, then larger QB penalty, then the
first declared context. The selected context is also used for the context-only comparator,
so its identity increment is measured on the same difficulty specification.

This remains a penalized point estimate. Separate penalties do not resolve co-occurrence,
and selection on one inner season is not proof that the selected priors are universally best.

## Natural partnership transfer cohorts

A new partnership requires both players to have appeared in earlier training seasons,
but never together there. It is separated from three unseen-player groups and known
partnerships before scoring; cohort membership never reads the outcome. Cohorts partition
the scored targets exactly. Loss comparisons resample games within a cohort, with no
interval below two games and a small-sample flag below ten. Empty groups report null
metrics instead of a score of zero.

This measures natural chronological transfer. It does not create randomized partner
assignments, withhold every historical instance of a selected pair, or remove selection
into a new partnership. A favorable new-pair score is evidence about predictive transfer,
not a causal teammate-swap effect.

## Conditional uncertainty for selected player contrasts

A contrast bootstrap resamples whole training games and refits with fixed block penalties.
The reference population is fixed rather than resampled. Replicated games receive new keys,
preserving canonical play uniqueness without dropping repeated rows. All requested players
must be observed. Disconnected player components or unsupported reference context cells
suppress an interval; missing players, disconnection, one-outcome samples, or nonconvergence
in any replicate likewise leave the interval null with failure counts. Successful replicates
are never silently substituted for the requested bootstrap distribution.

Graph connection is necessary but not sufficient for individual attribution. A narrow
conditional interval can coexist with prior-sensitive or confounded effects. These are
percentile intervals for regularized estimates under this specification, not posterior,
causal, or bias-corrected skill intervals. They omit selection uncertainty and uncertainty
in the reference population. Synthetic checks cover recovery with crossed partnerships,
prior sensitivity when partnerships are confounded, and support failures.

## A frozen passing study across expanding seasons

A study chooses context and two identity penalties once on its declared inner split.
Each outer season fits every declared earlier season, including previously scored outer
seasons, while keeping selection fixed. Every comparator sees identical held-out plays.
The five comparators are the historical linear context, selected context, selected context
with QB only, receiver only, and both. Single-block ablations retain the jointly selected
penalties, so they measure removal at that specification rather than optimized rivals.
The block fitter accepts an explicit subset for these ablations; an empty subset fits context only.

The report records the specification, complete grid, convergence, input coverage,
training partnership support, calibration and tails, clustered loss differences, and
transfer cohorts. Keyed predictions retain outcomes separately from the report. Requested
player contrasts use training targets, optionally restricted by position, as Q; no outer
context or outcome enters that reference population.

Reserved seasons are excluded from every split, and missing requested seasons fail.
Only development studies are supported: no program can establish that a human has never
inspected a season. A final untouched evaluation requires a separately documented,
prospective reservation and a frozen scientific protocol. No model is promoted by this
runner, and no ranking or causal interpretation is produced.

## Traceable passing-study artifacts

The JSON protocol admits only declared schema fields, rejects duplicate keys and nonfinite
constants, and validates every contrast before model fitting. The example config names
2022 training, 2023 selection, and 2024-2025 expanding development evaluations; it does not
claim to reserve an untouched holdout. Reserved seasons, when supplied, are excluded by
the study contract and Parquet season predicate before materialization.

Output paths are exclusive to one run. The exact protocol bytes, keyed predictions,
report, and per-output hashes are retained. A complete manifest is written last, while
exceptions retain a failure record. A bundle without a complete manifest is incomplete,
even if individual files exist. Hashes of the consumed input and package source identify
the actual working code independently of its Git commit; the recorded Git revision alone
does not assert a clean checkout. Input, source, and protocol hashes are checked again
after fitting to reject a mixed-state run. Runtime versions are recorded, not installed.

The configuration makes a run reproducible and reviewable; it does not pre-register a
scientific claim, prove inspection history, license new data, or promote a model. No
live network requests or automatic updates to forecast chains occur.

## Separate passing-context components

The fixed expanded basis is available as all eight combinations of nonlinear depth,
position, and defending-team indicators. Every combination retains the original linear
depth and situational context. This makes matched comparisons possible without changing
knots, selecting features from outcomes, or conflating opponent indicators with defense
quality. The declared context name travels with the design and fit; later rows reuse its
training levels and scaling. Linear and expanded behavior remain compatible with earlier
runs. Component loss differences describe refitted models, not causal credit for a feature.

## Identity increments against the other player block

A receiver increment compares the joint QB/receiver fit with QB-only, while a QB increment
compares joint with receiver-only. Each uses paired loss differences on the same plays,
resampling whole games with the protocol's seed, interval level and draw count. Reports
retain both references explicitly. These diagnostics are emitted overall and by natural
partnership cohort, rather than inferring receiver transfer from joint-versus-context gains.
Empty cohorts have no diagnostic; a single game has no interval. All three prediction
models must exist before a transfer cohort can report this comparison.

The ablations keep jointly selected penalties. They test predictive increments of refitted
blocks under that specification, not uniquely identified receiver skill, causal credit,
or performance of separately optimized alternatives. Multiple cohort intervals remain
development diagnostics and do not constitute independent confirmatory tests.

## Context diagnostics outside model selection

A passing protocol may declare context-only diagnostics independently of its inner
selection grid. They reuse the fold's training seasons, fixed basis and context penalty,
then score identical held-out targets. Existing fitted linear and selected contexts are
reused; other requested combinations are fitted once. The linear reference is always
included when diagnostics are enabled. Nonconverged diagnostic fits fail the complete
study rather than quietly dropping a comparison.

Reports include all matched pairs differing by exactly one of nonlinear depth, position,
or defending-team indicators. Differences are richer minus simpler model loss, with
paired game-bootstrap intervals. All eight combinations yield twelve such edges. Their
gains depend on what is already included, so they cannot be summed into percentages of
causal credit. No diagnostic result participates in selection or model promotion. The
example retains the original selection grid; any later choice informed by these outer
results needs separate validation. Optional diagnostic probabilities are saved with the
same play keys so each difference can be independently reproduced.

## Independent opponent shrinkage

Static defending-team intercepts can overfit when they share the weak penalty used for
well-measured situational context. The block completion fitter can shrink only those
one-hot columns while leaving the depth basis, position and player priors unchanged.
Omission retains the original behavior; explicit finite positive opponent penalties are
allowed only when the context includes defending-team terms. An empty identity tuple
with an empty penalty mapping fits context alone, allowing matched environment priors
across context-only, single-player-block and joint-player fits.

This is still a static team association. Shrinkage does not add defensive personnel,
coverage, season changes or causal interpretation, and it does not establish that an
opponent block transfers to a later season.

## Opponent priors chosen before outer evaluation

The inner passing search may cross separate opponent penalties with context and player
penalties. A context without defending-team terms has no opponent parameter and is fitted
once per player-penalty pair, not once for every irrelevant grid value. Explicit opponent
grids must contain unique finite positive values. Selected no-opponent contexts report
null, while omission of the option preserves the original shared-context selection API.
Nonconverged candidates remain in the evidence but cannot win.

Exact ties retain the existing preference for stronger player shrinkage and the earlier
declared context, then prefer stronger opponent shrinkage within that context. No outer
loss is used. This is selection on one development split, not proof of the best prior or
of transferable defensive quality; grid-edge choices require sensitivity checks.

## One selected environment prior across player comparisons

A validated study can search opponent shrinkage independently, then carry the winning
prior unchanged into every outer context and player fit. The context-only fit uses an
empty player block so it can retain the same environment prior. Player-bootstrap refits
also retain it and report it explicitly. Contexts lacking opponent terms have no opponent
prior; older protocols omit the new grid and keep their original shared-context behavior.
Optional context diagnostics apply the selected opponent prior when available, otherwise
they retain the context penalty, and do not select a new environment from outer outcomes.

The shrinkage-study example extends the inner candidate set to linear, depth,
depth-plus-position, depth-plus-opponent and fully expanded contexts. Player penalties
remain 10, 100 and 1000; opponent penalties are 1, 100 and 10000, giving 81 distinct
candidates. This is a declared development comparison prompted by inspected earlier
results, not a prospective confirmatory study. All outer player increments remain
conditional on the one jointly selected specification and its static opponent model.

## Numerical thread provenance

The offline runner records a fixed allowlist of numerical thread environment settings
at the start of fitting. These settings can affect execution cost and floating-point
reduction order, so package versions alone do not describe the numerical invocation.
Unset settings remain null. The runner does not alter them or inspect unrelated environment
variables. Recorded requests do not establish active library pool sizes when a caller
has already imported numerical libraries, and they do not promise bitwise reproducibility.
