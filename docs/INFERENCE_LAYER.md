# Inference research layer

`src/hierarchical_football/inference/` is the play-level layer of the research program. It turns
locally cached play-by-play into two canonical tables, audits what those tables can support, fits the
nested regularized baselines the plan declares, and puts clustered intervals on the comparisons. This
guide is the module-by-module contract: what each module promises, what each report field means, and
which checks hold it.

## Contents

- [What the layer is for](#what-the-layer-is-for)
- [Map of the layer](#map-of-the-layer)
- [Inference table contracts](#inference-table-contracts)
- [Inference table preparation](#inference-table-preparation)
- [Rushing attempt preparation](#rushing-attempt-preparation)
- [Inference table audits](#inference-table-audits)
- [Writing the inference tables](#writing-the-inference-tables)
- [Inference extractor command](#inference-extractor-command)
- [First-stage extractor script](#first-stage-extractor-script)
- [Team-game receiving panels](#team-game-receiving-panels)
- [Chronological team-game splits](#chronological-team-game-splits)
- [Team volume baselines](#team-volume-baselines)
- [Team-effect volume model](#team-effect-volume-model)
- [Baseline estimators](#baseline-estimators)
- [Baseline evaluation](#baseline-evaluation)
- [Partial pooling for identified players](#partial-pooling-for-identified-players)
- [Standardized contrasts and support](#standardized-contrasts-and-support)
- [Records of the first runs](#records-of-the-first-runs)
- [Reproducible passing studies](#reproducible-passing-studies)
- [Supported historical player comparisons](#supported-historical-player-comparisons)
- [What the layer does not do yet](#what-the-layer-does-not-do-yet)

## What the layer is for

The layer answers three questions about play-level outcomes, in this order.

1. **Conditional associations.** What does a passer, a receiver, or a runner do in the pre-play contexts
   he actually faced? The answer is a model term: an adjusted log-odds deviation for a completion, an
   adjusted-yard deviation for a carry. It is a statistical component, not a share of credit and not
   portable talent.
2. **Standardized contrasts.** What does the fitted model predict for two players under one shared
   distribution of contexts? Every contrast travels with the overlap between that distribution and what
   each player actually faced, and a player swap is never described as an intervention.
3. **Stability and transfer.** Do terms estimated on older seasons survive into later ones, and into
   pairings and blocking environments that never appeared together? The penalty study and the block
   ablation in [INFERENCE_MODEL_DEVELOPMENT.md](INFERENCE_MODEL_DEVELOPMENT.md) are the first answers.

The layer stops there. Mechanisms - target earning, placement, contested conversion, lane creation - are
named as future work and are not identified by these two tables, so no causal claim is made anywhere in
this layer, and no term is ever presented as a ranking of players.

The three questions come from a research plan drafted alongside this work and kept outside the repository,
which is why several records in `docs/` cite "the plan": they mean these questions and the decisions taken
to answer them. Where the plan and this guide disagree, the guide is the description of the code that
exists, and `docs/DECISIONS.md` records the policy behind each choice.

## Map of the layer

| Module | Responsibility | Focused checks |
| --- | --- | --- |
| `contracts.py` | Canonical columns, kinds, labels, and the read-only validator every consumer runs first | `tests.test_inference_contracts` |
| `prepare.py` | Cached play-by-play to `passing_targets` and `rushing_plays`, with a counted reason for every excluded play | `tests.test_inference_passing`, `tests.test_inference_rushing`, `tests.test_inference_build` |
| `audits.py` | Coverage, null shares, and the connectedness of the passer-receiver pairing graph | `tests.test_inference_audits` |
| `baselines.py` | Nested context and context-plus-identity baselines, one design builder, one reference frame | `tests.test_inference_baselines` |
| `partial_pooling.py` | A player penalty separate from the context penalty, chosen on an inner chronological split | `tests.test_inference_partial_pooling` |
| `evaluation.py` | Chronological split, completion and yardage metrics, paired game-clustered bootstrap intervals | `tests.test_inference_evaluation` |
| `design.py` | Standardized catch-rate and log-odds contrasts under a reference population, with support reporting | `tests.test_inference_design` |
| `panels.py` | Team-game target totals, available candidates, and the `OTHER` residual | `tests.test_inference_panels` |
| `splits.py` | The week cutoff that separates training team games from the weeks a model predicts | `tests.test_inference_splits` |
| `volume.py` | A pooled count law and a team-history reference for team target volume | `tests.test_inference_volume` |
| `team_effect.py` | A per-team deviation on the log mean, shrunk by a penalty chosen on an inner split | `tests.test_inference_team_effect` |
| `allocation.py` | Conserving joint allocation of supplied team target draws | `tests.test_inference_allocation` |
| `allocation_baseline.py` | Position-pooled proportional shares and matched position-only reference | `tests.test_inference_allocation_baseline` |
| `allocation_evaluation.py` | Exact conditional count likelihood and conserved-draw marginal calibration | `tests.test_inference_allocation_evaluation`, `tests.test_inference_allocation_sample_evaluation` |
| `situation.py` | Audited pre-play formation, tempo, score and clock join | `tests.test_inference_situation` |
| `context_support.py` | Shared model-category and numeric-range reference coverage | `tests.test_inference_context_support` |
| `estimability.py` | Data-only row-space checks for requested contrast arms | `tests.test_inference_estimability` |
| `comparisons.py` | Supported retrospective player contrasts across historical windows | `tests.test_inference_comparisons` |
| `simulation.py` | Synthetic passing scenarios with exact declared contrasts | `tests.test_inference_simulation` |
| `calibration.py` | Repeated known-truth contrast bias, coverage and availability | `tests.test_inference_calibration` |
| `study.py` / `study_io.py` | Chronological development studies and hashed offline bundles | `tests.test_inference_study`, `tests.test_inference_study_io` |
| `__main__.py` | Offline preparation, study, comparison and synthetic calibration commands | `tests.test_inference_cli`, `tests.test_inference_calibration_io` |

Every module is offline. They read the cache under `data/raw`, write under the output root they are
given, and never start a download; a missing partition raises instead.

## Inference table contracts

`hierarchical_football.inference.contracts` declares the two canonical play-level tables of the inferential research layer: `passing_targets` and `rushing_plays`. Each is one row per play with a fixed column list, label set, and key, so that preparation and modeling agree on the frame before any model is fitted.

`validate(frame, contract)` returns the frame unchanged when it satisfies the contract, and raises `ValueError` naming the table and the offending column otherwise. It rejects a missing required column, a column of the wrong kind (`int`, `number`, `text`, `flag`), a null season, week, team, play key, identity, outcome, or position, a completion flag outside `0` and `1`, a label outside the declared set, and a duplicate `(game_id, play_id)` key. It is read-only: it never drops a duplicate row, casts a dtype, or fills a value.

`canonical_position(tag)` maps the roster tags `QB`, `RB`, `HB`, `FB`, `WR`, and `TE` onto canonical labels and returns `UNKNOWN` for everything else, including null. Missing labels stay explicit: `pass_location`, `run_location`, `run_gap`, and `position` hold `UNKNOWN` rather than `null`, so no pass is assigned a side it may not have had and no rusher is assumed to be a running back.

The contract covers required columns only. `PASSING_CONTRACT.optional_columns` and `RUSHING_CONTRACT.optional_columns` list the charted and defensive fields the plan admits - catchability, drop, contested, screen, pressure, yards after catch, box count, personnel, formation. A table without them is valid; their coverage is a data-availability question, not a schema error.

Run the focused checks with `python -m unittest tests.test_inference_contracts -q`.


## Inference table preparation

`hierarchical_football.inference.prepare.passing_targets(pbp, *, positions=None)` turns locally cached play-by-play into the canonical `passing_targets` table and returns `(table, report)`. A kept play is a regular-season pass attempt with a known passer, a known receiver, a known possession team, and a recorded completion of `0` or `1`. Sacks, scrambles, throwaways, two-point attempts, aborted plays, and unidentified plays are removed and counted.

The report is the audit trail for that selection. `report["exclusions"]` maps each reason to the number of plays it removed, `report["exclusions_by_season"]` repeats that per season alongside source and target rows, and `report["missing_context"]` gives null counts and shares for `air_yards`, `yardline_100`, `ydstogo`, and `down`. `report["target_rows"]` plus the exclusions always equals the source row count.

Nothing is imputed and nothing is dropped in silence. An absent or unrecognised `pass_location` becomes `UNKNOWN` instead of a side, and a receiver the roster feed does not identify carries `UNKNOWN` position. A duplicate `(game_id, play_id)` key, a null play key, or an absent source column raises `ValueError` before any row is produced.

`roster_positions(rosters)` reduces a weekly roster feed to one canonical position per player, team, and week. It keeps regular-season rows, maps `HB` onto `RB`, and returns `UNKNOWN` for a tag it does not know. Passing the result as `positions=` fills the `position` column with the as-of-game label; two different positions for one player-week are refused rather than resolved by row order.

The table carries no charted field. Catchability, drop, contested, screen, pressure, and yards after catch arrive only after their per-season coverage is measured.

Run the focused checks with `python -m unittest tests.test_inference_passing -q`.


## Rushing attempt preparation

`hierarchical_football.inference.prepare.rushing_plays(pbp, *, positions=None, rusher_scope="backfield")` turns locally cached play-by-play into the canonical `rushing_plays` table and returns `(table, report)`. A kept play is a designed regular-season carry with a known team and a recorded signed yardage. Kneels, spikes, and scrambles are removed because they are recorded as runs but are not rushing attempts, and signed gains stay signed: a carry for minus three yards is a carry.

`rusher_scope` names whose carries the table is about, because a quarterback carry is a different question from a running back carry:

| Scope | Positions kept |
| --- | --- |
| `backfield` (default) | `RB`, `FB` |
| `non_qb` | `RB`, `FB`, `WR`, `TE` |
| `all` | every identified rusher, including `QB` and an untagged player |

The two narrower scopes need `positions`, and raise `ValueError` without them rather than returning an empty table. A carry outside the scope is counted under `outside_rusher_scope`, like every other exclusion, so `report["target_rows"]` plus the exclusions always equals the source row count. The report also carries `rusher_scope`, `scope_positions`, `missing_context` for `yardline_100`, `ydstogo`, and `down`, and the `UNKNOWN` counts for `run_location` and `run_gap`.

A declared scope label is not a promise that carries exist for it. The weekly roster feed tags no fullback in the four cached seasons (`position` holds `RB` but never `FB` or `HB`), so `backfield` keeps running backs only there, and `report["kept_positions"]` counts what each declared label actually kept. A `FB` count of zero is that fact stated rather than hidden.

Run the focused checks with `python -m unittest tests.test_inference_rushing -q`.


## Inference table audits

`hierarchical_football.inference.audits.table_coverage(table, contract)` reports what a canonical table holds: total rows, rows per season, the null count and null share of every column, and the optional charted columns that are actually present. `present_optional_columns` is empty for a table prepared without a charting join, which is how a reader can tell that catchability was never attached.

`pairing_support(table)` reports the evidence behind a passer-receiver separation. It counts passers, receivers, and distinct pairings, the pairings per receiver, the receivers who appear with exactly one quarterback, and the connected components of the pairing graph with the component that holds the most rows. Two components mean the graph does not connect the passing units: an effect estimated inside one of them rests on its own players and no crossover.

Both functions validate the table first, so an audit never describes a frame that preparation would have rejected. Neither function writes, deduplicates, or repairs anything.

Run the focused checks with `python -m unittest tests.test_inference_audits -q`.


## Writing the inference tables

`hierarchical_football.inference.prepare.build(raw, output, seasons, *, rusher_scope="backfield")` reads the cached `pbp` partitions under `raw`, joins cached `rosters_weekly` positions when they exist, and writes the canonical tables and their audits:

| Output | Contents |
| --- | --- |
| `<output>/passing_targets.parquet` | the canonical targeted-pass table, with a `.json` sidecar carrying rows, schema, and content hash |
| `<output>/rushing_plays.parquet` | the canonical rushing attempt table, sidecar included |
| `<output>/prepare_report.json` | the input manifest, both preparation reports, and the coverage and pairing audits |

`prepare_report.json` names every cached partition the tables were built from, with its `sha256`, so a fitted result can be traced to the exact inputs. The report holds `passing.preparation`, `passing.coverage`, `passing.pairing_support`, and `rushing.preparation` and `rushing.coverage`.

Reading is local only. `build` never starts a download, so a season without a cached `pbp` partition raises `FileNotFoundError` rather than quietly producing a smaller table. Without cached rosters, a position-scoped rushing table raises `ValueError` and the caller can either cache rosters or ask for `rusher_scope="all"`.

Run the focused checks with `python -m unittest tests.test_inference_build -q`.


## Inference extractor command

```powershell
uv run python -m hierarchical_football.inference prepare --seasons 2022 2023 2024 2025 --raw-root data/raw --output-root data/processed/inference --rusher-scope backfield
```

The `prepare` command builds both canonical tables from the cached partitions under `--raw-root` and prints a one-line JSON summary: `passing_targets` and `rushing_plays` row counts, the number of input partitions named in the manifest, and the total number of excluded plays across both tables. `--rusher-scope` accepts `backfield`, `non_qb`, or `all`.

The command reads only what is already cached. It has no refresh flag, so a missing `pbp` partition raises `FileNotFoundError` and exits non-zero instead of starting a download; use `python -m hierarchical_football.data ingest --seasons 2022 2023 2024 2025 --tables pbp rosters_weekly` first.

Run the focused checks with `python -m unittest tests.test_inference_cli -q`.


## First-stage extractor script

```powershell
uv run python prepare_inference_tables.py --raw-root data/raw --output-root data/processed/inference --seasons 2022 2023 2024 2025
```

The repository-root script is the command line the research plan documents. It takes the same four options as `python -m hierarchical_football.inference prepare`, calls the same `build` function, and prints the same one-line JSON summary. It requires the package to be importable, so run it with `uv run` from the repository root.

Run the focused checks with `python -m unittest tests.test_prepare_script -q`.


## Team-game receiving panels

`hierarchical_football.inference.panels` materializes the unit of observation the two opportunity
models share. The volume model asks how many targets a team's offense generates in a game; the
allocation model asks how that total splits across the receivers who were available. One row per team
game carries the first answer, one row per available candidate carries the second, and the `OTHER`
residual carries whatever the candidates do not explain.

`team_game_panel(passing, pbp, rosters)` returns `(panel, candidates)`. The panel has one row per
regular-season team game in the cached play-by-play, with `team`, `opponent`, `is_home`, `game_date`,
the recorded `targets`, the number of `candidates`, the `candidate_targets` they drew, and the
`other_targets` that went to nobody on the candidate list. The candidates frame has one row per
available candidate with a `candidate_index`, his `player_id`, his canonical `position`, and the
`targets` he drew.

`game_context(pbp)` reduces play-by-play to one row per regular-season game, and refuses a game whose
plays disagree about their week or date. `team_games(passing, games)` builds the team-game rows from
the canonical passing table, so a team game whose offense recorded no in-scope target keeps an observed
zero rather than a null, and a team game the cache does not contain gets no row at all. It refuses a
passing row whose game or week the games feed contradicts. `available_players(rosters)` keeps the
weekly roster's active rows with a modelled position: `WR`, `RB`, or `TE`. `candidates(passing,
rosters, panel)` joins the two and assigns the index. `conservation(panel, candidates)` returns the
split as a report and raises on any violation.

`OTHER` is a category, not an error term. A lineman reporting eligible, a quarterback catching a
deflection, or a receiver the roster feed does not place on the team that week draws a real target that
no candidate owns. Across the four cached seasons that is 149 targets of 68,408 (0.22%) in 104 team
games, and the allocation model has to carry it or the pool will not close.

The candidate index orders players by `player_id` inside each team game. It never orders them by
targets, because an index sorted by the label would put the answer in the row key.

Run the focused checks with `python -m unittest tests.test_inference_panels -q`.

## Chronological team-game splits

`hierarchical_football.inference.splits.team_game_split(panel, candidates, *, train_through)` cuts the
panel at one week into three bundles. `train` holds every team game at or before the cutoff with its
candidate rows and their targets; `predict` holds every team game after it with its candidate rows and
no label of any kind; `observed` holds the withheld labels for exactly the predicted team games. The
`manifest` names the cutoff, both week ranges, the row counts, and the label columns that were withheld.

The ordering key is `(season, week)`, not the week number, because week 18 of one season is older than
week 1 of the next. A cutoff is therefore a season and a week together.

A team game is the prediction unit, so both teams of a game and every teammate of a team game land on
the same side of the cutoff. The split refuses a panel whose game straddles the cutoff, a cutoff with no
training history, a cutoff with nothing left to predict, a cutoff whose team games have no candidates,
a team game that appears twice, and a candidate whose team game is not in the panel.

The prediction bundle drops the outcome columns rather than trusting a caller not to read them:
`targets`, `candidate_targets`, `other_targets`, and `candidates` are absent from it and returned in the
observed bundle instead. That is the mechanical half of the plan's as-of gate; the other half is that no
feature built from these bundles may be computed from a later week.

Run the focused checks with `python -m unittest tests.test_inference_splits -q`.

## Team volume baselines

`hierarchical_football.inference.volume` fits the comparison the team-volume model has to beat.
`fit_negative_binomial(counts)` returns a mean and a dispersion by maximum likelihood, with the variance
written `mu + mu**2 / k` so the two parameters are not confused with another library's. A sample without
spread is refused, because a dispersion cannot be identified from a constant.

`sample_negative_binomial(means, dispersion, *, draws, rng)` turns means into a `(rows, draws)` array of
counts, and `volume_scores(samples, observed)` reduces that array to MAE, CRPS, and 50% and 90% central
interval coverage and width. `team_means(panel, *, window=17)` gives each team its mean over its most
recent training team games. `volume_comparison(train_panel, predict_panel, observed_panel, *, draws,
seed, window)` consumes the three bundles a split produces, refuses a prediction panel that still carries
a label or an observed panel about other team games, and scores the pooled law and the team-history
reference on identical rows.

Measured on the cached seasons. The panel's 2,174 team games average 31.47 targets and range from 8 to
62. Cutting at the end of 2023 leaves 1,086 training team games and 1,088 to predict in 2024: the pooled
law fits a mean of 32.03 with dispersion 34.9 and scores CRPS 4.30 and MAE 6.09, against CRPS 4.43 and MAE
6.34 for the team-history reference. Cutting at the end of 2024 leaves 544 team games to predict in 2025,
and there the two baselines land within 0.02 CRPS of each other with the reference marginally ahead. A
team's own volume history is therefore not a reliable improvement over the league's count law in these
seasons, which is the bar a hierarchical volume model has to clear rather than assume.

Run the focused checks with `python -m unittest tests.test_inference_volume -q`.

## Team-effect volume model

`hierarchical_football.inference.team_effect` adds the plan's first real hierarchy: a negative binomial
whose log mean is a league intercept plus one deviation per team, with a ridge penalty on the deviations.
`fit_team_effect(panel, *, penalty)` returns the intercept, the deviations, the dispersion, the team order,
and whether the inner fit converged. `team_effect_means(fit, panel)` gives a mean per row and lets a team
the model has never seen fall back to the intercept. `count_log_score(means, dispersion, observed)` is the
proper score used for selection, so a penalty can be chosen without Monte Carlo noise.

`select_penalty(train_panel, *, penalties, inner_through)` fits on the training weeks at or before an inner
cutoff, scores the weeks after it, and returns the winning penalty with the whole table. The penalty is
never chosen on the rows the model is then scored on. `team_effect_comparison(train_panel, predict_panel,
observed_panel, *, inner_through, ...)` selects the penalty, refits on all the training weeks, and scores
the model beside the pooled law and the team-history reference on the same predicted team games. The
default grid runs from no shrinkage to 10,000, so an answer at the edge means the search needs another
order of magnitude rather than another grid.

Measured on the cached seasons, and the honest reading is a wash. Cutting at the end of 2023 and
predicting 2024, the inner split chooses a penalty of 100; the effect deviations have a standard deviation
of 0.078 and the model scores CRPS 4.42 against 4.30 for the pooled law, so it loses. Cutting at the end of
2024 and predicting 2025, the split chooses 1,000, the deviations shrink to 0.025, and the model scores
CRPS 4.27 against 4.29, so it wins by 0.02. The model asks for less team structure the more history it has,
which is what weak and unstable team effects look like rather than a hierarchy that needs a posterior to
show itself.

Run the focused checks with `python -m unittest tests.test_inference_team_effect -q`.

## Baseline estimators

`hierarchical_football.inference.baselines` fits the nested regularized baselines the research plan
declares, one pair per canonical table: context alone, then context plus identified players. Nothing
is split, sampled, or selected here. The caller declares the training seasons and the later season to
score, and the two baselines are fitted the same way on the same rows so that the difference between
them is the identity increment rather than an artifact of two unrelated models.

`fit_completion(table, *, identities=(), composites=None, penalty=1.0)` fits a logistic baseline for
`complete_pass`. The context is the context the contract and the plan declare: target depth, field
position, distance to go, down, and charted pass location - and
`fit_completion(table, identities=("qb_id", "receiver_id"))` adds one L2-penalized one-hot block per
identified passer and receiver. Nothing derived from the outcome is admitted, so a fitted probability is
a conversion rate under a context, not receiver talent: a receiver who earns targets and converts them at
the same rate as his peers is invisible to this model by construction. One caveat travels with that
sentence: `pass_location` is charted at the throw rather than before it, so it is the one context column
that describes what the passer did rather than what he faced. It is admitted because the plan names the
direction of the pass as part of the context; everything the outcome itself produces - the completion,
yards after the catch, catchability, a drop, a contested catch - is not.

`fit_rushing(table, *, identities=(), support=False, composites=None, penalty=1.0)` fits a ridge
baseline for signed `rushing_yards`. `support=True` adds the `team | run_location | run_gap`
composition as one column, which is the plan's support environment, and `identities=("rusher_id",)`
adds the runner. That support column is a team, a scheme, a blocking unit, and an opponent at once,
which is why it is named an environment and never a blocker grade. A squared penalty on a skewed
outcome is a first mean model; the plan leaves robust, mixture, and quantile models open.

`passing_design` and `rushing_design` build the design matrix on their own, and passing an earlier
design as `reference=` reuses its columns, its standardization, and its level sets on a later frame. A
category the reference never saw contributes a zero row and is counted in
`design.notes["unseen_categorical_rows"]`; a null continuous value is filled with the reference median
and counted in `notes["filled_continuous"]`, with its own indicator column when the reference frame
had nulls at all. A player the reference never saw gets no column and is counted in
`notes["unseen_identity_rows"]`, so a later season never invents a coefficient for a newcomer.

`FittedBaseline.identity_terms()` returns one entry per identified player: the fitted term, the rows
behind it, its share of the table, and `nested`, which is `True` when the player only ever appeared
inside one level of the first composite column and cannot therefore be separated from that
environment. Entries are ordered by identity column and player name, never by term size, because an
order by size would read as a ranking of players. The L2 penalty centres the terms of one block on
zero, so a term is a deviation from the average identified player, and a player with thin evidence is
pulled back toward that average instead of being handed a rate of his own.

`report()` returns the fit facts: rows, features, the penalty, the level counts, the block penalties
when the fit split them, and `in_sample` error. The in-sample error is named that way on purpose: it is
there to catch a broken fit, and a model with a column per player fits the rows it memorised by
construction. `in_sample` also carries `converged` and `iterations`. A fit that stopped at the iteration
cap is still returned - the weak end of a penalty grid has to stay measurable - and it warns with a
`RuntimeWarning` when the cap was reached, because a coefficient that is not at an optimum is only useful
evidence if the reader knows that. The cap is 5000 iterations rather than the solver's default, since the
identity designs at the weakest penalties need more than a thousand on a full season.

Run the focused checks with `python -m unittest tests.test_inference_baselines -q`.


## Baseline evaluation

`hierarchical_football.inference.evaluation` scores a fitted baseline on a later season and puts an
interval on the comparison. `chronological_split(table, *, train_seasons, holdout_season)` returns the
earlier rows to fit and the later season to score, and refuses a holdout season that is not strictly
later, an empty side, no named training season, or a table the contract would reject.

Plays are clustered inside games, so every interval here resamples whole clusters instead of rows.
`clustered_metric(losses, *, clusters)` gives the percentile interval of one model's mean row loss, and
`bootstrap_difference(first, second, *, clusters)` gives the paired interval of the loss difference: a
positive estimate means the first model is worse, and an interval that stays above zero is evidence
against "no difference" in the losses. `clustered_comparison(losses, *, clusters, reference=...)` reports
every model's interval plus each model's paired difference against the reference, which is the
context-only baseline unless the caller names another.

`row_losses(observed, predicted, *, task)` produces the per-row losses that both intervals consume: the
log loss of a completion, or the absolute error of a carry. `completion_metrics` adds the Brier score, a
reliability table, and the share of rows punished with a log loss of two or more; `yardage_metrics` adds
RMSE, the median absolute error, and the share of residuals at ten yards or more. A model that is never
badly wrong is not the same as a model that is badly wrong a tenth of the time, and the mean alone cannot
tell them apart.

The estimand is a row-level average loss over a season drawn from the same games, which is why a drawn
cluster contributes all of its rows and the mean is taken over rows rather than over cluster means: games
do not carry the same number of plays. The cluster count travels with every interval, because a
percentile bootstrap over very few clusters is narrow: a handful of distinct resamples cannot reproduce
the sampling spread of a handful of games.

Run the focused checks with `python -m unittest tests.test_inference_evaluation -q`.


## Partial pooling for identified players

`hierarchical_football.inference.partial_pooling` gives the identity block its own penalty. In the
plain baselines one penalty covers every coefficient, so a prior strong enough to control 800 player
columns also flattens target depth and field position, which are the columns the model is supposed to
believe. `fit_pooled_completion(..., identity_penalty=..., context_penalty=...)` and
`fit_pooled_rushing(..., identity_penalty=..., context_penalty=...)` return the ordinary
`FittedBaseline`, so prediction, `identity_terms`, and the evaluation module work unchanged.

The implementation is a rescaling rather than a new solver. The identity columns are multiplied by
`sqrt(context_penalty / identity_penalty)` before the fit, and their fitted coefficients are multiplied
by the same factor afterwards. The linear predictor is unchanged, so the fitted object keeps the
ordinary design and the ordinary predict path, and equal block penalties reproduce the ordinary fit
exactly - which the tests assert rather than assume.

`select_identity_penalty(table, *, identities, task, train_seasons, validation_season, ...)` chooses the
player penalty on an inner chronological split: every grid point is scored on the same later season by
the primary metric, and ties go to the larger penalty so an uninformative grid lands on more shrinkage.
`scores` keeps every grid value, because a flat grid and a peaked one are different evidence. The season
that is later reported is never the season the penalty is chosen on.

A penalty here is a variance component, not a probability of skill. The fit records both penalties and
the column scale in `report()["block_penalties"]`.

Run the focused checks with `python -m unittest tests.test_inference_partial_pooling -q`.


## Standardized contrasts and support

`hierarchical_football.inference.design` answers the plan's second question: not what a player did in the
contexts he saw, but what the fitted model predicts for two players under one shared population of
pre-play contexts Q. `standardized_prediction(fitted, population, *, levels=..., scale=...)` averages the
fitted value over Q with any identity level forced onto every row, and
`standardized_contrast(fitted, population, *, identity, left, right, fixed=...)` reports the difference of
the two arms with the fixed levels recorded.

Two scales are offered because the plan names two quantities: `scale="probability"` is the catch rate
under Q, and `scale="log_odds"` is the adjusted log-odds deviation. They are not interchangeable.
Averaging a sigmoid is not the sigmoid of an average, so on the probability scale a fixed partner changes
a receiver contrast even in an additive model, while on the log-odds scale it does not.

Every contrast travels with its support. `context_cells` groups rows into comparable situations, and
`player_support(fitting_table, population, *, identity, player, ...)` reports the share of Q that a
player's own rows cover, the share they cover well, and the share they never faced at all, with the
unsupported cells named. `standardized_prediction` marks a player with no fitted column at all in
`unseen_identities`: his arm is the average identified player rather than a term of his own.

`contrast_sensitivity(fitter, population, ..., penalties=...)` refits across penalties and reports the
range of the contrast. It is labelled a penalty sensitivity range rather than a confidence interval,
because it measures one modelling choice and not sampling error.

Run the focused checks with `python -m unittest tests.test_inference_design -q`.

## Records of the first runs

| Document | What it records |
| --- | --- |
| [INFERENCE_DATA_AUDIT.md](INFERENCE_DATA_AUDIT.md) | Observed rows, exclusions, missing-value coverage, and pairing components of the four-season extraction, over the partitions it names by hash |
| [INFERENCE_BASELINE_AUDIT.md](INFERENCE_BASELINE_AUDIT.md) | Metrics and clustered intervals of the first baseline fit at one shared penalty, with a correction note for its superseded headline |
| [INFERENCE_MODEL_DEVELOPMENT.md](INFERENCE_MODEL_DEVELOPMENT.md) | The penalty study, per-block pooling, the block ablation, the grid extension, and the current reading of what transfers between seasons |
| [PASSING_STUDY_RESULTS.md](PASSING_STUDY_RESULTS.md) | The frozen CLI study with expanded context, separate identity penalties, partnership transfer, and secondary receiver-increment diagnostics |
| [PASSING_CONTEXT_RESULTS.md](PASSING_CONTEXT_RESULTS.md) | Context-component diagnostics plus standard QB/receiver conditional increments, with the original predictions preserved |
| [PASSING_SHRINKAGE_RESULTS.md](PASSING_SHRINKAGE_RESULTS.md) | Earlier-season selection over simpler contexts and independent opponent priors, with later-season player increments |
| [PASSING_PLAYER_COMPARISON.md](PASSING_PLAYER_COMPARISON.md) | Team-scoped retrospective player contrasts, shared population retention, prior sensitivity and a non-estimable example |
| [PASSING_CONTEXT_SUPPORT_RESULTS.md](PASSING_CONTEXT_SUPPORT_RESULTS.md) | Sensitivity to observed model-context coverage and its loss during resampling |
| [PASSING_CALIBRATION_RESULTS.md](PASSING_CALIBRATION_RESULTS.md) | Known-truth shrinkage bias and conditional interval coverage stress tests |
| [PASSING_ESTIMABILITY_RESULTS.md](PASSING_ESTIMABILITY_RESULTS.md) | Original-design log-odds and probability-arm estimability on the frozen Philadelphia references |
| [PASSING_SITUATION_RESULTS.md](PASSING_SITUATION_RESULTS.md) | Matched development prediction comparison from adding observed pre-play formation, tempo, score and clock |
| [TARGET_ALLOCATION_RESULTS.md](TARGET_ALLOCATION_RESULTS.md) | Conditional player-history allocation increments over position rates, exact conservation and count-calibration limits |

These are evidence from local runs rather than test expectations. Each names its inputs by hash, and a
refresh of the cache can move the numbers.

## What the layer does not do yet

- No predeclared untouched holdout: 2024 and 2025 have both been scored during development, so no
  forecasting claim should be read out of the records.
- No charting join: catchability, drops, contested catches, pressure, and defensive box count are not
  attached, and their per-season coverage is unmeasured.
- No participation join, and no reconciliation of derived game and team counts against a weekly table.
- No PyMC posterior: the team-effect volume model is a penalized maximum-likelihood fit, which is the
  frequentist cousin of the plan's hierarchy rather than its posterior, and player terms are regularized
  point estimates. Selected contrasts support conditional game-bootstrap intervals with fixed
  penalties and Q, plus separate penalty sensitivity; these are not skill or identification intervals.
  The layer never turns a term into a ranking. The team-game panel that the opportunity
  models need is built and checked for conservation, and the pooled, team-history, and team-effect volume
  models are fitted and scored on it. Conditional allocation now has a joint sampler and a fitted
  proportional baseline with a position-only reference; conditional development value has been
  measured under supplied rosters and observed totals, while operational forecast value is unverified.
  Learned substitution remains future work. The team effect is not promoted: it loses one season
  and wins the next by a twentieth of a CRPS point.
- No notebooks: the plan's marimo prototypes are not built and `marimo` is not a dependency.
  Passing studies now have a CLI and frozen JSON specifications. Earlier records in `docs/`
  were produced by scratch drivers; new runs can use the workflow below.
- The rushing support environment is a team, a scheme, a line, and an opponent at once, so it is named an
  environment and never a blocker grade; the completion model cannot see route or target selection.

## Reproducible passing studies

`python -m hierarchical_football.inference study-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_study.json --output-root data/experiments/passing_2024_2025`

The example freezes a development protocol: train on 2022, choose the passing specification
on 2023, refit on 2022-2023 for 2024, then 2022-2024 for 2025. Selection is not repeated on
the outer seasons. It compares the historical linear context to an optional expanded depth
basis with position and defending-team intercepts, choosing separate QB and receiver penalties
on the inner split. QB-only and receiver-only ablations retain their jointly selected penalties.

`study.PassingStudySpec` validates chronological seasons, penalties, contexts, uncertainty
settings and any reserved seasons. `study.run_passing_study` returns a report and keyed
predictions. `study_io.run_study_file` validates the versioned JSON wrapper and persists
the exact protocol, report, predictions, metadata and a complete manifest with source,
input, specification and output hashes plus runtime versions. Existing output paths fail;
an exception writes `failure.json` instead of a complete manifest. The runner is offline.

Each fold records convergence, calibration and tails, paired game-clustered loss differences,
training partnership support, and five disjoint transfer cohorts. A new partnership means
two known players who were never paired in training; unseen players are separate cohorts.
Groups below two games get no loss interval, and groups below ten are flagged.

`identity_increments` also compares the joint model against QB-only and receiver-only,
both overall and within each nonempty cohort. This separates receiver information beyond
QBs from joint-versus-context gains. Increments remain comparisons of refitted blocks at
jointly selected penalties, not individual skill measurements.

The separate `context_diagnostics` specification field enables context-only comparisons
without changing the selection grid. `examples/passing_context_study.json` declares all
eight combinations of nonlinear depth, position and opponent indicators. Each fold records
their fit reports, scores, and twelve matched one-component increments. Negative richer-minus-
simpler loss differences favor the added component. Diagnostic probabilities are retained
with the scored play keys. These overlapping comparisons are exploratory; they are not
additive causal feature attributions or a second selection on the outer seasons.

`opponent_penalties` optionally declares an independent prior grid for defending-team
terms. Contexts without those terms are evaluated once per player-penalty pair. The
selected prior is then shared by context-only, single-player-block, joint and contrast
refits. Existing protocols omit the grid and retain their shared context penalty.
`examples/passing_shrinkage_study.json` admits simpler depth contexts during inner
selection and evaluates 81 distinct context/player/opponent combinations. No outer
score selects a new specification. Static opponent effects still do not represent a
season-aware defense or causal player control.

Optional contrasts specify player IDs, a fixed partner, scale and reference position.
Their Q is the fold's training targets, never its scored targets. `design.bootstrap_contrast`
refits on sampled games with fixed penalties; disconnected components, unsupported context
cells, or any incomplete resample suppress the conditional interval with an explicit status.
Connection does not establish identification. No Bayesian posterior or causal credit is claimed.

The protocol always reports development status and `untouched_holdout_verified=false`.
Declaring a reserved season excludes it from computation; establishing that it has never
been inspected still requires a prospective research record outside this runner.

Run `python -m unittest tests.test_inference_context tests.test_inference_block_pooling tests.test_inference_transfer tests.test_inference_contrast_uncertainty tests.test_inference_study tests.test_inference_study_io -q`.

## Supported historical player comparisons

`compare-passing` consumes a versioned `comparison` object such as
`examples/passing_player_comparison.json`. It fits the declared team's targeted passes
in disjoint historical windows, then compares two players with a fixed partner under
one observed Q from the earliest window. Shared depth/down/direction cells must meet
the protocol exposure threshold for both joint arms in every period. The retained
share and all counts are reported; no shared cells yields no player estimates.

The result includes per-window conditional game-bootstrap intervals and a separate
QB/receiver penalty sensitivity grid. It records reference categories unseen in a
later model and descriptive sign checks. It does not estimate an interval for change
between windows. The team-local coefficients do not identify portable player talent.
Later exposures can restrict Q, so these reports are retrospective and are explicitly
marked `forecast_evaluation=false`.

`reference_policy="model_context"` optionally restricts the shared player-cell population
to categories and numeric source ranges represented in every original fitting window.
The default coarse policy preserves the earlier comparison and reports the same domain
audit without applying it. Final `reference_rows` and `reference_share` describe Q after
the chosen policy; earlier coarse exposure counts remain in `support`. No final rows
yields `no_shared_model_context` and no estimates. This is marginal coverage rather than
joint positivity, and the restriction changes the estimand.

The offline file runner saves the exact specification, report, reference population and
metadata through the common hashed bundle writer. A complete manifest means the workflow
finished; inspect the report status to see whether a contrast was estimable. The paired
examples include Philadelphia WRs Brown/Smith with Hurts and Baltimore TEs Andrews/Likely
with Jackson, selected without fitted outcome rankings.

Run `python -m unittest tests.test_inference_shared_population tests.test_inference_comparisons tests.test_inference_comparison_io -q`.

The [reference-context sensitivity results](PASSING_CONTEXT_SUPPORT_RESULTS.md)
compare the coarse and model-context policies on the same Philadelphia fits.
The stricter reference retains 152 of 687 original targets; point contrasts
change little, while 199/200 early and 197/200 later game refits still lose
reference context. These diagnostics motivate known-truth simulation before
claiming calibrated uncertainty for player skill.

## Known-truth contrast uncertainty checkpoint

The [synthetic calibration results](PASSING_CALIBRATION_RESULTS.md) compare
30 datasets per scenario at fixed player penalties of 1, 100 and 1000. In the
fully crossed scenario, 27/30 weak-penalty intervals contain the exact contrast,
versus 0/30 at each stronger penalty. A connected but depth-confounded design
also yields 0/30 at every penalty despite full bootstrap completion and coarse
support. These controlled results reveal shrinkage bias and structural aliasing;
they do not calibrate real-player uncertainty or choose a production prior.

## Repeating known-truth calibration

Run `python -m hierarchical_football.inference calibrate-passing --spec
examples/passing_calibration_study.json --output-root data/experiments/passing_calibration_new`
(on one command line). This reads no external NFL input. The frozen synthetic protocol
saves trial parquet, report, metadata and a complete-last manifest with source hashes and
runtime settings. Existing outputs are refused. The example's 30 datasets and 50 draws
per interval are a diagnostic setting, with finite Monte Carlo bounds reported explicitly;
interpret the first study in [PASSING_CALIBRATION_RESULTS.md](PASSING_CALIBRATION_RESULTS.md).

The [original-design estimability evidence](PASSING_ESTIMABILITY_RESULTS.md) separates
log-odds contrast estimability from probability-arm estimability on the frozen Philadelphia
reference populations. Its point estimates reproduce, and the stricter Q passes both
original-window arm checks; neither result certifies resampling support or skill coverage.

## Optional observed pre-play situation

Both `prepare_inference_tables.py` and the package `prepare` command accept
`--include-passing-situation`. The explicit option attaches shotgun, no-huddle,
start-of-play score differential and game seconds remaining after canonical target
selection and roster matching, using the audited source-key join. The build report
and passing sidecar record the option and enabled reports include missing-context
counts. It changes passing columns only; original target rows and outcomes remain.
Use an output such as `data/processed/inference_situation` to preserve frozen inputs.
This preparation option alone does not change model specifications or establish causal
player attribution. Run `python -m unittest tests.test_inference_situation_preparation -q`.

## Completion context with observed situation

The explicit `expanded_situation` variant extends expanded depth/position/opponent context
with shotgun/no-huddle categories and standardized linear pre-play score/clock terms.
It requires the optional enriched preparation input and validates fields in fitting and
prediction. The shared raw-source schema includes all four fields in model-context
reference coverage. Existing contexts and frozen protocols are unchanged. Chronological
selection and context diagnostics may declare the variant; its component increment is
relative to expanded context. These are target-conditional associations, not causal credit.
Run `python -m unittest tests.test_inference_situation_model -q`.

## Joint target allocation primitive

`allocation.sample_target_allocation` conserves each supplied integer team/draw total
across candidate targets and an explicit OTHER residual, retaining row/draw identities.
It consumes label-free prediction rosters with declared probabilities, validates pool
closure and samples jointly. Team/game and player-ID sorting removes arbitrary row-order
effects. This establishes the conservation gate for target allocation, while shares,
availability and volume remain supplied inputs. No learned allocation or pregame roster
timestamp certification is claimed. Run `python -m unittest tests.test_inference_allocation -q`.

## Proportional receiving allocation baseline

`allocation_baseline.fit_proportional_allocation` estimates positive position-pooled rates
from historical targets per eligible team-target exposure. Its predictor normalizes rates
over a supplied future WR/RB/TE roster and OTHER, returning keys and probability arrays for
the joint allocation sampler. Predictions exclude raw labels, follow the last training
week and cannot reuse training game IDs. Counts and roster metadata are validated.
Removing a candidate implements proportional redistribution. This is a fixed-share
conditional baseline, with no learned availability/substitution response or certified
pregame roster cutoff. It is not yet a demonstrated forecasting improvement.
Run `python -m unittest tests.test_inference_allocation_baseline -q`.

The fitted share predictor's `include_player_history=False` option retains its position
and OTHER rates while removing the individual historical contribution. This is a matched
role-only reference for future chronological allocation evaluation, with unchanged fitted
population and roster assumptions. Default predictions continue to use individual history.

## Conditional allocation likelihood evaluation

`allocation_evaluation.allocation_log_scores` checks exact prediction/observation keys
and scores the candidate/OTHER multinomial count vector conditional on each observed team
total. It returns keyed team-game likelihood scores and game-mean/target-weighted summaries.
Impossible positive-count/zero-probability events retain infinite scores and force explicit
infinite-loss/null aggregate reporting. Failed cases are never clipped or dropped. These
are conditional allocation diagnostics, not volume or availability forecasting evidence.
Run `python -m unittest tests.test_inference_allocation_evaluation -q`.

## Conditional allocation sample calibration

`conditional_allocation_sample_scores` scores keyed candidate/OTHER draw matrices only
after every draw conserves its observed team target total. It retains zero-target candidates,
reports separate MAE/CRPS and central 50%/90% coverage/width, and returns row scores for
further diagnostics. Empty candidate summaries are null. This explicitly conditional
evaluation does not measure volume or availability forecast quality or certify nominal
coverage. Run `python -m unittest tests.test_inference_allocation_sample_evaluation -q`.

## Conditional share variation sampler

`sample_target_allocation(..., concentration=A)` adds Dirichlet-multinomial allocation
variation with unchanged mean shares, structural zeros and exact per-draw team totals.
None preserves the original fixed-share multinomial samples, including seed compatibility.
Positive finite A is reported explicitly. This is declared predictive share variation,
not a fitted parameter posterior or availability model. Earlier-data concentration selection
and later likelihood/calibration evaluation remain separate requirements.
Run `python -m unittest tests.test_inference_allocation_dispersion -q`.
