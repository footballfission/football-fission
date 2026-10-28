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
- [Baseline estimators](#baseline-estimators)
- [Baseline evaluation](#baseline-evaluation)
- [Partial pooling for identified players](#partial-pooling-for-identified-players)
- [Standardized contrasts and support](#standardized-contrasts-and-support)
- [Records of the first runs](#records-of-the-first-runs)
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
| `__main__.py` | The `prepare` command over the local cache | `tests.test_inference_cli` |

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

These are evidence from local runs rather than test expectations. Each names its inputs by hash, and a
refresh of the cache can move the numbers.

## What the layer does not do yet

- No predeclared untouched holdout: 2024 and 2025 have both been scored during development, so no
  forecasting claim should be read out of the records.
- No charting join: catchability, drops, contested catches, pressure, and defensive box count are not
  attached, and their per-season coverage is unmeasured.
- No participation join, and no reconciliation of derived game and team counts against a weekly table.
- No posterior layer: player terms are regularized point estimates with a penalty sensitivity range, not
  intervals, and the layer never turns a term into a ranking. The team-game panel that the opportunity
  models need is built and checked for conservation, but no volume and no allocation model is fitted on
  it yet.
- No notebooks: the plan's marimo prototypes are not built and `marimo` is not a dependency, so the
  pooling and contrast stages are library functions with no entry point in the tree. The records in
  `docs/` reached them from scratch drivers, and `tests/test_inference_partial_pooling.py` and
  `tests/test_inference_design.py` are the only callers a reader can find.
- The rushing support environment is a team, a scheme, a line, and an opponent at once, so it is named an
  environment and never a blocker grade; the completion model cannot see route or target selection.
