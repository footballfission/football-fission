# Hierarchical Football

Hierarchical parametric NFL models for simulating player statistics and fantasy-point distributions, with a roadmap for team opportunity allocation and injury scenarios.

## What is included

- Existing WR, RB, TE, QB, kicker, and D/ST conditional stat distribution models.
- Local feature-matrix construction, scoring functions, forecast metrics, and clustered uncertainty helpers.
- A runnable WR example that accepts prepared training and prediction panels.
- The [team opportunity plan](docs/TEAM_OPPORTUNITY_PLAN.md), model design, and historical comparison results.
- A [project file inventory](PROJECT_FILES.json) describing the package layout.

The package contains model source code and supporting utilities. Train the models on prepared historical panels before generating forecasts.

## Install

From this folder, use either:

```powershell
uv sync
```

or, inside your chosen Python environment:

```powershell
python -m pip install -e .
```

Python 3.10 or newer is required. Dependencies are declared in `pyproject.toml`. Dependency versions are lower bounds rather than a reproduction lock. Record a lockfile and environment for future experiments.

## Run with prepared data

```powershell
uv run python examples/predict_wr.py --train data/wr_train.parquet --predict data/wr_next_week.parquet --features lag_targets lag_receptions team opponent week --output artifacts/wr_draws.npz
```

Use actual pregame feature names from your panel. The names above are illustrative, not a feature-building pipeline. Training requires historical outcomes; prediction requires only the selected features and time columns. The example verifies chronological separation and rejects raw outcome features, but cannot establish whether arbitrary input features were truly available at the prediction cutoff.

See [DATA_CONTRACT.md](docs/DATA_CONTRACT.md) before preparing inputs. Supply your own training and prediction panels; datasets and fitted weights are not bundled.

## Python API

```python
import numpy as np
from hierarchical_football import HierarchicalParametric
from hierarchical_football.scoring import SCRIMMAGE_COLUMNS, scrimmage

# train and upcoming are prepared Polars DataFrames.
model = HierarchicalParametric(
    feature_columns=("lag_targets", "team", "opponent", "week"),
    td_mode="joint",
)
fitted = model.fit(train)
draws = fitted.sample(upcoming, n_draws=200, rng=np.random.default_rng(0))
full_ppr = scrimmage(1.0)(draws, SCRIMMAGE_COLUMNS)
half_ppr = scrimmage(0.5)(draws, SCRIMMAGE_COLUMNS)
standard = scrimmage(0.0)(draws, SCRIMMAGE_COLUMNS)
```

Draws have shape `(player_rows, draws, statistics)`. Preserve player row order. Use each module's `STAT_COLUMNS` when scoring; the WR order is `common.STAT_COLUMNS`. The WR class default is still `td_mode="separate"`; the example explicitly selects `joint`, to use the per-catch touchdown process. Other position classes have their own options; see their source docstrings.

Other public classes are `RBChain`, `TEChain`, `QBChain`, `KickerChain`, and `DefenseChain`. Fit one position at a time. Scoring functions encode the research targets documented here; they do not cover every platform rule, bonus, or rare scoring category.

## Research direction and current limits

Start with [TEAM_OPPORTUNITY_PLAN.md](docs/TEAM_OPPORTUNITY_PLAN.md). It proposes a team target pool and roster-wide allocation feeding the existing conditional production stages. Team allocation, weekly availability mixtures, external opportunity injection, and coherent QB/receiver outcomes are future work, not implemented features in this package.

Current samplers produce player distributions. They do not form a team-constrained or game-wide joint model. Preserve the documented concerns about tail calibration, TE touchdown dependence, QB volume bias, and D/ST tiers. The comparison reports record historical research results. Their benchmark models, evaluation runners, and datasets are not included; the reports are reference material rather than runnable benchmarks.

## Folder map

```text
src/hierarchical_football/    Existing models and local support code
examples/predict_wr.py       Prepared-panel training and prediction example
docs/DATA_CONTRACT.md        Input/output requirements and data limitations
docs/TEAM_OPPORTUNITY_PLAN.md Future team/injury research roadmap
docs/*RESULTS.md             Historical comparison snapshots
PROJECT_FILES.json           Package file inventory
pyproject.toml              Standalone package configuration
```

Data ingestion and feature generation are the caller's responsibility. Start with the prepared-panel example and the data contract, then follow the team opportunity plan for future extensions.

## Repository layout and ignored paths

Tracked source lives in `src/hierarchical_football/`, including the `data` subpackage that ingests raw feeds and writes prepared panels. Everything a pipeline run generates stays untracked: `/data/` (raw cache, processed panels, week splits, forecasts), plus `artifacts/`, `*.parquet`, `*.npz`, `*.pkl`, `notes/`, and `.venv/`.

The `/data/` rule is anchored to the repository root, so it does not hide `src/hierarchical_football/data/`. It covers every file a run writes under `data/`, including the `.json` manifests and per-file sidecars that record row counts and content hashes; those describe one local run and are not shared inputs.

Run the focused checks with `python -m unittest tests.test_repository_hygiene -q`.

## Raw ingest cache

`hierarchical_football.data.ingest.ingest(raw, seasons, tables, *, refresh=False, loader=None)` stores one Parquet file per table and season at `raw/<table>/season=<year>/part.parquet`, with a `.json` sidecar carrying the row count, schema, and content hash. A cached partition is reused unless `refresh=True`; a partition whose download fails is left exactly as it was, and the run raises `RuntimeError` after writing its report.

`raw/ingest_report.json` accumulates history instead of describing only the last run. `partitions` holds one entry per table and season across every run that touched the cache, with `status` of `downloaded` or `cached`. `runs` appends one summary per invocation with its `started_at`, `seasons`, `tables`, `downloaded` and `cached` counts, and its own `errors`. `started_at` is the first run that wrote the report and `updated_at` is the most recent one. The top-level `errors` list describes only the run that just finished, so a later successful run does not erase an earlier failure from `runs`.

The report describes which partitions this cache holds and when each was fetched. It does not certify that a cached partition was the state of the source at a requested `as_of`; every partition sidecar records `historical_asof_verified: false`.

Run the focused checks with `python -m unittest tests.test_ingest_history -q`.

## Roster extension for unplayed weeks

`assemble(..., extend_rosters=True)` and `build --extend-rosters` fill scheduled team-weeks that have no roster snapshot with each team's most recent earlier snapshot from the same season. The weekly roster feed lags the schedule, so a feed that stops at week 7 still leaves weeks 8-18 in `schedules`.

Every player row carries `roster_carried_forward`. An observed snapshot is `False`; a copied row is `True`, and `roster_present` stays `True` with the copied `roster_status`. The assemble report counts `carried_forward_roster_rows` and `extended_roster_team_weeks`, and `build` records the `extend_rosters` setting in `build_report.json`.

Extension never fills week 1 of a season, never invents a team that has no earlier snapshot, and never overwrites a week that already has one. It does not know about trades, signings, or practice-squad moves made after the copied snapshot.

Run the focused checks with `python -m unittest tests.test_roster_extension -q`.

## Prediction bundle metadata

`split_week` writes `train.parquet`, `prediction.parquet`, `observed.parquet`, `features.json`, and `split_report.json`. The prediction bundle holds identity and timing (`player_id`, `team`, `position`, `season`, `week`, `game_id`, `kickoff`, `prediction_timestamp`), roster context (`roster_status`, `roster_carried_forward`), and the feature list. Outcomes and `fantasy_points` go to `observed.parquet` only.

Roster columns are metadata, not model input: the feature list is filtered by `model_features`, which never admits a roster column, so a bundle cannot leak availability into a fit. Restricting the bundle to `roster_status == "ACT"` is the caller's decision, and it is what turns a roster-wide panel into a list of players who could plausibly be started.

Run the focused checks with `python -m unittest tests.test_data_pipeline -q`.

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

The command reads only what is already cached. It has no refresh flag, so a missing `pbp` partition raises `FileNotFoundError` and exits non-zero instead of starting a download; use `python -m hierarchical_football.data ingest --tables pbp rosters_weekly` first.

Run the focused checks with `python -m unittest tests.test_inference_cli -q`.

## First-stage extractor script

```powershell
uv run python prepare_inference_tables.py --raw-root data/raw --output-root data/processed/inference --seasons 2022 2023 2024 2025
```

The repository-root script is the command line the research plan documents. It takes the same four options as `python -m hierarchical_football.inference prepare`, calls the same `build` function, and prints the same one-line JSON summary. It requires the package to be importable, so run it with `uv run` from the repository root.

Run the focused checks with `python -m unittest tests.test_prepare_script -q`.

## Inference extraction audit

`docs/INFERENCE_DATA_AUDIT.md` records the observed counts from the first four-season extraction: kept rows by season, every exclusion reason with its count, missing-value coverage, and the connected components of the passer-receiver pairing graph. It is evidence from one local run over the cached partitions it names by hash, not a test expectation or a model result.

Run the focused checks with `python -m unittest tests.test_inference_build -q`.

## Baseline estimators

`hierarchical_football.inference.baselines` fits the nested regularized baselines the research plan
declares, one pair per canonical table: context alone, then context plus identified players. Nothing
is split, sampled, or selected here. The caller declares the training seasons and the later season to
score, and the two baselines are fitted the same way on the same rows so that the difference between
them is the identity increment rather than an artifact of two unrelated models.

`fit_completion(table, *, identities=(), composites=None, penalty=1.0)` fits a logistic baseline for
`complete_pass`. The context is the pre-play context the contract declares - target depth, field
position, distance to go, down, and charted pass location - and
`fit_completion(table, identities=("qb_id", "receiver_id"))` adds one L2-penalized one-hot block per
identified passer and receiver. No completion-derived field is admitted, so a fitted probability is a
conversion rate under a context, not receiver talent: a receiver who earns targets and converts them
at the same rate as his peers is invisible to this model by construction.

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

`report()` returns the fit facts: rows, features, the penalty, the level counts, and `in_sample`
error. The in-sample error is named that way on purpose. It is there to catch a broken fit, and a
model with a column per player fits the rows it memorised by construction.

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
against "no difference" in the losses. `clustered_comparison(models, *, clusters, reference=...)` reports
every model's interval plus each model's paired difference against the reference, which is the
context-only baseline unless the caller names another.

`row_losses(observed, predicted, *, task)` produces the per-row losses that both intervals consume: the
log loss of a completion, or the absolute error of a carry. `completion_metrics` adds the Brier score and
a reliability table, and `yardage_metrics` adds RMSE, the median absolute error, and the share of
residuals at ten yards or more, so a model that is never badly wrong is not confused with a model that is
badly wrong a tenth of the time.

The estimand is a row-level average loss over a season drawn from the same games, which is why a drawn
cluster contributes all of its rows and the mean is taken over rows rather than over cluster means: games
do not carry the same number of plays. The cluster count travels with every interval, because a
percentile bootstrap over very few clusters is narrow: a handful of distinct resamples cannot reproduce
the sampling spread of a handful of games.

Run the focused checks with `python -m unittest tests.test_inference_evaluation -q`.

## Baseline fit audit

`docs/INFERENCE_BASELINE_AUDIT.md` records the observed metrics and the paired game-clustered
intervals from the first baseline fit on the cached canonical tables. Its headline finding is negative
and useful: on the 2024 and 2025 splits the passer-receiver identity block does not improve held-out
completion loss, and the support environment and the runner block cost mean absolute error rather than
saving it. It is evidence from one local run over the partitions it names by hash, not a test
expectation.
