# Hierarchical Football

Hierarchical parametric NFL models for simulating player statistics and fantasy-point distributions, with a roadmap for team opportunity allocation and injury scenarios.

## What is included

- Existing WR, RB, TE, QB, kicker, and D/ST conditional stat distribution models.
- Local feature-matrix construction, scoring functions, forecast metrics, and clustered uncertainty helpers.
- A runnable WR example that accepts prepared training and prediction panels.
- The [team opportunity plan](docs/TEAM_OPPORTUNITY_PLAN.md), model design, and historical comparison results.
- A play-level [inference research layer](docs/INFERENCE_LAYER.md): canonical tables built from cached
  play-by-play, coverage and pairing audits, regularized baselines with per-block pooling, clustered
  holdout evaluation, standardized contrasts, and the team-game receiving panels with candidate rosters
  that the opportunity models read.
- The [policy decisions](docs/DECISIONS.md) behind each behaviour, and a
  [project file inventory](PROJECT_FILES.json) of the tracked tree, kept honest by a check in
  `tests/test_repository_hygiene.py`.

The package contains model source code and supporting utilities. Train the models on prepared historical panels before generating forecasts.

## Install

From this folder, use either:

```powershell
uv sync
```

or, with `uv`'s own installer inside the environment it manages:

```powershell
uv pip install -e .
```

A `uv`-managed virtual environment has no `pip` module, so `python -m pip install -e .` fails there.
Use it only in a non-`uv` environment that already provides `pip`.

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

The position classes share a small gain law, `GainMixture` (one copy per position module). `sample_gains` draws per-opportunity gains and returns the offset of each group; `sample_totals` sums one group per row and returns zero for a group of size zero, which is what a caller with an empty group needs. Other public classes are `RBChain`, `TEChain`, `QBChain`, `KickerChain`, and `DefenseChain`. Fit one position at a time. Scoring functions encode the research targets documented here; they do not cover every platform rule, bonus, or rare scoring category.

## Research direction and current limits

Start with [TEAM_OPPORTUNITY_PLAN.md](docs/TEAM_OPPORTUNITY_PLAN.md). It proposes a team target pool and roster-wide allocation feeding the existing conditional production stages. Team allocation, weekly availability mixtures, external opportunity injection, and coherent QB/receiver outcomes are future work, not implemented features in this package.

Current samplers produce player distributions. They do not form a team-constrained or game-wide joint model. Preserve the documented concerns about tail calibration, TE touchdown dependence, QB volume bias, and D/ST tiers. The comparison reports record historical research results. Their benchmark models, evaluation runners, and datasets are not included; the reports are reference material rather than runnable benchmarks.

## Folder map

```text
prepare_inference_tables.py            Root entry point for the inference extractor
src/hierarchical_football/             Position models, shared support code, and the data package
src/hierarchical_football/data/        Raw ingest, roster history, panels, splits, and reports
src/hierarchical_football/inference/   Canonical play tables, audits, baselines, pooling, contrasts
examples/predict_wr.py                 Prepared-panel training and prediction example
tests/                                 Offline checks, one module per unit under test
docs/DATA_CONTRACT.md                  Input/output requirements and data limitations
docs/DECISIONS.md                      Policy records, one per behaviour change
docs/PIPELINE_VALIDATION.md            Pipeline verification record
docs/INFERENCE_LAYER.md                Module-by-module guide to the inference layer
docs/INFERENCE_DATA_AUDIT.md           Observed counts of the first extraction
docs/INFERENCE_BASELINE_AUDIT.md       Observed metrics of the first baseline fit
docs/INFERENCE_MODEL_DEVELOPMENT.md    Penalty, pooling, and block evidence
docs/TEAM_OPPORTUNITY_PLAN.md          Future team/injury research roadmap
docs/*RESULTS.md                       Historical comparison snapshots
PROJECT_FILES.json                     Tracked file inventory
pyproject.toml, uv.lock                Package configuration and the resolved environment
```

Data ingestion and feature generation are the caller's responsibility. Start with the prepared-panel example and the data contract, then follow the team opportunity plan for future extensions.

## Repository layout and ignored paths

Tracked source lives in `src/hierarchical_football/`, including the `data` subpackage that ingests raw feeds and writes prepared panels, and the `inference` subpackage that prepares the play-level tables of the research program. Everything a pipeline run generates stays untracked: `/data/` (raw cache, processed panels, week splits, forecasts), plus `artifacts/`, `*.parquet`, `*.npz`, `*.pkl`, whatever scratch directory the local ignore rule names, and `.venv/`.

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

## Inference research layer

`src/hierarchical_football/inference/` is the play-level layer of the research program: canonical tables built from the cached play-by-play, audits of what those tables support, nested baseline estimators with per-block pooling, chronological holdout evaluation with game-clustered intervals, and standardized contrasts under a reference population.

```powershell
uv run python -m hierarchical_football.inference prepare --seasons 2022 2023 2024 2025 --raw-root data/raw --output-root data/processed/inference --rusher-scope backfield
```

The command reads only what is already cached and prints one line of JSON. [docs/INFERENCE_LAYER.md](docs/INFERENCE_LAYER.md) holds the module-by-module contracts, the meaning of every report field, the estimator and contrast policy, and the checks to run for each module.

Three documents record what the first runs observed, each with the inputs it read named by hash:

| Document | Record |
| --- | --- |
| [INFERENCE_DATA_AUDIT.md](docs/INFERENCE_DATA_AUDIT.md) | Rows, exclusions, missing-value coverage, and pairing components of the four-season extraction |
| [INFERENCE_BASELINE_AUDIT.md](docs/INFERENCE_BASELINE_AUDIT.md) | Metrics and clustered intervals of the first baseline fit, at one shared penalty |
| [INFERENCE_MODEL_DEVELOPMENT.md](docs/INFERENCE_MODEL_DEVELOPMENT.md) | The penalty study, per-block pooling, the block ablation, and the corrected reading of what transfers |

Run the focused checks with `python -m unittest tests.test_inference_contracts tests.test_inference_passing tests.test_inference_rushing tests.test_inference_audits tests.test_inference_build tests.test_inference_cli tests.test_prepare_script tests.test_inference_baselines tests.test_inference_evaluation tests.test_inference_partial_pooling tests.test_inference_design -q`.

## Expanded passing context

`fit_completion(..., context="expanded")` uses a fixed piecewise-linear target-depth basis
with knots at 0, 5, 10, 20, and 30 yards, plus position and defending-team intercepts.
`passing_design` accepts the same option. Training scaling and category levels are reused
on later rows; new categories contribute zero. Unknown context names raise `ValueError`.
The default `linear` context preserves the historical model. These are target-conditional
associations, including direction and depth observed at the throw, not pregame features.

Run the focused checks with `python -m unittest tests.test_inference_context -q`.

## Separate passing identity penalties

`fit_block_completion(table, identity_penalties={"qb_id": 100, "receiver_id": 100},
context="expanded")` fits independent QB and receiver priors with a separate context penalty.
`select_passing_penalties` evaluates their Cartesian grid and declared context candidates
on an inner chronological split, returning every score and solver status. Only converged
fits can win; invalid penalties, missing blocks, and empty grids raise `ValueError`.
Keep its validation season earlier than the season used for final comparison.

Run the focused checks with `python -m unittest tests.test_inference_block_pooling -q`.

## Passing partnership transfer

`evaluation.passing_transfer(train, holdout, predictions=...)` scores five disjoint
cohorts: known pairs, new pairs of known players, unseen QBs, unseen receivers, and both
unseen. It returns row assignments, cohort counts, completion metrics, and paired
game-clustered loss comparisons. Empty cohorts have no metrics; fewer than two games
have no interval, and cohorts below ten games are flagged. Nonchronological seasons,
invalid probabilities, and misaligned prediction lengths raise `ValueError`.

Run the focused checks with `python -m unittest tests.test_inference_transfer -q`.

## Conditional player-contrast uncertainty

`design.bootstrap_contrast(train, population, identity="receiver_id", left=..., right=...,
fixed={"qb_id": ...}, identity_penalties=...)` refits completion models on resampled
training games. It reports a percentile interval conditional on fixed penalties, context,
and reference population, plus context support and failures. At least two games and twenty
draws are required. Unobserved identities raise `ValueError`; disconnected partnerships,
unsupported reference contexts, or any incomplete resample suppress the interval with an
explicit status. These intervals do not quantify causal identification or tuning uncertainty.

Run the focused checks with `python -m unittest tests.test_inference_contrast_uncertainty -q`.

## Chronological passing studies

`study.run_passing_study(table, spec=PassingStudySpec(...))` composes inner selection,
expanding later-season fits, five model ablations, calibration, pairing audits, and
partnership-transfer comparisons. It returns a report plus keyed row predictions.
Selection happens once; each outer fit uses only declared earlier seasons. Single-block
ablations use the selected block penalty, not a separately optimized ablation.
Optional preselected contrasts use training targets as their fixed reference population.

The frozen specification requires increasing, disjoint training/selection/evaluation
seasons, positive finite penalties, and at least twenty bootstrap draws. Reserved seasons
cannot overlap any requested season. Undeclared rows are filtered before validation;
missing requested seasons and nonconverged outer fits fail. Reports always state
`purpose="development"` and `untouched_holdout_verified=false`: exclusion alone cannot
certify a season's inspection history.

Run the focused checks with `python -m unittest tests.test_inference_study -q`.

## Frozen passing-study files

Run the offline workflow with a JSON protocol:

```powershell
python -m hierarchical_football.inference study-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_study.json --output-root data/experiments/passing_2024_2025
```

`study_io.run_study_file(input_path, specification_path, output_root)` validates the
versioned specification before creating a new output directory. Unknown fields, duplicate
JSON keys, invalid contrasts, nonfinite values, and existing output paths are refused.
The example uses already-inspected development seasons; it reserves no untouched data.
Add preselected comparisons to `contrasts`, for example
`{"identity":"receiver_id","left":"PLAYER_A","right":"PLAYER_B","fixed":{"qb_id":"QB_ID"},"population_position":"WR"}`,
using actual IDs observed in every fold's training set.

A complete run contains `report.json`, keyed `predictions.parquet` with a metadata sidecar,
an exact copy of `specification.json`, and `manifest.json` written last. The manifest records
input/specification/output hashes, package-source hashes, Git revision when available,
Python and numerical-library versions. Source, input, or protocol changes during fitting
fail the run. Failed runs retain `failure.json` without a complete manifest and require a
new output path for retry. No files are downloaded and no trained forecast model is replaced.

Run the focused checks with `python -m unittest tests.test_inference_study_io -q`.

The [first reproducible passing study](docs/PASSING_STUDY_RESULTS.md) records the cached
2022-2025 run: expanded context is selected, and pooled player identities add a small
completion-loss improvement, including on new partnerships. Receiver-specific transfer
beyond the QB model remains uncertain. These are development findings, not player grades.
