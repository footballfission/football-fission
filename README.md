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

The current play-level work adds [observed situation context](docs/PASSING_SITUATION_RESULTS.md)
with small completion-loss gains, strongest in the 2024 development fold. It also includes
[known-truth interval calibration](docs/PASSING_CALIBRATION_RESULTS.md) and
[player-contrast estimability audits](docs/PASSING_ESTIMABILITY_RESULTS.md). These distinguish
conditional estimator uncertainty from uncertainty about player skill.

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

The [context sensitivity experiment](docs/PASSING_CONTEXT_SUPPORT_RESULTS.md) finds
nearly unchanged Philadelphia point contrasts after restricting Q, but frequent
opponent-category loss during game resampling. Interval completion does not certify
context coverage or calibrated uncertainty about player skill.

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

## Passing context components

`fit_completion(..., context=...)` and `passing_design` accept `depth`, `position`,
`opponent`, `depth_position`, `depth_opponent`, and `position_opponent`, alongside
`linear` and `expanded`. All retain linear depth and the original situational columns;
`depth` adds the fixed nonlinear hinges, `position` adds receiver position, and `opponent`
adds defending-team intercepts. `expanded` includes all three. Names and reference designs
retain the declared combination. The historical defaults and expanded predictions are unchanged.

Run the focused checks with `python -m unittest tests.test_inference_context -q`.

## Conditional identity increments

`evaluation.identity_increments(losses, clusters=...)` compares `both` against `qb`
(receiver information beyond the QB model) and against `receiver` (QB information beyond
receivers). Negative loss differences favor adding the block. Inputs must contain finite,
row-aligned losses for all three models and one game key per row. Fewer than two games
produce no interval. Passing studies include these comparisons overall and within every
nonempty transfer cohort; generic transfer calls without all three model keys report null.

Run the focused checks with `python -m unittest tests.test_inference_transfer tests.test_inference_study -q`.

## Passing context diagnostics

`PassingStudySpec(context_diagnostics=(...))` declares context-only diagnostics separate
from the model-selection candidates in `contexts`. The default empty tuple adds no fits.
When enabled, the linear reference is included automatically. Each fold records metrics,
fit reports, paired loss differences against linear context, and every declared pair that
differs by one component. Saved predictions include `p_context_diagnostic_<context>`.
Unknown or duplicate diagnostic names and nonconverged diagnostic fits fail the run.

`examples/passing_context_study.json` preserves the original inner selection grid while
adding all eight diagnostics and standard conditional identity increments. Run it with
the same `study-passing` command and a fresh output directory. Outer diagnostics do not
select a new specification, establish feature importance percentages, or certify causality.

Run the focused checks with `python -m unittest tests.test_inference_study -q`.

The [context-component study](docs/PASSING_CONTEXT_RESULTS.md) finds that nonlinear
depth supplies the consistent context gain; position adds little measurable value and
static opponent effects are unstable. Receiver information beyond QBs remains uncertain
on new partnerships. The study preserves the original model predictions and does not
promote a diagnostic winner from inspected seasons.

## Separate opponent shrinkage

`fit_block_completion(..., opponent_penalty=...)` gives defending-team intercepts a
finite positive penalty independent of player and situational effects. An explicit
opponent penalty requires a context containing `opponent`. Omission preserves the earlier
shared-context behavior. Reports record the effective extra block penalty and fit name.
`identities=(), identity_penalties={}` fits context only with the same opponent prior,
so a player comparison need not change its environment regularization.

Run the focused checks with `python -m unittest tests.test_inference_block_pooling -q`.

## Opponent-prior selection

`select_passing_penalties(..., opponent_penalties=(1, 100, 10000))` extends the inner
chronological search with independent opponent shrinkage. Contexts without opponent
terms are evaluated once per player-penalty pair; their selected opponent penalty is
null. Opponent contexts evaluate every declared opponent value. Invalid or duplicate
grid values fail. Omitting this argument preserves the historical search and output keys.
All candidate losses and convergence flags remain visible; later seasons do not enter selection.

Run the focused checks with `python -m unittest tests.test_inference_block_pooling -q`.

## Studies with matched opponent priors

`PassingStudySpec(opponent_penalties=(...))` enables the independent opponent grid through
the file runner. The selected prior is reused for context-only, QB-only, receiver-only,
joint fits, and conditional contrast refits. Context diagnostics use it where they include
opponent terms; when selection has no opponent block, diagnostics retain the context penalty.
Omitting the field preserves older protocols. Invalid grids fail specification validation.

`examples/passing_shrinkage_study.json` admits linear and simpler depth combinations with
the original player grid and an opponent grid of 1, 100 and 10000: 81 inner candidates.
Selection still fits 2022 and scores 2023 before expanding evaluations in 2024 and 2025.
No diagnostic or outer loss chooses the context or prior.

Run the focused checks with `python -m unittest tests.test_inference_study tests.test_inference_contrast_uncertainty -q`.

## Numerical runtime provenance

Study manifests record an allowlist of numerical thread environment settings alongside
package versions. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and `NUMEXPR_NUM_THREADS` are captured at
run start; unset values are null. These are requested environment settings, not verified
active thread-pool sizes. Other environment variables are not copied into the manifest.

Run the focused checks with `python -m unittest tests.test_inference_study_io -q`.

The [independent-shrinkage study](docs/PASSING_SHRINKAGE_RESULTS.md) admits simpler contexts
but selects expanded context with opponent penalty 100 on the earlier validation season.
Overall completion loss improves slightly in both scored seasons; receiver-specific transfer
on new partnerships remains uncertain. The finding is development evidence.

## Shared passing reference populations

`design.shared_passing_population(windows, population, identity=..., left=..., right=...,
fixed=..., min_rows=5)` keeps observed reference rows only in depth/down/pass-direction
cells supported by both player arms in every named fitting window. A fixed partner is
counted jointly with the player. It returns the retained population plus exposure counts,
cell definitions and the retained share. No shared cells produces a typed empty population
and explicit status. Outcomes do not choose cells. Invalid identities, thresholds or
empty reference populations raise `ValueError`.

Run the focused checks with `python -m unittest tests.test_inference_shared_population -q`.

## Retrospective player comparisons

`comparisons.run_passing_comparison(table, spec=PassingComparisonSpec(...))` compares two
players with one fixed partner across disjoint historical windows on one shared observed
population. Fits use the declared team's targeted passes; Q starts with the earliest
window, optionally restricted by receiver position, then passes the joint-support screen.
Fixed priors and a context penalty of 1 are reused across periods. Each window reports a
conditional game-bootstrap interval and a separate QB/receiver prior sensitivity grid.

The specification validates windows, IDs, priors, support and uncertainty settings.
Missing declared team seasons fail. No shared support returns no estimates. Failed or
incomplete fits retain explicit status; sensitivity failures suppress the range. The report
records context categories absent in a later design. Direction checks are descriptive,
with no interval for the between-window change and no claim of stable talent.

Run the focused checks with `python -m unittest tests.test_inference_comparisons -q`.

## Frozen player-comparison files

```powershell
python -m hierarchical_football.inference compare-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_player_comparison.json --output-root data/experiments/player_comparison
```

`study_io.run_comparison_file` validates the versioned `comparison` object and persists
`report.json`, the exact protocol, `reference_population.parquet` with its metadata, and
a complete manifest. Existing paths, unknown fields and invalid protocols fail before
fitting. Workflow failures retain `failure.json`. A valid comparison with no shared support
still produces a complete artifact bundle containing an explicit non-estimable report and
an empty typed reference frame. It produces no player estimates or forecast predictions.

The examples preselect Brown versus Smith with Hurts (Philadelphia WR targets) and
Andrews versus Likely with Jackson (Baltimore TE targets), using 2022-2023 and 2024-2025
windows. They are team-scoped retrospective examples, not league-wide talent comparisons.
Reference construction and bootstrap arm reports use the same declared support threshold.

Run the focused checks with `python -m unittest tests.test_inference_comparison_io tests.test_inference_comparisons -q`.

The [first supported player comparisons](docs/PASSING_PLAYER_COMPARISON.md) demonstrate
both an estimable but uncertain Brown/Smith contrast on a narrow shared target mix and
an Andrews/Likely case with no shared support under the declared rule. These retrospective
examples expose prior sensitivity and context gaps; they are not player grades.

## Reference model-context support

`context_support.model_context_population(windows, population, context=...)` retains
reference rows whose modeled categories occur and numeric source values fall inside
observed ranges in every fitting window. It audits down, direction and declared optional
position/opponent categories plus depth, field position and distance. Missing/nonfinite
numeric values fail the screen. Counts, observed domains and retained share are reported;
per-column failure counts can overlap. Unknown context names and empty fitting windows fail.
This is marginal domain coverage, not joint density or player-specific causal support.

Run the focused checks with `python -m unittest tests.test_inference_context_support -q`.

## Comparison reference policy

`PassingComparisonSpec(reference_policy="model_context")` applies marginal model-context
coverage after the joint player-cell screen and before any estimation. Its final reference
must have categories and numeric ranges represented in every fitting window. No retained
rows yields `no_shared_model_context` and no player estimates. The default `coarse` policy
preserves prior behavior while still reporting the model-context audit. `reference_rows`
and `reference_share` describe the actual final Q; `support` retains the earlier coarse
screen's counts, and `context_support` reports the second-stage audit.

`examples/passing_player_context_comparison.json` declares the stricter Philadelphia
comparison without changing its priors, windows, or exposure threshold. Restricting Q
changes the comparison population; it does not demonstrate that player ability changed.

Run the focused checks with `python -m unittest tests.test_inference_comparisons -q`.

## Contrast resampling context coverage

`bootstrap_contrast` reports the original reference model-context audit plus diagnostics
for converged eligible bootstrap refits: evaluated draws, draws with context failures,
maximum failed reference rows, mean failed row share, and per-column counts of draws
with unseen categories, numeric extrapolation or nonfinite reference values. Counts can
overlap. These diagnostics preserve the regularized estimator and its interval; they do
not silently reinterpret a completed bootstrap as complete covariate coverage.

Run the focused checks with `python -m unittest tests.test_inference_contrast_uncertainty -q`.

## Known-truth passing simulation

`simulation.simulate_passing(games=20, repeats=4, seed=0, scenario="crossed")`
generates canonical synthetic targets, an outcome-independent two-depth Q and its exact
receiver probability contrast with a fixed QB. The additive logistic generating model
is returned explicitly. `nested` isolates disconnected partnerships; `depth_confounded`
crosses partners while aliasing receiver and depth inside a shared coarse cell. All player
and game IDs are synthetic. Independent Bernoulli outcomes have no latent game effects;
these controlled scenarios do not validate NFL talent estimates.

Run `python -m unittest tests.test_inference_simulation -q`.

## Conditional contrast calibration

`calibration.run_calibration(CalibrationSpec(...))` repeats known-truth synthetic
passing datasets across declared scenarios and fixed player penalties. Each dataset and
game-resampling seed is shared across penalties. The result includes a trial table,
exact DGP contrasts, point-estimate bias/RMSE, interval availability, conditional coverage,
mean width and Wilson 95% bounds for Monte Carlo coverage uncertainty. Both identity
blocks receive the declared penalty; the linear context penalty is 1 and Q stays fixed.

Unavailable intervals and single-outcome trials stay in the denominator. Coverage among
available intervals and the share of all trials returning an interval containing truth
are separate fields. Missing point estimates also remain visible. Neither nominal interval
level nor a favorable toy result establishes real-player uncertainty calibration.

Run `python -m unittest tests.test_inference_calibration -q`.

The [known-truth calibration experiment](docs/PASSING_CALIBRATION_RESULTS.md)
shows strong-shrinkage intervals missing the DGP contrast despite complete
resampling support, plus a connected but depth-confounded failure case. These
results document a limit of conditional estimator intervals, not NFL player
coverage or a reason to select priors from this toy experiment.

## Data-only contrast estimability

`estimability.audit_contrast_estimability(fitted, population, identity=..., left=...,
right=..., fixed=...)` checks requested predictors against the training design's row
space, including its intercept. It reports numerical rank, nullity, per-arm failed rows,
and whether the mean log-odds contrast is estimable without prior information. Redundant
dummy columns alone do not invalidate a contrast. Probability-arm checks are a sufficient
condition for the probability contrast, distinct from a linear log-odds difference.

The audit ignores fitted coefficients and penalties. Passing establishes model-based
structural estimability, not causal identification, precision, context density, or interval
coverage. `prior_dependent` exposes a restriction supplied by regularization; it does not
make the regularized point estimate undefined. Numerical tolerance is recorded explicitly.

Run `python -m unittest tests.test_inference_estimability -q`.

## Contrast structural estimability diagnostics

`bootstrap_contrast` includes `estimability` for the original fitting design and the
requested reference arms, including reports returned early for disconnected players or
unsupported cells. A completed conditional bootstrap can coexist with `prior_dependent`
structural predictions. The audit does not remove draws, alter point estimates or redefine
the estimator's conditional interval. Inspect both diagnostics and the bootstrap status;
`status="ok"` alone does not establish data-only estimability or player-skill coverage.

Run `python -m unittest tests.test_inference_contrast_uncertainty -q`.

## Frozen synthetic calibration bundles

`study_io.run_calibration_file(specification_path, output_root)` accepts schema version 1
and a `calibration` object containing `CalibrationSpec` fields. It saves exact protocol
bytes, report, trial parquet and metadata through the shared non-overwriting bundle writer.
The manifest declares `input={"kind":"synthetic"}`: the generating code and protocol are
the input provenance. Source and specification changes during a run prevent completion;
errors leave `failure.json`. Unknown fields, duplicate keys and invalid protocols fail
before creating outputs. A completed study can validly report no available intervals.

Run `python -m unittest tests.test_inference_calibration_io -q`.

## Passing calibration command

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m hierarchical_football.inference calibrate-passing --spec examples/passing_calibration_study.json --output-root data/experiments/passing_calibration_new
```

The command runs a frozen synthetic protocol without an external data input. The example
matches the first calibration study's design: 30 datasets per scenario, 50 bootstrap
refits, three priors and three support scenarios. It takes several minutes. Set thread
environment variables before Python starts. Use a fresh output directory for each run.
Reports, per-trial results and provenance are saved before a complete manifest is written.
A complete manifest does not mean intervals are calibrated; inspect coverage, availability,
Monte Carlo bounds and the declared DGP. The console explicitly marks synthetic runs.

Run `python -m unittest tests.test_inference_calibration_io -q`.

## Calibration estimability diagnostics

Calibration trial tables retain original-design estimability status, log-odds estimability
and probability-arm estimability. Summaries count these statuses separately from bootstrap
completion. A depth-confounded scenario can return every conditional interval while all
contrasts are structurally prior dependent. Single-outcome trials have no fitted audit and
are explicitly `not_evaluated`, which never counts as passing. These fields do not change
simulated outcomes, refits, point estimates, intervals or coverage denominators.

The [original-design Philadelphia audit](docs/PASSING_ESTIMABILITY_RESULTS.md)
finds the later coarse probability arms prior dependent on 76 reference rows, while
the stricter Q passes both original-window arm checks. The log-odds contrast passes
in both cases. Passing structural checks does not remove shrinkage bias or establish
skill-interval coverage.

## Comparison interpretation summary

Historical comparison summaries place `uncertainty` beside descriptive direction checks.
It declares the conditional regularized-estimator interval target, leaves skill coverage
unverified, and reports original-design estimability on the requested scale plus each
window's interval status and refit context-failure count. Probability checks require arm
predictor estimability; log-odds checks use the average linear contrast. An `ok` interval
status can coexist with prior dependence or context loss. No estimates or intervals change.

## Audited passing situation join

`situation.attach_passing_situation(targets, pbp)` returns the same canonical target rows,
order, original columns and outcomes with shotgun, no-huddle, start-of-play score
differential and game seconds remaining attached. It also returns counts and missing
context. Source keys must be unique and integral, and every target must match the source
season and offense. Existing situation columns cannot be overwritten. Binary flags are
nullable Int8; score and clock are nullable Float64. Nulls are preserved; nonfinite observed
values, invalid flags and negative clocks fail. Only the declared fields are selected.

Definitions follow [nflfastR](https://nflfastr.com/reference/fast_scraper.html).
These observed pre-play controls are associations, not causal credit. This join alone
does not change any existing model or frozen experiment.

Run `python -m unittest tests.test_inference_situation -q`.

## Optional passing situation preparation

`passing_targets(..., include_situation=True)` and `build(..., include_situation=True)`
attach the audited four-field pre-play join after the existing target selection and roster
join. Both preparation entry points accept `--include-passing-situation`, for example:

```powershell
python -m hierarchical_football.inference prepare --seasons 2022 2023 2024 2025 --raw-root data/raw --output-root data/processed/inference_situation --include-passing-situation
```

The default passing columns and target population stay the same. The preparation report
and passing metadata record the option; enabled reports include source matches and missing
situation counts. Rushing columns stay the same. Requested source fields must exist and
pass the audited join. Use a separate output directory to retain earlier frozen datasets.

Run `python -m unittest tests.test_inference_situation_preparation tests.test_prepare_script -q`.

## Passing context source schema

`baselines.passing_context_sources(context)` defines raw continuous and categorical
inputs for both completion designs and model-context domain audits. Fixed depth hinges
reuse the air-yards source. This shared definition preserves all existing context column
orders and coverage behavior while preventing future modeled inputs from being omitted
from the reference screen.

Run `python -m unittest tests.test_inference_context_sources -q`.

## Passing situation model variant

`context="expanded_situation"` adds nullable shotgun/no-huddle categorical terms and
linear score/clock terms to the existing expanded completion context. Its four fields
must be explicitly prepared; missing fields and invalid observed values fail in fitting
and prediction. Null numeric values use the existing training-median design behavior
with reported fills; flags use the existing declared unknown category. Training scaling
and levels are reused for later predictions. Existing context variants stay the same.

Model-context coverage uses the same source schema, so the new flags, score and clock
enter reference category/range checks. The variant is available for declared chronological
selection and context-only diagnostics, including its increment over expanded context.
It models associations conditional on a target, not causal formation effects or player
skill uncertainty. No predictive improvement on NFL data is claimed from its synthetic test.

Run `python -m unittest tests.test_inference_situation_model tests.test_inference_study -q`.

## Frozen passing situation evaluation

`examples/passing_situation_reference_study.json` and
`examples/passing_situation_study.json` declare matched development studies on the same
`data/processed/inference_situation/passing_targets.parquet` input. They differ only in
expanded versus expanded_situation context. Player and opponent penalties are fixed at
100, context penalty at 1, with 2022 training, 2023 inner scoring and expanding 2024/2025
folds. No priors are retuned. Both use 2,000 game-bootstrap draws, seed 0 and 95% intervals.

The predeclared primary comparison is new-minus-reference joint-model completion log
loss; negative favors added situation. Context-only and identity-transfer differences
are secondary diagnostics. The target population is unchanged. These previously inspected
seasons are development evidence, with no untouched-holdout or player-skill claim.
Run each with `study-passing --input ... --spec ... --output-root <fresh-directory>`.
