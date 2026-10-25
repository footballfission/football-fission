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
