# Prepared panel contract

## Rows and time

Supply Polars DataFrames with one player-game row per position (team-game rows for D/ST). Include integer `season` and `week`; fitting uses them for its internal chronological shape-calibration split. Include stable player/team/game identifiers for joining outputs even when they are not selected features.

The models do not construct rosters, fetch data, build lags, determine eligibility, or enforce forecast cutoffs. Retain zero-opportunity eligible players and distinguish inactive, DNP, missing statistics, and bye weeks deliberately. Avoid duplicate player-game rows.

Every selected feature must exist in both training and prediction data and represent knowledge available at the cutoff. Do not select same-game targets, yards, touchdowns, snaps, or routes. The feature adapter recognizes `player_id`, `team`, `opponent`, `roof`, and `surface` as categorical. Other features are converted to numeric; extend that adapter deliberately if adding categorical features. Unknown test categories become missing. Entirely missing/unusable training features are dropped.

The package does not automatically determine whether selected features are available before kickoff. Explicit `feature_columns` are the caller's responsibility; passing a raw outcome as a feature can silently leak the answer.

## Historical target statistics

Training requires every column in the relevant module's `STAT_COLUMNS`. All targets must be observed and appropriately cleaned; do not fill unknown outcomes with zero merely to pass validation. Counts should be integer-valued and nonnegative; yards can be negative. Reconcile receptions with targets and TDs with exposures before fitting.

| Position | Module | Required target columns |
| --- | --- | --- |
| WR | `hierarchical`, order in `common` | targets, receptions, receiving_yards, receiving_tds, carries, rushing_yards, rushing_tds |
| RB | `chains_rb` | carries, rushing_yards, rushing_tds, targets, receptions, receiving_yards, receiving_tds; use module order |
| TE | `chains_te` | targets, receptions, receiving_yards, receiving_tds, carries, rushing_yards, rushing_tds |
| QB | `chains_qb` | attempts, completions, passing_yards, passing_tds, interceptions, carries, rushing_yards, rushing_tds |
| K | `chains_k` | fg_att, fg_made, fg_made_40_49, fg_made_50_59, fg_made_60_, pat_att, pat_made |
| D/ST | `chains_dst` | def_sacks, def_interceptions, def_fumbles, def_tds, def_safeties, blocked_kicks, special_teams_tds, points_allowed |

For kickers, also supply makes and misses for bands `0_19`, `20_29`, `30_39`, `40_49`, `50_59`, and `60_` as `fg_made_<band>` and `fg_missed_<band>`. The inherited code uses pooled-rate fallbacks if miss counters are absent; missing make counters are treated as zero. Those fallbacks can hide incomplete data, so audit the complete band schema instead of relying on them.

`def_fumbles` means recoveries in this model's target definition, not forced fumbles. QB targets and scrimmage targets use different scoring coverage; consult `scoring.py` rather than assuming universal fantasy scoring.

## Calibration

Without a supplied `frozen` shape layer, each model reserves an internal late-history calibration segment, fits calibration-stage means on earlier rows, estimates distribution shapes, then refits means on all training data. Adequate historical observations and positive-exposure examples are needed; tiny or all-zero samples may fail or fall back to priors.

Each position module exposes `estimate_frozen_shapes`; review its signature before use. It estimates from strictly earlier seasons relative to the requested holdout season. Never estimate shapes from the prediction season's future outcomes. Internal calibration may divide rows within a week; use deliberately separated prior-season frozen calibration if whole-week shape separation is required.

## Outputs and interpretation

`fitted.sample(test, n_draws=..., rng=...)` returns raw statistic draws shaped `(rows, draws, len(STAT_COLUMNS))` in that module's order. Store column names and row identifiers alongside arrays. `scoring.scrimmage(alpha)` accepts named columns, so it can score RB arrays in RB order without positional assumptions.

The example saves identifiers/time, stat names, stat draws, and three point-scoring arrays. It does not save a trained-model checkpoint. No draws imply coherent cross-player dependence until the team-level plan is implemented.

The metrics operate on sample arrays shaped `(rows, draws)` and observed vectors shaped `(rows,)`. CRPS measures distribution quality; MAE of the sample mean measures average absolute point error. Injury scenario evaluation must distinguish known-state diagnostic forecasts from actual forecasts made under uncertain pregame availability.
