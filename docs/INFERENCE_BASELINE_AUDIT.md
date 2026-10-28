# First baseline fit and holdout comparison

> **Correction.** The headline below - that the passer-receiver identity block does not improve
> held-out completion loss - holds at the one shared penalty of 1.0 this run used, and it does not
> survive a penalty chosen on an earlier season. `docs/INFERENCE_MODEL_DEVELOPMENT.md` records the
> follow-up: at a penalty selected on 2022 -> 2023 the same block improves both scored seasons with
> intervals below zero, and on the inner season the fit at penalty 1.0 was already scoring worse than
> the context-only model. Everything else in this record stands, including the rushing result, which
> was re-measured at the selected penalty and still holds.

Evidence from one local run of `hierarchical_football.inference.baselines` and
`hierarchical_football.inference.evaluation` over the cached canonical tables. It is not a
test expectation and not the research program's final result.

## Inputs and protocol

| Input | Rows | sha256 |
| --- | --- | --- |
| `passing_targets` | 68,408 | `ebe6f593139df4ca...` |
| `rushing_plays` | 47,256 | `3cba617f38c0b905...` |

- Seasons 2022 and 2023 fit, 2024 is scored; then 2022 to 2024 fit, 2025 is scored.
- Models: an L2 logistic baseline for `complete_pass` (penalty 1.0, tolerance 1e-8) and a
  ridge baseline for signed `rushing_yards` (alpha 1.0). One penalty value, no search.
- Intervals: paired game-clustered percentile bootstrap, cluster = `game_id`, 2000 draws, seed 0.
- Primary metrics: log loss for completion, mean absolute error for rushing. Brier, RMSE,
  the residual tail share, and a reliability table are supporting output.

| Split | Train rows | Holdout rows | Holdout games |
| --- | --- | --- | --- |
| passing 2024 | 34,788 | 17,011 | 272 |
| passing 2025 | 51,799 | 16,609 | 272 |
| rushing 2024 | 23,560 | 11,830 | 272 |
| rushing 2025 | 35,390 | 11,866 | 272 |

## Passing: completion probability

Held-out season 2024:

| Model | Features | Log loss | Brier |
| --- | --- | --- | --- |
| context | 11 | 0.5711 | 0.1925 |
| context+position | 16 | 0.5709 | 0.1924 |
| context+identity | 792 | 0.5749 | 0.1940 |
| context+position+identity | 797 | 0.5748 | 0.1940 |

Paired difference against `context` (positive means a larger loss, so a worse model):

| Model | Log loss difference | 95% interval |
| --- | --- | --- |
| context+position | -0.0001 | [-0.0004, +0.0001] |
| context+identity | +0.0038 | [+0.0018, +0.0059] |
| context+position+identity | +0.0038 | [+0.0017, +0.0059] |

Held-out season 2025:

| Model | Features | Log loss | Brier |
| --- | --- | --- | --- |
| context | 11 | 0.5797 | 0.1964 |
| context+position | 16 | 0.5797 | 0.1964 |
| context+identity | 940 | 0.5816 | 0.1972 |
| context+position+identity | 945 | 0.5818 | 0.1973 |

Paired difference against `context` (positive means a larger loss, so a worse model):

| Model | Log loss difference | 95% interval |
| --- | --- | --- |
| context+position | -0.0000 | [-0.0003, +0.0003] |
| context+identity | +0.0019 | [-0.0001, +0.0038] |
| context+position+identity | +0.0021 | [+0.0001, +0.0041] |

## Rushing: signed yards

Held-out season 2024:

| Model | Features | MAE | RMSE | Share of residuals at 10 yards or more |
| --- | --- | --- | --- | --- |
| context | 14 | 3.6273 | 6.1572 | 0.047 |
| context+support | 247 | 3.6625 | 6.1894 | 0.047 |
| context+support+runner | 434 | 3.7339 | 6.2334 | 0.050 |

Paired difference against `context`:

| Model | MAE difference (yards) | 95% interval |
| --- | --- | --- |
| context+support | +0.0352 | [+0.0217, +0.0487] |
| context+support+runner | +0.1066 | [+0.0834, +0.1318] |

Held-out season 2025:

| Model | Features | MAE | RMSE | Share of residuals at 10 yards or more |
| --- | --- | --- | --- | --- |
| context | 14 | 3.5984 | 6.2477 | 0.047 |
| context+support | 248 | 3.6276 | 6.2669 | 0.047 |
| context+support+runner | 470 | 3.6774 | 6.2861 | 0.048 |

Paired difference against `context`:

| Model | MAE difference (yards) | 95% interval |
| --- | --- | --- |
| context+support | +0.0292 | [+0.0170, +0.0413] |
| context+support+runner | +0.0790 | [+0.0586, +0.0989] |


## What these numbers support

1. On these splits the passer-receiver identity block does not improve held-out completion loss. In
   2024 it costs +0.0038 nats with an interval that stays above zero. In 2025 it costs +0.0019 with an
   interval that touches zero, so that season alone is not decisive, while the same block with the
   position composite costs +0.0021 with an interval just above zero. Both seasons point the same way,
   and the direction is a cost rather than an absence of an effect.
2. The per-position mean is neutral. It moves log loss by -0.0001 and -0.0000 nats with intervals that
   straddle zero, so a position intercept neither helps nor hurts these splits.
3. The support environment and the runner both cost mean absolute error in both seasons, with intervals
   that stay above zero: +0.0352 and +0.0292 yards for the environment, +0.1066 and +0.0790 yards for
   the environment plus the runner. The contextual ridge transfers better than either model that carries
   team or player columns.
4. The residual tail share barely moves for rushing, from 0.047 to 0.050 and 0.048, so the extra error
   is spread through the body of the distribution rather than produced by a few extreme carries.

The reliability table says where the completion model disagrees with the season it is scoring. In 2024
the context baseline reads the 0.6-0.7 bin well and the 0.7-0.8 bin low (0.751 predicted against 0.793
observed on 6,139 targets), it over-predicts the 0.1-0.2 bin (0.164 against 0.242 on 149 targets), and
its extreme bins are thin: 3 rows above 0.9 and 3 below 0.1. Adding identity widens the top bin to 153
rows and mis-reads it (0.914 predicted against 0.791 observed), which is one visible shape of the extra
loss the interval reports.

## What these numbers do not support

- No ranking of players, and no claim that passers or runners do not matter. The identity block is a
  season-to-season one-hot proxy at one penalty. It tests whether those identities carry predictive
  information about the play outcome that survives into a later season, which is a different question
  from how good a player is.
- The completion model sees no target selection, route, or catchability. A receiver who earns targets and
  converts them at the same rate as his peers is invisible to it by construction, which is the plan's own
  warning about selling a conversion model as receiver talent.
- One penalty value, fixed before the run, on both sides. A penalty searched on the holdout, a
  per-position model, a stronger prior, or a hierarchical structure could move these orderings. Nothing
  here selects a winning model: the intervals report a loss difference, and the protocol decides what a
  difference means.
- Seasons 2024 and 2025 were already inspected in `docs/INFERENCE_DATA_AUDIT.md`, so they are model
  development seasons, not the plan's predeclared untouched holdout. That holdout is still to be
  reserved, and no forecasting claim should be made from this table yet.
- Every interval covers the difference in a row-level loss over a season drawn from the same games. None
  of them is an interval for a player's term.
- This is one run over the cached partitions named by hash. A different cached state would move the
  numbers, and the manifest records what was read rather than where it could be fetched again.

## Runner and environment terms, descriptively

The 2022-2023 fit with the support environment and the runner identifies 187 runners with a median of
47 carries, over 233 support combinations. A runner appears with a median of seven of those
combinations, and only 8 runners ever carried behind a single one, which is where the `nested` flag
applies.

The fitted runner terms run from -3.38 to +4.42 yards around a median of +0.10, with the middle 80% of
them inside -1.44 to +1.34 yards. Each term is a shift in expected yards on a carry under the same
context and support environment, and it is the model's own regularized parameter rather than a grade.
The spread is what a reader should notice: it is the size of the block that cost 0.107 yards of MAE out
of season.

## Reproducing this table

The canonical tables are rebuilt from the cache named above:

```powershell
uv run python -m hierarchical_football.inference prepare --seasons 2022 2023 2024 2025 `
  --raw-root data/raw --output-root data/processed/inference --rusher-scope backfield
```

The comparison is then assembled from the committed modules - a chronological split, one fit per model,
per-row losses, and the paired game-clustered comparison:

```python
import polars as pl
from hierarchical_football.inference.baselines import fit_completion
from hierarchical_football.inference.contracts import PASSING_CONTRACT
from hierarchical_football.inference.evaluation import (
    chronological_split, clustered_comparison, completion_metrics, row_losses,
)

targets = pl.read_parquet("data/processed/inference/passing_targets.parquet")
split = chronological_split(targets, train_seasons=[2022, 2023], holdout_season=2024,
                            contract=PASSING_CONTRACT)
observed = split["holdout"]["complete_pass"].to_numpy().astype(float)
losses, metrics = {}, {}
for name, kwargs in (("context", {}), ("context+position", {"composites": {"position": ("position",)}}),
                     ("context+identity", {"identities": ("qb_id", "receiver_id")})):
    probability = fit_completion(split["train"], **kwargs).predict(split["holdout"])
    losses[name] = row_losses(observed, probability, task="completion")
    metrics[name] = completion_metrics(observed, probability)
clustered_comparison(losses, clusters=split["holdout"]["game_id"].to_list(), draws=2000, seed=0)
```

The same steps with `fit_rushing`, `RUSHING_CONTRACT`, `yardage_metrics`, and `task="yards"` produce the
rushing table. The run behind this record took a few minutes on a 12-core Windows box, dominated by the
logistic fits; the bootstrap itself is cheap because a replicate is two inner products over the drawn
clusters.
