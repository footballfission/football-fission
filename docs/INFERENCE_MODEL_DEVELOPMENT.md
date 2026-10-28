# Baseline model development: the penalty and the blocks

Evidence from the model-development runs on this branch. It corrects one headline of
`docs/INFERENCE_BASELINE_AUDIT.md`, and it is still development evidence rather than a forecasting
result: the seasons it scores were inspected during preparation, no untouched holdout has been
reserved, and nothing here ranks players.

## What changed and why

The first fit put one penalty on every coefficient, including 792 one-hot columns for passers and
receivers, and that single penalty was fixed at 1.0. The identity block then cost held-out loss and the
audit recorded that as its headline. Two further questions decide whether that reading survives:

1. Was 1.0 the right amount of shrinkage? A weaker or stronger prior changes how much of the player
   block the model keeps.
2. Should the player block share a penalty with the pre-play context at all? Shrinking 800 player
   columns enough to stay stable out of season also flattens target depth and field position, which are
   the columns the model exists to condition on.

The second question is answered in code: `hierarchical_football.inference.partial_pooling` gives the
identity block its own variance component. The first question is answered by selection on an inner
chronological split.

## Protocol

- Selection: fit season 2022, score season 2023, primary metric only (log loss for completion, mean
  absolute error for carries). This season is never the season that is later reported.
- Grid: 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, extended to 10, 30, 100, 300, 1000, 3000 for the edge check.
- Outer evaluation at the selected penalty: fit 2022-2023, score 2024; fit 2022-2024, score 2025.
- Intervals: paired game-clustered percentile bootstrap, cluster `game_id`, 2000 draws, seed 0,
  272 games per holdout season. A positive difference means the added block carries the larger loss.
- Tables: `passing_targets` (68,408 rows) and `rushing_plays` (47,256 rows), the same cached canonical
  tables named by hash in `docs/INFERENCE_BASELINE_AUDIT.md`.

## One shared penalty, chosen on 2023

The 2023 grid, per model:

| Passing model | 0.03 | 0.1 | 0.3 | 1.0 | 3.0 | 10.0 | 30.0 | 100.0 | own optimum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| context | **0.58171** | 0.58171 | 0.58171 | 0.58171 | 0.58171 | 0.58171 | 0.58173 | 0.58184 | 0.03 |
| context+position | **0.58160** | 0.58160 | 0.58161 | 0.58161 | 0.58163 | 0.58164 | 0.58167 | 0.58179 | 0.03 |
| context+identity | 0.63179 | 0.60896 | 0.59811 | 0.59030 | 0.58540 | 0.58241 | **0.58141** | 0.58142 | 30 |
| context+position+identity | 0.63162 | 0.60890 | 0.59808 | 0.59027 | 0.58534 | 0.58232 | **0.58133** | 0.58135 | 30 |

Two things in that grid matter more than the winner. The context-only model barely moves across four
orders of magnitude of penalty, which is what a handful of well-measured pre-play columns should do. The
identity model without shrinkage is far worse than dropping the block altogether - 0.63179 against
0.58171 at penalty 0.03 - so the player block needs a prior, and one shared penalty cannot serve both
blocks at once. The pre-declared tiebreak across a task's models gave a shared value of 30.

At that shared penalty the block stops costing loss:

| Passing model | 2024 log loss | 2024 difference vs context | 2025 log loss | 2025 difference vs context |
| --- | --- | --- | --- | --- |
| context | 0.57109 | - | 0.57969 | - |
| context+position | 0.57091 | -0.00017 [-0.00035, -0.00000] | 0.57948 | -0.00021 [-0.00042, -0.00000] |
| context+identity | 0.56966 | -0.00143 [-0.00226, -0.00060] | 0.57865 | -0.00104 [-0.00202, -0.00001] |
| context+position+identity | 0.56950 | -0.00159 [-0.00246, -0.00074] | 0.57856 | -0.00113 [-0.00216, -0.00004] |

The same rows at the penalty the first audit used reproduce that audit:

| Passing model | 2024 log loss at penalty 1.0 | 2024 difference | 2025 log loss at penalty 1.0 | 2025 difference |
| --- | --- | --- | --- | --- |
| context+identity | 0.57487 | +0.00380 [+0.00185, +0.00586] | 0.58158 | +0.00188 [-0.00010, +0.00379] |
| context+position+identity | 0.57484 | +0.00377 [+0.00171, +0.00588] | 0.58185 | +0.00215 [+0.00006, +0.00413] |

So the cost recorded in the audit was an artifact of one untuned shared penalty, not a failure of player
identity to transfer. At a penalty chosen on an earlier season the passer-receiver block improves held-out
completion loss in both scored seasons, and the intervals stay below zero. The inner season had already
said so: at penalty 1.0 the identity model scored 0.59030 on 2023 against 0.58171 for context.

## Per-block penalties

`select_identity_penalty` on the same inner split, with the context penalty held at 1.0:

| Block | 2023 grid | selected | 2024 difference | 2025 difference |
| --- | --- | --- | --- | --- |
| passers and receivers | 10: 0.58241, 30: 0.58139, **100: 0.58128**, 300: 0.58145, 1000: 0.58161, 3000: 0.58167 | 100 | -0.00120 [-0.00169, -0.00072] | -0.00101 [-0.00164, -0.00035] |
| passers only | 10: 0.58170, 30: 0.58156, **100: 0.58151**, 300: 0.58157, 1000: 0.58165 | 100 | -0.00082 [-0.00123, -0.00040] | -0.00064 [-0.00108, -0.00016] |
| receivers only | 10: 0.58246, 30: 0.58151, **100: 0.58143**, 300: 0.58157, 1000: 0.58166 | 100 | -0.00052 [-0.00083, -0.00018] | -0.00055 [-0.00097, -0.00010] |

Three facts from that table:

- 100 is an interior optimum on the inner split, not a boundary artifact: the score rises again at 300.
- Each block carries a little transferable information on its own, and the two together are close to the
  sum of their separate gains (-0.00120 against -0.00082 and -0.00052 in 2024).
- The receiver block holds 652 columns in 2024 and the passer block 151, and the smaller block carries the
  larger gain. Column count is not evidence.

The outer curve, reported as a sensitivity and not as a selection:

| Identity penalty (context penalty 1.0) | 2024 log loss | 2024 difference | 2025 log loss | 2025 difference |
| --- | --- | --- | --- | --- |
| 10 | 0.57039 | -0.00068 [-0.00190, +0.00051] | 0.57916 | -0.00054 [-0.00188, +0.00082] |
| 30 | 0.56963 | -0.00143 [-0.00226, -0.00060] | 0.57866 | -0.00104 [-0.00202, -0.00001] |
| 100 | 0.56986 | -0.00120 [-0.00169, -0.00072] | 0.57869 | -0.00101 [-0.00164, -0.00035] |
| 300 | 0.57037 | -0.00070 [-0.00095, -0.00045] | 0.57904 | -0.00066 [-0.00100, -0.00030] |
| 1000 | 0.57078 | -0.00029 [-0.00038, -0.00019] | 0.57940 | -0.00030 [-0.00045, -0.00015] |
| 3000 | 0.57096 | -0.00011 [-0.00014, -0.00007] | 0.57958 | -0.00012 [-0.00017, -0.00006] |

The gain is a shrinkage-dependent band, not a plateau: the block helps most between 30 and 100 and its
contribution decays to almost nothing by 3000, where the block is effectively removed. The inner split
chose 100, the outer peak sits at 30, and the distance between them is well inside the interval width.
No penalty outside the selected value was used to choose anything.

## The rushing side does not get the same reprieve

At a shared penalty of 100, which is where the same selection procedure lands for carries, and it lands
at the grid edge (the score keeps improving to 100 and beyond):

| Rushing model | 2024 difference vs context | 2025 difference vs context |
| --- | --- | --- |
| context+support | +0.00504 [-0.00177, +0.01146] | +0.00925 [+0.00189, +0.01663] |
| context+support+runner | +0.02751 [+0.01711, +0.03827] | +0.03820 [+0.02632, +0.04995] |

And measured against the support environment rather than against context, so that the runner block is the
only change:

| Runner penalty (context penalty 1.0, support included) | 2024 difference vs support | 2025 difference vs support |
| --- | --- | --- |
| 30 | +0.03400 [+0.02350, +0.04650] | +0.03923 [+0.02869, +0.05002] |
| 100 | +0.01996 [+0.01253, +0.02842] | +0.02880 [+0.02109, +0.03672] |
| 1000 | +0.00308 [+0.00131, +0.00489] | +0.00759 [+0.00512, +0.01004] |

The runner block's cost does decay with shrinkage, and it never turns into a gain. The inner split says
the same thing from the other side: on 2023 the score improves monotonically all the way to the grid edge
at 3000, where the runner terms are nearly gone. A block whose best penalty is "shrink me to nothing" is
a block that carries no transferable information on this outcome.

## Scale, so the size of these effects is not overstated

| Quantity | 2024 | 2025 |
| --- | --- | --- |
| Completion base rate (log loss of the training-season rate) | 0.62437 | 0.63032 |
| Context model log loss | 0.57109 | 0.57969 |
| Carries: mean of the training season (MAE) | 3.69428 | 3.66923 |
| Context model MAE | 3.62744 | 3.59851 |

The context model's edge over the naive rate is 0.053 nats, and the passer-receiver block's gain at the
selected penalty is 0.0012 of it, about 2.7% of the edge. On carries, the runner block's cost at penalty
100 is 0.0275 yards against a context edge of 0.067 yards, about 40% of the edge. The two sides are not
the same size of claim.

Two further facts from the same runs: at the selected shared penalty the identity terms have a standard
deviation of 0.075 log-odds, against 0.374 at penalty 1.0, so the block that helps is a heavily shrunk
one; and the share of rows whose own log loss is 2 or more rose from 0.32% to 0.42% in 2024 and from
0.46% to 0.50% in 2025 when the block was added. The average improves while the worst rows get slightly
worse, which is why `evaluation.completion_metrics` now reports that tail share beside the mean.

Unseen players are common enough to matter, and the two windows differ. Training on 2022-2023 and
scoring 2024: 13.5% of targets have a passer the training seasons never saw, 13.3% have an unseen
receiver, 24.8% have one or the other, and 11.4% of carries have an unseen runner. Training on 2022-2024
and scoring 2025: 13.8%, 14.4%, 24.9%, and 18.6% of carries. A row with an unseen player falls back to
the context, so a quarter of the scored plays put a ceiling on what any identity block can do between
seasons whatever the prior.

## What this evidence supports

1. A heavily shrunk passer and receiver intercept block carries a small amount of information that
   survives into a later season. Both held-out seasons agree, the intervals stay below zero, and each
   block contributes on its own.
2. The earlier counter-claim was a penalty artifact at one shared penalty of 1.0, and the record in
   `docs/INFERENCE_BASELINE_AUDIT.md` now points here.
3. The runner block does not carry transferable information on `rushing_yards` at any penalty tried, and
   the support environment is at best neutral and probably a small cost.
4. Identity terms need their own variance component. Sharing one penalty with the pre-play context is
   what produced the wrong answer.

## What this evidence does not support

- No ranking of players, and no reading of a term as skill. The gain is a property of a block of
  regularized intercepts, not of an individual passer or receiver.
- No forecasting claim. 2024 and 2025 were inspected during preparation, and no untouched holdout has
  been reserved.
- No statement that the selected penalty is the best one. It is the best on a grid, chosen on one inner
  split, and the outer curve is flatter than the selection step implies.
- No completion claim about target earning, route, catchability, or pressure: the model still sees no
  charted field and no route exposure.
- No causal statement about any player swap.
- Nothing about the crossing point between penalties 10 and 30 on the scored seasons: the outer rows were
  computed at 30 and 100, and locating the crossing would take another predeclared row.

## What would move this next

1. Reserve a predeclared untouched chronological holdout and stop scoring it during development.
2. Test the identity penalty per position, and let the receivers' term depend on the passer rather than
   being strictly additive.
3. Add the defended-side and scheme columns the plan names, then re-run the same selection procedure to
   see whether the identity block still carries information once the opponent is modeled.
4. Extend the pooling to a hierarchy with a per-season player component, so that a player who changes
   teams is not one fixed intercept.
5. Decide the rushing question the other way round: the environment cost is small, so the interesting
   target may be a better outcome model (robust or quantile) rather than another identity block.

## Reproducing the headline table

The canonical tables are read from the cache, the player penalty is chosen on 2022 -> 2023, and the
scored seasons are 2024 and 2025:

```python
import polars as pl
from hierarchical_football.inference.baselines import fit_completion
from hierarchical_football.inference.contracts import PASSING_CONTRACT
from hierarchical_football.inference.evaluation import (
    chronological_split, clustered_comparison, row_losses,
)
from hierarchical_football.inference.partial_pooling import (
    fit_pooled_completion, select_identity_penalty,
)

targets = pl.read_parquet("data/processed/inference/passing_targets.parquet")
selection = select_identity_penalty(targets, identities=("qb_id", "receiver_id"), task="completion",
                                    train_seasons=[2022], validation_season=2023,
                                    penalties=(10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0))
penalty = selection["selected"]

for holdout_season, train_seasons in ((2024, [2022, 2023]), (2025, [2022, 2023, 2024])):
    split = chronological_split(targets, train_seasons=train_seasons,
                                holdout_season=holdout_season, contract=PASSING_CONTRACT)
    observed = split["holdout"]["complete_pass"].to_numpy().astype(float)
    losses = {"context": row_losses(observed, fit_completion(split["train"]).predict(split["holdout"]),
                                    task="completion")}
    pooled = fit_pooled_completion(split["train"], identities=("qb_id", "receiver_id"),
                                   identity_penalty=penalty, context_penalty=1.0)
    losses["pooled_identity"] = row_losses(observed, pooled.predict(split["holdout"]), task="completion")
    comparison = clustered_comparison(losses, clusters=split["holdout"]["game_id"].to_list(),
                                      reference="context", draws=2000, seed=0)
    print(holdout_season, {name: round(float(values.mean()), 5) for name, values in losses.items()},
          {name: [round(value, 5) for value in
                  (entry["estimate"], entry["lower"], entry["upper"])]
           for name, entry in comparison["differences"].items()})
```

The shared-penalty cross-check follows the same shape with `fit_completion(..., identities=..., penalty=30.0)`,
and the running-block table with `fit_pooled_rushing(..., support=True)`. The drivers behind the recorded
numbers were scratch scripts kept outside the index by a local ignore rule; every model, metric,
interval, and selection rule they used is in the committed modules.
