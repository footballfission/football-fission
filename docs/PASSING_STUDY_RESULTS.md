# Reproducible passing development study

Observed November 3, 2026 using the committed offline runner. These are development
results on previously inspected seasons, not an untouched evaluation, player ranking,
or causal credit allocation.

## Protocol and provenance

```powershell
python -m hierarchical_football.inference study-passing --input data/processed/inference/passing_targets.parquet --spec examples/passing_study.json --output-root data/experiments/passing_development_20261103
```

The output directory is local and ignored. Replays require a fresh output path. Its
complete manifest records exact specification bytes, source/runtime versions and output
hashes. Generated data and predictions are not committed.

| Item | Value |
| --- | --- |
| Code revision | `d70b543b1857d1105214f410bdd1674d4a9aacba` |
| Input | 68,408 canonical targets, regular seasons 2022-2025 |
| Input SHA-256 | `ebe6f593139df4cadcbb6d0b07e7b797f9603ad350f34c1d0bb2f70ffa54c98c` |
| Specification SHA-256 | `766f234c30251e32d8b4ab14d3696f9a171091026118d3d28029e961e6e74d1f` |
| Report SHA-256 | `aa55b1a581001aa35760e3b3307b230e13a3e02d17c4a8dd6ef06e704e1d7784` |
| Predictions SHA-256 | `7ffe1f04ae1a1d29a8b82ea09e7e4d69ea9c4848fdd2f9534a72739a1c2da6ff` |
| Runtime | Python 3.13.12; NumPy 2.5.4; SciPy 1.18.1; Polars 2.0.0; scikit-learn 1.9.1 |

Selection fits 2022 and scores 2023 using completion log loss. The declared grid crosses
linear/expanded context with independent QB and receiver penalties of 10, 100 and 1000:
18 candidates, all converged. The context penalty is 1.0.

The selected context is **expanded**, with **QB and receiver penalties both 100**,
interior grid values. Its inner loss is 0.578624 versus 0.581282 for the best linear
candidate. Expanded context with QB penalty 10 and receiver penalty 100 scores 0.578674,
so the exact QB penalty is weakly distinguished. Supporting separate penalties does not
establish an empirical need for unequal penalties.

Expanded context combines fixed depth hinges at 0, 5, 10, 20 and 30 yards, position, and
defending-team intercepts. This run does not separate those additions' contributions.
The selected specification stays fixed across expanding outer folds:

| Scored season | Training seasons | Train targets | Scored targets | Scored games |
| --- | --- | ---: | ---: | ---: |
| 2024 | 2022-2023 | 34,788 | 17,011 | 272 |
| 2025 | 2022-2024 | 51,799 | 16,609 | 272 |

All outer fits converged. Intervals below are paired game-clustered percentile bootstrap
intervals, 2,000 draws, seed 0, level 95%. Negative loss differences favor the added block.

## Completion loss

| Model | 2024 log loss | 2025 log loss |
| --- | ---: | ---: |
| Historical linear context | 0.571067 | 0.579700 |
| Selected expanded context | 0.568988 | 0.576260 |
| Expanded context + QB | 0.568067 | 0.575774 |
| Expanded context + receiver | 0.568395 | 0.575797 |
| Expanded context + both | 0.567665 | 0.575480 |

Single-block ablations retain the jointly selected penalty; they are not independently
optimized alternatives.

| Difference versus expanded context | 2024 estimate [95% interval] | 2025 estimate [95% interval] |
| --- | --- | --- |
| QB | -0.000920 [-0.001378, -0.000453] | -0.000486 [-0.000959, -0.000003] |
| Receiver | -0.000592 [-0.000917, -0.000281] | -0.000463 [-0.000882, -0.000024] |
| Both | -0.001323 [-0.001841, -0.000814] | -0.000780 [-0.001401, -0.000140] |

The difficulty expansion improves more than the player blocks do. Player identities still
add a small improvement in both seasons. These comparisons evaluate adding and refitting
a block, not a player effect with all other coefficients held fixed.

## Natural partnership transfer

The five cohorts partition each scored season. New pairs contain two previously observed
players who were never paired in training. Games can appear in multiple cohorts; the
intervals are not independent experiments. All differences below compare both player
blocks against expanded context.

| Cohort | 2024 targets / games | 2024 difference [95% interval] | 2025 targets / games | 2025 difference [95% interval] |
| --- | --- | --- | --- | --- |
| Known pair | 7,387 / 245 | -0.002318 [-0.003121, -0.001471] | 7,585 / 250 | -0.000664 [-0.001747, +0.000459] |
| New pair, both known | 5,404 / 254 | -0.001008 [-0.001880, -0.000150] | 4,892 / 259 | -0.001123 [-0.002265, -0.000078] |
| Unseen QB only | 1,953 / 82 | -0.000112 [-0.001214, +0.000946] | 1,748 / 85 | -0.000990 [-0.002431, +0.000437] |
| Unseen receiver only | 1,918 / 239 | +0.000334 [-0.000951, +0.001577] | 1,843 / 233 | +0.000430 [-0.000911, +0.001753] |
| Both unseen | 349 / 69 | -0.001002 [-0.003018, +0.000832] | 541 / 67 | -0.002751 [-0.004660, -0.000836] |

The combined model improves new-pair loss in both seasons, but this does **not** isolate
receiver-specific transfer. A secondary diagnostic after the primary run adds receivers
to the QB model on new pairs: -0.000327 [-0.000944, +0.000269] in 2024 and -0.000602
[-0.001292, +0.000090] in 2025. Both intervals cross zero. These diagnostics did not
influence selection.

Unseen identities contribute zero player adjustments, but context coefficients are refitted
with the identity model. Gains in the both-unseen group therefore provide no evidence
of information about those new players.

The secondary receiver-increment diagnostic is reproducible from saved predictions:

```python
import polars as pl
from hierarchical_football.inference.evaluation import bootstrap_difference, row_losses

rows = pl.read_parquet("data/experiments/passing_development_20261103/predictions.parquet")
for season in (2024, 2025):
    frame = rows.filter((pl.col("season") == season) & (pl.col("transfer_cohort") == "new_pair"))
    y = frame["complete_pass"].to_numpy()
    losses = {name: row_losses(y, frame["p_" + name].to_numpy(), task="completion")
              for name in ("both", "qb")}
    print(season, bootstrap_difference(losses["both"], losses["qb"],
                                       clusters=frame["game_id"].to_list(), draws=2000, seed=0))
```

## Calibration, support and limits

The share of targets with individual log loss at least 2 falls from 0.335% and 0.464%
under linear context to 0.135% and 0.151% under expanded context. Adding players raises
it slightly to 0.182% and 0.181%, despite improving average loss. The report retains full
calibration bins and Brier scores.

Training audits count 182 receivers with one QB before 2024 and 207 before 2025. Predictive
improvement of a block does not validate those individuals' attribution. No individual
NFL contrast was selected here: the protocol's `contrasts` list is empty. Conditional
bootstrap machinery is checked with synthetic crossed and confounded partnerships; no
real-player interval coverage claim is established.

All saved output hashes were verified. Predictions contain 33,620 unique play keys with
finite probabilities. All 364 offline tests and syntax checks passed. Ruff is not installed
in the project environment, so no Ruff result is claimed.

Next work should test receiver-specific transfer with greater precision, separate the
expanded context additions, inspect supported preselected player contrasts, consider
player-season variation, and document a prospective untouched evaluation before promotion.
Charting, route exposure, scheme and richer defensive context remain absent. There is no
full Bayesian posterior, target-earning model, or evidence of causal player-swap effects.
