# Passing situation development evidence

## Matched frozen studies

Two protocols committed before reading results compare `expanded` context with
`expanded_situation`, which adds shotgun/no-huddle categories and linear pre-play
score/clock terms. See [the reference protocol](../examples/passing_situation_reference_study.json)
and [the situation protocol](../examples/passing_situation_study.json).

Both fix QB, receiver and opponent penalties at 100 and context penalty at 1.
Training starts in 2022, the one-candidate inner score uses 2023, and expanding
development folds score 2024 and 2025. Priors and populations were not retuned.
The predeclared primary comparison is situation-minus-reference joint-model
completion log loss, separately by season. Negative favors added situation.
Secondary results include context-only fits and partnership-transfer diagnostics.
Each paired interval resamples whole games 2,000 times with seed 0 and level 95%.

The enriched input `data/processed/inference_situation/passing_targets.parquet`
was prepared at `dcb691b`. Its 68,408 targets exactly reproduce every original
canonical column and outcome; all four added fields are complete and finite.
The audited join matches all targets to 197,362 cached play-by-play rows without
changing the target population. Its SHA-256 is
`82b62c3cfe7318209f961d9b850aad1bac4b5e59969e45b36e8a7046e6fab7ae`.
The selected score field is at the start of the play, as documented by
[nflfastR](https://nflfastr.com/reference/fast_scraper.html).

## Primary prediction comparison

| Development season | Targets / games | Reference joint log loss | Situation joint log loss | Paired difference [95% interval] |
| --- | --- | ---: | ---: | --- |
| 2024 | 17,011 / 272 | 0.567116390 | 0.566589895 | -0.000526495 [-0.000969419, -0.000062448] |
| 2025 | 16,609 / 272 | 0.575310090 | 0.574997559 | -0.000312531 [-0.000754323, 0.000109874] |

Both point differences favor added situation. The 2024 interval excludes zero;
the 2025 interval includes it. These are small gains, with uncertain repeatability
in the later fold. Joint Brier scores also move from 0.191037222 to 0.190806464
in 2024 and from 0.194752971 to 0.194617136 in 2025.

All outer fits converge. The situation joint design has 840 columns in the first
fold and 988 in the second, six more than its matched reference. The reference's
joint probabilities exactly reproduce the preceding shrinkage study, with maximum
absolute difference zero on aligned plays. The observed gain is therefore not a
change in the earlier baseline or target extraction.

## Secondary context and transfer checks

Context-only completion loss improves by -0.000669374 in 2024, with paired interval
[-0.001149911, -0.000160062], and by -0.000386351 in 2025, with interval
[-0.000884915, 0.000088648]. This supports a small contextual contribution rather
than assigning the entire gain to player terms.

Among two previously observed players forming a new partnership, joint-model
situation-minus-reference differences are -0.000678436 in 2024 (5,404 targets,
254 games) and -0.000628416 in 2025 (4,892 targets, 259 games). Their intervals,
[-0.001581504, 0.000207623] and [-0.001480210, 0.000215892], both include zero.

The receiver block's increment beyond QB within the situation model also remains
uncertain on new partnerships:

| Season | Joint minus QB-only loss | Paired 95% interval |
| --- | ---: | --- |
| 2024 | -0.000313433 | [-0.000917669, 0.000266594] |
| 2025 | -0.000642977 | [-0.001339453, 0.000031063] |

Adding controls does not establish portable receiver skill. These diagnostics are
secondary, have not been adjusted for multiple comparisons, and cannot turn
observational target conversion into causal credit. Completion is conditional on
the observed target, depth and direction; this does not evaluate weekly target
earning or fantasy-point distributions. Both evaluation seasons were previously
inspected during development, so no untouched-holdout or forecast promotion follows.
The [calibration limits](PASSING_CALIBRATION_RESULTS.md) on player intervals remain.

## Artifact provenance and verification

Both studies and the paired check ran at
`4e7905be71af435be7c0d5e412ccf0f751a16450`, with both numerical thread variables
set to 1 before Python startup. Complete manifests record input/specification,
package-source and output hashes, runtime versions and thread settings.

| Ignored local bundle | Manifest SHA-256 |
| --- | --- |
| `data/experiments/passing_situation_reference_20261103` | `7dc4930c3596ea776330a7b3ac65cb733a4289f6194d98289ca95f3ca080f39e` |
| `data/experiments/passing_situation_20261103` | `18b1d8641b003a9e7500801e8c2c11f9fb53578a745206c83ec78189390e97b1` |
| `data/experiments/passing_situation_paired_20261103` | `c96d76c997237fa1cbe023f5395325adab3dd4c11e0551756554df9d468d8f62` |

The paired report SHA-256 is
`6d7b17e1b0eddc82cde7d624c9910facf01a43742d64db31ae21ee7e5f83695a`.
All source and output hashes were verified, along with original input hashes,
unique play keys, aligned outcomes/cohorts, and convergence of outer models.
The frozen paired runner is retained in its bundle. Generated artifacts remain
ignored; neither original canonical data nor earlier completed runs were overwritten.
