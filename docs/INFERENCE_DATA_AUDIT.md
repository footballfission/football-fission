# Canonical inference extraction audit

Observed results from one local run of the first-stage extractor, recorded as evidence rather
than as a test expectation. The numbers describe the cached partitions named by hash below; a
refresh of the cache can change them, and nothing here validates an analytical claim.

## What was run

Run from the repository root with the canonical tables written to `data/processed/inference/`:

```powershell
uv run python prepare_inference_tables.py --raw-root data/raw --output-root data/processed/inference --seasons 2022 2023 2024 2025
```

The command printed one line:

```
{"passing_targets": 68408, "rushing_plays": 47256, "inputs": 8, "excluded_plays": 279060}
```

The extraction read eight cached partitions and joined `rosters_weekly` positions. The build
recorded these input hashes in `prepare_report.json`:

| Partition | sha256 |
| --- | --- |
| `pbp/season=2022/part.parquet` | `02c05c964a3f095eddafe0863739dfd338f42ef47e33dc6aee7e9626f3bfc271` |
| `pbp/season=2023/part.parquet` | `b48a21d516f4813a8f3878eecc9f2448fc95fe5c2e6ee686b3c418afed1b3537` |
| `pbp/season=2024/part.parquet` | `2e2a840888c18263754687548689421edbebfc5b61ac33f0ce1322b582f61948` |
| `pbp/season=2025/part.parquet` | `1453e383d17fe82c751fb4a8f38eb5ba438c6f3d0a175fcee18f2aafac8861e2` |
| `rosters_weekly/season=2022/part.parquet` | `457e3ddf2da03085e1098c870220785708f7ebc17d1c37380cc10eca14867910` |
| `rosters_weekly/season=2023/part.parquet` | `d854979a836565b550fd8552e9cfb827b2b1993dd4487ce7e3ee04157bc534b5` |
| `rosters_weekly/season=2024/part.parquet` | `ff856cadec4950ffe5e43a703af073502f120062efaf11b5d29c730debc369a2` |
| `rosters_weekly/season=2025/part.parquet` | `a8a6022796dc9c50f645a00c250c2f719ff2c974439f883e41e3fa2da84f38c2` |

## Kept rows

| Table | Rows | 2022 | 2023 | 2024 | 2025 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `passing_targets` | 68,408 | 17,306 | 17,482 | 17,011 | 16,609 |
| `rushing_plays` (backfield scope) | 47,256 | 11,830 | 11,730 | 11,830 | 11,866 |

Both tables are about 17,100 targeted passes and 11,800 backfield carries per season, which is the
shape the plan expects for a four-season regular-season extraction.

## Removed plays

The source has 197,362 cached plays per table because both tables read the same play-by-play.

| Exclusion reason | Passing | Rushing |
| --- | ---: | ---: |
| `not_a_pass_play` / `not_a_run_play` | 106,328 | 128,522 |
| `postseason` | 9,080 | 9,080 |
| `sack` | 5,307 | - |
| `scramble` | 4,380 | 4,380 |
| `missing_receiver` | 2,943 | - |
| `kneel` | - | 1,692 |
| `outside_rusher_scope` | - | 5,231 |
| `two_point_attempt` | 509 | 509 |
| `aborted_play` | 407 | 407 |
| `spike` | - | 285 |
| `no_pass_attempt`, `missing_quarterback`, `missing_team`, `invalid_outcome` | 0 | - |
| `no_rush_attempt`, `missing_rusher`, `missing_yardage`, `missing_team` | - | 0 |

Every removal is accounted for by a named reason, and no source row was dropped for an unreported
cause: `target_rows` plus the exclusions equals `source_rows` in both reports.

The 5,231 carried-outside-scope plays are the quarterback, receiver, and untagged carries the
default backfield scope leaves out. They are the material for the `non_qb` and `all` sensitivity
cases, not an error.

## Coverage inside the kept rows

| Column | Missing in `passing_targets` | Missing in `rushing_plays` |
| --- | ---: | ---: |
| `air_yards`, `yardline_100`, `ydstogo`, `down` | 0 | 0 |
| `pass_location` (`UNKNOWN` label) | 1 | - |
| `run_location` (`UNKNOWN` label) | - | 14 |
| `run_gap` (`UNKNOWN` label) | - | 12,271 (26.0%) |
| `position` (`UNKNOWN` label) | 57 | 0 |

Two findings matter for later stages:

- The pre-play context the models need is complete: air yards, field position, distance, and down
  are recorded for every kept play in all four seasons.
- `run_gap` is not. One carry in four has no gap label, so a gap-level contrast cannot treat
  carries as if the gap were known; it has to keep `UNKNOWN` as its own group or restrict the
  sample and say so.

`pass_location` is recorded for all but one pass, but it is the *charted* side of the pass, so it
stays out of a completion model as a pre-play predictor.

## Pairing support in `passing_targets`

| Quantity | Value |
| --- | ---: |
| Quarterbacks | 191 |
| Receivers | 875 |
| Distinct pairings | 2,954 |
| Pairings per receiver (min / median / max) | 1 / 3 / 13 |
| Receivers with exactly one quarterback | 221 |
| Quarterbacks with exactly one receiver | 53 |
| Connected components of the pairing graph | 8 |
| Largest component | 977 players, 184 quarterbacks, 867 receivers, 68,400 rows (99.99% of rows) |
| Components with a single pairing | 6 |

The pairing graph is effectively connected: one component holds 99.99% of the targets, so a
passer-versus-receiver comparison is not blocked by disconnected passing units in this window.
The concentration is elsewhere - 221 receivers (about a quarter of them) appear with only one
quarterback, and 53 quarterbacks throw to only one receiver, so those players' estimates rest on a
single partnership whatever the graph looks like globally. Six components consist of players who
only ever appeared with each other; a split inside those is not identified by these data.

## What this audit does not establish

- It is not a validation of any model. No baseline or Bayesian fit has been run on these tables.
- It does not reconcile derived game and team counts against an independent weekly table.
- It does not audit charting: FTN catchability, drops, contested catches, pressure, box count, and
  participation are not joined at this stage, and their per-season coverage is still unmeasured.
- It does not certify that cached partitions are the source's state at an earlier date; the ingest
  sidecars record `historical_asof_verified: false`.
- It says nothing about whether a targeted-pass model measures receiver talent, because the table
  contains no route exposure for receivers who were not targeted.
