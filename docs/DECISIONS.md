# Policy decisions

Records the policies this package commits to, and the limits of each. It is not
a changelog: each section describes why the current behavior is what it is.

## Tracked and generated paths

Generated datasets, run manifests, and scratch notes are never tracked. The
`/data/` ignore rule is anchored to the repository root, so it does not hide
`src/hierarchical_football/data/`, which is ordinary source.

Every frame stored by `write_frame` writes a `.json` manifest beside its
Parquet file, and `ingest` and `build` write their reports beside the trees they
describe. All of those paths live under `data/`, so all of them are ignored.

The rule does not stop a caller from writing somewhere tracked. Pointing
`--raw-root` or `--output` at a tracked directory still produces files that git
will see.

## Ingest report history

`raw/ingest_report.json` records each invocation rather than replacing the previous one: a run appends a summary to `runs` and merges its partition results into `partitions`, so a cache assembled over several invocations still lists every partition it holds.

The merge is keyed by `(table, season)` and the newest record for a key wins. A partition downloaded in one run and served from cache in a later one reports `cached`, because the entry describes the last touch rather than the first fetch.

This is not a complete audit trail. It keeps one entry per partition and per run, not the intermediate state of a failed download, and it does not hash the report itself. A deleted report restarts the history; earlier runs are not recoverable.

## Roster extension

A team-week with a scheduled game and no roster snapshot inherits the most recent earlier snapshot from the same season when extension is requested. The copied row keeps its original `roster_status` and is marked `roster_carried_forward`, so a consumer can still separate observed rosters from inferred ones.

Extension is opt-in. With it off, a week without a snapshot produces no player rows, because the panel is assembled from rosters and the schedule; inventing participation would make an unsnapshotted week look modeled.

It does not model roster change. A trade, signing, injury designation, or practice-squad move after the copied week is invisible, and a season's first week is never inferred from a later one.

| Situation | Result |
| --- | --- |
| Week has its own snapshot | used unchanged, `roster_carried_forward` false |
| Week has none, a same-season earlier snapshot exists | copied from the newest such snapshot, flagged true |
| Week has none, no earlier snapshot exists | no player rows |
| Extension not requested | no player rows for unsnapshotted weeks |

## Prediction bundle metadata

A prediction bundle carries what identifies a row and what its cutoff was; it never carries the outcome. Roster status and the carried-forward flag travel with the bundle because eligibility is a caller decision, not a model output.

Roster columns stay out of `features.json`. They are context a consumer may filter on, not signals a model may fit.

The bundle does not say who will play. `roster_status` is the last observed designation, and for a week built by roster extension it is inherited from an earlier snapshot; an inactive list published after the cutoff never appears.

## Inference table contracts

A canonical inference table is validated, never repaired. A consumer that receives a
`passing_targets` or `rushing_plays` frame knows that the play key is unique, that both
identities and the outcome are present, and that every label is either declared in the
contract or `UNKNOWN`. Preparation owns those guarantees; validation only proves them.

`UNKNOWN` is a label, not a missing value. A pass whose charted location is absent, and a
player the weekly roster feed does not tag with a known position, carry `UNKNOWN` so an
analysis can count them instead of dropping or guessing them. A null in those columns is a
contract violation, because it would be indistinguishable from a failed join.

The contract covers required columns only. The optional charted and defensive fields -
catchability, drop, contested, screen, pressure, yards after catch, box count, personnel,
formation - vary in coverage by season, so a table without them is valid and their absence
is a data-availability question rather than a schema error.

| Situation | Result |
| --- | --- |
| Required column absent | `ValueError` naming the table and the missing columns |
| Column of the wrong kind | `ValueError` naming the column and the expected kind |
| Null key, season, week, team, identity, outcome, or position | `ValueError` naming the column and its null count |
| Label outside the declared set | `ValueError` listing the unexpected labels |
| Duplicate `(game_id, play_id)` | `ValueError` with the duplicate count; no row is dropped |
| Trick-play reception by a player tagged `QB` | kept; analyses choose their own position subset |

It does not certify that play-by-play classified a play correctly, that a snap belongs to
the player credited with it, or that the optional charted fields are trustworthy in the
seasons where they exist.

## Passing target selection

A targeted pass enters the canonical table when it is a regular-season pass attempt, play-by-play
names the passer, the receiver, and the possession team, and the completion flag is `0` or `1`.
Everything else is removed and counted, never repaired. Reason order decides which reason is
credited: postseason, two-point attempt, aborted play, sack, scramble, not a pass play, no pass
attempt, missing quarterback, missing receiver, missing team, invalid outcome.

Sacks and scrambles are removed because the play has no target a receiver could have converted;
a scramble is a run in the record but a failed pass for this question, and counting it as a
failed completion would penalise a passer for leaving the pocket. Two-point attempts and aborted
plays are removed because their situation is not comparable to a normal down. A completion flag
that is missing or outside `0` and `1` is an invalid outcome rather than an incompletion, so a
corrupt source row never becomes evidence about a passer.

| Source play | Kept | Reason when removed |
| --- | --- | --- |
| Regular-season pass attempt, flag `0` or `1`, identities and team present | yes | - |
| Sack, scramble, throwaway with no receiver | no | `sack`, `scramble`, `missing_receiver` |
| Two-point attempt, aborted play, non-pass play, no pass attempt | no | `two_point_attempt`, `aborted_play`, `not_a_pass_play`, `no_pass_attempt` |
| Postseason play | no | `postseason` |
| Passer or possession team unidentified | no | `missing_quarterback`, `missing_team` |
| Completion flag missing or not `0` or `1` | no | `invalid_outcome` |

The table measures conversion once a pass is targeted. It cannot measure whether the target should
have existed: play-by-play records the receiver of the attempt, not the routes run by every other
eligible receiver. Missing `air_yards`, `yardline_100`, `ydstogo`, and `down` are retained and
counted in the report rather than filled, so a later model can decide whether to use them.

Charted fields are absent from this stage. Their coverage varies by season, and joining them
before that coverage is measured would put a season-dependent definition inside the canonical
table.
