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
