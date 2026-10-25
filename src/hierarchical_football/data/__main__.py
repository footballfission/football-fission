"""Command line for local data ingestion and panel building."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .ingest import DEFAULT_TABLES, TABLES, ingest
from .panels import build
from .schema import POSITION_COLUMNS
from .splits import split_week


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("ingest", help="Download/cache explicitly selected raw partitions")
    download.add_argument("--seasons", nargs="+", required=True, type=int)
    download.add_argument("--tables", nargs="+", choices=TABLES, default=DEFAULT_TABLES)
    download.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    download.add_argument("--refresh", action="store_true")
    panel = commands.add_parser("build", help="Build panels from local raw partitions; no network calls")
    panel.add_argument("--seasons", nargs="+", required=True, type=int)
    panel.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    panel.add_argument("--output", type=Path, default=Path("data/processed"))
    panel.add_argument("--positions", nargs="+", choices=POSITION_COLUMNS, default=list(POSITION_COLUMNS))
    panel.add_argument("--zero-missing-player-stats", action="store_true",
        help="Explicitly assume roster players missing from a completed team's stat feed had zero counters")
    panel.add_argument("--as-of", help="Timezone-aware time limiting completed outcomes; default current UTC")
    panel.add_argument("--prediction-time", help="Timezone-aware cutoff for unfinished games; historical rows retain Wednesday cutoffs")
    split = commands.add_parser("split", help="Prepare one week with cutoff-filtered training and outcome-free prediction rows")
    split.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    split.add_argument("--position", choices=POSITION_COLUMNS, required=True)
    split.add_argument("--season", type=int, required=True)
    split.add_argument("--week", type=int, required=True)
    split.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest(args.raw_root, args.seasons, args.tables, refresh=args.refresh)
        print(json.dumps({"partitions": len(result["partitions"]), "errors": result["errors"]}))
    elif args.command == "split":
        print(json.dumps(split_week(args.processed_root, args.output, args.position, args.season, args.week)))
    else:
        result = build(args.raw_root, args.output, args.seasons, positions=args.positions,
            zero_missing=args.zero_missing_player_stats, prediction_time=args.prediction_time, as_of=args.as_of)
        print(json.dumps({"positions": result["positions"], "assumed_zero_rows": result["assumed_zero_rows"],
                          "unknown_outcome_rows": result["unknown_outcome_rows"]}))


if __name__ == "__main__":
    main()
