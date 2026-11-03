"""Offline inference table preparation and passing development studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare import RUSHER_SCOPES, build


def prepare_summary(report):
    """The console summary of one build: what was written and how much was removed."""
    return {
        "passing_targets": report["passing"]["preparation"]["target_rows"],
        "rushing_plays": report["rushing"]["preparation"]["target_rows"],
        "inputs": len(report["inputs"]),
        "excluded_plays": sum(sum(block["preparation"]["exclusions"].values())
                              for block in (report["passing"], report["rushing"])),
    }


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare",
                                  help="Build the canonical inference tables from cached raw partitions")
    command.add_argument("--seasons", nargs="+", required=True, type=int)
    command.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    command.add_argument("--output-root", type=Path, default=Path("data/processed/inference"))
    command.add_argument("--rusher-scope", choices=sorted(RUSHER_SCOPES), default="backfield")
    study = commands.add_parser("study-passing", help="Run a frozen passing development specification")
    study.add_argument("--input", required=True, type=Path)
    study.add_argument("--spec", required=True, type=Path)
    study.add_argument("--output-root", required=True, type=Path)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "study-passing":
        from .study_io import run_study_file
        manifest = run_study_file(args.input, args.spec, args.output_root)
        print(json.dumps({"status": manifest["status"], "output_root": str(args.output_root)}))
        return
    print(json.dumps(prepare_summary(
        build(args.raw_root, args.output_root, args.seasons, rusher_scope=args.rusher_scope))))


if __name__ == "__main__":
    main()
