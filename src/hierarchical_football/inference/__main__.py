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
    command.add_argument('--include-passing-situation', action='store_true',
                         help='Attach audited formation, tempo, score and clock context to passing targets')
    study = commands.add_parser("study-passing", help="Run a frozen passing development specification")
    study.add_argument("--input", required=True, type=Path)
    study.add_argument("--spec", required=True, type=Path)
    study.add_argument("--output-root", required=True, type=Path)
    comparison = commands.add_parser("compare-passing", help="Compare supported player contrasts across historical windows")
    comparison.add_argument("--input", required=True, type=Path)
    comparison.add_argument("--spec", required=True, type=Path)
    comparison.add_argument("--output-root", required=True, type=Path)
    calibration = commands.add_parser('calibrate-passing', help='Check known-truth synthetic contrast coverage')
    calibration.add_argument('--spec', required=True, type=Path)
    calibration.add_argument('--output-root', required=True, type=Path)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'calibrate-passing':
        from .study_io import run_calibration_file
        manifest = run_calibration_file(args.spec, args.output_root)
        print(json.dumps({'status': manifest['status'], 'synthetic': True, 'output_root': str(args.output_root)}))
        return
    if args.command == "compare-passing":
        from .study_io import run_comparison_file
        manifest = run_comparison_file(args.input, args.spec, args.output_root)
        print(json.dumps({"status": manifest["status"], "output_root": str(args.output_root)}))
        return
    if args.command == "study-passing":
        from .study_io import run_study_file
        manifest = run_study_file(args.input, args.spec, args.output_root)
        print(json.dumps({"status": manifest["status"], "output_root": str(args.output_root)}))
        return
    print(json.dumps(prepare_summary(
        build(args.raw_root, args.output_root, args.seasons, rusher_scope=args.rusher_scope,
              include_situation=args.include_passing_situation))))


if __name__ == "__main__":
    main()
