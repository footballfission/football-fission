"""First-stage extractor: cached play-by-play into the canonical inference tables.

Run from a checkout root:

    uv run python prepare_inference_tables.py --seasons 2022 2023 2024 2025

The same work is available as `python -m hierarchical_football.inference prepare`; this script
exists so the research plan's documented command line works from the repository root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierarchical_football.inference.__main__ import prepare_summary
from hierarchical_football.inference.prepare import RUSHER_SCOPES, build


def parser():
    document = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    document.add_argument("--seasons", nargs="+", required=True, type=int)
    document.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    document.add_argument("--output-root", type=Path, default=Path("data/processed/inference"))
    document.add_argument("--rusher-scope", choices=sorted(RUSHER_SCOPES), default="backfield")
    document.add_argument('--include-passing-situation', action='store_true',
                          help='Attach audited formation, tempo, score and clock context to passing targets')
    return document


def main(argv=None):
    args = parser().parse_args(argv)
    report = build(args.raw_root, args.output_root, args.seasons, rusher_scope=args.rusher_scope,
                   include_situation=args.include_passing_situation)
    print(json.dumps(prepare_summary(report)))


if __name__ == "__main__":
    main()
