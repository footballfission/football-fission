"""CLI for offensive opportunity latent vs identity experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_opportunity_experiment, write_results_markdown
from .pools import POOLS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/experiments/opportunity_latent_identity"),
    )
    parser.add_argument("--train-seasons", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument("--evaluation-seasons", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--validation-season", type=int, default=2023)
    parser.add_argument("--pools", nargs="+", choices=list(POOLS), default=list(POOLS))
    parser.add_argument("--volume-draws", type=int, default=500)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--results-markdown",
        type=Path,
        default=Path("docs/OPPORTUNITY_EXPERIMENT_RESULTS.md"),
    )
    args = parser.parse_args(argv)
    report = run_opportunity_experiment(
        raw_root=args.raw_root,
        output_root=args.output_root,
        train_seasons=tuple(args.train_seasons),
        evaluation_seasons=tuple(args.evaluation_seasons),
        validation_season=args.validation_season,
        pools=tuple(args.pools),
        volume_draws=args.volume_draws,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    write_results_markdown(report, args.results_markdown)
    summary = {
        "output_root": str(args.output_root),
        "results_markdown": str(args.results_markdown),
        "available_seasons": report["available_seasons"],
        "evaluation_seasons": report["evaluation_seasons"],
        "pools": report["pools"],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
