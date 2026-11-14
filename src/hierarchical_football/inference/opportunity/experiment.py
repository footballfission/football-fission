"""Fit and evaluate latent vs identity opportunity models."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import polars as pl

from ...data.store import atomic_json, write_frame
from ..allocation_baseline import fit_proportional_allocation
from ..allocation_evaluation import allocation_log_scores
from ..evaluation import bootstrap_difference
from ..panels import TEAM_GAME_KEYS
from .features import (
    attach_model_features,
    build_defense_latents,
    build_skill_history,
    vacated_role_features,
)
from .pools import POOLS, build_opportunity_panels
from .shares import fit_identity_shares, fit_latent_shares, select_identity_ridge
from .totals import fit_identity_volume, fit_latent_volume, score_volume_predictions


def _as_target_schema(panel, candidates):
    """Rename pool totals so existing allocation scorers can be reused."""
    observed_panel = panel.select(
        *TEAM_GAME_KEYS,
        "season", "week",
        pl.col("total").alias("targets"),
        "candidates",
        pl.col("candidate_total").alias("candidate_targets"),
        pl.col("other_total").alias("other_targets"),
    )
    observed_candidates = candidates.select(
        *TEAM_GAME_KEYS,
        "season", "week",
        "candidate_index",
        "player_id",
        "position",
        pl.col("count").alias("targets"),
    )
    return observed_panel, observed_candidates


def _load_seasons(raw_root, seasons):
    pbp = pl.concat([
        pl.read_parquet(Path(raw_root) / f"pbp/season={year}/part.parquet")
        for year in seasons
    ], how="diagonal_relaxed")
    rosters = pl.concat([
        pl.read_parquet(Path(raw_root) / f"rosters_weekly/season={year}/part.parquet")
        for year in seasons
    ], how="diagonal_relaxed")
    return pbp, rosters


def _season_mask(frame, seasons):
    return pl.col("season").is_in(list(seasons))


def run_opportunity_experiment(
    *,
    raw_root=Path("data/raw"),
    output_root=Path("data/experiments/opportunity_latent_identity"),
    train_seasons=(2022, 2023),
    evaluation_seasons=(2024, 2025),
    validation_season=2023,
    pools=POOLS,
    share_ridge=1.0,
    identity_grid=(1.0, 10.0, 100.0),
    volume_draws=500,
    bootstrap_draws=1000,
    seed=0,
):
    """Chronological latent-vs-identity comparison for shares and totals.

    Development selection uses seasons in `train_seasons` with an inner holdout
    at `validation_season` for the identity penalty. Confirmation seasons are
    scored only after selection. Stage D injury haircuts are deferred unless a
    separate injury-timestamp audit is supplied.
    """
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    seasons = list(train_seasons) + list(evaluation_seasons)
    available = []
    for year in seasons:
        if (raw_root / f"pbp/season={year}/part.parquet").exists() and (
                raw_root / f"rosters_weekly/season={year}/part.parquet").exists():
            available.append(year)
    if not available:
        raise FileNotFoundError(f"no cached pbp/rosters seasons under {raw_root}")
    missing = sorted(set(seasons) - set(available))
    train_seasons = [y for y in train_seasons if y in available]
    evaluation_seasons = [y for y in evaluation_seasons if y in available]
    if not train_seasons or not evaluation_seasons:
        raise ValueError(
            f"need train and evaluation seasons in cache; available={available}, missing={missing}"
        )

    pbp, rosters = _load_seasons(raw_root, available)
    panel, candidates, events, skill_players, ol_counts, audit = build_opportunity_panels(pbp, rosters)
    skills = build_skill_history(candidates, panel)
    vacated = vacated_role_features(skill_players, skills)
    defense = build_defense_latents(events, panel)
    team_feats, player_feats = attach_model_features(
        panel, candidates, skills, vacated, defense, ol_counts
    )

    write_frame(output_root / "team_games.parquet", panel)
    write_frame(output_root / "candidates.parquet", candidates)
    write_frame(output_root / "defense_latents.parquet", defense)
    atomic_json(output_root / "panel_audit.json", audit)

    child_seeds = np.random.SeedSequence(seed).spawn(len(evaluation_seasons))
    folds = []
    for year, child in zip(evaluation_seasons, child_seeds):
        years = [y for y in train_seasons if y < year] + [
            y for y in evaluation_seasons if y < year
        ]
        if validation_season in years and validation_season < year:
            select_train = [y for y in years if y < validation_season] or [
                y for y in years if y != validation_season
            ][:1]
            select_valid = [validation_season] if validation_season in years else []
        else:
            select_train = years[:-1] or years
            select_valid = years[-1:] if len(years) > 1 else []

        fold_report = {
            "evaluation_season": year,
            "train_seasons": years,
            "selection_train_seasons": select_train,
            "selection_valid_seasons": select_valid,
            "pools": {},
        }
        sample_seed, bootstrap_seed = [int(v) for v in child.generate_state(2)]
        for pool in pools:
            print(f"fitting {pool} evaluation={year} train={years}", flush=True)
            train_teams = team_feats.filter(
                (pl.col("pool") == pool) & pl.col("season").is_in(years)
            )
            train_players = player_feats.filter(
                (pl.col("pool") == pool) & pl.col("season").is_in(years)
            )
            obs_teams = team_feats.filter(
                (pl.col("pool") == pool) & (pl.col("season") == year)
            )
            obs_players = player_feats.filter(
                (pl.col("pool") == pool) & (pl.col("season") == year)
            )
            observed_panel, observed_candidates = _as_target_schema(
                panel.filter((pl.col("pool") == pool) & (pl.col("season") == year)),
                candidates.filter((pl.col("pool") == pool) & (pl.col("season") == year)),
            )
            future_panel = observed_panel.drop(
                "targets", "candidates", "candidate_targets", "other_targets"
            )
            future_candidates = observed_candidates.drop("targets")

            # Proportional baselines for context (targets-like schema).
            train_panel_obs, train_cand_obs = _as_target_schema(
                panel.filter((pl.col("pool") == pool) & pl.col("season").is_in(years)),
                candidates.filter((pl.col("pool") == pool) & pl.col("season").is_in(years)),
            )
            proportional = fit_proportional_allocation(train_panel_obs, train_cand_obs)
            prop_history = proportional.predict(future_panel, future_candidates, include_player_history=True)
            prop_position = proportional.predict(future_panel, future_candidates, include_player_history=False)

            # Identity ridge selection on earlier seasons only.
            if select_valid and select_train:
                selection = select_identity_ridge(
                    team_feats.filter(pl.col("pool") == pool),
                    player_feats.filter(pl.col("pool") == pool),
                    pool,
                    train_mask=pl.col("season").is_in(select_train),
                    valid_mask=pl.col("season").is_in(select_valid),
                    ridge=share_ridge,
                    grid=identity_grid,
                )
                identity_ridge = selection["identity_ridge"]
            else:
                selection = {"identity_ridge": 10.0, "scores": [], "note": "fallback_default"}
                identity_ridge = 10.0

            latent_share = fit_latent_shares(train_teams, train_players, pool, ridge=share_ridge)
            identity_share = fit_identity_shares(
                train_teams, train_players, pool, ridge=share_ridge, identity_ridge=identity_ridge
            )
            latent_pred = latent_share.predict(obs_teams, obs_players)
            identity_pred = identity_share.predict(obs_teams, obs_players)

            share_scores = {}
            loss_tables = {}
            for name, pred in (
                ("position_only", prop_position),
                ("player_history", prop_history),
                ("latent", latent_pred),
                ("identity", identity_pred),
            ):
                scored = allocation_log_scores(pred, observed_panel, observed_candidates)
                share_scores[name] = scored["report"]
                loss_tables[name] = scored["scores"].sort(list(TEAM_GAME_KEYS))

            finite = all(share_scores[n]["status"] == "ok" for n in ("latent", "identity"))
            primary = None
            if finite:
                primary = bootstrap_difference(
                    loss_tables["identity"]["negative_log_likelihood"].to_numpy(),
                    loss_tables["latent"]["negative_log_likelihood"].to_numpy(),
                    clusters=loss_tables["identity"]["game_id"].to_list(),
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                    level=0.95,
                )

            # Volume comparison with share parameters held fixed conceptually:
            # report team-pool CRPS for latent vs identity volume arms.
            latent_volume = fit_latent_volume(train_teams, train_players, pool)
            identity_volume = fit_identity_volume(
                train_teams, train_players, pool, identity_ridge=identity_ridge
            )
            lv = latent_volume.sample(obs_teams, obs_players, draws=volume_draws, seed=sample_seed)
            iv = identity_volume.sample(obs_teams, obs_players, draws=volume_draws, seed=sample_seed)
            truth = obs_teams.sort("game_id", "team")["total"].to_numpy().astype(float)
            # Align sample rows to obs_teams order used in predict.
            volume_report = {
                "latent": score_volume_predictions(lv["samples"], truth),
                "identity": score_volume_predictions(iv["samples"], truth),
                "latent_fit": latent_volume.report(),
                "identity_fit": identity_volume.report(),
            }

            fold_report["pools"][pool] = {
                "team_games": obs_teams.height,
                "candidate_rows": obs_players.height,
                "identity_selection": selection,
                "share_models": {
                    "latent": latent_share.report(),
                    "identity": identity_share.report(),
                    "proportional_baseline": proportional.report(),
                },
                "share_scores": share_scores,
                "primary_identity_minus_latent_nll": primary,
                "primary_status": "ok" if finite else "infinite_loss",
                "volume_scores": volume_report,
            }
            print(
                f"completed {pool} {year}: latent NLL={share_scores['latent'].get('mean_negative_log_likelihood')} "
                f"identity NLL={share_scores['identity'].get('mean_negative_log_likelihood')}",
                flush=True,
            )
        folds.append(fold_report)

    defense_encoding = {
        "features": [
            "def_lag_targets_allowed",
            "def_lag_rush_yards_allowed",
            "def_lag_success_allowed",
        ],
        "method": (
            "Expanding prior-game means of targets allowed, rush yards allowed, and play-success "
            "allowed when the opponent was on defense. Shrunk toward the expanding league mean "
            "with 4 pseudo-games. Only earlier (season, week) games enter; same-game excluded."
        ),
        "used_by": ["latent_shares", "identity_shares", "latent_volume", "identity_volume"],
        "sources": "play-by-play aggregates from cached nflverse pbp partitions",
    }
    report = {
        "purpose": "opportunity_latent_vs_identity",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "available_seasons": available,
        "missing_requested_seasons": missing,
        "train_seasons": train_seasons,
        "evaluation_seasons": evaluation_seasons,
        "pools": list(pools),
        "population": ["QB", "RB", "FB", "WR", "TE"],
        "defense_latents": defense_encoding,
        "panel_audit": audit,
        "folds": folds,
        "deferred": {
            "stage_d_injury": (
                "Injury practice-status haircut and scenario mixture deferred: no injuries "
                "partition audit in this run; unknown report status is not treated as healthy."
            ),
            "participation_on_field_rates": (
                "load_participation on-field rates not ingested; skill vector uses lagged pool "
                "shares and prior-game exposure only."
            ),
            "preferred_dev_window_2016_2023": (
                "Local cache lacked 2016-2021; fit used available seasons and documented gaps."
                if missing or min(available) > 2016 else "not applicable"
            ),
        },
        "interpretation": (
            "Conditional share scores use realized pool size and supplied weekly ACT rosters. "
            "Volume CRPS uses roster/defense-conditional NB means. Defense latents are shared by "
            "both arms. No causal injury effects are claimed."
        ),
    }
    atomic_json(output_root / "report.json", report)
    atomic_json(output_root / "defense_encoding.json", defense_encoding)
    return report


def write_results_markdown(report, path):
    """Write a short results note from an experiment report."""
    path = Path(path)
    lines = [
        "# Offensive opportunity experiment results",
        "",
        "Status: fitted comparison of latent-only vs identity-aware opportunity models.",
        "Protocol: [OPPORTUNITY_EXPERIMENT.md](OPPORTUNITY_EXPERIMENT.md).",
        "",
        "## What was fitted",
        "",
        f"- Population: {', '.join(report['population'])} (OL as volume covariates only).",
        f"- Pools: {', '.join(report['pools'])} with OTHER residual and conservation.",
        f"- Available seasons in cache: {report['available_seasons']}.",
        f"- Training seasons: {report['train_seasons']}; evaluation: {report['evaluation_seasons']}.",
        "- **Latent arm**: lagged skill vector + position + vacated-role features + defense latents; no player intercept.",
        "- **Identity arm**: same latents + partially pooled player intercept (shares); team and QB intercepts on volume only.",
        "",
        "## Defense latent encoding",
        "",
        report["defense_latents"]["method"],
        "",
        "Features: " + ", ".join(f"`{f}`" for f in report["defense_latents"]["features"]) + ".",
        "",
        "## Share metrics (mean team-game multinomial NLL, realized pool size)",
        "",
        "| Season | Pool | Position-only | Player-history | Latent | Identity | Identity − latent [95% CI] |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for fold in report["folds"]:
        year = fold["evaluation_season"]
        for pool, block in fold["pools"].items():
            scores = block["share_scores"]
            primary = block.get("primary_identity_minus_latent_nll")
            if primary and "estimate" in primary:
                delta = (
                    f"{primary['estimate']:.4f} "
                    f"[{primary['lower']:.4f}, {primary['upper']:.4f}]"
                )
            else:
                delta = block.get("primary_status", "n/a")
            lines.append(
                "| {year} | {pool} | {pos:.4f} | {hist:.4f} | {lat:.4f} | {ident:.4f} | {delta} |".format(
                    year=year,
                    pool=pool,
                    pos=scores["position_only"]["mean_negative_log_likelihood"],
                    hist=scores["player_history"]["mean_negative_log_likelihood"],
                    lat=scores["latent"]["mean_negative_log_likelihood"],
                    ident=scores["identity"]["mean_negative_log_likelihood"],
                    delta=delta,
                )
            )
    lines += [
        "",
        "## Volume metrics (team-pool CRPS)",
        "",
        "| Season | Pool | Latent CRPS | Identity CRPS | Latent MAE | Identity MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for fold in report["folds"]:
        year = fold["evaluation_season"]
        for pool, block in fold["pools"].items():
            vol = block["volume_scores"]
            lines.append(
                "| {year} | {pool} | {lcrps:.4f} | {icrps:.4f} | {lmae:.4f} | {imae:.4f} |".format(
                    year=year,
                    pool=pool,
                    lcrps=vol["latent"]["crps"],
                    icrps=vol["identity"]["crps"],
                    lmae=vol["latent"]["mae"],
                    imae=vol["identity"]["mae"],
                )
            )
    lines += [
        "",
        "## Deferred / blockers",
        "",
    ]
    for key, value in report.get("deferred", {}).items():
        lines.append(f"- **{key}**: {value}")
    lines += [
        "",
        "## Reading",
        "",
        report["interpretation"],
        "",
        "Negative identity−latent NLL means identity assigns higher probability to the observed "
        "allocation after the skill vector and defense latents are included. Volume CRPS compares "
        "roster/defense-conditional pool totals; share parameters are not re-tuned to help volume.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
