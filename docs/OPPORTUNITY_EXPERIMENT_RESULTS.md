# Offensive opportunity experiment results

Status: fitted comparison of latent-only vs identity-aware opportunity models.
Protocol: [OPPORTUNITY_EXPERIMENT.md](OPPORTUNITY_EXPERIMENT.md).
Frozen CLI spec: [opportunity_latent_identity.json](../examples/opportunity_latent_identity.json).

## What was fitted

- Population: QB, RB, FB, WR, TE (OL as volume covariates only).
- Pools: targets, carries, dropbacks with OTHER residual and conservation.
- Available seasons in local cache: 2022–2025 (preferred 2016–2023 selection window not fully cached).
- Training: 2022–2023 (2025 fold expands through 2024). Evaluation / confirmation: 2024 and 2025.
- **Latent arm**: lagged skill vector + position + vacated-role features + defense latents; no player intercept.
- **Identity arm**: same latents + partially pooled player intercept (shares); team and QB intercepts on volume only.
- Share scores condition on realized pool size and supplied weekly ACT rosters. Volume uses roster/defense-conditional negative binomial means.

Run:

```powershell
uv run python -m hierarchical_football.inference.opportunity --train-seasons 2022 2023 --evaluation-seasons 2024 2025
```

## Defense latent encoding

Both arms receive the same opponent defense latents, built only from earlier games:

| Feature | Definition |
| --- | --- |
| `def_lag_targets_allowed` | Prior-game mean targets allowed when the opponent was on defense |
| `def_lag_rush_yards_allowed` | Prior-game mean rush yards allowed |
| `def_lag_success_allowed` | Prior-game mean play-success rate allowed |

Each rate shrinks toward the expanding league mean with 4 pseudo-games. Same-game and future games are excluded. Source: cached nflverse play-by-play aggregates (not `team_stats_week` in this fit).

## Share metrics (mean team-game multinomial NLL)

| Season | Pool | Position-only | Player-history | Latent | Identity | Identity − latent [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2024 | targets | 27.9225 | 21.9702 | 20.3583 | 19.8925 | -0.4658 [-0.5428, -0.3858] |
| 2024 | carries | 22.6337 | 15.3742 | 14.7837 | 14.4534 | -0.3302 [-0.4628, -0.1697] |
| 2024 | dropbacks | 22.8880 | 14.9293 | 9.9526 | 10.6821 | 0.7295 [0.2559, 1.2313] |
| 2025 | targets | 27.4772 | 21.1112 | 20.2602 | 19.8413 | -0.4189 [-0.5321, -0.2938] |
| 2025 | carries | 22.2124 | 14.9064 | 14.0446 | 13.7470 | -0.2977 [-0.4812, -0.1011] |
| 2025 | dropbacks | 22.9032 | 14.5418 | 10.9078 | 10.8962 | -0.0117 [-0.5012, 0.4211] |

### Share reading

- Both latent and identity beat sticky player-history on targets and carries.
- On targets and carries, identity still improves on the latent vector (negative paired NLL; intervals exclude 0).
- On dropbacks in 2024, latent beats identity; in 2025 they are tied. Sticky QB intercepts do not help after the dropback skill coordinate and defense latents are included.

## Volume metrics (team-pool CRPS)

| Season | Pool | Latent CRPS | Identity CRPS | Latent MAE | Identity MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024 | targets | 4.4299 | 4.2955 | 6.2236 | 6.0006 |
| 2024 | carries | 4.1402 | 4.1071 | 5.9689 | 5.9637 |
| 2024 | dropbacks | 4.7711 | 4.6153 | 6.7086 | 6.4748 |
| 2025 | targets | 4.3303 | 4.3514 | 6.1537 | 6.1813 |
| 2025 | carries | 3.9787 | 3.9835 | 5.6496 | 5.6705 |
| 2025 | dropbacks | 4.7129 | 4.6888 | 6.7548 | 6.7035 |

Identity volume (team + QB intercepts) helps 2024 pool totals slightly; 2025 is essentially tied. Share parameters were not re-tuned to help volume.

## Panel audit (2022–2025)

| Pool | Team-games | Total | Candidate total | OTHER | OTHER mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| targets | 2174 | 68745 | 68685 | 60 | 0.087% |
| carries | 2174 | 57118 | 57062 | 56 | 0.098% |
| dropbacks | 2174 | 77022 | 76854 | 168 | 0.218% |

Promoting QB/FB into the target candidate set leaves almost no residual target mass.

## Deferred / blockers

- **Stage D injury**: practice-status haircut and scenario mixture deferred — no injuries-partition timestamp audit in this run; unknown report status is not treated as healthy.
- **Participation on-field rates**: `load_participation` not used in the skill vector yet (ingest table registered; lagged pool shares + prior-game exposure only).
- **2016–2023 selection window**: local cache had 2022–2025 only; earlier seasons were not downloaded for this fit.
- Artifacts under `data/experiments/opportunity_latent_identity/` (report, panels, defense latents) stay local per `.gitignore`.

## Interpretation limits

Conditional share scores are not volume or availability forecasts. Defense latents are leakage-safe opponent history, not causal defensive grades. Scenario differences are model conditionals; this writeup does not claim causal injury effects.
