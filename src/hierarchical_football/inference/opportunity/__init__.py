"""Offensive opportunity share and total models (latent vs identity).

See docs/OPPORTUNITY_EXPERIMENT.md for the research protocol and
docs/OPPORTUNITY_EXPERIMENT_RESULTS.md for fitted comparisons.
"""
from .experiment import run_opportunity_experiment
from .pools import POOLS, SKILL_POSITIONS, build_opportunity_panels
from .shares import fit_identity_shares, fit_latent_shares
from .totals import fit_identity_volume, fit_latent_volume

__all__ = [
    "POOLS",
    "SKILL_POSITIONS",
    "build_opportunity_panels",
    "fit_identity_shares",
    "fit_identity_volume",
    "fit_latent_shares",
    "fit_latent_volume",
    "run_opportunity_experiment",
]
