"""Outcome-independent marginal domain checks for a fixed reference population."""
from __future__ import annotations

import numpy as np
import polars as pl

from .baselines import PASSING_CONTEXT_COMPONENTS, passing_context_sources
from .contracts import PASSING_CONTRACT, column_labels, validate


def model_context_population(windows, population, *, context="expanded"):
    """Retain Q rows whose modeled categories and numeric ranges occur in every window.

    This screens the source variables of fixed depth hinges, not outcome-derived values.
    Marginal ranges and categories do not establish joint density, player-specific support,
    or causal identification. Per-column failure counts can overlap.
    """
    if not windows or context not in PASSING_CONTEXT_COMPONENTS:
        raise ValueError("model context support needs fitting windows and a declared passing context")
    validate(population, PASSING_CONTRACT)
    sources = passing_context_sources(context)
    keep = np.ones(population.height, dtype=bool)
    reports = {}
    for name, table in windows.items():
        validate(table, PASSING_CONTRACT)
        if not table.height:
            raise ValueError("model context support cannot derive a domain from an empty fitting window")
        within = np.ones(population.height, dtype=bool)
        unseen, outside, nonfinite, bounds, levels = {}, {}, {}, {}, {}
        for column in sources['categorical']:
            observed = set(column_labels(table, column))
            supported = np.array([value in observed for value in column_labels(population, column)], dtype=bool)
            unseen[column] = int((~supported).sum())
            levels[column] = sorted(observed)
            within &= supported
        for column in sources['continuous']:
            observed = table[column].cast(pl.Float64).to_numpy()
            observed = observed[np.isfinite(observed)]
            values = population[column].cast(pl.Float64).to_numpy()
            finite = np.isfinite(values)
            nonfinite[column] = int((~finite).sum())
            lower, upper = ((float(observed.min()), float(observed.max())) if observed.size else (None, None))
            supported = ((values >= lower) & (values <= upper) & finite if observed.size else
                         np.zeros(population.height, dtype=bool))
            outside[column] = int((finite & ~supported).sum())
            bounds[column] = {"lower": lower, "upper": upper, "observed_rows": int(observed.size)}
            within &= supported
        reports[name] = {"rows_with_any_failure": int((~within).sum()),
                         "unseen_categorical_rows": unseen, "outside_continuous_rows": outside,
                         "nonfinite_continuous_rows": nonfinite, "bounds": bounds, "levels": levels}
        keep &= within
    selected = population.filter(pl.Series(keep))
    return {"population": selected, "report": {
        "status": "ok" if selected.height else "no_shared_model_context", "context": context,
        "source_rows": population.height, "retained_rows": selected.height,
        "retained_share": selected.height / population.height if population.height else None,
        "windows": reports,
        "note": "marginal model category/range screen; not joint positivity or a density estimate; counts overlap"}}
