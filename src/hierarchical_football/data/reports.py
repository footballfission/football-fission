"""Candidate injury report joins; source timestamps still require independent audit."""
from __future__ import annotations
import pandas as pd
import polars as pl


def latest_known_injury_status(players: pl.DataFrame, reports: pl.DataFrame) -> pl.DataFrame:
    """Latest report strictly before each row's cutoff; unknown status remains null.

    A report's stated modification time is not proof of historical publication.
    The output is a candidate scenario input, excluded from model feature lists.
    """
    result = players.to_pandas().copy()
    result["candidate_injury_status"] = None
    result["candidate_report_time"] = pd.NaT
    if reports.is_empty() or not {"gsis_id", "season", "week", "date_modified", "report_status"}.issubset(reports.columns):
        return pl.from_pandas(result)
    source = reports.to_pandas().rename(columns={"gsis_id": "player_id"})
    source["date_modified"] = pd.to_datetime(source.date_modified, utc=True, errors="coerce").dt.tz_localize(None)
    source = source.dropna(subset=["player_id", "date_modified"])
    keys = ["player_id", "season", "week"]
    if "team" in source:
        keys.append("team")
    result["_report_row"] = range(len(result))
    joined = result[["_report_row", "prediction_timestamp", *keys]].merge(
        source[[*keys, "date_modified", "report_status"]], on=keys, how="inner")
    joined = joined.loc[joined.date_modified.lt(joined.prediction_timestamp)].copy()
    # Conflicting reports at one timestamp have no defensible arbitrary winner.
    conflicts = joined.groupby(["_report_row", "date_modified"], dropna=False).report_status.nunique(dropna=False)
    if conflicts.gt(1).any():
        raise ValueError("Conflicting injury reports share the same timestamp")
    latest = joined.sort_values("date_modified").drop_duplicates("_report_row", keep="last").set_index("_report_row")
    result["candidate_injury_status"] = result._report_row.map(latest.report_status)
    result["candidate_report_time"] = result._report_row.map(latest.date_modified)
    return pl.from_pandas(result.drop(columns="_report_row"))
