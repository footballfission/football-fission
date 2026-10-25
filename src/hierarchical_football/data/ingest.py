"""Explicit network ingestion; panel building never downloads implicitly."""
from __future__ import annotations
from pathlib import Path
import importlib.metadata
import polars as pl
from .store import atomic_json, utc_now, write_frame

DEFAULT_TABLES = ("schedules", "player_stats_week", "rosters_weekly")
TABLES = (*DEFAULT_TABLES, "team_stats_week", "snap_counts", "ff_playerids", "injuries", "depth_charts", "pbp")


def ingest(raw: Path, seasons, tables=DEFAULT_TABLES, *, refresh=False, loader=None):
    if not seasons or any(s < 1999 or s > 2100 for s in seasons):
        raise ValueError("Specify seasons between 1999 and 2100")
    unknown = set(tables) - set(TABLES)
    if unknown:
        raise ValueError(f"Unknown tables: {sorted(unknown)}")
    if loader is None:
        import nflreadpy as nfl
        functions = {
            "schedules": nfl.load_schedules,
            "player_stats_week": lambda s: nfl.load_player_stats(s, summary_level="week"),
            "team_stats_week": lambda s: nfl.load_team_stats(s, summary_level="week"),
            "rosters_weekly": nfl.load_rosters_weekly, "snap_counts": nfl.load_snap_counts,
            "ff_playerids": lambda s: nfl.load_ff_playerids(),
            "injuries": nfl.load_injuries, "depth_charts": nfl.load_depth_charts,
            "pbp": nfl.load_pbp,
        }
        loader = lambda table, season: functions[table](season)
    report = {"started_at": utc_now(), "partitions": [], "errors": []}
    for table in tables:
        for season in sorted(set(seasons)):
            path = raw / table / f"season={season}" / "part.parquet"
            if path.exists() and not refresh:
                report["partitions"].append({"table": table, "season": season, "status": "cached"})
                continue
            try:
                frame = loader(table, season)
                if not isinstance(frame, pl.DataFrame) or frame.is_empty():
                    raise ValueError("Source returned no rows")
                if "season" in frame.columns:
                    frame = frame.filter(pl.col("season").cast(pl.Int32) == season)
                else:
                    frame = frame.with_columns(pl.lit(season).alias("season"))
                if frame.is_empty():
                    raise ValueError("No rows match requested season")
                write_frame(path, frame, metadata={"table": table, "season": season,
                    "retrieved_at": utc_now(), "historical_asof_verified": False})
                report["partitions"].append({"table": table, "season": season, "status": "downloaded", "rows": frame.height})
            except Exception as error:
                report["errors"].append({"table": table, "season": season, "error": str(error)})
    try:
        report["nflreadpy_version"] = importlib.metadata.version("nflreadpy")
    except importlib.metadata.PackageNotFoundError:
        report["nflreadpy_version"] = None
    atomic_json(raw / "ingest_report.json", report)
    if report["errors"]:
        raise RuntimeError(f"{len(report['errors'])} source partitions failed; see {raw / 'ingest_report.json'}")
    return report
