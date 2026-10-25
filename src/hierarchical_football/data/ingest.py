"""Explicit network ingestion; panel building never downloads implicitly."""
from __future__ import annotations
from pathlib import Path
import importlib.metadata
import json
import polars as pl
from .store import atomic_json, utc_now, write_frame

DEFAULT_TABLES = ("schedules", "player_stats_week", "rosters_weekly")
TABLES = (*DEFAULT_TABLES, "team_stats_week", "snap_counts", "ff_playerids", "injuries", "depth_charts", "pbp")


def _prior_report(path: Path) -> dict:
    """The previous report, or an empty one when absent or unreadable."""
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return prior if isinstance(prior, dict) else {}


def _merge_partitions(previous, current):
    """One entry per table and season; the newest record for a key wins."""
    merged = {}
    for entry in (*previous, *current):
        if isinstance(entry, dict) and {"table", "season"} <= set(entry):
            merged[(entry["table"], entry["season"])] = entry
    return [merged[key] for key in sorted(merged)]


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
    report_path = raw / "ingest_report.json"
    previous = _prior_report(report_path)
    started_at = utc_now()
    report = {"started_at": previous.get("started_at", started_at),
              "updated_at": started_at, "partitions": [], "errors": [],
              "runs": list(previous.get("runs", []))}
    run_partitions = report["partitions"]
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
    report["runs"].append({
        "started_at": started_at,
        "seasons": sorted(set(seasons)),
        "tables": list(tables),
        "downloaded": sum(1 for entry in run_partitions if entry["status"] == "downloaded"),
        "cached": sum(1 for entry in run_partitions if entry["status"] == "cached"),
        "errors": report["errors"],
    })
    report["partitions"] = _merge_partitions(previous.get("partitions", []), run_partitions)
    report["updated_at"] = utc_now()
    try:
        report["nflreadpy_version"] = importlib.metadata.version("nflreadpy")
    except importlib.metadata.PackageNotFoundError:
        report["nflreadpy_version"] = None
    atomic_json(report_path, report)
    if report["errors"]:
        raise RuntimeError(f"{len(report['errors'])} source partitions failed; see {report_path}")
    return report
