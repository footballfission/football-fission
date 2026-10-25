"""Season-partitioned local storage with atomic writes and content manifests."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import uuid
import polars as pl


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def content_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_frame(path: Path, frame: pl.DataFrame, *, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary)
        digest = content_hash(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(path.with_suffix(".json"), dict(
        written_at=utc_now(), rows=frame.height, sha256=digest,
        schema={k: str(v) for k, v in frame.schema.items()}, **(metadata or {})))


def read_table(raw: Path, table: str, seasons, *, required=True):
    static = raw / table / "part.parquet"
    if table == "ff_playerids" and static.exists():
        return pl.read_parquet(static)
    paths = [raw / table / f"season={s}" / "part.parquet" for s in seasons]
    missing = [str(p) for p in paths if not p.exists()]
    if missing and required:
        raise FileNotFoundError("Missing raw partitions: " + ", ".join(missing))
    frames = [pl.read_parquet(p) for p in paths if p.exists()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
