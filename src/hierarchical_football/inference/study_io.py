"""Validated file specifications and traceable, non-overwriting offline study bundles."""
from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import polars as pl

from ..data.store import atomic_json, content_hash, utc_now, write_frame
from .study import PassingStudySpec, run_passing_study

THREAD_ENVIRONMENT_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate specification key {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def _read_specification(path):
    raw = path.read_bytes()
    config = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(config, dict) or set(config) - {"schema_version", "study", "contrasts", "contrast_draws"}:
        raise ValueError("unknown study specification fields")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("specification schema_version must be 1")
    if not isinstance(config.get("study"), dict):
        raise ValueError("specification needs a study object")
    try:
        spec = PassingStudySpec(**config["study"])
    except (TypeError, KeyError) as error:
        raise ValueError(f"invalid study specification: {error}") from error
    contrasts = config.get("contrasts", [])
    if not isinstance(contrasts, list):
        raise ValueError("contrasts must be a list")
    for request in contrasts:
        required = {"identity", "left", "right"}
        allowed = required | {"fixed", "population_position", "scale"}
        if not isinstance(request, dict) or set(request) - allowed or not required <= set(request):
            raise ValueError("contrast needs identity, left, right and only declared optional fields")
        if (request["identity"] not in ("qb_id", "receiver_id") or
                any(not isinstance(request[key], str) or not request[key].strip() for key in required)):
            raise ValueError("contrast identities and player labels must be nonempty passing labels")
        fixed = request.get("fixed", {})
        if (not isinstance(fixed, dict) or set(fixed) - {"qb_id", "receiver_id"} or
                request["identity"] in fixed or
                any(not isinstance(value, str) or not value.strip() for value in fixed.values())):
            raise ValueError("contrast fixed levels must name other passing identities")
        if request.get("scale", "probability") not in ("probability", "log_odds"):
            raise ValueError("contrast scale must be probability or log_odds")
        if request.get("population_position") not in (None, "WR", "TE", "RB", "QB", "FB", "UNKNOWN"):
            raise ValueError("contrast population_position must be a canonical position")
    draws = config.get("contrast_draws", 200)
    if type(draws) is not int or draws < 20:
        raise ValueError("contrast_draws must be an integer of at least 20")
    return raw, spec, contrasts, draws


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {path.relative_to(root).as_posix(): content_hash(path)
            for path in sorted(root.rglob("*.py"))}


def _revision():
    try:
        result = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_study_file(input_path, specification_path, output_root):
    """Write a new offline run with exact config, source/input/output hashes and runtime versions.

    Validate configuration before creating the output directory. Existing output paths are
    refused. A manifest is written last; a failed run retains failure.json instead. No network
    access, model promotion, or untouched-holdout certification is performed.
    """
    input_path, specification_path, output_root = map(Path, (input_path, specification_path, output_root))
    raw, spec, contrasts, contrast_draws = _read_specification(specification_path)
    return _run_bundle(
        input_path, specification_path, output_root, raw=raw, requested_seasons=spec.requested_seasons,
        execute=lambda table: run_passing_study(table, spec=spec, contrasts=contrasts, contrast_draws=contrast_draws))


def run_comparison_file(input_path, specification_path, output_root):
    """Persist a validated retrospective comparison and its exact shared reference rows."""
    from .comparisons import PassingComparisonSpec, run_passing_comparison

    raw = Path(specification_path).read_bytes()
    config = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if (not isinstance(config, dict) or set(config) != {"schema_version", "comparison"} or
            type(config.get("schema_version")) is not int or config["schema_version"] != 1 or
            not isinstance(config.get("comparison"), dict)):
        raise ValueError("comparison protocol needs schema_version 1 and a comparison object")
    try:
        spec = PassingComparisonSpec(**config["comparison"])
    except (TypeError, KeyError) as error:
        raise ValueError(f"invalid comparison specification: {error}") from error
    return _run_bundle(
        input_path, specification_path, output_root, raw=raw, requested_seasons=spec.requested_seasons,
        execute=lambda table: run_passing_comparison(table, spec=spec), frame_key="reference_population",
        frame_filename="reference_population.parquet", team=spec.team)


def _run_bundle(input_path, specification_path, output_root, *, raw, requested_seasons, execute,
                 frame_key="predictions", frame_filename="predictions.parquet", team=None):
    """Common publication boundary; public workflows validate their own protocol first."""
    input_path, specification_path, output_root = map(Path, (input_path, specification_path, output_root))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        source_hashes, input_hash = _source_hashes(), content_hash(input_path)
        thread_environment = {key: os.environ.get(key) for key in THREAD_ENVIRONMENT_KEYS}
        spec_hash = hashlib.sha256(raw).hexdigest()
        started_at, revision = utc_now(), _revision()
        (output_root / "specification.json").write_bytes(raw)
        # Predicate pushdown excludes undeclared/reserved seasons before materialization.
        lazy = pl.scan_parquet(input_path).filter(pl.col("season").is_in(requested_seasons))
        if team is not None:
            lazy = lazy.filter(pl.col("team") == team)
        result = execute(lazy.collect())
        if (_source_hashes() != source_hashes or content_hash(input_path) != input_hash or
                content_hash(specification_path) != spec_hash):
            raise ValueError("source, input or specification changed during the study")
        json.dumps(result["report"], allow_nan=False)
        atomic_json(output_root / "report.json", result["report"])
        write_frame(output_root / frame_filename, result[frame_key])
        outputs = {name: content_hash(output_root / name) for name in
                   ("specification.json", "report.json", frame_filename, Path(frame_filename).with_suffix(".json").name)}
        manifest = {"schema_version": 1, "status": "complete", "started_at": started_at,
                    "completed_at": utc_now(), "input": {"path": str(input_path.resolve()), "sha256": input_hash},
                    "specification_sha256": spec_hash, "code_revision": revision,
                    "source_hashes": source_hashes, "outputs": outputs,
                    "environment": {"python": platform.python_version(),
                                    "thread_environment": thread_environment, "packages": {
                        package: importlib.metadata.version(package)
                        for package in ("numpy", "scipy", "polars", "scikit-learn")}}}
        atomic_json(output_root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        atomic_json(output_root / "failure.json", {"status": "failed", "failed_at": utc_now(),
                                                   "error_type": type(error).__name__, "error": str(error)})
        raise
