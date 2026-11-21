"""Check development/solver identity before spending confirmation compute.

This is a necessary provenance gate, not a numerical validation or a readiness
certificate. It reads requests, never rewrites their recorded source hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from SciML.builders.build_dataset_v11_pilot import (
    PIPELINE_ONLY_FINGERPRINT_NAMES,
    fingerprint_logical_path,
)


PHYSICS_SOURCES = (
    "SciML/builders/Run_SCIML_V11_Pilot.m",
    "GlobalXi_TimeStep_Core.m",
    "Run_GlobalXi_Simulink_Cycles.m",
    "Run_Simulink_Exchange_Hook.m",
    "Build_Simulink_Exchange_Params.m",
    "Configure_Exchange_Pipe_Initial_Pressure.m",
    "Configure_Exchange_Consistency_Initialization.m",
    "nixunhuan_solver101_internal.slx",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _aggregate(sources: dict[str, str]) -> str:
    # Identical logical-path/NUL/hash/NUL convention to the dataset builder.
    digest = hashlib.sha256()
    for path, value in sorted(sources.items()):
        digest.update(path.encode("utf-8") + b"\0" + value.encode("ascii") + b"\0")
    return digest.hexdigest()


def _recorded_sources(request: dict[str, Any]) -> dict[str, str]:
    entries = request.get("implementation_fingerprint")
    if not isinstance(entries, list) or not entries:
        raise ValueError("request has no implementation fingerprint")
    config_name = fingerprint_logical_path(str(request.get("source_config_path", ""))).split("/")[-1]
    excluded = PIPELINE_ONLY_FINGERPRINT_NAMES | {config_name}
    sources: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed request fingerprint entry")
        path = fingerprint_logical_path(str(entry.get("path", "")))
        if path.split("/")[-1] in excluded:
            continue
        value = str(entry.get("sha256", ""))
        if path in sources or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"duplicate or invalid physics source: {path}")
        sources[path] = value
    if set(sources) != set(PHYSICS_SOURCES):
        raise ValueError("request does not bind the exact required physics source set")
    return sources


def audit_physics_identity(
    repo: Path,
    development_quality: Path,
    candidate_reports: dict[str, Path],
) -> dict[str, Any]:
    repo = repo.resolve()
    quality_path = development_quality.resolve()
    case_index = quality_path.parent / "case_index.csv"
    errors: list[str] = []
    current: dict[str, str] = {}
    for name in PHYSICS_SOURCES:
        path = repo / name
        if not path.is_file():
            errors.append(f"current physics source missing: {name}")
        else:
            current[name] = _sha256(path)
    quality = _json(quality_path)
    source_hash = str(quality.get("source_content_sha256", ""))
    if len(source_hash) != 64:
        errors.append("development quality lacks a source-content hash")
    if set(candidate_reports) != {"wp2", "wp3", "wp4"}:
        errors.append("exactly WP2, WP3 and WP4 development reports are required")
    for name, path in candidate_reports.items():
        report = _json(path)
        if report.get("dataset_source_content_sha256") != source_hash:
            errors.append(f"{name} candidate was not trained on the selected development dataset")

    with case_index.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    fingerprints: set[str] = set()
    mismatches: dict[str, set[str]] = {}
    request_count = 0
    for row in rows:
        case_id = row.get("case_id", "<missing>")
        # The builder records the locally aggregated artifact path, even when
        # the request's implementation paths refer to another machine.
        source_artifact = Path(row.get("source_artifact", ""))
        if not source_artifact.is_absolute():
            source_artifact = repo / source_artifact
        request_path = source_artifact.parent / "sciml_case_request.json"
        if not request_path.is_file():
            errors.append(f"{case_id}: source request missing: {request_path}")
            continue
        if _sha256(request_path) != row.get("source_request_sha256"):
            errors.append(f"{case_id}: source request hash differs from development index")
            continue
        try:
            recorded = _recorded_sources(_json(request_path))
        except (ValueError, OSError) as exc:
            errors.append(f"{case_id}: {exc}")
            continue
        request_count += 1
        fingerprint = _aggregate(recorded)
        fingerprints.add(fingerprint)
        if fingerprint != row.get("physics_implementation_fingerprint_sha256"):
            errors.append(f"{case_id}: physics fingerprint differs from source request")
        for name, value in recorded.items():
            if current.get(name) != value:
                mismatches.setdefault(name, set()).add(value)
    if not rows:
        errors.append("development index is empty")
    if len(fingerprints) != 1:
        errors.append("development requests must have one physics fingerprint")
    if mismatches:
        errors.append("development/current physics differ: " + ", ".join(sorted(mismatches)))
    return {
        "schema_version": "sciml_v12_physics_preflight_v1",
        "passed": not errors,
        "development_case_count": len(rows),
        "verified_request_count": request_count,
        "development_dataset_source_content_sha256": source_hash,
        "development_physics_fingerprints": sorted(fingerprints),
        "current_physics_fingerprint_sha256": _aggregate(current) if len(current) == len(PHYSICS_SOURCES) else "",
        "current_physics_sources": current,
        "development_case_index_sha256": _sha256(case_index),
        "source_mismatches": [
            {"path": name, "recorded_sha256": sorted(values), "current_sha256": current.get(name, "")}
            for name, values in sorted(mismatches.items())
        ],
        "errors": errors,
    }


def require_physics_identity(repo: Path, artifacts: dict[str, Path]) -> dict[str, Any]:
    audit = audit_physics_identity(
        repo, artifacts["development_quality_report"],
        {wp: artifacts[f"{wp}_development_report"] for wp in ("wp2", "wp3", "wp4")},
    )
    if not audit["passed"]:
        raise RuntimeError("pre-confirmation physics identity refused: " + "; ".join(audit["errors"][:8]))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--development-root", type=Path, default=Path("SciML/datasets/v1.1-paper220"))
    parser.add_argument("--reports-root", type=Path, default=Path("SciML/results/development"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else args.repo / path
    reports_root = resolve(args.reports_root)
    try:
        audit = audit_physics_identity(args.repo, resolve(args.development_root) / "quality_report.json", {
            "wp2": reports_root / "wp2_v12_development.json",
            "wp3": reports_root / "wp3_v12_neural_temporal_candidate.json",
            "wp4": reports_root / "wp4_v12_neural_candidate.json",
        })
    except (OSError, ValueError) as exc:
        audit = {"schema_version": "sciml_v12_physics_preflight_v1", "passed": False, "errors": [str(exc)]}
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
