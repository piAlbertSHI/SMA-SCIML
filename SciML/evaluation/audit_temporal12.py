"""Strict, fail-closed audit for the registered Temporal12 evidence set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_CASES = 12
EXPECTED_CYCLES = 26
EXPECTED_STAGE_TRANSITIONS = 104
STRICT_COLUMNS = (
    "run_ok",
    "observation_contract_ok",
    "physics_ok",
    "paper_grade_valid",
    "export_provenance_available",
    "export_artifact_hash_matches",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _nonblank_fingerprints(rows: list[dict[str, str]]) -> set[str]:
    return {
        row.get("physics_implementation_fingerprint_sha256", "").strip().lower()
        for row in rows
        if row.get("physics_implementation_fingerprint_sha256", "").strip()
    }


def audit(
    temporal_audit_root: Path,
    paper_audit_root: Path,
    temporal_config: Path,
) -> dict[str, Any]:
    """Return a machine-readable report; ``passed`` is true only for all gates."""

    errors: list[str] = []
    quality = json.loads((temporal_audit_root / "quality_report.json").read_text(encoding="utf-8"))
    temporal_rows = _rows(temporal_audit_root / "case_index.csv")
    cycle_rows = _rows(temporal_audit_root / "cycle_index.csv")
    paper_rows = _rows(paper_audit_root / "case_index.csv")
    config = json.loads(temporal_config.read_text(encoding="utf-8"))
    configured = {str(case["case_id"]): int(case["cycle_count"]) for case in config["cases"]}

    if len(configured) != EXPECTED_CASES:
        errors.append(f"registered Temporal12 config has {len(configured)} unique cases, expected 12")
    observed_ids = [row.get("case_id", "") for row in temporal_rows]
    if len(temporal_rows) != EXPECTED_CASES or len(set(observed_ids)) != EXPECTED_CASES:
        errors.append("Temporal12 must contain exactly 12 unique case rows")
    if set(observed_ids) != set(configured):
        errors.append("Temporal12 case rows do not exactly match the registered config")
    if len({row.get("condition_id", "") for row in temporal_rows}) != EXPECTED_CASES:
        errors.append("Temporal12 must contain 12 distinct physical conditions")

    for column in STRICT_COLUMNS:
        failures = sorted(row.get("case_id", "") for row in temporal_rows if not _bool(row.get(column)))
        if failures:
            errors.append(f"{column} is not true for: {', '.join(failures)}")

    expected_quality = {
        "case_count": EXPECTED_CASES,
        "expected_case_count": EXPECTED_CASES,
        "export_provenance_complete_count": EXPECTED_CASES,
        "run_ok_count": EXPECTED_CASES,
        "observation_contract_ok_count": EXPECTED_CASES,
        "physics_ok_count": EXPECTED_CASES,
        "paper_grade_valid_count": EXPECTED_CASES,
        "cycle_sample_count": EXPECTED_CYCLES,
        "observed_stage_transition_count": EXPECTED_STAGE_TRANSITIONS,
        "physics_implementation_fingerprint_count": 1,
    }
    for key, expected in expected_quality.items():
        if int(quality.get(key, -1)) != expected:
            errors.append(f"quality_report.{key}={quality.get(key)!r}, expected {expected}")
    if not _bool(quality.get("expected_matrix_complete")):
        errors.append("quality_report.expected_matrix_complete is false")
    if quality.get("missing_expected_case_ids"):
        errors.append("quality_report contains missing expected Temporal12 cases")
    if quality.get("unexpected_case_ids"):
        errors.append("quality_report contains unexpected Temporal12 cases")

    if len(cycle_rows) != EXPECTED_CYCLES:
        errors.append(f"Temporal12 has {len(cycle_rows)} cycle samples, expected 26")
    cycle_counts = Counter(row.get("case_id", "") for row in cycle_rows)
    for case_id, expected_cycles in configured.items():
        indices = sorted(
            int(row.get("cycle_index", 0))
            for row in cycle_rows
            if row.get("case_id") == case_id
        )
        if cycle_counts[case_id] != expected_cycles or indices != list(range(1, expected_cycles + 1)):
            errors.append(f"{case_id} does not contain the registered {expected_cycles} ordered cycles")
    if any(not _bool(row.get("physics_ok")) for row in cycle_rows):
        errors.append("one or more Temporal12 cycle samples are not physics-valid")

    temporal_fingerprints = _nonblank_fingerprints(temporal_rows)
    paper_fingerprints = _nonblank_fingerprints(paper_rows)
    if len(temporal_fingerprints) != 1:
        errors.append("Temporal12 does not have one nonblank physics fingerprint")
    if len(paper_fingerprints) != 1:
        errors.append("Paper220 audit does not have one nonblank physics fingerprint")
    if temporal_fingerprints != paper_fingerprints:
        errors.append("Temporal12 physics fingerprint differs from Paper220")

    return {
        "schema_version": "sciml_v11_temporal12_strict_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passed": not errors,
        "errors": errors,
        "case_count": len(temporal_rows),
        "distinct_condition_count": len({row.get("condition_id", "") for row in temporal_rows}),
        "cycle_sample_count": len(cycle_rows),
        "stage_transition_count": 4 * len(cycle_rows),
        "physics_implementation_fingerprint_sha256": next(iter(temporal_fingerprints), ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-audit-root", type=Path, required=True)
    parser.add_argument("--paper-audit-root", type=Path, required=True)
    parser.add_argument("--temporal-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.temporal_audit_root, args.paper_audit_root, args.temporal_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
