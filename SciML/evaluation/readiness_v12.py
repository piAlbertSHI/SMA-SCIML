"""Fail-closed final readiness audit for the frozen WP2--WP4 v1.2 campaign.

The v1.1 development corpus and the untouched v1.2 confirmation corpora have
different evidentiary roles.  This audit keeps those roles explicit: at least
200 independent physics-valid development conditions are still required, all
50 Confirm50 conditions and all 12 Confirm-Temporal12 conditions must pass the
registered data/provenance gates, and each work package must provide a report
that is bound to the pre-confirmation freeze and to the relevant confirmation
dataset hashes.  Missing evidence is a failure, never an implicit pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from SciML.evaluation.confirmation_freeze_v12 import verify_manifest


REQUIRED_TRUE_FIELDS = (
    "run_ok",
    "physics_ok",
    "observation_contract_ok",
    "paper_grade_valid",
    "request_config_matches_current",
    "request_self_contained",
    "export_provenance_available",
    "export_artifact_hash_matches",
)
WP_REPORTS = {
    "wp2": Path("SciML/results/wp2_v12_confirmation.json"),
    "wp3": Path("SciML/results/wp3_v12_confirmation.json"),
    "wp4": Path("SciML/results/wp4_v12_confirmation.json"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _dataset_evidence(
    root: Path,
    config_path: Path,
    *,
    expected_cases: int,
    expected_cycles: int | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    for name in (
        "case_index.csv",
        "quality_report.json",
        "provenance.json",
        "artifact_manifest.json",
    ):
        if not (root / name).is_file():
            return {}, [f"missing confirmation dataset artifact: {root / name}"]
    rows = _rows(root / "case_index.csv")
    quality = _json(root / "quality_report.json")
    provenance = _json(root / "provenance.json")
    artifact_manifest = _json(root / "artifact_manifest.json")
    config = _json(config_path)
    configured = {str(case["case_id"]): int(case["cycle_count"]) for case in config["cases"]}
    observed_ids = [str(row.get("case_id", "")) for row in rows]
    conditions = {str(row.get("condition_id", "")) for row in rows}
    if len(configured) != expected_cases:
        errors.append(f"config has {len(configured)} unique cases, expected {expected_cases}")
    if len(rows) != expected_cases or len(set(observed_ids)) != expected_cases:
        errors.append(f"dataset must contain exactly {expected_cases} unique case rows")
    if set(observed_ids) != set(configured):
        errors.append("dataset case IDs do not exactly match the frozen config")
    if len(conditions) != expected_cases or "" in conditions:
        errors.append(f"dataset must contain {expected_cases} independent conditions")
    for field in REQUIRED_TRUE_FIELDS:
        failed = sorted(row.get("case_id", "") for row in rows if not _as_bool(row.get(field)))
        if failed:
            errors.append(f"{field} is not true for: {', '.join(failed)}")
    exact_quality = {
        "case_count": expected_cases,
        "expected_case_count": expected_cases,
        "run_ok_count": expected_cases,
        "physics_ok_count": expected_cases,
        "paper_grade_valid_count": expected_cases,
        "observation_contract_ok_count": expected_cases,
        "export_provenance_complete_count": expected_cases,
        "physics_implementation_fingerprint_count": 1,
    }
    for key, expected in exact_quality.items():
        if int(quality.get(key, -1)) != expected:
            errors.append(f"quality_report.{key}={quality.get(key)!r}, expected {expected}")
    if not _as_bool(quality.get("expected_matrix_complete")):
        errors.append("quality_report.expected_matrix_complete is false")
    if quality.get("missing_expected_case_ids") or quality.get("unexpected_case_ids"):
        errors.append("quality_report records missing or unexpected cases")
    confirmation_gate = quality.get("confirmation_quality_gate", {})
    if not isinstance(confirmation_gate, dict) or confirmation_gate.get("passed") is not True:
        errors.append("quality_report.confirmation_quality_gate.passed is not true")
    source_hash = str(quality.get("source_content_sha256", ""))
    if len(source_hash) != 64 or source_hash != str(
        provenance.get("source_content_sha256", "")
    ):
        errors.append("quality/provenance source-content hash mismatch")
    if str(provenance.get("expected_config_sha256", "")).lower() != _sha256(
        config_path
    ):
        errors.append("dataset provenance is not bound to the selected config")
    if str(artifact_manifest.get("source_content_sha256", "")) != source_hash:
        errors.append("artifact manifest source-content hash mismatch")
    manifest_artifacts = artifact_manifest.get("artifacts", [])
    if not isinstance(manifest_artifacts, list) or not manifest_artifacts:
        errors.append("artifact manifest has no bound artifacts")
    else:
        for artifact in manifest_artifacts:
            if not isinstance(artifact, dict):
                errors.append("artifact manifest contains an invalid entry")
                continue
            path = root / str(artifact.get("path", ""))
            if not path.is_file() or _sha256(path) != str(artifact.get("sha256", "")):
                errors.append(f"artifact manifest hash mismatch: {path.name}")
    cycle_rows: list[dict[str, str]] = []
    if expected_cycles is not None:
        cycle_path = root / "cycle_index.csv"
        if not cycle_path.is_file():
            errors.append(f"missing temporal cycle table: {cycle_path}")
        else:
            cycle_rows = _rows(cycle_path)
            if len(cycle_rows) != expected_cycles:
                errors.append(f"temporal dataset has {len(cycle_rows)} cycles, expected {expected_cycles}")
            for case_id, count in configured.items():
                indices = sorted(
                    int(row.get("cycle_index", 0))
                    for row in cycle_rows
                    if row.get("case_id") == case_id
                )
                if indices != list(range(1, count + 1)):
                    errors.append(f"{case_id} lacks its exact ordered {count}-cycle evidence")
            if any(not _as_bool(row.get("physics_ok")) for row in cycle_rows):
                errors.append("one or more confirmation cycles are not physics-valid")
    fingerprint = {
        str(row.get("physics_implementation_fingerprint_sha256", "")).strip().lower()
        for row in rows
        if str(row.get("physics_implementation_fingerprint_sha256", "")).strip()
    }
    if len(fingerprint) != 1:
        errors.append("dataset does not contain exactly one nonblank physics fingerprint")
    return {
        "root": str(root.resolve()),
        "case_count": len(rows),
        "condition_count": len(conditions - {""}),
        "cycle_count": len(cycle_rows),
        "source_content_sha256": str(quality.get("source_content_sha256", "")),
        "physics_implementation_fingerprint_sha256": next(iter(fingerprint), ""),
        "passed": not errors,
    }, errors


def audit_v12_readiness(
    repo: Path,
    development_root: Path,
    confirm50_root: Path,
    temporal12_root: Path,
    freeze_manifest: Path,
    report_paths: dict[str, Path] | None = None,
    confirm50_config: Path = Path("SciML/configs/wp1_v12_confirm50.json"),
    temporal12_config: Path = Path("SciML/configs/wp1_v12_confirm_temporal12.json"),
) -> dict[str, Any]:
    repo = repo.resolve()
    development_root = _resolve(repo, development_root)
    confirm50_root = _resolve(repo, confirm50_root)
    temporal12_root = _resolve(repo, temporal12_root)
    freeze_path = _resolve(repo, freeze_manifest)
    confirm50_config = _resolve(repo, confirm50_config)
    temporal12_config = _resolve(repo, temporal12_config)
    errors: list[str] = []
    try:
        freeze = verify_manifest(repo, freeze_manifest)
    except Exception as exc:
        freeze = {}
        errors.append(f"confirmation freeze verification failed: {type(exc).__name__}: {exc}")
    freeze_hash = _sha256(freeze_path) if freeze_path.is_file() else ""
    if freeze:
        for name, path in (
            ("confirm50_config", confirm50_config),
            ("confirm_temporal12_config", temporal12_config),
        ):
            binding = freeze.get("artifact_bindings", {}).get(name, {})
            if not path.is_file() or binding.get("sha256") != _sha256(path):
                errors.append(f"selected {name} differs from the frozen campaign")

    development: dict[str, Any] = {}
    if not (development_root / "case_index.csv").is_file() or not (
        development_root / "quality_report.json"
    ).is_file():
        errors.append("development dataset audit is missing")
    else:
        dev_rows = _rows(development_root / "case_index.csv")
        dev_valid = [
            row
            for row in dev_rows
            if all(_as_bool(row.get(field)) for field in REQUIRED_TRUE_FIELDS)
        ]
        dev_conditions = {row.get("condition_id", "") for row in dev_valid} - {""}
        dev_fingerprints = {
            str(row.get("physics_implementation_fingerprint_sha256", "")).strip().lower()
            for row in dev_valid
            if str(row.get("physics_implementation_fingerprint_sha256", "")).strip()
        }
        development = {
            "physics_valid_case_count": len(dev_valid),
            "physics_valid_condition_count": len(dev_conditions),
            "physics_implementation_fingerprint_sha256": next(iter(dev_fingerprints), ""),
            "passed": len(dev_conditions) >= 200 and len(dev_fingerprints) == 1,
        }
        if len(dev_conditions) < 200:
            errors.append(f"development corpus has {len(dev_conditions)} strict valid conditions; >=200 required")
        if len(dev_fingerprints) != 1:
            errors.append("development corpus lacks one consistent physics fingerprint")

    confirm, confirm_errors = _dataset_evidence(
        confirm50_root,
        confirm50_config,
        expected_cases=50,
        expected_cycles=None,
    )
    temporal, temporal_errors = _dataset_evidence(
        temporal12_root,
        temporal12_config,
        expected_cases=12,
        expected_cycles=26,
    )
    errors.extend(f"Confirm50: {error}" for error in confirm_errors)
    errors.extend(f"Confirm-Temporal12: {error}" for error in temporal_errors)
    fingerprint_values = (
        development.get("physics_implementation_fingerprint_sha256", ""),
        confirm.get("physics_implementation_fingerprint_sha256", ""),
        temporal.get("physics_implementation_fingerprint_sha256", ""),
    )
    fingerprints = {value for value in fingerprint_values if value}
    fingerprint_consistent = all(fingerprint_values) and len(fingerprints) == 1
    if not fingerprint_consistent:
        errors.append("development, Confirm50 and Confirm-Temporal12 physics fingerprints differ")

    reports: dict[str, Any] = {}
    selected_reports = WP_REPORTS if report_paths is None else report_paths
    for wp in ("wp2", "wp3", "wp4"):
        path = _resolve(repo, selected_reports[wp])
        report_errors: list[str] = []
        payload: dict[str, Any] = {}
        if not path.is_file():
            report_errors.append(f"missing report: {path}")
        else:
            try:
                payload = _json(path)
            except Exception as exc:
                report_errors.append(f"unreadable report: {type(exc).__name__}: {exc}")
        binding = freeze.get("candidate_bindings", {}).get(wp, {}) if freeze else {}
        if payload:
            for key in ("confirmatory", "readiness_eligible", "paper_grade_ready"):
                if payload.get(key) is not True:
                    report_errors.append(f"{key} is not true")
            if payload.get("freeze_manifest_sha256") != freeze_hash:
                report_errors.append("freeze manifest hash mismatch")
            for key in ("implementation_sha256", "parameter_sha256"):
                if payload.get(key) != binding.get(key):
                    report_errors.append(f"{key} differs from frozen candidate")
            checkpoint_text = str(payload.get("checkpoint", ""))
            checkpoint = _resolve(repo, Path(checkpoint_text)) if checkpoint_text else None
            if checkpoint is None or not checkpoint.is_file():
                report_errors.append("checkpoint is missing")
            elif payload.get("checkpoint_sha256") != _sha256(checkpoint):
                report_errors.append("checkpoint SHA-256 mismatch")
            if payload.get("checkpoint_roundtrip_verified") is not True:
                report_errors.append("checkpoint round-trip is not verified")
            bindings = payload.get("confirmation_bindings", {})
            frozen_artifacts = freeze.get("artifact_bindings", {}) if freeze else {}
            if wp in {"wp2", "wp4"} and bindings.get(
                "confirm50_source_content_sha256"
            ) != confirm.get("source_content_sha256"):
                report_errors.append("Confirm50 dataset binding mismatch")
            reported_confirm50_config = bindings.get(
                "confirm50_config_sha256",
                payload.get("confirmation_config_sha256"),
            )
            if wp in {"wp2", "wp4"} and reported_confirm50_config != frozen_artifacts.get(
                "confirm50_config", {}
            ).get("sha256"):
                report_errors.append("Confirm50 config binding mismatch")
            if wp in {"wp3", "wp4"} and bindings.get(
                "confirm_temporal12_source_content_sha256"
            ) != temporal.get("source_content_sha256"):
                report_errors.append("Confirm-Temporal12 dataset binding mismatch")
            if wp in {"wp3", "wp4"} and bindings.get(
                "confirm_temporal12_config_sha256"
            ) != frozen_artifacts.get("confirm_temporal12_config", {}).get("sha256"):
                report_errors.append("Confirm-Temporal12 config binding mismatch")
            if payload.get("blocking_reasons"):
                report_errors.append("report retains blocking reasons")
        reports[wp] = {
            "path": str(path.resolve()),
            "passed": not report_errors,
            "errors": report_errors,
        }
        errors.extend(f"{wp.upper()}: {error}" for error in report_errors)

    return {
        "schema_version": "sciml_v12_readiness_v1",
        "passed": not errors,
        "development": development,
        "confirm50": confirm,
        "confirm_temporal12": temporal,
        "physics_fingerprint_consistent": fingerprint_consistent,
        "freeze_manifest_sha256": freeze_hash,
        "reports": reports,
        "blocking_reasons": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--development-root", type=Path, default=Path("SciML/datasets/v1.1-paper220")
    )
    parser.add_argument(
        "--confirm50-root", type=Path, default=Path("SciML/datasets/v1.2-confirm50")
    )
    parser.add_argument(
        "--temporal12-root",
        type=Path,
        default=Path("SciML/datasets/v1.2-confirm-temporal12"),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("SciML/results/v12_confirmation_freeze.json"),
    )
    parser.add_argument(
        "--confirm50-config",
        type=Path,
        default=Path("SciML/configs/wp1_v12_confirm50.json"),
    )
    parser.add_argument(
        "--temporal12-config",
        type=Path,
        default=Path("SciML/configs/wp1_v12_confirm_temporal12.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("SciML/results/readiness_v12.json")
    )
    args = parser.parse_args()
    report = audit_v12_readiness(
        args.repo,
        args.development_root,
        args.confirm50_root,
        args.temporal12_root,
        args.freeze_manifest,
        confirm50_config=args.confirm50_config,
        temporal12_config=args.temporal12_config,
    )
    output = _resolve(args.repo.resolve(), args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
