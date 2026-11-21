"""Emit the non-confirmatory WP2-v1.2 hydraulic-feature audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sklearn

from .cycle_proxy_v12_development import (
    DEVELOPMENT_VERSION,
    RAW_DEPLOYMENT_FEATURES,
    TARGETS,
    fit_development_proxy,
    fit_frozen_candidate,
    load_frozen_checkpoint,
    save_frozen_checkpoint,
)


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("cycle_proxy_v12_development.py"),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_index_hashes(dataset_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("cycle_index.parquet", "cycle_index.csv"):
        path = dataset_root / name
        if path.is_file():
            hashes[name] = _sha256(path)
    if not hashes:
        for name in ("case_index.parquet", "case_index.csv"):
            path = dataset_root / name
            if path.is_file():
                hashes[name] = _sha256(path)
    if not hashes:
        raise FileNotFoundError("WP2-v1.2 training index is missing")
    return hashes


def _validate_confirmation_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_target_version") != "1.2-confirm50":
        raise ValueError("WP2-v1.2 freeze requires the Confirm50 configuration")
    cases = payload.get("cases", [])
    if len(cases) != 50:
        raise ValueError("Confirm50 configuration must contain exactly 50 cases")
    forbidden = {
        "physics_ok",
        "pump_work_available",
        "Qcool_J",
        "Wpump_J",
        "COP",
        "SCP_W_per_kg",
        "deltaT_hot_cold_max_K",
    }
    for case in cases:
        leaked = forbidden.intersection(case)
        if leaked:
            raise ValueError(
                f"Confirm50 configuration contains observed/result fields: {sorted(leaked)}"
            )
        missing = set(RAW_DEPLOYMENT_FEATURES[:5]) - set(case)
        if missing:
            raise ValueError(f"Confirm50 case lacks controls: {sorted(missing)}")
    return payload


def run(dataset_root: Path, output: Path) -> dict[str, object]:
    result = fit_development_proxy(dataset_root)
    frozen_candidate = fit_frozen_candidate(dataset_root)
    quality = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )
    report = asdict(result)
    report.update(
        {
            "schema_version": "sciml_wp2_v12_development_v2",
            "confirmatory": False,
            "readiness_eligible": False,
            "replaces_v11_gate": False,
            "dataset_source_content_sha256": quality["source_content_sha256"],
            "implementation_sha256": implementation_sha256(),
            "parameter_sha256": frozen_candidate.parameter_sha256(),
            "frozen_candidate_status": "fitted_on_development_data_not_yet_checkpoint_frozen",
            "evidence_boundary": (
                "This model was developed after inspecting Paper220 v1.1 failures. "
                "Its positive OOD skills are development diagnostics only; a new "
                "untouched confirmation matrix is required before any paper-grade claim."
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def freeze_confirmation_candidate(
    dataset_root: Path,
    confirmation_config: Path,
    development_output: Path,
    checkpoint: Path,
    manifest: Path,
) -> dict[str, Any]:
    """Freeze the exact candidate and protocol before Confirm50 is observed."""

    confirmation = _validate_confirmation_config(confirmation_config)
    development = run(dataset_root, development_output)
    if not development["development_gate_audit"]["development_freeze_ready"]:
        raise RuntimeError("WP2-v1.2 development folds do not pass the frozen gate")
    quality = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )
    implementation_hash = implementation_sha256()
    confirmation_hash = _sha256(confirmation_config)
    metadata = {
        "dataset_source_content_sha256": str(quality["source_content_sha256"]),
        "implementation_sha256": implementation_hash,
        "confirmation_config_sha256": confirmation_hash,
    }
    model = fit_frozen_candidate(dataset_root)
    checkpoint_hash = save_frozen_checkpoint(checkpoint, model, metadata=metadata)
    # A load-and-rehash here makes the freeze fail closed if serialization did
    # not preserve the exact prediction state.
    restored = load_frozen_checkpoint(checkpoint, expected_metadata=metadata)
    if restored.parameter_sha256() != model.parameter_sha256():
        raise RuntimeError("WP2-v1.2 checkpoint changed fitted parameters")
    payload: dict[str, Any] = {
        "schema_version": "sciml_wp2_v12_confirmation_freeze_v1",
        "work_package": DEVELOPMENT_VERSION,
        "evidence_status": "candidate_frozen_before_confirm50_results",
        "confirmatory": False,
        "readiness_eligible": False,
        "paper_grade_ready": False,
        "confirmation_results_observed": False,
        "candidate_refit_after_confirmation_permitted": False,
        "training_corpus": {
            "role": "development_only_after_v11_failure_analysis",
            "dataset_source_content_sha256": metadata[
                "dataset_source_content_sha256"
            ],
            "index_file_sha256": _dataset_index_hashes(dataset_root),
            "condition_count": model.training_condition_count,
            "cycle_count": model.training_cycle_count,
        },
        "candidate": {
            "implementation_sha256": implementation_hash,
            "parameter_sha256": model.parameter_sha256(),
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": checkpoint_hash,
            "python_sklearn_version": sklearn.__version__,
            "targets": list(TARGETS),
            "feature_contract": development["feature_contract"],
            "target_estimators": development["target_estimators"],
        },
        "development_evidence": {
            "report_file": development_output.name,
            "report_sha256": _sha256(development_output),
            "report_implementation_sha256": development["implementation_sha256"],
            "report_parameter_sha256": development["parameter_sha256"],
            "fold_audit": development["development_gate_audit"],
            "claim_boundary": (
                "Passing Paper220 development folds does not confirm the candidate."
            ),
        },
        "confirmation_protocol": {
            "configuration_file": confirmation_config.name,
            "configuration_sha256": confirmation_hash,
            "configured_case_count": len(confirmation["cases"]),
            "configuration_contains_observed_results": False,
            "load_checkpoint_without_refit": True,
            "all_configured_cases_must_complete": True,
            "all_evaluated_cases_must_be_physics_valid_and_pump_complete": True,
            "evaluation_targets": list(TARGETS),
            "positive_pooled_skill_vs_frozen_training_mean_required_for_every_target": True,
            "positive_skill_required_for_every_target_in_each_material_group": True,
            "positive_skill_required_for_every_target_in_each_geometry_group": True,
            "confirmation_results_must_not_select_features_or_parameters": True,
            "failure_action": (
                "retain failure as confirmation evidence; any revised model requires "
                "a new version and a new untouched confirmation matrix"
            ),
        },
        "blocking_reasons": [
            "Confirm50 has not been run and its outcome has not been evaluated.",
            "The frozen candidate remains ineligible for v1.1 or paper-grade readiness.",
        ],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def verify_confirmation_freeze(
    dataset_root: Path,
    confirmation_config: Path,
    development_output: Path,
    checkpoint: Path,
    manifest: Path,
) -> dict[str, Any]:
    """Fail closed if any frozen code/data/config/model evidence changed."""

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sciml_wp2_v12_confirmation_freeze_v1":
        raise ValueError("unsupported WP2-v1.2 confirmation freeze manifest")
    if payload.get("confirmatory") is not False or payload.get(
        "confirmation_results_observed"
    ) is not False:
        raise ValueError("WP2-v1.2 freeze manifest crosses the evidence boundary")
    _validate_confirmation_config(confirmation_config)
    quality = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )
    expected = {
        "dataset_source_content_sha256": str(quality["source_content_sha256"]),
        "implementation_sha256": implementation_sha256(),
        "confirmation_config_sha256": _sha256(confirmation_config),
    }
    candidate = payload["candidate"]
    training = payload["training_corpus"]
    protocol = payload["confirmation_protocol"]
    checks = {
        "dataset_source_hash_matches": training[
            "dataset_source_content_sha256"
        ]
        == expected["dataset_source_content_sha256"],
        "dataset_index_hashes_match": training["index_file_sha256"]
        == _dataset_index_hashes(dataset_root),
        "implementation_hash_matches": candidate["implementation_sha256"]
        == expected["implementation_sha256"],
        "confirmation_config_hash_matches": protocol["configuration_sha256"]
        == expected["confirmation_config_sha256"],
        "development_report_hash_matches": payload["development_evidence"][
            "report_sha256"
        ]
        == _sha256(development_output),
        "checkpoint_hash_matches": candidate["checkpoint_sha256"]
        == _sha256(checkpoint),
    }
    if not all(checks.values()):
        failures = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"WP2-v1.2 freeze verification failed: {failures}")
    model = load_frozen_checkpoint(checkpoint, expected_metadata=expected)
    checks["parameter_hash_matches"] = (
        model.parameter_sha256() == candidate["parameter_sha256"]
    )
    checks["all_checks_passed"] = all(checks.values())
    if not checks["all_checks_passed"]:
        raise ValueError("WP2-v1.2 frozen parameter hash mismatch")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SciML/results/development/wp2_v12_development.json"),
    )
    parser.add_argument("--freeze-confirmation", action="store_true")
    parser.add_argument(
        "--confirmation-config",
        type=Path,
        default=Path("SciML/configs/wp1_v12_confirm50.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("SciML/results/development/wp2_v12_confirmation_candidate.pkl"),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("SciML/results/development/wp2_v12_confirmation_freeze.json"),
    )
    args = parser.parse_args()
    if args.freeze_confirmation:
        report = freeze_confirmation_candidate(
            args.dataset_root,
            args.confirmation_config,
            args.output,
            args.checkpoint,
            args.freeze_manifest,
        )
    else:
        report = run(args.dataset_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
