"""Evaluate the frozen WP3-v1.2 candidate on untouched Confirm-Temporal12.

This evaluator is intentionally prediction-only.  It verifies the global
pre-confirmation manifest and the exact frozen checkpoint, rejects incomplete
or provenance-invalid confirmation data, and never calls ``fit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from SciML.evaluation.confirmation_freeze_v12 import verify_manifest
from SciML.models.v11.run_wp3_state_space import (
    LAYOUT,
    _autonomous_multicycle_rollout,
    _metric,
    _rmse_skill,
)
from SciML.models.v11.state_space import TransitionRecord, build_transition_corpus
from SciML.models.v12.run_wp3_temporal_development import implementation_sha256
from SciML.models.v12.neural_state_space_candidate import load_frozen_checkpoint


SCHEMA = "sciml_wp3_v12_confirm_temporal12_v1"
TEMPORAL_SLICES = {
    "all_state": None,
    "sma_temperature": LAYOUT.sma_T,
    "stage2_htf_temperature": LAYOUT.htf_stage2_T,
    "stage4_htf_temperature": LAYOUT.htf_stage4_T,
}
STRUCTURED_SLICES = {
    "all_state": None,
    "sma_temperature": LAYOUT.sma_T,
}
REQUIRED_CASE_GATES = (
    "run_ok",
    "physics_ok",
    "observation_contract_ok",
    "paper_grade_valid",
    "pump_work_available",
    "htf_internal_full_field_available",
    "request_config_matches_current",
    "request_self_contained",
    "export_provenance_available",
    "export_artifact_hash_matches",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve(repo: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _read_table(root: Path, stem: str) -> pd.DataFrame:
    parquet = root / f"{stem}.parquet"
    csv = root / f"{stem}.csv"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    if csv.is_file():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"missing confirmation table: {stem}")


def _bool_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"confirmation case index lacks {name}")
    values = frame[name]
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"confirmation case gate {name} is not Boolean")
    return normalized.isin(("true", "1"))


def _validate_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("dataset_target_version") != "1.2-confirm-temporal12":
        raise ValueError("unexpected Confirm-Temporal12 target version")
    policy = config.get("matrix_policy", {})
    if policy.get("selection_uses_observed_outcome") is not False:
        raise ValueError("confirmation selection is not outcome independent")
    if policy.get("confirmation_results_must_not_select_or_tune_model") is not True:
        raise ValueError("confirmation config permits retrospective tuning")
    cases = config.get("cases")
    if not isinstance(cases, list) or len(cases) != 12:
        raise ValueError("Confirm-Temporal12 must contain exactly 12 cases")
    identifiers = [str(case.get("case_id", "")) for case in cases]
    if any(not value for value in identifiers) or len(set(identifiers)) != 12:
        raise ValueError("Confirm-Temporal12 case IDs must be unique and non-empty")
    cycles = sorted(int(case.get("cycle_count", 0)) for case in cases)
    if cycles != [2] * 10 + [3] * 2:
        raise ValueError("Confirm-Temporal12 must contain ten 2-cycle and two 3-cycle cases")
    if sorted(str(case.get("material_id")) for case in cases).count("recA") != 6:
        raise ValueError("Confirm-Temporal12 must contain six recA cases")
    if sorted(str(case.get("material_id")) for case in cases).count("recAs") != 6:
        raise ValueError("Confirm-Temporal12 must contain six recAs cases")
    return cases


def _validate_confirmation_dataset(
    dataset_root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cases = _validate_config(config)
    frame = _read_table(dataset_root, "case_index")
    if frame["case_id"].astype(str).duplicated().any():
        raise ValueError("confirmation case index contains duplicate case IDs")
    expected = {str(case["case_id"]): case for case in cases}
    observed = set(frame["case_id"].astype(str))
    if observed != set(expected):
        raise ValueError(
            "confirmation dataset does not exactly match the 12 frozen case IDs: "
            f"missing={sorted(set(expected) - observed)}, extra={sorted(observed - set(expected))}"
        )
    by_id = frame.assign(case_id=frame["case_id"].astype(str)).set_index("case_id")
    numeric_fields = (
        "radius_m",
        "thickness_m",
        "pressure_Pa",
        "flush_volume_target_ratio",
        "exchange_duration_s",
    )
    for case_id, registered in expected.items():
        row = by_id.loc[case_id]
        if str(row["material_id"]) != str(registered["material_id"]):
            raise ValueError(f"confirmation material mismatch: {case_id}")
        if int(row["cycle_count"]) != int(registered["cycle_count"]):
            raise ValueError(f"confirmation cycle-count mismatch: {case_id}")
        for field in numeric_fields:
            if not np.isclose(
                float(row[field]), float(registered[field]), rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"confirmation {field} mismatch: {case_id}")
    failed = {
        gate: frame.loc[~_bool_series(frame, gate), "case_id"].astype(str).tolist()
        for gate in REQUIRED_CASE_GATES
    }
    failed = {gate: values for gate, values in failed.items() if values}
    if failed:
        raise RuntimeError(f"confirmation cases fail strict gates: {failed}")
    for field in (
        "source_artifact_sha256",
        "source_artifact_semantic_sha256",
        "source_request_sha256",
        "request_config_sha256",
        "source_run_result_sha256",
    ):
        if field not in frame or not frame[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise RuntimeError(f"confirmation provenance hash is missing or invalid: {field}")

    quality = _json(dataset_root / "quality_report.json")
    source_hash = quality.get("source_content_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("confirmation quality report lacks source_content_sha256")
    required_quality_true = ("all_required_fields_finite",)
    bad_quality = [name for name in required_quality_true if quality.get(name) is not True]
    confirmation_gate = quality.get("confirmation_quality_gate", {})
    if not isinstance(confirmation_gate, dict) or confirmation_gate.get("passed") is not True:
        bad_quality.append("confirmation_quality_gate")
    if bad_quality:
        raise RuntimeError(f"confirmation quality gates failed: {bad_quality}")
    for count_name in (
        "case_count",
        "artifact_complete_case_count",
        "physics_ok_count",
        "paper_grade_valid_count",
        "observation_contract_ok_count",
        "request_config_match_count",
        "request_self_contained_count",
        "export_provenance_complete_count",
    ):
        if int(quality.get(count_name, -1)) != 12:
            raise RuntimeError(f"confirmation quality count is not 12: {count_name}")
    return frame.reset_index(drop=True), quality


def _skill_report(
    prediction: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    slices: dict[str, slice | None],
) -> dict[str, Any]:
    model_metrics = {
        name: _metric(prediction, target, slc) for name, slc in slices.items()
    }
    baseline_metrics = {
        name: _metric(baseline, target, slc) for name, slc in slices.items()
    }
    skills = {
        name: _rmse_skill(model_metrics[name], baseline_metrics[name])
        for name in slices
    }
    return {
        "model": model_metrics,
        "stage_mean_baseline": baseline_metrics,
        "skill_vs_stage_mean": skills,
        "all_required_skills_positive": all(value > 0.0 for value in skills.values()),
    }


def _group_reports(
    evaluated: list[TransitionRecord],
    prediction: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    attribute: str,
    slices: dict[str, slice | None],
) -> dict[str, Any]:
    groups = sorted({str(getattr(record, attribute)) for record in evaluated})
    reports = []
    for group in groups:
        selected = np.asarray(
            [str(getattr(record, attribute)) == group for record in evaluated]
        )
        report = _skill_report(
            prediction[selected], baseline[selected], target[selected], slices
        )
        reports.append(
            {
                "group": group,
                "condition_count": len(
                    {record.condition_id for record, use in zip(evaluated, selected) if use}
                ),
                "transition_count": int(selected.sum()),
                **report,
            }
        )
    return {
        "group_attribute": attribute,
        "group_count": len(groups),
        "groups": reports,
        "all_groups_all_required_skills_positive": all(
            report["all_required_skills_positive"] for report in reports
        ),
    }


def run(
    repo: Path,
    dataset_root: Path,
    confirmation_config: Path,
    freeze_manifest: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    """Run prediction-only confirmation; no model fitting is permitted here."""

    repo = repo.resolve()
    dataset_root = _resolve(repo, dataset_root)
    confirmation_config = _resolve(repo, confirmation_config)
    freeze_manifest = _resolve(repo, freeze_manifest)
    checkpoint = _resolve(repo, checkpoint)
    output = _resolve(repo, output)
    manifest = verify_manifest(repo, freeze_manifest)
    config_binding = manifest.get("artifact_bindings", {}).get(
        "confirm_temporal12_config"
    )
    if not isinstance(config_binding, dict) or _sha256(
        confirmation_config
    ) != config_binding.get("sha256"):
        raise RuntimeError("Confirm-Temporal12 config differs from the global freeze")
    config = _json(confirmation_config)
    _, quality = _validate_confirmation_dataset(dataset_root, config)
    confirmation_source_hash = str(quality["source_content_sha256"])
    if confirmation_source_hash == manifest.get(
        "development_dataset_source_content_sha256"
    ):
        raise RuntimeError("confirmation dataset is not independent of development data")

    wp3_binding = manifest.get("candidate_bindings", {}).get("wp3")
    if not isinstance(wp3_binding, dict):
        raise ValueError("freeze manifest lacks the WP3 candidate binding")
    report_path = repo / Path(str(wp3_binding["report_path"]))
    development = _json(report_path)
    if development.get("development_freeze_audit", {}).get(
        "development_freeze_ready"
    ) is not True:
        raise RuntimeError("WP3 development candidate was not freeze-ready")
    current_implementation = implementation_sha256()
    if current_implementation != wp3_binding.get("implementation_sha256"):
        raise RuntimeError("WP3 implementation changed after the global freeze")
    checkpoint_hash = _sha256(checkpoint)
    checkpoint_record = development.get("checkpoint", {})
    if checkpoint_hash != checkpoint_record.get("file_sha256"):
        raise RuntimeError("WP3 checkpoint hash differs from the frozen report")
    if checkpoint_record.get("roundtrip_parameter_sha256_match") is not True or checkpoint_record.get(
        "roundtrip_prediction_exact_match"
    ) is not True:
        raise RuntimeError("WP3 checkpoint lacked exact pre-confirmation round-trip evidence")
    metadata = {
        "dataset_source_content_sha256": str(
            manifest["development_dataset_source_content_sha256"]
        ),
        "implementation_sha256": current_implementation,
    }
    model = load_frozen_checkpoint(checkpoint, expected_metadata=metadata)
    parameter_hash = model.parameter_sha256()
    if parameter_hash != wp3_binding.get("parameter_sha256"):
        raise RuntimeError("WP3 checkpoint parameter hash differs from the global freeze")

    records = build_transition_corpus(dataset_root)
    evaluated, prediction, baseline, trust, _ = _autonomous_multicycle_rollout(
        model, records
    )
    if len({record.condition_id for record in evaluated}) != 12 or len(evaluated) != 56:
        raise RuntimeError("Confirm-Temporal12 autonomous rollout is incomplete")
    target = np.stack([record.state_out for record in evaluated])
    temporal = _skill_report(
        prediction, baseline, target, TEMPORAL_SLICES
    )
    condition = _group_reports(
        evaluated, prediction, baseline, target, "condition_id", TEMPORAL_SLICES
    )
    material = _group_reports(
        evaluated, prediction, baseline, target, "material_id", STRUCTURED_SLICES
    )
    geometry = _group_reports(
        evaluated, prediction, baseline, target, "geometry_id", STRUCTURED_SLICES
    )
    physical_constraints = {
        "finite": bool(np.isfinite(prediction).all()),
        "xi_in_unit_interval": bool(
            np.all(
                (prediction[:, LAYOUT.xi] >= 0.0)
                & (prediction[:, LAYOUT.xi] <= 1.0)
            )
        ),
        "clamped_u_edge_max_abs_m": float(
            np.max(np.abs(prediction[:, LAYOUT.u.stop - 1]))
        ),
        "clamped_w_edge_max_abs_m": float(
            np.max(np.abs(prediction[:, LAYOUT.w.stop - 1]))
        ),
    }
    checks = {
        "temporal_all_required_skills_positive": bool(
            temporal["all_required_skills_positive"]
        ),
        "each_condition_all_temporal_skills_positive": bool(
            condition["group_count"] == 12
            and condition["all_groups_all_required_skills_positive"]
        ),
        "each_material_all_structured_skills_positive": bool(
            material["group_count"] == 2
            and material["all_groups_all_required_skills_positive"]
        ),
        "each_geometry_all_structured_skills_positive": bool(
            geometry["group_count"] == 5
            and geometry["all_groups_all_required_skills_positive"]
        ),
        "physical_constraints_pass": bool(
            physical_constraints["finite"]
            and physical_constraints["xi_in_unit_interval"]
            and physical_constraints["clamped_u_edge_max_abs_m"] == 0.0
            and physical_constraints["clamped_w_edge_max_abs_m"] == 0.0
        ),
        "checkpoint_roundtrip_verified": True,
        "all_12_conditions_and_56_temporal_transitions_present": True,
    }
    blocking_reasons = sorted(name for name, passed in checks.items() if not passed)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "work_package": "WP3-v1.2-confirmation",
        "confirmatory": True,
        "readiness_eligible": True,
        "paper_grade_ready": not blocking_reasons,
        "freeze_manifest_sha256": _sha256(freeze_manifest),
        "implementation_sha256": current_implementation,
        "parameter_sha256": parameter_hash,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_roundtrip_verified": True,
        "confirmation_dataset_source_content_sha256": confirmation_source_hash,
        "confirmation_bindings": {
            "confirm_temporal12_config_sha256": _sha256(confirmation_config),
            "confirm_temporal12_source_content_sha256": confirmation_source_hash,
        },
        "confirmation_results_used_for_fit_or_tuning": False,
        "preregistered_gate_contract": {
            "temporal_pooled": (
                "strictly positive skill versus the frozen stage-mean baseline "
                "for all-state, SMA temperature, stage-2 HTF and stage-4 HTF"
            ),
            "each_condition": (
                "the same four temporal skills must be strictly positive in each "
                "of the 12 independent physical conditions"
            ),
            "each_material": (
                "all-state and SMA-temperature skills must be strictly positive "
                "within recA and recAs separately"
            ),
            "each_geometry": (
                "all-state and SMA-temperature skills must be strictly positive "
                "within every registered geometry group"
            ),
            "completion": "all 12 cases and all 56 cycle-2/3 transitions are required",
            "model_update_policy": "checkpoint load only; no refit or confirmation tuning",
        },
        "registered_gate_audit": {
            "checks": checks,
            "all_checks_passed": not blocking_reasons,
        },
        "temporal": temporal,
        "condition_groups": condition,
        "material_groups": material,
        "geometry_groups": geometry,
        "physical_constraints": physical_constraints,
        "trust": {
            "count": len(trust),
            "minimum": float(np.min(trust)),
            "median": float(np.median(trust)),
            "maximum": float(np.max(trust)),
        },
        "blocking_reasons": blocking_reasons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--confirmation-config",
        type=Path,
        default=Path("SciML/configs/wp1_v12_confirm_temporal12.json"),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("SciML/results/v12_confirmation_freeze.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("SciML/results/development/wp3_v12_neural_confirmation_candidate.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SciML/results/wp3_v12_confirmation.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.repo,
                args.dataset_root,
                args.confirmation_config,
                args.freeze_manifest,
                args.checkpoint,
                args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
