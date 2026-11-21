"""Paper-grade readiness audit for dataset v1.1-pilot and WP2--WP4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


SCIML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = SCIML_ROOT / "datasets" / "v1.1-pilot-r3"
WP2_MIN_VALID_CASES = 200
WP3_MIN_VALID_TRANSITIONS = 200
WP4_MIN_VALID_CASES = 200
PAPER_MIN_VALID_CONDITIONS = 200
ENGINEERING_MIN_VALID_CASES = 20
ENGINEERING_MIN_VALID_CONDITIONS = 20
ENGINEERING_MIN_TRANSITIONS = 80
MODEL_RESULT_PATTERNS = {
    "wp2": "wp2_v11*.json",
    "wp3": "wp3_v11*.json",
    "wp4": "wp4_v11*.json",
}
MODEL_IMPLEMENTATION_FILES = {
    "wp2": ("cycle_proxy.py", "run_cycle_proxy.py"),
    "wp3": ("state_space.py", "run_wp3_state_space.py"),
    "wp4": ("spatiotemporal_operator.py", "run_wp4_spatiotemporal.py"),
}


def _table(root: Path, stem: str = "case_index") -> pd.DataFrame:
    try:
        return pd.read_parquet(root / f"{stem}.parquet")
    except (ImportError, OSError):
        return pd.read_csv(root / f"{stem}.csv")


def _matching_model_result(
    work_package: str, dataset_source_content_sha256: str
) -> tuple[Path | None, dict[str, Any] | None]:
    digest = hashlib.sha256()
    for name in sorted(MODEL_IMPLEMENTATION_FILES[work_package]):
        path = SCIML_ROOT / "models" / "v11" / name
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    current_implementation_sha256 = digest.hexdigest()
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((SCIML_ROOT / "results").glob(MODEL_RESULT_PATTERNS[work_package])):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            str(payload.get("dataset_source_content_sha256", "")) == dataset_source_content_sha256
            and str(payload.get("implementation_sha256", ""))
            == current_implementation_sha256
        ):
            matches.append((path, payload))
    if not matches:
        return None, None
    return max(matches, key=lambda item: item[0].stat().st_mtime_ns)


def _checkpoint_evidence(
    payload: dict[str, Any], dataset_source_content_sha256: str
) -> dict[str, Any]:
    checkpoint_text = str(payload.get("checkpoint", ""))
    checkpoint = Path(checkpoint_text) if checkpoint_text else None
    evidence: dict[str, Any] = {
        "checkpoint_exists": bool(checkpoint and checkpoint.is_file()),
        "checkpoint_dataset_hash_matches": False,
        "checkpoint_implementation_hash_matches": False,
        "checkpoint_parameter_hash_matches": False,
        "checkpoint_state_complete": False,
        "checkpoint_verified": False,
        "checkpoint_error": "",
    }
    if checkpoint is None or not checkpoint.is_file():
        return evidence
    try:
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint_payload, dict):
            raise TypeError("checkpoint payload is not a mapping")
        evidence["checkpoint_dataset_hash_matches"] = (
            str(checkpoint_payload.get("dataset_source_content_sha256", ""))
            == dataset_source_content_sha256
        )
        evidence["checkpoint_implementation_hash_matches"] = (
            str(checkpoint_payload.get("implementation_sha256", ""))
            == str(payload.get("implementation_sha256", ""))
        )
        evidence["checkpoint_parameter_hash_matches"] = (
            str(checkpoint_payload.get("parameter_sha256", ""))
            == str(payload.get("parameter_sha256", ""))
        )
        work_package = str(payload.get("work_package", ""))
        required_keys: set[str] = set()
        expected_schema = ""
        if work_package == "WP3-v1.1":
            expected_schema = "sciml_wp3_checkpoint_v2"
            required_keys = {
                "state_dict",
                "feature_mean",
                "feature_scale",
                "residual_scale",
                "stage_baseline",
                "stage_trust_radius",
                "stage_training_features",
                "residual_gain",
                "training_ids",
            }
        elif work_package == "WP4-v1.1":
            expected_schema = "sciml_wp4_checkpoint_v2"
            required_keys = {
                "state_dict",
                "branch_mean",
                "branch_scale",
                "sma_pod_basis",
                "htf_pod_basis",
                "stage_mean",
                "stage_training_branch",
                "stage_trust_radius",
                "coefficient_mean",
                "coefficient_scale",
                "temporal_sma_mean_gain",
                "temporal_htf_mean_gain",
                "training_ids",
            }
        evidence["checkpoint_state_complete"] = bool(
            expected_schema
            and str(checkpoint_payload.get("checkpoint_schema_version", ""))
            == expected_schema
            and required_keys <= set(checkpoint_payload)
        )
        evidence["checkpoint_verified"] = bool(
            evidence["checkpoint_dataset_hash_matches"]
            and evidence["checkpoint_implementation_hash_matches"]
            and evidence["checkpoint_parameter_hash_matches"]
            and evidence["checkpoint_state_complete"]
        )
    except Exception as exc:  # readiness must report corrupt/stale checkpoints
        evidence["checkpoint_error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def audit_v11_readiness(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = Path(root)
    quality = json.loads((root / "quality_report.json").read_text(encoding="utf-8"))
    cases = _table(root)
    cycles = _table(root, "cycle_index") if (root / "cycle_index.csv").exists() else pd.DataFrame()
    valid = cases[cases["physics_ok"].astype(bool)].copy()
    valid_condition_count = int(valid["condition_id"].nunique())
    valid_cycles = cycles[cycles["physics_ok"].astype(bool)].copy() if len(cycles) else cycles
    required_targets = (
        "Qcool_J",
        "W1_J",
        "Wpump_J",
        "COP",
        "SCP_W_per_kg",
        "deltaT_hot_cold_max_K",
    )
    finite_counts = {
        name: int(np.isfinite(pd.to_numeric(valid[name], errors="coerce")).sum())
        for name in required_targets
    }
    pump_complete = int(valid["pump_work_available"].astype(bool).sum())
    full_cycle_inputs_present = all(
        name in cases.columns
        for name in ("radius_m", "thickness_m", "pressure_Pa", "flush_volume_target_ratio", "exchange_duration_s")
    )

    stage_fields_complete = True
    exchange_sma_spatiotemporal_complete = True
    sma_htf_fields_distinct = True
    pressure_command_complete = True
    htf_internal_full_field_available = bool(
        len(valid) > 0 and valid["htf_internal_full_field_available"].astype(bool).all()
    )
    observed_transition_count = 0
    with h5py.File(root / "fields.h5", "r") as h5:
        for row in valid.to_dict(orient="records"):
            group = h5[str(row["hdf5_group"])]
            expected = {
                "stage1_loading": ("u_m_t", "w_m_t", "T_K_t", "xi_t"),
                "stage2_hold_exchange": ("sma_strip_T_K_t",),
                "stage3_unloading": ("u_m_t", "w_m_t", "T_K_t", "xi_t"),
                "stage4_cooldown_exchange": ("sma_strip_T_K_t",),
            }
            for stage, names in expected.items():
                stage_fields_complete &= stage in group and all(name in group[stage] for name in names)
            for stage in ("stage2_hold_exchange", "stage4_cooldown_exchange"):
                sma = np.asarray(group[stage]["sma_strip_T_K_t"])
                exchange_sma_spatiotemporal_complete &= sma.shape == (50, 101) and np.isfinite(sma).all()
                htf = np.asarray(group[stage].get("htf_internal_T_K_t", []))
                htf_internal_full_field_available &= htf.shape == (50, 101) and np.isfinite(htf).all()
                sma_htf_fields_distinct &= sma.shape == htf.shape and not np.array_equal(sma, htf)
                pressure = np.asarray(group[stage].get("pressure_pi_command_MPa_t", []))
                pressure_command_complete &= pressure.shape == (50,) and np.isfinite(pressure).all()
            cycles_group = group.get("cycles")
            if cycles_group is None:
                observed_transition_count += 4
                continue
            expected_cycles = int(row["cycle_count"])
            if len(cycles_group) != expected_cycles:
                stage_fields_complete = False
            for cycle_group in cycles_group.values():
                for stage, names in expected.items():
                    stage_fields_complete &= stage in cycle_group and all(
                        name in cycle_group[stage] for name in names
                    )
                for stage in ("stage2_hold_exchange", "stage4_cooldown_exchange"):
                    sma = np.asarray(cycle_group[stage]["sma_strip_T_K_t"])
                    htf = np.asarray(cycle_group[stage].get("htf_internal_T_K_t", []))
                    exchange_sma_spatiotemporal_complete &= sma.shape == (50, 101) and np.isfinite(sma).all()
                    htf_internal_full_field_available &= htf.shape == (50, 101) and np.isfinite(htf).all()
                    sma_htf_fields_distinct &= sma.shape == htf.shape and not np.array_equal(sma, htf)
                    pressure = np.asarray(cycle_group[stage].get("pressure_pi_command_MPa_t", []))
                    pressure_command_complete &= pressure.shape == (50,) and np.isfinite(pressure).all()
                observed_transition_count += 4

    multi_cycle_valid_cases = int((valid["cycle_count"] >= 2).sum()) if len(valid) else 0
    valid_materials = sorted(valid["material_id"].unique().tolist())
    valid_geometries = int(valid[["radius_m", "thickness_m"]].drop_duplicates().shape[0])
    cycle_targets_finite = bool(
        len(valid_cycles) > 0
        and np.isfinite(
            valid_cycles[
                ["Qcool_J", "W1_J", "Wpump_J", "COP", "SCP_W_per_kg", "deltaT_hot_cold_max_K"]
            ].to_numpy(float)
        ).all()
    )
    multi_cycle_independent = False
    if len(valid_cycles):
        for _, group in valid_cycles.groupby("case_id"):
            if len(group) >= 2:
                values = group.sort_values("cycle_index")[["Qcool_J", "COP", "Wpump_J"]].to_numpy(float)
                multi_cycle_independent |= bool(np.max(np.abs(np.diff(values, axis=0))) > 1e-12)

    implementation_evidence: dict[str, Any] = {}
    dataset_source_content_sha256 = str(quality.get("source_content_sha256", ""))
    for name in MODEL_RESULT_PATTERNS:
        path, payload = _matching_model_result(name, dataset_source_content_sha256)
        if path is not None and payload is not None:
            checkpoint = _checkpoint_evidence(payload, dataset_source_content_sha256)
            implementation_evidence[name] = {
                "result_exists": True,
                "result_path": str(path.resolve()),
                "dataset_hash_matches": True,
                "implementation_hash_matches": True,
                "paper_grade_ready": bool(payload.get("paper_grade_ready", False)),
                **checkpoint,
            }
        else:
            implementation_evidence[name] = {
                "result_exists": False,
                "result_path": "",
                "dataset_hash_matches": False,
                "implementation_hash_matches": False,
                "paper_grade_ready": False,
                "checkpoint_exists": False,
                "checkpoint_dataset_hash_matches": False,
                "checkpoint_implementation_hash_matches": False,
                "checkpoint_parameter_hash_matches": False,
                "checkpoint_state_complete": False,
                "checkpoint_verified": False,
                "checkpoint_error": "",
            }
    prototype_chain_complete = all(
        item["result_exists"] and (name == "wp2" or item["checkpoint_verified"])
        for name, item in implementation_evidence.items()
    )
    physics_fingerprint_count = int(
        quality.get("physics_implementation_fingerprint_count", 0)
    )
    pipeline_fingerprint_count = int(
        quality.get("pipeline_implementation_fingerprint_count", 0)
    )
    physics_implementation_consistent = physics_fingerprint_count == 1
    export_provenance_complete_count = int(
        quality.get("export_provenance_complete_count", 0)
    )
    export_provenance_complete = export_provenance_complete_count == len(cases)

    wp2_ready = (
        full_cycle_inputs_present
        and len(valid) >= WP2_MIN_VALID_CASES
        and valid_condition_count >= PAPER_MIN_VALID_CONDITIONS
        and pump_complete == len(valid)
        and all(count == len(valid) for count in finite_counts.values())
        and cycle_targets_finite
        and pressure_command_complete
        and physics_implementation_consistent
        and export_provenance_complete
        and implementation_evidence["wp2"]["paper_grade_ready"]
    )
    wp3_ready = (
        stage_fields_complete
        and observed_transition_count >= WP3_MIN_VALID_TRANSITIONS
        and valid_condition_count >= PAPER_MIN_VALID_CONDITIONS
        and multi_cycle_valid_cases > 0
        and multi_cycle_independent
        and physics_implementation_consistent
        and export_provenance_complete
        and implementation_evidence["wp3"]["checkpoint_verified"]
        and implementation_evidence["wp3"]["paper_grade_ready"]
    )
    wp4_ready = (
        exchange_sma_spatiotemporal_complete
        and htf_internal_full_field_available
        and sma_htf_fields_distinct
        and len(valid) >= WP4_MIN_VALID_CASES
        and valid_condition_count >= PAPER_MIN_VALID_CONDITIONS
        and len(valid_materials) >= 2
        and valid_geometries >= 2
        and physics_implementation_consistent
        and export_provenance_complete
        and implementation_evidence["wp4"]["checkpoint_verified"]
        and implementation_evidence["wp4"]["paper_grade_ready"]
    )
    wp2_engineering_ready = (
        full_cycle_inputs_present
        and len(valid) >= ENGINEERING_MIN_VALID_CASES
        and valid_condition_count >= ENGINEERING_MIN_VALID_CONDITIONS
        and pump_complete == len(valid)
        and all(count == len(valid) for count in finite_counts.values())
        and cycle_targets_finite
        and pressure_command_complete
        and physics_implementation_consistent
        and export_provenance_complete
        and implementation_evidence["wp2"]["result_exists"]
    )
    wp3_engineering_ready = (
        stage_fields_complete
        and observed_transition_count >= ENGINEERING_MIN_TRANSITIONS
        and valid_condition_count >= ENGINEERING_MIN_VALID_CONDITIONS
        and multi_cycle_valid_cases > 0
        and multi_cycle_independent
        and physics_implementation_consistent
        and export_provenance_complete
        and implementation_evidence["wp3"]["result_exists"]
        and implementation_evidence["wp3"]["checkpoint_verified"]
    )
    wp4_engineering_ready = (
        exchange_sma_spatiotemporal_complete
        and htf_internal_full_field_available
        and sma_htf_fields_distinct
        and len(valid) >= ENGINEERING_MIN_VALID_CASES
        and valid_condition_count >= ENGINEERING_MIN_VALID_CONDITIONS
        and len(valid_materials) >= 2
        and valid_geometries >= 3
        and physics_implementation_consistent
        and export_provenance_complete
        and implementation_evidence["wp4"]["result_exists"]
        and implementation_evidence["wp4"]["checkpoint_verified"]
    )

    blockers: list[str] = []
    if len(valid) < WP2_MIN_VALID_CASES:
        blockers.append(f"WP2 has {len(valid)} physics-valid cases; minimum gate is {WP2_MIN_VALID_CASES}.")
    if valid_condition_count < PAPER_MIN_VALID_CONDITIONS:
        blockers.append(
            f"Dataset has {valid_condition_count} independent physics-valid conditions; "
            f"paper-grade minimum is {PAPER_MIN_VALID_CONDITIONS}."
        )
    if pump_complete < len(valid):
        blockers.append("WP2 pump work is unavailable for one or more physics-valid cases.")
    if multi_cycle_valid_cases == 0:
        blockers.append("WP3 has no physics-valid multi-cycle case for autonomous drift evaluation.")
    if observed_transition_count < WP3_MIN_VALID_TRANSITIONS:
        blockers.append(
            f"WP3 has {observed_transition_count} valid observed stage transitions; minimum gate is {WP3_MIN_VALID_TRANSITIONS}."
        )
    if not htf_internal_full_field_available:
        blockers.append("WP4 still lacks a time-resolved HTF internal full field.")
    if not sma_htf_fields_distinct:
        blockers.append("WP4 SMA and HTF full fields are not demonstrably distinct.")
    if not pressure_command_complete:
        blockers.append("WP2 pressure-command evidence is incomplete for one or more exchange stages.")
    if not multi_cycle_independent:
        blockers.append("WP3 multi-cycle artifact does not demonstrate an independently changing cycle.")
    if not physics_implementation_consistent:
        blockers.append(
            f"Dataset contains {physics_fingerprint_count} physics implementation fingerprints; exactly one is required."
        )
    if not export_provenance_complete:
        blockers.append(
            f"Only {export_provenance_complete_count}/{len(cases)} cases have verified export provenance."
        )
    if valid_geometries < 2:
        blockers.append("WP4 has fewer than two physics-valid geometries.")
    if len(valid) < WP4_MIN_VALID_CASES:
        blockers.append(f"WP4 has {len(valid)} physics-valid cases; minimum gate is {WP4_MIN_VALID_CASES}.")
    for name, evidence in implementation_evidence.items():
        if not evidence["result_exists"]:
            blockers.append(
                f"{name.upper()} has no result bound to the current dataset and implementation hashes."
            )
        elif name != "wp2" and not evidence["checkpoint_verified"]:
            blockers.append(
                f"{name.upper()} checkpoint does not match the current dataset, implementation and parameter hashes."
            )
        elif not evidence["paper_grade_ready"]:
            blockers.append(
                f"{name.upper()} model report has not passed its registered OOD/temporal paper-grade gate."
            )

    engineering_blockers: list[str] = []
    if len(valid) < ENGINEERING_MIN_VALID_CASES:
        engineering_blockers.append(
            f"Only {len(valid)} physics-valid cases; engineering minimum is {ENGINEERING_MIN_VALID_CASES}."
        )
    if valid_condition_count < ENGINEERING_MIN_VALID_CONDITIONS:
        engineering_blockers.append(
            f"Only {valid_condition_count} independent physics-valid conditions; "
            f"engineering minimum is {ENGINEERING_MIN_VALID_CONDITIONS}."
        )
    if observed_transition_count < ENGINEERING_MIN_TRANSITIONS:
        engineering_blockers.append(
            f"Only {observed_transition_count} valid stage transitions; engineering minimum is {ENGINEERING_MIN_TRANSITIONS}."
        )
    if len(valid_materials) < 2:
        engineering_blockers.append("Engineering corpus has fewer than two physics-valid materials.")
    if valid_geometries < 3:
        engineering_blockers.append("Engineering corpus has fewer than three physics-valid geometries.")
    if multi_cycle_valid_cases == 0 or not multi_cycle_independent:
        engineering_blockers.append("Engineering corpus lacks an independent physics-valid multi-cycle object.")
    if not pressure_command_complete or pump_complete < len(valid):
        engineering_blockers.append("Engineering pressure/pump evidence is incomplete.")
    if not htf_internal_full_field_available or not sma_htf_fields_distinct:
        engineering_blockers.append("Engineering SMA/HTF dual-field evidence is incomplete or ambiguous.")
    if not physics_implementation_consistent or not export_provenance_complete:
        engineering_blockers.append("Engineering provenance consistency is incomplete.")
    if not prototype_chain_complete:
        engineering_blockers.append(
            "Engineering WP2-WP4 results are missing or stale for the current dataset/code hashes."
        )

    return {
        "schema_version": "sciml_v11_readiness_v3",
        "dataset_root": str(root.resolve()),
        "case_count": int(len(cases)),
        "cycle_sample_count": int(len(cycles)),
        "physics_valid_case_count": int(len(valid)),
        "physics_valid_condition_count": valid_condition_count,
        "physics_valid_cycle_sample_count": int(len(valid_cycles)),
        "wp2": {
            "paper_grade_ready": wp2_ready,
            "minimum_valid_cases": WP2_MIN_VALID_CASES,
            "minimum_valid_conditions": PAPER_MIN_VALID_CONDITIONS,
            "full_cycle_inputs_present": full_cycle_inputs_present,
            "finite_target_counts_on_valid_cases": finite_counts,
            "pump_complete_valid_cases": pump_complete,
            "cycle_targets_finite": cycle_targets_finite,
            "pressure_command_complete": pressure_command_complete,
        },
        "wp3": {
            "paper_grade_ready": wp3_ready,
            "four_stage_observation_schema_complete": stage_fields_complete,
            "valid_observed_stage_transitions": observed_transition_count,
            "minimum_valid_stage_transitions": WP3_MIN_VALID_TRANSITIONS,
            "minimum_valid_conditions": PAPER_MIN_VALID_CONDITIONS,
            "valid_multi_cycle_cases": multi_cycle_valid_cases,
            "multi_cycle_independent": multi_cycle_independent,
        },
        "wp4": {
            "paper_grade_ready": wp4_ready,
            "sma_strip_spatiotemporal_field_complete": exchange_sma_spatiotemporal_complete,
            "htf_internal_full_field_available": htf_internal_full_field_available,
            "sma_htf_fields_distinct": sma_htf_fields_distinct,
            "physics_valid_materials": valid_materials,
            "physics_valid_geometry_count": valid_geometries,
            "minimum_valid_cases": WP4_MIN_VALID_CASES,
            "minimum_valid_conditions": PAPER_MIN_VALID_CONDITIONS,
        },
        "implementation_evidence": implementation_evidence,
        "prototype_chain_complete": prototype_chain_complete,
        "provenance_consistency": {
            "physics_implementation_fingerprint_count": physics_fingerprint_count,
            "pipeline_implementation_fingerprint_count": pipeline_fingerprint_count,
            "physics_implementation_consistent": physics_implementation_consistent,
            "export_provenance_complete_count": export_provenance_complete_count,
            "export_provenance_complete": export_provenance_complete,
        },
        "paper_grade_gate": {
            "passed": wp2_ready and wp3_ready and wp4_ready,
            "blocking_reasons": blockers,
        },
        "engineering_gate": {
            "passed": wp2_engineering_ready and wp3_engineering_ready and wp4_engineering_ready,
            "wp2_ready": wp2_engineering_ready,
            "wp3_ready": wp3_engineering_ready,
            "wp4_ready": wp4_engineering_ready,
            "minimum_valid_cases": ENGINEERING_MIN_VALID_CASES,
            "minimum_valid_conditions": ENGINEERING_MIN_VALID_CONDITIONS,
            "minimum_valid_stage_transitions": ENGINEERING_MIN_TRANSITIONS,
            "blocking_reasons": engineering_blockers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_v11_readiness(args.dataset_root)
    text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
