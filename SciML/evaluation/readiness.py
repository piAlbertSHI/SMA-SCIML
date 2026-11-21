"""Audit whether dataset v1.0 supports the claims planned for WP2--WP4.

This module deliberately distinguishes a runnable research prototype from a
paper-grade generalization claim.  It does not mutate the released dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


SCIML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = SCIML_ROOT / "datasets" / "v1.0"

WP2_DESIGN_TARGETS = {
    "Q_lat_J",
    "W1_J",
    "eta_Q_over_W1",
    "avg_xi",
    "max_w_mm",
    "sigma_max_MPa",
    "design_valid",
}
WP2_FULL_CYCLE_INPUTS = {
    "V_star",
    "frequency_Hz",
    "exchange_time_s",
}
WP2_FULL_CYCLE_TARGETS = {
    "Q_cool_J",
    "W_pump_J",
    "COP",
    "SCP_W_per_kg",
    "delta_T_K",
}
REQUIRED_OPERATOR_FIELDS = {
    "sma_T_in_51_K",
    "sma_T_out_51_K",
    "htf_directional_T_in_101_K",
    "htf_directional_T_out_101_K",
}


def _hash_arrays(group: h5py.Group, names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        if name not in group:
            digest.update(f"missing:{name}".encode("utf-8"))
            continue
        array = np.ascontiguousarray(np.asarray(group[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _cross_role_hashes(records: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    roles: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["role"] not in {"train", "test"}:
            continue
        roles[record[key]].add(record["role"])
        samples[record[key]].append(record["sample_id"])
    return [
        {"sha256": value, "roles": sorted(roles[value]), "sample_ids": sorted(samples[value])}
        for value in sorted(roles)
        if len(roles[value]) > 1
    ]


def audit_wp2_wp4_readiness(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> dict[str, Any]:
    """Return evidence-backed readiness and claim-boundary checks."""

    root = Path(dataset_root)
    scalars = pd.read_parquet(root / "scalar_cases.parquet")
    operators = pd.read_parquet(root / "operator_samples.parquet")
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    protocol = manifest["schemes"]["protocol_frequency_holdout"]["assignments"]
    protocol_scheme = manifest["schemes"]["protocol_frequency_holdout"]
    operator_rows = operators.set_index("sample_id")
    row_protocols_by_role = {
        role: sorted({
            str(operator_rows.loc[sample_id, "protocol_id"])
            for sample_id, assigned_role in protocol.items()
            if assigned_role == role
        })
        for role in ("train", "test", "golden_validation")
    }
    protocol_declarations_match_rows = (
        protocol_scheme.get("train_protocols") == row_protocols_by_role["train"]
        and protocol_scheme.get("test_protocols") == row_protocols_by_role["test"]
        and protocol_scheme.get("golden_validation_protocols")
        == row_protocols_by_role["golden_validation"]
    )

    field_records: list[dict[str, str]] = []
    required_fields_complete = True
    duration_metadata_matches_hdf5 = True
    with h5py.File(root / "fields.h5", "r") as handle:
        for row in operators.to_dict(orient="records"):
            group = handle[row["hdf5_group"]]
            required_fields_complete &= REQUIRED_OPERATOR_FIELDS.issubset(group.keys())
            time_local = np.asarray(group.get("time_local_s", []), dtype=np.float64)
            observed_duration = float(row.get("observed_stage_duration_s", np.nan))
            duration_metadata_matches_hdf5 &= bool(
                time_local.size >= 2
                and np.isfinite(time_local).all()
                and np.all(np.diff(time_local) >= 0.0)
                and np.isclose(
                    observed_duration,
                    float(np.max(time_local) - np.min(time_local)),
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            field_records.append(
                {
                    "sample_id": row["sample_id"],
                    "role": protocol[row["sample_id"]],
                    "field_input_sha256": _hash_arrays(
                        group, ("sma_T_in_51_K", "htf_directional_T_in_101_K")
                    ),
                    "full_input_sha256": _hash_arrays(
                        group,
                        (
                            "sma_T_in_51_K",
                            "htf_directional_T_in_101_K",
                            "side_state_values",
                            "topology_valve_mask",
                        ),
                    ),
                }
            )

    field_overlap = _cross_role_hashes(field_records, "field_input_sha256")
    full_overlap = _cross_role_hashes(field_records, "full_input_sha256")
    train_test = operators[
        operators["sample_id"].map(protocol).isin({"train", "test"})
    ]
    train_runs = set(
        operators.loc[operators["sample_id"].map(protocol) == "train", "physical_case_id"]
    )
    test_runs = set(
        operators.loc[operators["sample_id"].map(protocol) == "test", "physical_case_id"]
    )

    scalar_columns = set(scalars.columns)
    operator_stages = sorted(int(value) for value in operators["stage_index"].unique())
    operator_materials = sorted(operators["material_id"].dropna().unique().tolist())
    operator_geometries = sorted(operators["geometry_id"].dropna().unique().tolist())

    wp2_design_ready = WP2_DESIGN_TARGETS.issubset(scalar_columns)
    missing_full_inputs = sorted(WP2_FULL_CYCLE_INPUTS - scalar_columns)
    missing_full_targets = sorted(WP2_FULL_CYCLE_TARGETS - scalar_columns)
    wp2_full_ready = not missing_full_inputs and not missing_full_targets
    wp3_interface_ready = required_fields_complete and set(operator_stages).issuperset({2, 4})
    wp3_rollout_ready = set(operator_stages).issuperset({1, 2, 3, 4})
    wp4_interface_ready = (
        wp3_interface_ready
        and not full_overlap
        and protocol_declarations_match_rows
        and duration_metadata_matches_hdf5
    )
    wp4_generalization_ready = (
        wp4_interface_ready
        and len(operator_materials) >= 2
        and len(operator_geometries) >= 2
        and len(train_runs) >= 2
        and len(test_runs) >= 2
    )

    blocking_reasons = []
    if not wp2_full_ready:
        blocking_reasons.append("WP2 full-cycle controls/targets are absent from scalar_cases.")
    if not wp3_rollout_ready:
        blocking_reasons.append("WP3 has exchange checkpoints for stages 2/4 only; autonomous four-stage rollout is unproven.")
    if not wp4_generalization_ready:
        blocking_reasons.append("WP4 corpus has one material/geometry baseline and one train/test protocol run each.")
    if field_overlap:
        blocking_reasons.append("Field-only inputs repeat across protocol train/test; no-side-state results carry leakage risk.")
    if not protocol_declarations_match_rows:
        blocking_reasons.append("Operator protocol declarations do not match row-level assignments.")
    if not duration_metadata_matches_hdf5:
        blocking_reasons.append("Operator tabular durations do not match retained HDF5 time grids.")

    return {
        "dataset_version": "1.0",
        "wp2": {
            "design_law_baseline_ready": wp2_design_ready,
            "full_cycle_performance_ready": wp2_full_ready,
            "available_design_targets": sorted(WP2_DESIGN_TARGETS & scalar_columns),
            "missing_full_cycle_inputs": missing_full_inputs,
            "missing_full_cycle_targets": missing_full_targets,
        },
        "wp3": {
            "exchange_interface_prototype_ready": wp3_interface_ready,
            "autonomous_four_stage_rollout_ready": wp3_rollout_ready,
            "observed_stage_indices": operator_stages,
        },
        "wp4": {
            "stage_conditioned_interface_prototype_ready": wp4_interface_ready,
            "cross_material_geometry_generalization_ready": wp4_generalization_ready,
            "operator_samples": int(len(operators)),
            "physical_runs": int(operators["physical_case_id"].nunique()),
            "protocol_train_physical_runs": len(train_runs),
            "protocol_test_physical_runs": len(test_runs),
            "materials": operator_materials,
            "geometries": operator_geometries,
            "required_fields_complete": bool(required_fields_complete),
            "protocol_declarations_match_rows": protocol_declarations_match_rows,
            "duration_metadata_matches_hdf5": duration_metadata_matches_hdf5,
            "protocols_by_role": row_protocols_by_role,
            "field_only_cross_role_duplicates": field_overlap,
            "full_state_cross_role_duplicates": full_overlap,
            "protocol_train_test_samples": int(len(train_test)),
        },
        "paper_grade_gate": {
            "passed": (
                wp2_full_ready
                and wp3_rollout_ready
                and wp4_generalization_ready
                and not field_overlap
                and protocol_declarations_match_rows
                and duration_metadata_matches_hdf5
            ),
            "blocking_reasons": blocking_reasons,
        },
    }


if __name__ == "__main__":
    print(json.dumps(audit_wp2_wp4_readiness(), ensure_ascii=False, indent=2))
