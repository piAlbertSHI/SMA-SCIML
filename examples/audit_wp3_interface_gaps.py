"""Summarize archived state-to-field interface gaps without model refitting."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _max_values(mapping):
    return {key: float(value) for key, value in mapping.items()}


def run(output):
    contract_path = ROOT / "evidence/archive/wp3-input-contract-r1/case_contracts.json"
    source_summary_path = ROOT / "evidence/archive/wp3-input-contract-r1/summary.json"
    scalar_path = ROOT / "data/scalar63.npz"
    prior_audit_path = ROOT / "evidence/local_studies/wp3_state_field_contract_20261123/audit.json"
    if Path(output).exists():
        raise FileExistsError("Preserve previous WP3 audits and choose a new output directory")

    contracts = read_json(contract_path)
    source_summary = read_json(source_summary_path)
    prior_audit = read_json(prior_audit_path)
    with np.load(scalar_path, allow_pickle=False) as source:
        case_ids = source["case_ids"].tolist()
        splits = source["split"].tolist()
    split_by_case = dict(zip(case_ids, splits))
    if len(contracts) != 63 or set(split_by_case) != {row["case_id"] for row in contracts}:
        raise ValueError("Interface contracts and the 63-case scalar index do not align")

    gap_rows = []
    declared_rows = []
    for contract in contracts:
        case_id = contract["case_id"]
        split = split_by_case[case_id]
        for name, value in contract["declared_initial_differences"].items():
            declared_rows.append({"case_id": case_id, "split": split, "field": name, "gap": float(value)})
        for stage_record in contract["stages"]:
            stage = int(stage_record["stage"])
            for field, value in stage_record["canonical_vs_observed_start_max_abs"].items():
                gap_rows.append({
                    "case_id": case_id,
                    "split": split,
                    "stage": stage,
                    "field": field,
                    "gap": float(value),
                })
            if stage_record["requested_vs_observed_control_max_abs"] != 0.0:
                raise ValueError(f"Control mismatch in {case_id}/stage{stage}")

    declared_by_field = {}
    for row in declared_rows:
        declared_by_field.setdefault(row["field"], []).append(row["gap"])
    stage_by_key = {}
    for row in gap_rows:
        stage_by_key.setdefault((row["stage"], row["field"]), []).append(row["gap"])

    declared_summary = {
        field: {
            "case_count": len(values),
            "nonzero_case_count": int(np.sum(np.asarray(values) > 0)),
            "max_abs_gap": float(np.max(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
        }
        for field, values in sorted(declared_by_field.items())
    }
    stage_summary = {
        f"stage{stage}:{field}": {
            "case_count": len(values),
            "nonzero_case_count": int(np.sum(np.asarray(values) > 0)),
            "max_abs_gap": float(np.max(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
        }
        for (stage, field), values in sorted(stage_by_key.items())
    }
    stage4_path = [row for row in gap_rows if row["stage"] == 4 and row["field"] == "stage4_path_pipe_T_state_in_K"]
    stage4_path_sorted = sorted(stage4_path, key=lambda row: row["gap"], reverse=True)
    top_cases = stage4_path_sorted[:10]
    request_rmse = source_summary["initialization_sensitivity"]
    if source_summary["cases_with_stage4_future_initial_difference"] != 63:
        raise ValueError("Source summary no longer reports all stage-4 future gaps")
    if prior_audit["input_boundary"]["requested_vs_observed_controls_max_abs"] != 0.0:
        raise ValueError("Prior WP3 audit does not confirm exact control matching")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "stage_interface_gaps.csv", gap_rows, ["case_id", "split", "stage", "field", "gap"])
    write_csv(output / "declared_initial_gaps.csv", declared_rows, ["case_id", "split", "field", "gap"])
    audit = {
        "status": "verified_wp3_interface_gap_distribution",
        "learning_only": True,
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "case_count": len(contracts),
        "split_counts": {split: int(splits.count(split)) for split in sorted(set(splits))},
        "control_contract": {
            "requested_vs_observed_max_abs": source_summary["requested_vs_observed_controls_max_abs"],
            "all_63_cases_exact": True,
        },
        "declared_record_vs_request_gaps": declared_summary,
        "stage_start_gap_distribution": stage_summary,
        "stage4_path_pipe_top_cases": top_cases,
        "record_vs_request_initialization_rmse": request_rmse,
        "interpretation": {
            "stage4_path_pipe_gap_is_interface_diagnostic": True,
            "reported_as_formal_model_error_lower_bound": False,
            "record_derived_state_and_request_only_state_are_distinct_inputs": True,
            "hidden_storage_and_restart_ownership_closed": False,
        },
        "source_sha256": {
            "case_contracts": sha256(contract_path),
            "source_summary": sha256(source_summary_path),
            "scalar_case_index": sha256(scalar_path),
            "prior_wp3_audit": sha256(prior_audit_path),
            "audit_code": sha256(Path(__file__).resolve()),
        },
    }
    write_json(output / "audit.json", audit)
    write_json(output / "manifest.json", {
        "sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    })
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "split_counts": result["split_counts"],
        "stage4_path_pipe_max_gap_K": result["stage_start_gap_distribution"]["stage4:stage4_path_pipe_T_state_in_K"]["max_abs_gap"],
        "stage4_path_pipe_nonzero_cases": result["stage_start_gap_distribution"]["stage4:stage4_path_pipe_T_state_in_K"]["nonzero_case_count"],
        "fresh_training_calls": result["fresh_training_calls"],
        "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
