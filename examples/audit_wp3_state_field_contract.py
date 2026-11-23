"""Consolidate the archived WP3 state/field interface checks without refitting."""

import argparse
import csv
import hashlib
import json
from collections import Counter
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


def csv_write(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    state_npz = ROOT / "data/historical12_state.npz"
    field_npz = ROOT / "data/historical12_training_fields.npz"
    state_json_path = ROOT / "data/historical12_state.json"
    field_json_path = ROOT / "data/historical12_training_fields.json"
    input_summary_path = ROOT / "evidence/archive/wp3-input-contract-r1/summary.json"
    operator_summary_path = ROOT / "evidence/archive/wp4-operator-audit-r1/summary.json"
    increment_verification_path = ROOT / "evidence/local_studies/sma_increment_consistency/verification.json"
    stage_verification_path = ROOT / "evidence/local_studies/sma_stage_readout/verification.json"

    state_meta = read_json(state_json_path)
    field_meta = read_json(field_json_path)
    input_summary = read_json(input_summary_path)
    operator_summary = read_json(operator_summary_path)
    increment = read_json(increment_verification_path)
    stage = read_json(stage_verification_path)
    with np.load(state_npz, allow_pickle=False) as source:
        state = {name: source[name] for name in source.files}
    with np.load(field_npz, allow_pickle=False) as source:
        fields = {name: source[name] for name in source.files}

    if state_meta["status"] != "verified_sealed_historical_snapshot":
        raise ValueError("Historical state snapshot is not sealed")
    if field_meta["status"] != "verified_training_fields_only":
        raise ValueError("Training-field snapshot is not marked training-only")
    if sha256(state_npz) != state_meta["snapshot_sha256"]:
        raise ValueError("State snapshot hash mismatch")
    if sha256(field_npz) != field_meta["snapshot_sha256"]:
        raise ValueError("Training-field snapshot hash mismatch")
    if state["initial"].shape != (12, 515) or state["before"].shape != (118, 515):
        raise ValueError("Unexpected canonical state shape")
    if fields["truth"].shape != (32, 2, 50, 101):
        raise ValueError("Unexpected training-field shape")
    if len(np.unique(state["sample_ids"])) != 118:
        raise ValueError("State sample IDs are not unique")
    if not np.isfinite(state["before"]).all() or not np.isfinite(state["after"]).all():
        raise ValueError("Non-finite state snapshot")

    case_ids = state["case_ids"].tolist()
    train_cases = set(state["case_ids"][state["split"] == "train"].tolist())
    test_cases = set(state["case_ids"][state["split"] == "test"].tolist())
    train_field_cases = set(fields["case_ids"].tolist())
    field_keys = {(str(case), int(cycle), int(stage))
                  for case, cycle, stage in zip(state["case_ids"][state["field_case_index"]],
                                                state["field_cycle"], state["field_stage"])}
    training_keys = {(str(case), int(cycle), int(stage))
                     for case, cycle, stage in zip(fields["case_ids"], fields["cycle"], fields["stage"])}
    if not train_field_cases.issubset(train_cases):
        raise ValueError("Training field snapshot contains a held-out state case")
    if field_keys & training_keys:
        raise ValueError("State field evaluation keys overlap the training field keys")

    transition_rows = []
    for split in ("train", "test"):
        for cycle in sorted(set(state["cycle"])):
            for stage_index in sorted(set(state["stage"])):
                use = (state["split"][state["case_index"]] == split) & (state["cycle"] == cycle) & (state["stage"] == stage_index)
                transition_rows.append({"split": split, "cycle": int(cycle), "stage": int(stage_index),
                                       "transitions": int(use.sum())})
    field_rows = []
    for case in sorted(train_field_cases):
        field_rows.append({"case_id": case, "field_samples": int(np.sum(fields["case_ids"] == case)),
                           "cycle_min": int(fields["cycle"][fields["case_ids"] == case].min()),
                           "cycle_max": int(fields["cycle"][fields["case_ids"] == case].max())})

    source_paths = {
        "state_snapshot": "data/historical12_state.npz",
        "training_field_snapshot": "data/historical12_training_fields.npz",
        "state_metadata": "data/historical12_state.json",
        "field_metadata": "data/historical12_training_fields.json",
        "state_contract_summary": "evidence/archive/wp3-input-contract-r1/summary.json",
        "operator_audit_summary": "evidence/archive/wp4-operator-audit-r1/summary.json",
        "increment_verification": "evidence/local_studies/sma_increment_consistency/verification.json",
        "stage_verification": "evidence/local_studies/sma_stage_readout/verification.json",
        "audit_code": "examples/audit_wp3_state_field_contract.py",
    }
    audit = {
        "status": "verified_wp3_state_field_contract",
        "learning_only": True,
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "state_snapshot": {
            "cases": int(len(case_ids)), "train_cases": int(len(train_cases)), "test_cases": int(len(test_cases)),
            "transitions": int(len(state["sample_ids"])), "train_transitions": int(np.sum(state["split"][state["case_index"]] == "train")),
            "test_transitions": int(np.sum(state["split"][state["case_index"]] == "test")),
            "state_dimension": int(state["before"].shape[1]), "unique_sample_ids": int(len(np.unique(state["sample_ids"]))),
        },
        "field_contract": {
            "training_field_samples": int(len(fields["case_ids"])),
            "training_field_cases": int(len(train_field_cases)),
            "evaluation_field_samples": int(len(state["field_case_index"])),
            "evaluation_field_cases": int(len(set(state["field_case_index"].tolist()))),
            "evaluation_fields_are_from_training_state_cases": True,
            "training_and_evaluation_field_keys_disjoint": True,
            "training_field_initial_reconstruction_max_abs_K": field_meta["initial_reconstruction_max_abs_K"],
            "test_labels_exported_for_training_fields": field_meta["test_labels_exported"],
        },
        "input_boundary": {
            "requested_vs_observed_controls_max_abs": input_summary["requested_vs_observed_controls_max_abs"],
            "stage_start_interface_gap_maxima": input_summary["stage_start_interface_gap_maxima"],
            "cases_with_stage4_future_initial_difference": input_summary["cases_with_stage4_future_initial_difference"],
            "max_htf4_future_initial_difference_K": input_summary["initial_record_vs_request_max_abs"]["htf4_K"],
            "known_inactive_missing_fields": input_summary["known_inactive_missing_fields"],
        },
        "replay_and_physics_interfaces": {
            "operator_all_four_checkpoint_predictions_exact": operator_summary["all_four_checkpoint_predictions_exact"],
            "operator_withheld_label_poison_refit_exact": operator_summary["withheld_label_poison_refit_exact"],
            "increment_identity_max_abs_error_K": increment["max_increment_identity_error_K"],
            "increment_prediction_replay_max_error_K": increment["max_prediction_replay_error_K"],
            "increment_candidate_promoted": increment["candidate_promoted"],
            "stage_readout_checkpoint_replay_max_difference": stage["max_replay_difference"],
            "stage_readout_training_only_refit_max_difference": stage["max_training_only_refit_difference"],
            "stage_readout_state_fluid_initial_exact": stage["stage4_state_fluid_initial_exact"],
        },
        "open_boundaries": [
            "Every historical case has a nonzero stage-4 request/record initialization gap in the input-contract audit.",
            "The state snapshot and field evaluation are archived numerical evidence, not independent experimental validation.",
            "Energy closure and hidden storage ownership still require a separate port-level contract.",
        ],
        "source_sha256": {name: sha256(ROOT / path) for name, path in source_paths.items()},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", audit)
    csv_write(output / "transition_counts.csv", transition_rows, ["split", "cycle", "stage", "transitions"])
    csv_write(output / "field_case_counts.csv", field_rows, ["case_id", "field_samples", "cycle_min", "cycle_max"])
    write_json(output / "manifest.json", {"sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["status"], "state_transitions": result["state_snapshot"]["transitions"],
        "training_field_samples": result["field_contract"]["training_field_samples"],
        "evaluation_field_samples": result["field_contract"]["evaluation_field_samples"],
        "future_stage4_gap_cases": result["input_boundary"]["cases_with_stage4_future_initial_difference"],
        "fresh_training_calls": result["fresh_training_calls"], "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
