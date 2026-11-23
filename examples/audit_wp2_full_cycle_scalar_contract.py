"""Audit the archived 63-case full-cycle scalar contract without refitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260910, 20260911, 20260912)
MODEL_NAMES = ("ensemble5", "ridge", "mlp")
ARCHIVED_TARGETS = ("Qcool_J", "Wpump_J", "COP", "SCP_W_per_kg", "deltaT_hot_cold_max_K")
FULL_CYCLE_TARGETS = ARCHIVED_TARGETS + ("Wtotal_for_COP_J",)
MISSING_INPUTS = (
    "V_star",
    "cycle_frequency",
    "failure_label_linked_to_case_id",
    "independent_Q_environment_J",
    "independent_Q_internal_pipe_fluid_J",
)


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


def _assert_close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=label)


def run(output):
    scalar_path = ROOT / "data/scalar63.npz"
    metrics_path = ROOT / "data/metrics.json"
    archive_root = ROOT / "evidence/archive/wp2-strong-r1"
    protocol_path = archive_root / "protocol.json"
    summary_path = archive_root / "summary.json"
    if Path(output).exists():
        raise FileExistsError("Preserve previous WP2 audits and choose a new output directory")

    protocol = read_json(protocol_path)
    archived_summary = read_json(summary_path)
    metrics = read_json(metrics_path)
    with np.load(scalar_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}

    if arrays["x"].shape != (63, 7) or arrays["y"].shape != (63, 6):
        raise ValueError("Unexpected full-cycle scalar shapes")
    if arrays["feature_names"].tolist() != protocol["feature_names"]:
        raise ValueError("Full-cycle feature contract differs from archived protocol")
    if arrays["target_names"].tolist() != list(FULL_CYCLE_TARGETS):
        raise ValueError("Full-cycle target contract differs from the saved scalar snapshot")
    if len(np.unique(arrays["case_ids"])) != 63:
        raise ValueError("Case IDs are not unique")
    if not np.isfinite(arrays["x"]).all() or not np.isfinite(arrays["y"]).all():
        raise ValueError("Full-cycle scalar snapshot contains non-finite values")

    split_counts = {str(name): int(np.sum(arrays["split"] == name)) for name in np.unique(arrays["split"])}
    if split_counts != {"train": 37, "calibration": 13, "test": 13}:
        raise ValueError(f"Unexpected split counts: {split_counts}")
    if protocol["target_names"] != list(ARCHIVED_TARGETS):
        raise ValueError("Archived five-target model contract changed")
    if protocol["seeds"] != list(SEEDS):
        raise ValueError("Archived seed contract changed")

    target_index = {name: index for index, name in enumerate(arrays["target_names"])}
    qcool = arrays["y"][:, target_index["Qcool_J"]]
    wpump = arrays["y"][:, target_index["Wpump_J"]]
    cop = arrays["y"][:, target_index["COP"]]
    scp = arrays["y"][:, target_index["SCP_W_per_kg"]]
    wtotal = arrays["y"][:, target_index["Wtotal_for_COP_J"]]
    identity_residuals = {
        "COP_minus_Qcool_div_Wtotal": float(np.max(np.abs(cop - qcool / wtotal))),
        "Qcool_minus_COP_times_Wtotal": float(np.max(np.abs(qcool - cop * wtotal))),
    }
    if identity_residuals["COP_minus_Qcool_div_Wtotal"] != 0.0:
        raise ValueError("COP identity is not exact in the archived scalar snapshot")
    if identity_residuals["Qcool_minus_COP_times_Wtotal"] > 1e-12:
        raise ValueError("Reverse COP identity exceeds floating-point tolerance")
    if np.any(qcool <= 0) or np.any(wpump <= 0) or np.any(wtotal <= 0) or np.any(cop <= 0) or np.any(scp <= 0):
        raise ValueError("Expected positive full-cycle scalar targets are not all positive")

    reports = []
    for seed in SEEDS:
        report = read_json(archive_root / f"s{seed}/report.json")
        if (report["training_count"], report["calibration_count"], report["test_count"]) != (37, 13, 13):
            raise ValueError(f"Archived report count mismatch for seed {seed}")
        reports.append(report)
    for model_name in MODEL_NAMES:
        seed_rows = [report["model_scores"][model_name]["test"]["rmse"] for report in reports]
        mean = np.mean(seed_rows, axis=0)
        sample_std = np.std(seed_rows, axis=0, ddof=1)
        archived = archived_summary["comparison"][model_name]
        _assert_close(mean, archived["seed_mean_test_rmse"], f"{model_name} mean mismatch")
        _assert_close(sample_std, archived["seed_std_test_rmse"], f"{model_name} standard deviation mismatch")

    if metrics.get("scalar", {}).get("target_names") != list(ARCHIVED_TARGETS):
        raise ValueError("metrics.json scalar target contract mismatch")
    if metrics["scalar"]["comparison"] != archived_summary["comparison"]:
        raise ValueError("metrics.json and archived WP2 scalar summary differ")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        "status": "verified_wp2_full_cycle_scalar_contract",
        "learning_only": True,
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "scope": {
            "cases": 63,
            "split_counts": split_counts,
            "archived_model_targets": list(ARCHIVED_TARGETS),
            "saved_auxiliary_target": "Wtotal_for_COP_J",
            "saved_features": arrays["feature_names"].tolist(),
            "missing_or_unlinked_inputs": list(MISSING_INPUTS),
            "failure_labels_joined_to_scalar_cases": False,
        },
        "contract_checks": {
            "unique_case_ids": 63,
            "all_values_finite": True,
            "positive_target_ranges": True,
            "export_identity_max_abs_residual": identity_residuals,
            "archived_three_seed_reports_reconciled": True,
            "metrics_json_reconciled": True,
        },
        "target_ranges": {
            name: {"min": float(arrays["y"][:, index].min()), "max": float(arrays["y"][:, index].max())}
            for name, index in target_index.items()
        },
        "reliability_boundary": {
            "model_scores_are_archived_descriptive_results": True,
            "independent_new_design_confirmation": False,
            "full_energy_closure": False,
            "failure_label_reliability": "not evaluable from scalar63 snapshot because failure labels are not linked",
        },
        "source_sha256": {
            "data/scalar63.npz": sha256(scalar_path),
            "data/metrics.json": sha256(metrics_path),
            "evidence/archive/wp2-strong-r1/protocol.json": sha256(protocol_path),
            "evidence/archive/wp2-strong-r1/summary.json": sha256(summary_path),
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
        "cases": result["scope"]["cases"],
        "split_counts": result["scope"]["split_counts"],
        "archived_model_targets": result["scope"]["archived_model_targets"],
        "cop_identity_max_abs_residual": result["contract_checks"]["export_identity_max_abs_residual"],
        "fresh_training_calls": result["fresh_training_calls"],
        "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
