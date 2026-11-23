"""Audit the archived WP2 design-law scalar benchmark without refitting models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMES = ("iid", "geometry_group_holdout", "material_group_holdout", "combined_ood_holdout")


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
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def best_regression(rows, scheme, target, metric="rmse_mean"):
    candidates = [row for row in rows if row["scheme"] == scheme and row["task"] == "regression"
                  and row["target"] == target and row["role"] == "test"
                  and row.get(metric) is not None]
    return min(candidates, key=lambda row: float(row[metric]))


def best_classification(rows, scheme, metric, maximize=True):
    candidates = [row for row in rows if row["scheme"] == scheme and row["task"] == "classification"
                  and row["role"] == "test" and row.get(metric) is not None]
    return (max if maximize else min)(candidates, key=lambda row: float(row[metric]))


def run(solver_repo, output):
    solver_repo = Path(solver_repo).resolve()
    result_path = solver_repo / "SciML/results/wp2_scalar/wp2_results.json"
    checkpoint_path = solver_repo / "SciML/results/wp2_scalar/wp2_checkpoint.json"
    progress_path = solver_repo / "SciML/results/wp2_scalar/wp2_progress.jsonl"
    config_path = solver_repo / "SciML/configs/wp2_scalar.json"
    code_path = solver_repo / "SciML/models/baselines/scalar_benchmarks.py"
    dataset_root = solver_repo / "SciML/datasets/v1.0"
    result = read_json(result_path)
    checkpoint = read_json(checkpoint_path)
    config = read_json(config_path)
    rows = result["summary"]
    raw_rows = result["results"]

    if result.get("subtask_completion_status") != "completed":
        raise ValueError("Archived WP2 scalar subtask is not marked completed")
    if result.get("full_cycle_wp2_status") != "incomplete":
        raise ValueError("Full-cycle WP2 boundary was not retained")
    if tuple(config["split_schemes"]) != SCHEMES:
        raise ValueError("Archived scalar result does not contain the four required split schemes")
    if tuple(checkpoint["completed_schemes"]) != SCHEMES:
        raise ValueError("WP2 checkpoint does not cover every configured split scheme")
    if len(raw_rows) != 1176 or len(rows) != 392:
        raise ValueError(f"Unexpected archived row counts: raw={len(raw_rows)}, summary={len(rows)}")

    provenance = result["provenance"]["hashes"]
    source_hashes = {
        "code_sha256": sha256(code_path),
        "config_sha256": sha256(config_path),
        "scalar_cases_sha256": sha256(dataset_root / "scalar_cases.csv"),
        "split_manifest_sha256": sha256(dataset_root / "split_manifest.json"),
        "provenance_sha256": sha256(dataset_root / "provenance.json"),
        "results_sha256": sha256(result_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "progress_sha256": sha256(progress_path),
    }
    recorded_hashes = {
        "code_sha256": provenance["code_sha256"],
        "config_sha256": provenance["config_sha256"],
        "scalar_cases_sha256": provenance["data_sha256"]["scalar_cases.csv"],
        "split_manifest_sha256": provenance["data_sha256"]["split_manifest.json"],
        "provenance_sha256": provenance["data_sha256"]["provenance.json"],
    }
    if any(source_hashes[key] != value for key, value in recorded_hashes.items()):
        raise ValueError("Archived WP2 provenance hashes do not match the adjacent source files")

    targets = list(config["regression"]["targets"])
    best_rows = []
    ridge_rows = []
    shifts = []
    for scheme in SCHEMES:
        for target in targets:
            best = best_regression(rows, scheme, target)
            ridge = next(row for row in rows if row["scheme"] == scheme and row["task"] == "regression"
                         and row["target"] == target and row["model"] == "linear_ridge"
                         and row["role"] == "test")
            best_rows.append({
                "scheme": scheme, "target": target, "model": best["model"],
                "rmse_mean": best["rmse_mean"], "r2_mean": best["r2_mean"],
                "mean_relative_error_mean": best["mean_relative_error_mean"],
                "n_eval": best["n_eval"],
            })
            ridge_rows.append({
                "scheme": scheme, "target": target, "rmse_mean": ridge["rmse_mean"],
                "r2_mean": ridge["r2_mean"], "mean_relative_error_mean": ridge["mean_relative_error_mean"],
            })
            shifts.append({
                "target": target, "scheme": scheme, "best_test_rmse": best["rmse_mean"],
                "ridge_test_rmse": ridge["rmse_mean"],
            })

    classification_rows = []
    for scheme in SCHEMES:
        prevalence = float(np.mean([
            row["invalid_prevalence"] for row in raw_rows
            if row["scheme"] == scheme and row["task"] == "classification" and row["role"] == "test"
        ]))
        for metric, maximize in (("invalid_auprc_mean", True), ("invalid_f1_mean", True),
                                 ("invalid_recall_mean", True), ("balanced_accuracy_mean", True),
                                 ("brier_mean", False), ("ece_mean", False)):
            best = best_classification(rows, scheme, metric, maximize=maximize)
            classification_rows.append({
                "scheme": scheme, "metric": metric, "model": best["model"],
                "value": best[metric], "invalid_prevalence": prevalence,
                "n_eval": best["n_eval"],
            })

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / "regression_best_test.csv", best_rows,
              ["scheme", "target", "model", "rmse_mean", "r2_mean", "mean_relative_error_mean", "n_eval"])
    csv_write(output / "regression_ridge_test.csv", ridge_rows,
              ["scheme", "target", "rmse_mean", "r2_mean", "mean_relative_error_mean"])
    csv_write(output / "classification_best_test.csv", classification_rows,
              ["scheme", "metric", "model", "value", "invalid_prevalence", "n_eval"])
    csv_write(output / "distribution_shift.csv", shifts, ["target", "scheme", "best_test_rmse", "ridge_test_rmse"])
    audit = {
        "status": "verified_archived_wp2_scalar_reliability",
        "learning_only": True,
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "source_repository": "adjacent gx101r1 workspace",
        "source_scope": {
            "dataset_version": result["dataset_version"],
            "source_result": "SciML/results/wp2_scalar/wp2_results.json",
            "source_checkpoint": "SciML/results/wp2_scalar/wp2_checkpoint.json",
            "source_config": "SciML/configs/wp2_scalar.json",
            "dataset_files": ["scalar_cases.csv", "split_manifest.json", "provenance.json"],
        },
        "integrity": {
            "provenance_hashes_match": True,
            "configured_schemes": list(SCHEMES),
            "raw_result_rows": len(raw_rows),
            "summary_rows": len(rows),
            "checkpoint_completed_schemes": checkpoint["completed_schemes"],
        },
        "scheme_counts": {
            scheme: {
                "train": next(row["n_train"] for row in raw_rows if row["scheme"] == scheme and row["role"] == "test"),
                "test": next(row["n_eval"] for row in raw_rows if row["scheme"] == scheme and row["role"] == "test"),
                "validation": next(row["n_eval"] for row in raw_rows if row["scheme"] == scheme and row["role"] == "validation"),
            }
            for scheme in SCHEMES
        },
        "regression_targets": targets,
        "classification_target": config["classification"]["target"],
        "full_cycle_wp2_status": result["full_cycle_wp2_status"],
        "full_cycle_missing_fields": result["full_cycle_wp2_missing_fields"],
        "provenance_hashes": source_hashes,
    }
    audit["provenance_hashes"]["audit_code_sha256"] = sha256(Path(__file__).resolve())
    write_json(output / "audit.json", audit)
    write_json(output / "manifest.json", {
        "sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    })
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.solver_repo, args.output)
    print(json.dumps({
        "status": result["status"],
        "schemes": result["integrity"]["configured_schemes"],
        "raw_result_rows": result["integrity"]["raw_result_rows"],
        "summary_rows": result["integrity"]["summary_rows"],
        "full_cycle_wp2_status": result["full_cycle_wp2_status"],
        "fresh_training_calls": result["fresh_training_calls"],
        "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
