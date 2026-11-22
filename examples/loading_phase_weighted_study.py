"""Diagnose recAs 0.9 arrivals and select a near-saturation weighted phase head."""

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from examples.loading_field_benchmark import RADIUS, ROOT, TIME, TI, training_arrays, score
from examples.loading_phase_study import (
    BASE_FILES, BENCHMARK, LEVELS, TOL, arrival_times, front_scores, load_snapshot,
)
from SciML.models.v13.loading_field_operator import PODRidgeLoading
from SciML.models.v13.loading_phase_reconstruction import MaterialPressurePhaseRidge
from SciML.models.v13.loading_phase_weighted import SaturationWeightedPhaseRidge
from tools.audit_project_corpora import read_json, sha256
from tools.compare_solver_archives import write


THRESHOLD = .9
SATURATION_WEIGHT_GRID = (40.0, 64.0, 100.0)
BLEND_GRID = ((.8, .85), (.8, .88), (.8, .92))
SOURCE_FILES = (
    "examples/loading_phase_weighted_study.py", "examples/loading_phase_weighted_report.py",
    "examples/loading_phase_study.py", "SciML/models/v13/loading_phase_weighted.py",
    "SciML/models/v13/loading_phase_reconstruction.py", "SciML/models/v13/loading_field_operator.py",
    "SciML/models/v12/learning_coordinate_operator.py",
)


def protocol_for(base, source_protocol):
    manifest = read_json(base / "manifest.json")["sha256"]
    return {
        "study": "recAs near-saturation phase arrival diagnosis and weighted readout",
        "base_file_sha256": {name: manifest[name] for name in BASE_FILES},
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "source_baseline_protocol_sha256": sha256(base / "protocol.json"),
        "case_counts": source_protocol["case_counts"],
        "split_key": source_protocol["split_key"],
        "split_stratification": source_protocol["stratification"],
        "native_grid": source_protocol["native_grid"],
        "fit_time_indices": source_protocol["train_time_indices"],
        "fit_radius_indices": source_protocol["train_radius_indices"],
        "frozen_pod_rank": source_protocol["pod_ridge"]["rank_each_channel"],
        "ridge_alpha": 1.0,
        "saturation_threshold": .8,
        "saturation_weight_grid": list(SATURATION_WEIGHT_GRID),
        "blend_grid": [list(pair) for pair in BLEND_GRID],
        "fit_scope": "Phase readout only. Baseline POD, mechanical fields and temperature field are frozen.",
        "weight_scope": "recAs training points with phase >= 0.8 receive additional field-regression weight.",
        "selection": "Validation recAs 0.9 missed radial nodes, then transition-band RMSE, then common-arrival MAE.",
        "arrival_threshold": THRESHOLD,
        "arrival_definition": "First threshold crossing by linear interpolation between native time samples.",
        "future_frame_excluded": True,
        "fresh_solver_calls": 0,
    }


def recas_arrival_summary(case_ids, controls, truth, baseline, candidate):
    expected = arrival_times(truth, THRESHOLD)
    result = {}
    for name, prediction in (("baseline", baseline), ("weighted", candidate)):
        arrival = arrival_times(prediction, THRESHOLD)
        reachable = np.isfinite(expected)
        available = np.isfinite(arrival)
        common = reachable & available
        errors = np.abs(arrival - expected)
        per_case = []
        for i, case_id in enumerate(case_ids):
            target_nodes = np.flatnonzero(reachable[i])
            common_nodes = np.flatnonzero(common[i])
            missed_nodes = np.flatnonzero(reachable[i] & ~available[i])
            false_nodes = np.flatnonzero(~reachable[i] & available[i])
            valid_errors = errors[i, common[i]]
            per_case.append({
                "case_id": str(case_id), "radius_mm": float(controls[i, 0] * 1000),
                "peak_pressure_MPa": float(controls[i, 2] / 1e6),
                "reference_reached_nodes": int(len(target_nodes)),
                "common_nodes": int(len(common_nodes)),
                "missed_nodes": [{"node": int(node), "r_over_R": float(RADIUS[node]),
                                  "reference_arrival_ms": float(expected[i, node])}
                                 for node in missed_nodes],
                "false_nodes": [{"node": int(node), "r_over_R": float(RADIUS[node]),
                                 "predicted_arrival_ms": float(arrival[i, node])}
                                for node in false_nodes],
                "common_mae_ms": float(valid_errors.mean()) if len(valid_errors) else None,
                "common_p95_ms": float(np.quantile(valid_errors, .95)) if len(valid_errors) else None,
                "common_error_max_ms": float(valid_errors.max()) if len(valid_errors) else None,
            })
        valid = errors[common]
        result[name] = {
            "reference_reached_nodes": int(reachable.sum()), "common_nodes": int(common.sum()),
            "missed_nodes": int((reachable & ~available).sum()),
            "false_nodes": int((~reachable & available).sum()),
            "common_mae_ms": float(valid.mean()) if len(valid) else None,
            "common_p95_ms": float(np.quantile(valid, .95)) if len(valid) else None,
            "per_case": per_case,
        }
    result["change"] = {
        "missed_nodes_reduction": result["baseline"]["missed_nodes"] - result["weighted"]["missed_nodes"],
        "common_mae_change_ms": result["weighted"]["common_mae_ms"] - result["baseline"]["common_mae_ms"],
    }
    return result


def arrival_rows(case_ids, controls, truth, baseline, candidate, split):
    expected = arrival_times(truth, THRESHOLD)
    baseline_arrival = arrival_times(baseline, THRESHOLD)
    candidate_arrival = arrival_times(candidate, THRESHOLD)
    rows = []
    for i, case_id in enumerate(case_ids):
        if controls[i, 3] != 1.0:
            continue
        for node in range(len(RADIUS)):
            ref = expected[i, node]
            base = baseline_arrival[i, node]
            weighted = candidate_arrival[i, node]
            if not (np.isfinite(ref) or np.isfinite(base) or np.isfinite(weighted)):
                continue
            rows.append({
                "split": split, "case_id": str(case_id), "radius_mm": float(controls[i, 0] * 1000),
                "thickness_um": float(controls[i, 1] * 1e6), "peak_pressure_MPa": float(controls[i, 2] / 1e6),
                "r_over_R": float(RADIUS[node]), "radial_node": int(node),
                "reference_arrival_ms": float(ref) if np.isfinite(ref) else "",
                "baseline_arrival_ms": float(base) if np.isfinite(base) else "",
                "weighted_arrival_ms": float(weighted) if np.isfinite(weighted) else "",
                "baseline_missed": bool(np.isfinite(ref) and not np.isfinite(base)),
                "weighted_missed": bool(np.isfinite(ref) and not np.isfinite(weighted)),
                "baseline_abs_error_ms": float(abs(base - ref)) if np.isfinite(base) and np.isfinite(ref) else "",
                "weighted_abs_error_ms": float(abs(weighted - ref)) if np.isfinite(weighted) and np.isfinite(ref) else "",
            })
    return rows


def csv_text(rows):
    buffer = io.StringIO(newline="")
    fieldnames = [
        "split", "case_id", "radius_mm", "thickness_um", "peak_pressure_MPa", "r_over_R", "radial_node",
        "reference_arrival_ms", "baseline_arrival_ms", "weighted_arrival_ms", "baseline_missed", "weighted_missed",
        "baseline_abs_error_ms", "weighted_abs_error_ms",
    ]
    writer = csv.DictWriter(buffer, fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_manifest(out):
    manifest = out / "manifest.json"
    if manifest.exists():
        manifest.unlink()
    write(manifest, {
        "sha256": {p.name: sha256(p) for p in sorted(out.iterdir())
                   if p.is_file() and p.name != "manifest.json"}
    })


def evaluate_partition(data, baseline, candidate, split):
    use = data["split"] == split
    controls, truth, case_ids = (data[key][use] for key in ("controls", "fields", "case_ids"))
    baseline_fields = baseline.predict(controls, TIME, RADIUS)
    candidate_fields = candidate.predict(controls, TIME, RADIUS)
    np.testing.assert_array_equal(baseline_fields[:, :3], candidate_fields[:, :3])
    np.testing.assert_array_equal(baseline_fields[:, :, 0], truth[:, :, 0])
    phases = {
        "clipped": np.clip(baseline_fields[:, 3], 0.0, 1.0),
        "baseline": baseline_fields[:, 3], "weighted": candidate_fields[:, 3],
    }
    band = (truth[:, 3] >= LEVELS[0]) & (truth[:, 3] <= LEVELS[-1])
    front = front_scores(phases, truth[:, 3], band)
    field_metrics = {}
    for name, fields in (("baseline", baseline_fields), ("weighted", candidate_fields)):
        metrics = score(fields, truth, baseline.pod.parameters["field_scale"], case_ids, controls)
        metrics["phase_temporal_decreases"] = int((np.diff(fields[:, 3], axis=1) < -1e-12).sum())
        field_metrics[name] = metrics
    recas = controls[:, 3] == 1.0
    arrival = recas_arrival_summary(case_ids[recas], controls[recas], truth[recas, 3],
                                    phases["baseline"][recas], phases["weighted"][recas])
    return {
        "case_count": int(use.sum()), "reference_transition_points": int(band.sum()),
        "metrics": {name: {"field_rmse": field_metrics[name]["rmse"],
                            "phase_outside_bounds_fraction": field_metrics[name]["phase_outside_bounds_fraction_tol_1e-6"],
                            "phase_temporal_decreases": field_metrics[name]["phase_temporal_decreases"],
                            "front": front[name]}
                    for name in ("baseline", "weighted")},
        "recAs_threshold_0p9": arrival,
        "case_ids": case_ids.tolist(),
    }, arrival_rows(case_ids, controls, truth[:, 3], phases["baseline"], phases["weighted"], split)


def candidate_key(validation):
    result = validation["recAs_threshold_0p9"]["weighted"]
    front = validation["metrics"]["weighted"]["front"]["rmse"]
    return [result["missed_nodes"], front, result["common_mae_ms"]]


def run(base, out):
    from examples.loading_phase_weighted_report import draw_arrival

    started = time.perf_counter()
    data, _ = load_snapshot(base)
    protocol = protocol_for(base, read_json(base / "protocol.json"))
    out.mkdir(parents=False, exist_ok=False)
    write(out / "protocol.json", protocol)
    train_controls, normalized_change, _, _, _ = training_arrays(data)
    base_model = PODRidgeLoading.load(base / "pod_ridge.npz")
    baseline = MaterialPressurePhaseRidge.fit(base_model, train_controls, normalized_change[:, 3])
    candidate_reports = {}
    candidate_models = {}
    fit_seconds = 0.0
    for weight_strength in SATURATION_WEIGHT_GRID:
        for blend_start, blend_end in BLEND_GRID:
            before = time.perf_counter()
            model = SaturationWeightedPhaseRidge.fit(
                baseline, train_controls, normalized_change[:, 3],
                saturation_threshold=.8, weight_strength=weight_strength,
                blend_start=blend_start, blend_end=blend_end,
            )
            fit_seconds += time.perf_counter() - before
            label = f"weight_{int(weight_strength)}_blend_{blend_start:.2f}_{blend_end:.2f}"
            report, _ = evaluate_partition(data, baseline, model, "validation")
            candidate_reports[label] = {
                "weight_strength": weight_strength, "blend_start": blend_start, "blend_end": blend_end,
                "selection_key": candidate_key(report), "metrics": report["metrics"],
                "recAs_threshold_0p9": report["recAs_threshold_0p9"],
            }
            candidate_models[label] = model
    selected_label = min(candidate_reports, key=lambda label: tuple(candidate_reports[label]["selection_key"]))
    selected = candidate_models[selected_label]
    test_report, test_rows = evaluate_partition(data, baseline, selected, "test")
    validation_report, selected_validation_rows = evaluate_partition(data, baseline, selected, "validation")
    test_controls = data["controls"][data["split"] == "test"]
    test_case_ids = data["case_ids"][data["split"] == "test"]
    test_truth = data["fields"][data["split"] == "test", 3]
    base_test_phase = baseline.predict(test_controls, TIME, RADIUS)[:, 3]
    selected_test_phase = selected.predict(test_controls, TIME, RADIUS)[:, 3]
    draw_arrival(out / "phase_saturation_arrival.png", test_case_ids, test_controls, test_truth,
                 base_test_phase, selected_test_phase)
    selected.save(out / "saturation_weighted_phase.npz")
    (out / "arrival_diagnostics.csv").write_text(csv_text(test_rows + selected_validation_rows), encoding="utf-8", newline="\n")
    summary = {
        "status": "completed", "selection": selected_label, "selection_key": candidate_key(validation_report),
        "candidate_validation": candidate_reports, "validation": validation_report, "test": test_report,
        "selected_checkpoint": "saturation_weighted_phase.npz", "fit_count": len(candidate_reports),
        "fit_seconds": fit_seconds, "runtime_seconds": time.perf_counter() - started, "fresh_solver_calls": 0,
        "selected_channels_preserved_exactly": True,
    }
    write(out / "summary.json", summary)
    write_manifest(out)
    return summary


def verify(base, out):
    for name, digest in read_json(out / "manifest.json")["sha256"].items():
        assert sha256(out / name) == digest, name
    data, _ = load_snapshot(base)
    assert protocol_for(base, read_json(base / "protocol.json")) == read_json(out / "protocol.json")
    # Re-run the fixed validation grid, then replay the saved selected head.
    train_controls, normalized_change, _, _, _ = training_arrays(data)
    base_model = PODRidgeLoading.load(base / "pod_ridge.npz")
    baseline = MaterialPressurePhaseRidge.fit(base_model, train_controls, normalized_change[:, 3])
    saved = read_json(out / "summary.json")
    reports = {}
    for weight_strength in SATURATION_WEIGHT_GRID:
        for blend_start, blend_end in BLEND_GRID:
            label = f"weight_{int(weight_strength)}_blend_{blend_start:.2f}_{blend_end:.2f}"
            model = SaturationWeightedPhaseRidge.fit(
                baseline, train_controls, normalized_change[:, 3], .8, weight_strength, blend_start, blend_end,
            )
            report, _ = evaluate_partition(data, baseline, model, "validation")
            reports[label] = {
                "weight_strength": weight_strength, "blend_start": blend_start, "blend_end": blend_end,
                "selection_key": candidate_key(report), "metrics": report["metrics"],
                "recAs_threshold_0p9": report["recAs_threshold_0p9"],
            }
    assert min(reports, key=lambda label: tuple(reports[label]["selection_key"])) == saved["selection"]
    selected_model = SaturationWeightedPhaseRidge.load(out / saved["selected_checkpoint"], baseline)
    test_report, test_rows = evaluate_partition(data, baseline, selected_model, "test")
    validation_report, validation_rows = evaluate_partition(data, baseline, selected_model, "validation")
    expected = {"status": "completed", "selection": saved["selection"], "selection_key": saved["selection_key"],
                "candidate_validation": reports, "validation": validation_report, "test": test_report,
                "selected_checkpoint": saved["selected_checkpoint"], "fit_count": saved["fit_count"],
                "selected_channels_preserved_exactly": saved["selected_channels_preserved_exactly"],
                "fresh_solver_calls": 0}
    for key, value in expected.items():
        if key in ("candidate_validation", "validation", "test"):
            from examples.verify_loading_field_benchmark import compare
            compare(value, saved[key])
        else:
            assert value == saved[key], key
    assert csv_text(test_rows + validation_rows) == (out / "arrival_diagnostics.csv").read_text(encoding="utf-8")
    verification = {
        "status": "verified", "selection_replayed": saved["selection"],
        "candidate_grid_replayed": len(reports), "test_arrival_rows_replayed": len(test_rows),
        "test_channels_exact": True, "test_phase_bounds_and_time_monotonicity": True,
        "fresh_solver_calls": 0,
    }
    verification_path = out / "verification.json"
    if verification_path.exists():
        verification_path.unlink()
    write(verification_path, verification)
    write_manifest(out)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BENCHMARK)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = run(args.base, args.output) if args.output else verify(args.base, args.verify)
    print(json.dumps({
        "status": result["status"], "selection": result.get("selection", result.get("selection_replayed")),
        "test_0p9_missed_baseline": result.get("test", {}).get("recAs_threshold_0p9", {}).get("baseline", {}).get("missed_nodes"),
        "test_0p9_missed_weighted": result.get("test", {}).get("recAs_threshold_0p9", {}).get("weighted", {}).get("missed_nodes"),
        "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
