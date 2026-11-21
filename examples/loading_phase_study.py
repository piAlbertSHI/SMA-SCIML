"""Locate POD phase violations and compare bounded first-loading phase reconstructions."""

import argparse
import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from examples.loading_field_benchmark import RADIUS, RI, ROOT, TIME, TI, score, stratified_split, training_arrays
from examples.verify_loading_field_benchmark import BENCHMARK, compare
from SciML.models.v13.loading_field_operator import PODRidgeLoading
from SciML.models.v13.loading_phase_reconstruction import (
    MaterialPressurePhaseRidge, constrained_phase_fields, resample_phase,
)
from tools.audit_project_corpora import read_json, sha256
from tools.compare_solver_archives import write


TIME_MS = 100.0 * TIME
TOL = 1e-6
LEVELS = (0.1, 0.5, 0.9)
METHODS_FOR_SELECTION = ("clipped", "monotone_linear", "monotone_cubic", "material_pressure_monotone")
SOURCE_FILES = (
    "examples/loading_phase_study.py", "examples/loading_phase_report.py",
    "SciML/models/v13/loading_phase_reconstruction.py", "examples/loading_field_benchmark.py",
    "SciML/models/v13/loading_field_operator.py", "SciML/models/v12/learning_coordinate_operator.py",
)
BASE_FILES = ("loading_data.npz", "pod_ridge.npz", "test_predictions.npz", "protocol.json", "summary.json")


def load_snapshot(base):
    manifest = read_json(base / "manifest.json")["sha256"]
    for name in BASE_FILES:
        if sha256(base / name) != manifest[name]:
            raise ValueError(f"Original benchmark hash mismatch: {name}")
    with np.load(base / "loading_data.npz", allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    original = read_json(base / "protocol.json")
    np.testing.assert_array_equal(data["split"], stratified_split(data["condition_ids"], data["controls"]))
    for name, ids in original["condition_splits"].items():
        assert data["condition_ids"][data["split"] == name].tolist() == ids
    assert np.min(np.diff(data["fields"][data["split"] == "train", 3], axis=1)) == 0.0
    protocol = {
        "study": "Bounds and transformation fronts during first loading on a fixed numerical archive",
        "base_file_sha256": {name: manifest[name] for name in BASE_FILES},
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "case_counts": original["case_counts"],
        "native_grid": original["native_grid"],
        "fit_grid": [len(TI), len(RI)],
        "basis_rank": original["pod_ridge"]["rank_each_channel"],
        "fit_budget": "Shared frozen rank-eight POD; one Ridge coefficient fit per readout, alpha=1. No basis refit.",
        "new_ridge_fits": 1,
        "fit_scope": "Phase only; the pressure mean for each material uses the training partition.",
        "methods": {
            "raw": "Original POD/Ridge with bilinear interpolation.",
            "clipped": "Original native-grid phase clipped pointwise to [0, 1].",
            "monotone_linear": "Bounded isotonic projection at the original POD phase anchors; linear resampling.",
            "monotone_cubic": "Same constrained anchors; shape-preserving cubic time and linear radial interpolation.",
            "material_pressure_raw": "Material-centered pressure and its interaction in the phase Ridge; raw ablation.",
            "material_pressure_monotone": "Material-dependent pressure Ridge with bounded isotonic anchors and linear resampling.",
        },
        "violation_tolerance": TOL,
        "temporal_decrease_tolerance": 1e-12,
        "transition_band": [LEVELS[0], LEVELS[-1]],
        "arrival_thresholds": LEVELS,
        "selection_candidates": METHODS_FOR_SELECTION,
        "selection": "Minimum validation RMSE inside the reference transition band, evaluated before the fixed test retest.",
        "arrival_comparison": "First crossing by linear interpolation of native samples; common reachable radial nodes across all predictors and reference. Report missed and false crossings separately.",
        "fresh_solver_calls": 0,
    }
    return data, protocol


def arrival_times(phase, level):
    """First threshold crossing at each radius, with NaN for no crossing."""
    reached = phase >= level
    first = np.argmax(reached, axis=1)
    lower = np.maximum(first - 1, 0)
    before = np.take_along_axis(phase, lower[:, None, :], axis=1)[:, 0]
    after = np.take_along_axis(phase, first[:, None, :], axis=1)[:, 0]
    weight = np.divide(level - before, after - before, out=np.zeros_like(before), where=first > 0)
    arrival = TIME_MS[lower] + weight * (TIME_MS[first] - TIME_MS[lower])
    return np.where(reached.any(axis=1), arrival, np.nan)


def phase_violations(phase, truth, case_ids, controls, split):
    lower, upper = phase < -TOL, phase > 1.0 + TOL
    outside = lower | upper
    rows = []
    for case, frame, node in np.argwhere(outside):
        rows.append({
            "split": split, "case_id": str(case_ids[case]), "material": "recAs" if controls[case, 3] else "recA",
            "radius_mm": float(1000 * controls[case, 0]), "peak_pressure_MPa": float(controls[case, 2] / 1e6),
            "frame": int(frame), "radial_node": int(node), "time_ms": float(TIME_MS[frame]),
            "r_over_R": float(RADIUS[node]), "r_mm": float(1000 * controls[case, 0] * RADIUS[node]),
            "direction": "below" if lower[case, frame, node] else "above",
            "predicted_phase": float(phase[case, frame, node]), "reference_phase": float(truth[case, frame, node]),
        })
    cases = []
    for i in np.flatnonzero(outside.any(axis=(1, 2))):
        frames, nodes = np.nonzero(outside[i])
        cases.append({
            "case_id": str(case_ids[i]), "below": int(lower[i].sum()), "above": int(upper[i].sum()),
            "time_ms_range": [float(TIME_MS[frames.min()]), float(TIME_MS[frames.max()])],
            "r_over_R_range": [float(RADIUS[nodes.min()]), float(RADIUS[nodes.max()])],
        })
    details = {"points": int(outside.sum()), "future_points": int(phase[:, 1:].size),
               "cases": sorted(cases, key=lambda row: -(row["below"] + row["above"]))}
    for name, mask in (("below", lower), ("above", upper)):
        _, frames, nodes = np.nonzero(mask)
        reference = truth[mask]
        details[name] = {
            "points": int(mask.sum()), "cases": int(mask.any(axis=(1, 2)).sum()),
            "points_by_material": {label: int(mask[controls[:, 3] == code].sum())
                                   for code, label in enumerate(("recA", "recAs"))},
            "at_exact_reference_zero": int((reference == 0).sum()),
            "at_reference_within_tolerance_of_one": int((reference >= 1.0 - TOL).sum()),
            "inside_reference_transition_band": int(((reference >= LEVELS[0]) & (reference <= LEVELS[-1])).sum()),
            "time_ms_min_median_max": np.quantile(TIME_MS[frames], [0, .5, 1]).tolist() if len(frames) else [],
            "r_over_R_min_median_max": np.quantile(RADIUS[nodes], [0, .5, 1]).tolist() if len(nodes) else [],
        }
    return details, rows


def basis_diagnostic(pod, raw, truth, band):
    """Give the saved phase basis exact coarse labels to diagnose truncation and interpolation."""
    params = pod.parameters
    coarse = truth[:, TI][:, :, RI]
    scale, mean, basis = (params[name][3] for name in ("field_scale", "mean_change", "basis"))
    flat = coarse.reshape(len(truth), -1) / scale
    modal = (((flat - mean) @ basis.T) @ basis + mean).reshape(coarse.shape) * scale
    projected = resample_phase(modal, TIME[TI], RADIUS[RI], TIME, RADIUS)
    interpolated = resample_phase(coarse, TIME[TI], RADIUS[RI], TIME, RADIUS)
    return {
        "raw_front_rmse": float(np.sqrt(np.mean((raw[band] - truth[band]) ** 2))),
        "oracle_modal_front_rmse": float(np.sqrt(np.mean((projected[band] - truth[band]) ** 2))),
        "oracle_interpolation_front_rmse": float(np.sqrt(np.mean((interpolated[band] - truth[band]) ** 2))),
        "oracle_violations": int(((projected < -TOL) | (projected > 1 + TOL)).sum()),
        "same_direction_as_raw_below": int(((raw < -TOL) & (projected < -TOL)).sum()),
        "same_direction_as_raw_above": int(((raw > 1 + TOL) & (projected > 1 + TOL)).sum()),
        "interpolation_only_violations": int(((interpolated < -TOL) | (interpolated > 1 + TOL)).sum()),
    }


def front_scores(phases, truth, band):
    band_count = band.sum(axis=(1, 2))
    assert np.all(band_count)
    result = {}
    for name, phase in phases.items():
        error_sq = (phase - truth) ** 2
        error_case = np.sqrt(np.sum(error_sq * band, axis=(1, 2)) / band_count)
        result[name] = {
            "rmse": float(np.sqrt(np.mean(error_sq[band]))),
            "case_rmse": error_case.tolist(), "case_rmse_median": float(np.median(error_case)), "arrivals": {},
        }
    for level in LEVELS:
        expected = arrival_times(truth, level)
        times = {name: arrival_times(phase, level) for name, phase in phases.items()}
        reachable = np.isfinite(expected)
        common = reachable & np.logical_and.reduce([np.isfinite(value) for value in times.values()])
        for name, value in times.items():
            got = np.isfinite(value)
            error = np.abs(value[common] - expected[common])
            result[name]["arrivals"][str(level)] = {
                "reference_reached": int(reachable.sum()), "common_reached": int(common.sum()),
                "missed": int((reachable & ~got).sum()), "false": int((~reachable & got).sum()),
                "common_mae_ms": float(error.mean()), "common_p95_ms": float(np.quantile(error, .95)),
            }
    clipped_case_rmse = np.asarray(result["clipped"]["case_rmse"])
    for method in result.values():
        change = np.array(method["case_rmse"]) - clipped_case_rmse
        method["cases_improved_vs_clipped"] = int((change < -1e-12).sum())
        method["cases_worsened_vs_clipped"] = int((change > 1e-12).sum())
    return result


def evaluate_partition(base, data, pod, fitted, split):
    use = data["split"] == split
    controls, truth, ids = (data[key][use] for key in ("controls", "fields", "case_ids"))
    raw = pod.predict(controls, TIME, RADIUS)
    if split == "test":
        with np.load(base / "test_predictions.npz", allow_pickle=False) as stored:
            np.testing.assert_array_equal(raw, stored["pod_ridge"])
    raw_phase, reference = raw[:, 3], truth[:, 3]
    coarse = pod.predict(controls, TIME[TI], RADIUS[RI])[:, 3]
    conditioned = fitted.predict(controls, TIME, RADIUS)
    np.testing.assert_array_equal(conditioned[:, :3], raw[:, :3])
    phases = {
        "raw": raw_phase, "clipped": np.clip(raw_phase, 0, 1),
        **constrained_phase_fields(coarse, TIME[TI], RADIUS[RI], TIME, RADIUS),
        "material_pressure_raw": resample_phase(fitted.phase_anchors(controls), TIME[TI], RADIUS[RI], TIME, RADIUS),
        "material_pressure_monotone": conditioned[:, 3],
    }
    band = (reference >= LEVELS[0]) & (reference <= LEVELS[-1])
    fronts = front_scores(phases, reference, band)
    violations, rows = phase_violations(raw_phase, reference, ids, controls, split)
    scores = {}
    for name, phase in phases.items():
        candidate = raw.copy()
        candidate[:, 3] = phase
        np.testing.assert_array_equal(candidate[:, :3], raw[:, :3])
        if name in METHODS_FOR_SELECTION:
            assert (phase >= 0).all() and (phase <= 1).all()
        falls = int((np.diff(phase, axis=1) < -1e-12).sum())
        if name in ("monotone_linear", "monotone_cubic", "material_pressure_monotone"):
            assert falls == 0
        field = score(candidate, truth, pod.parameters["field_scale"], ids, controls)
        scores[name] = {
            "field_rmse": field["rmse"], "phase_outside_bounds_fraction": field["phase_outside_bounds_fraction_tol_1e-6"],
            "phase_temporal_decreases": falls, "front": fronts[name],
        }
    original_scores = read_json(base / "summary.json")["scores"]["pod_ridge"][split]
    for field, error in scores["raw"]["field_rmse"].items():
        np.testing.assert_allclose(error, original_scores["rmse"][field]["mean"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(violations["points"] / violations["future_points"],
                               original_scores["phase_outside_bounds_fraction_tol_1e-6"]["mean"], rtol=0, atol=1e-12)
    diagnostics = {"case_ids": ids.tolist(), "reference_transition_points": int(band.sum()),
                   "reference_phase_decreases": int((np.diff(reference, axis=1) < -1e-12).sum()),
                   "original_violations": violations, "rank_diagnostic": basis_diagnostic(pod, raw_phase, reference, band),
                   "metrics": scores}
    display = (controls, reference, phases, band)
    return diagnostics, rows, display


def evaluate(base, data, checkpoint=None):
    pod = PODRidgeLoading.load(base / "pod_ridge.npz")
    controls, change, mean, scale, field_scale = training_arrays(data)
    for name, values in (("control_mean", mean), ("control_scale", scale), ("field_scale", field_scale),
                         ("coordinates_time", TIME[TI]), ("coordinates_radius", RADIUS[RI])):
        np.testing.assert_array_equal(values, pod.parameters[name])
    start = time.perf_counter()
    fitted = MaterialPressurePhaseRidge.fit(pod, controls, change[:, 3])
    fit_time = time.perf_counter() - start
    if checkpoint is not None:
        replayed = MaterialPressurePhaseRidge.load(checkpoint, pod)
        for name in ("pressure_centers", "coefficients", "intercept"):
            np.testing.assert_array_equal(getattr(fitted, name), getattr(replayed, name))
        fitted = replayed

    validation, val_rows, _ = evaluate_partition(base, data, pod, fitted, "validation")
    selected = min(METHODS_FOR_SELECTION, key=lambda name: validation["metrics"][name]["front"]["rmse"])
    test, test_rows, plot_inputs = evaluate_partition(base, data, pod, fitted, "test")
    report = {
        "status": "completed", "validation_selected": selected, "validation": validation, "test": test,
        "test_baseline_predictions_replayed_exactly": True, "nonphase_fields_preserved_exactly": True,
        "training_pressure_centers_MPa": (fitted.pressure_centers / 1e6).tolist(),
        "train_only_ridge_fits": 1, "fresh_solver_calls": 0,
    }
    return report, val_rows + test_rows, fitted, fit_time, plot_inputs


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def run(base, out):
    from examples.loading_phase_report import draw

    start = time.perf_counter()
    data, protocol = load_snapshot(base)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", protocol)
    report, rows, fitted, fit_time, plot_inputs = evaluate(base, data)
    fitted.save(out / "phase_ridge.npz")
    (out / "phase_violations.csv").write_text(csv_text(rows), encoding="utf-8", newline="\n")
    draw(out / "phase_front_diagnostics.png", *plot_inputs, report["validation_selected"], TOL)
    report.update(fit_seconds=fit_time, runtime_seconds=time.perf_counter() - start)
    write(out / "summary.json", report)
    write(out / "manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return report


def verify(base, out):
    for name, digest in read_json(out / "manifest.json")["sha256"].items():
        assert sha256(out / name) == digest, name
    data, protocol = load_snapshot(base)
    assert json.loads(json.dumps(protocol)) == read_json(out / "protocol.json")
    report, rows, _, _, _ = evaluate(base, data, checkpoint=out / "phase_ridge.npz")
    saved = read_json(out / "summary.json")
    compare(report, {k: v for k, v in saved.items() if k not in ("fit_seconds", "runtime_seconds")})
    assert csv_text(rows) == (out / "phase_violations.csv").read_text(encoding="utf-8")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BENCHMARK)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        report = run(args.base, args.output) if args.output else verify(args.base, args.verify)
    print(json.dumps({
        "status": "verified" if args.verify else report["status"],
        "validation_selected": report["validation_selected"],
        "original_test_violation_count": report["test"]["original_violations"]["points"],
        "test_front_rmse": {name: values["front"]["rmse"] for name, values in report["test"]["metrics"].items()},
        "fit_seconds": report["fit_seconds"], "study_runtime_seconds": report["runtime_seconds"],
        "fresh_solver_calls": report["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
