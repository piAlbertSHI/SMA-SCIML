"""Select a central-radial correction for the recAs saturation arrival tail."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from examples.loading_field_benchmark import RADIUS, ROOT, TIME, training_arrays, score
from examples.loading_phase_study import BASE_FILES, BENCHMARK, LEVELS, front_scores, load_snapshot
from examples.loading_phase_weighted_study import (
    arrival_rows, csv_text, recas_arrival_summary, write_manifest,
)
from SciML.models.v13.loading_field_operator import PODRidgeLoading
from SciML.models.v13.loading_phase_reconstruction import MaterialPressurePhaseRidge
from SciML.models.v13.loading_phase_weighted import SaturationWeightedPhaseRidge
from SciML.models.v13.loading_phase_central import CentralSaturationPhaseRidge
from tools.audit_project_corpora import read_json, sha256
from tools.compare_solver_archives import write


WEIGHTED_BASE = ROOT / "evidence/local_studies/loading_phase_weighted_20261121"
CENTRAL_RADIUS_GRID = (.32, .40)
CENTRAL_STRENGTH_GRID = (120.0, 200.0, 400.0)
BLEND_GRID = ((.80, .95), (.85, .95))
RADIAL_SMOOTHING = .04
SOURCE_FILES = (
    "examples/loading_phase_central_study.py", "examples/loading_phase_central_report.py",
    "examples/loading_phase_weighted_study.py", "SciML/models/v13/loading_phase_central.py",
    "SciML/models/v13/loading_phase_weighted.py", "SciML/models/v13/loading_phase_reconstruction.py",
    "SciML/models/v13/loading_field_operator.py",
)


def load_base(data):
    train_controls, normalized_change, _, _, _ = training_arrays(data)
    pod = PODRidgeLoading.load(BENCHMARK / "pod_ridge.npz")
    baseline_phase = MaterialPressurePhaseRidge.fit(pod, train_controls, normalized_change[:, 3])
    base = SaturationWeightedPhaseRidge.load(WEIGHTED_BASE / "saturation_weighted_phase.npz", baseline_phase)
    return train_controls, normalized_change, base


def protocol_for(base_protocol):
    weighted_manifest = read_json(WEIGHTED_BASE / "manifest.json")
    for name, digest in weighted_manifest["sha256"].items():
        assert sha256(WEIGHTED_BASE / name) == digest, name
    return {
        "study": "Central-radial correction for recAs near-saturation arrival",
        "base_file_sha256": {name: sha256(BENCHMARK / name) for name in BASE_FILES},
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "source_loading_protocol_sha256": sha256(BENCHMARK / "protocol.json"),
        "weighted_base_manifest_sha256": sha256(WEIGHTED_BASE / "manifest.json"),
        "weighted_base_summary_sha256": sha256(WEIGHTED_BASE / "summary.json"),
        "case_counts": base_protocol["case_counts"],
        "split_key": base_protocol["split_key"],
        "native_grid": base_protocol["native_grid"],
        "fit_time_indices": base_protocol["train_time_indices"],
        "fit_radius_indices": base_protocol["train_radius_indices"],
        "frozen_pod_rank": base_protocol["pod_ridge"]["rank_each_channel"],
        "base_readout": "Selected saturation-weighted phase head, frozen before central correction.",
        "central_threshold": .8,
        "base_weight_strength": 100.0,
        "central_radius_grid": list(CENTRAL_RADIUS_GRID),
        "central_strength_grid": list(CENTRAL_STRENGTH_GRID),
        "blend_grid": [list(pair) for pair in BLEND_GRID],
        "radial_smoothing": RADIAL_SMOOTHING,
        "selection": "Validation transition-band RMSE, then common-arrival MAE, then recAs 0.9 missed radial nodes.",
        "arrival_threshold": .9,
        "fresh_solver_calls": 0,
    }


def evaluate_partition(data, baseline, candidate, split):
    use = data["split"] == split
    controls, truth, case_ids = (data[key][use] for key in ("controls", "fields", "case_ids"))
    base_fields = baseline.predict(controls, TIME, RADIUS)
    candidate_fields = candidate.predict(controls, TIME, RADIUS)
    np.testing.assert_array_equal(base_fields[:, :3], candidate_fields[:, :3])
    np.testing.assert_array_equal(base_fields[:, :, 0], truth[:, :, 0])
    phases = {"clipped": np.clip(base_fields[:, 3], 0, 1),
              "baseline": base_fields[:, 3], "central": candidate_fields[:, 3]}
    band = (truth[:, 3] >= LEVELS[0]) & (truth[:, 3] <= LEVELS[-1])
    front = front_scores(phases, truth[:, 3], band)
    metrics = {}
    for name, fields in (("baseline", base_fields), ("central", candidate_fields)):
        values = score(fields, truth, baseline.pod.parameters["field_scale"], case_ids, controls)
        metrics[name] = {
            "field_rmse": values["rmse"],
            "phase_outside_bounds_fraction": values["phase_outside_bounds_fraction_tol_1e-6"],
            "phase_temporal_decreases": int((np.diff(fields[:, 3], axis=1) < -1e-12).sum()),
            "front": front[name],
        }
    recas = controls[:, 3] == 1.0
    arrivals = recas_arrival_summary(case_ids[recas], controls[recas], truth[recas, 3],
                                     phases["baseline"][recas], phases["central"][recas])
    arrivals["central"] = arrivals.pop("weighted")
    rows = arrival_rows(case_ids, controls, truth[:, 3], phases["baseline"], phases["central"], split)
    for row in rows:
        row["central_arrival_ms"] = row.pop("weighted_arrival_ms")
        row["central_missed"] = row.pop("weighted_missed")
        row["central_abs_error_ms"] = row.pop("weighted_abs_error_ms")
    return {
        "case_count": int(use.sum()), "reference_transition_points": int(band.sum()),
        "metrics": metrics, "recAs_threshold_0p9": arrivals, "case_ids": case_ids.tolist(),
    }, rows


def central_csv_text(rows):
    normalized = []
    for row in rows:
        value = dict(row)
        value["weighted_arrival_ms"] = value.pop("central_arrival_ms")
        value["weighted_missed"] = value.pop("central_missed")
        value["weighted_abs_error_ms"] = value.pop("central_abs_error_ms")
        normalized.append(value)
    return csv_text(normalized).replace("weighted_arrival_ms", "central_arrival_ms") \
        .replace("weighted_missed", "central_missed") \
        .replace("weighted_abs_error_ms", "central_abs_error_ms")


def selection_key(report):
    arrival = report["recAs_threshold_0p9"]["central"]
    return [report["metrics"]["central"]["front"]["rmse"], arrival["common_mae_ms"],
            arrival["missed_nodes"]]


def run(base, out):
    from examples.loading_phase_central_report import draw_arrival

    started = time.perf_counter()
    data, _ = load_snapshot(base)
    base_protocol = read_json(base / "protocol.json")
    protocol = protocol_for(base_protocol)
    out.mkdir(parents=False, exist_ok=False)
    write(out / "protocol.json", protocol)
    train_controls, normalized_change, baseline = load_base(data)
    candidates, candidate_models = {}, {}
    fit_seconds = 0.0
    for radius in CENTRAL_RADIUS_GRID:
        for strength in CENTRAL_STRENGTH_GRID:
            for blend_start, blend_end in BLEND_GRID:
                before = time.perf_counter()
                model = CentralSaturationPhaseRidge.fit(
                    baseline, train_controls, normalized_change[:, 3],
                    saturation_threshold=.8, base_weight_strength=100.0,
                    central_weight_strength=strength, central_radius=radius,
                    blend_start=blend_start, blend_end=blend_end,
                    radial_smoothing=RADIAL_SMOOTHING,
                )
                fit_seconds += time.perf_counter() - before
                label = f"radius_{radius:.2f}_strength_{int(strength)}_blend_{blend_start:.2f}_{blend_end:.2f}"
                report, _ = evaluate_partition(data, baseline, model, "validation")
                candidates[label] = {
                    "central_radius": radius, "central_weight_strength": strength,
                    "blend_start": blend_start, "blend_end": blend_end,
                    "selection_key": selection_key(report), "metrics": report["metrics"],
                    "recAs_threshold_0p9": report["recAs_threshold_0p9"],
                }
                candidate_models[label] = model
    selected_label = min(candidates, key=lambda label: tuple(candidates[label]["selection_key"]))
    selected = candidate_models[selected_label]
    test_report, test_rows = evaluate_partition(data, baseline, selected, "test")
    validation_report, validation_rows = evaluate_partition(data, baseline, selected, "validation")
    test_mask = data["split"] == "test"
    test_controls, test_ids, test_truth = data["controls"][test_mask], data["case_ids"][test_mask], data["fields"][test_mask, 3]
    draw_arrival(out / "phase_central_arrival.png", test_ids, test_controls, test_truth,
                 baseline.predict(test_controls, TIME, RADIUS)[:, 3],
                 selected.predict(test_controls, TIME, RADIUS)[:, 3])
    selected.save(out / "central_weighted_phase.npz")
    (out / "arrival_diagnostics.csv").write_text(central_csv_text(test_rows + validation_rows), encoding="utf-8", newline="\n")
    summary = {
        "status": "completed", "selection": selected_label,
        "selection_key": selection_key(validation_report), "candidate_validation": candidates,
        "validation": validation_report, "test": test_report,
        "selected_checkpoint": "central_weighted_phase.npz", "fit_count": len(candidates),
        "fit_seconds": fit_seconds, "runtime_seconds": time.perf_counter() - started,
        "fresh_solver_calls": 0, "selected_channels_preserved_exactly": True,
    }
    write(out / "summary.json", summary)
    write_manifest(out)
    return summary


def verify(base, out):
    for name, digest in read_json(out / "manifest.json")["sha256"].items():
        assert sha256(out / name) == digest, name
    data, _ = load_snapshot(base)
    assert protocol_for(read_json(base / "protocol.json")) == read_json(out / "protocol.json")
    train_controls, normalized_change, baseline = load_base(data)
    saved = read_json(out / "summary.json")
    candidates = {}
    for radius in CENTRAL_RADIUS_GRID:
        for strength in CENTRAL_STRENGTH_GRID:
            for blend_start, blend_end in BLEND_GRID:
                label = f"radius_{radius:.2f}_strength_{int(strength)}_blend_{blend_start:.2f}_{blend_end:.2f}"
                model = CentralSaturationPhaseRidge.fit(
                    baseline, train_controls, normalized_change[:, 3], .8, 100.0, strength, radius,
                    blend_start, blend_end, RADIAL_SMOOTHING,
                )
                report, _ = evaluate_partition(data, baseline, model, "validation")
                candidates[label] = {
                    "central_radius": radius, "central_weight_strength": strength,
                    "blend_start": blend_start, "blend_end": blend_end,
                    "selection_key": selection_key(report), "metrics": report["metrics"],
                    "recAs_threshold_0p9": report["recAs_threshold_0p9"],
                }
    assert min(candidates, key=lambda label: tuple(candidates[label]["selection_key"])) == saved["selection"]
    selected = CentralSaturationPhaseRidge.load(out / saved["selected_checkpoint"], baseline)
    test_report, test_rows = evaluate_partition(data, baseline, selected, "test")
    validation_report, validation_rows = evaluate_partition(data, baseline, selected, "validation")
    from examples.verify_loading_field_benchmark import compare
    compare(candidates, saved["candidate_validation"])
    compare(test_report, saved["test"])
    compare(validation_report, saved["validation"])
    assert central_csv_text(test_rows + validation_rows) == (out / "arrival_diagnostics.csv").read_text(encoding="utf-8")
    verification = {
        "status": "verified", "selection_replayed": saved["selection"],
        "candidate_grid_replayed": len(candidates), "test_arrival_rows_replayed": len(test_rows),
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
        "test_0p9_missed_central": result.get("test", {}).get("recAs_threshold_0p9", {}).get("central", {}).get("missed_nodes"),
        "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
