"""Compare four stage-one force/temperature field surrogates on one archived cohort."""

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from SciML.models.v13.loading_field_operator import FIELDS, INITIAL, NEURAL_METHODS, NeuralLoadingOperator, PODRidgeLoading
from tools.audit_project_corpora import CORPORA, read_json, sha256
from tools.compare_solver_archives import first_cycles, write


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = CORPORA["historical_development"]
PAIR_RESULT = ROOT / "evidence/local_studies/solver_version_pairs_20261121/summary.json"
SEEDS = (20261121, 20261122, 20261123)
EPOCHS = 24
BATCH_SIZE = 14
POD_RANK = 8
SPLIT_KEY = "sma-loading-stage-one/20261121"
TIME = np.linspace(0, 1, 50)
RADIUS = np.linspace(0, 1, 51)
TI = np.r_[np.arange(0, 50, 2), 49]
RI = np.arange(0, 51, 2)
METHODS = ("persistence", "pod_ridge", *NEURAL_METHODS)
SOURCE_FILES = (
    "examples/loading_field_benchmark.py", "SciML/models/v13/loading_field_operator.py",
    "SciML/models/v12/learning_coordinate_operator.py", "tools/compare_solver_archives.py",
    "tools/audit_project_corpora.py",
)


def stratified_split(conditions, controls):
    groups = defaultdict(list)
    for i, values in enumerate(controls):
        groups[(float(values[0]), float(values[3]))].append(i)
    if len(groups) != 10 or min(map(len, groups.values())) < 7:
        raise ValueError("Expected the registered two-material, five-radius development cohort")
    split = np.full(len(conditions), "train", dtype="<U10")
    for group in groups.values():
        order = sorted(group, key=lambda i: (hashlib.sha256(f"{SPLIT_KEY}/{conditions[i]}".encode()).hexdigest(), conditions[i]))
        split[order[:3]] = "test"
        split[order[3:6]] = "validation"
    return split


def snapshot(dataset):
    manifest, cases = first_cycles(dataset)
    pair = read_json(PAIR_RESULT)
    if (
        len(cases) != 200 or pair["pair_count"] != 54 or
        pair["dataset_source_content_sha256"][0] != manifest["source_content_sha256"] or
        sha256(dataset / "artifact_manifest.json") != pair["dataset_manifest_sha256"][0] or
        any(pair["stages"]["stage1_loading"][key]["exact_pairs"] != 54
            for key in ("u_mm", "w_mm", "sma_K", "phase_fraction"))
    ):
        raise ValueError("Loading archive is not the one selected after the paired audit")
    ids = sorted(cases)
    controls, fields, case_ids, fingerprints = [], [], [], set()
    with h5py.File(dataset / "fields.h5", "r") as source:
        for condition in ids:
            case, cycle = cases[condition]
            material = {"recA": 0.0, "recAs": 1.0}[case["material_id"]]
            r, h, peak = [float(case[key]) for key in ("radius_m", "thickness_m", "pressure_Pa")]
            group = source[cycle["hdf5_cycle_group"] + "/stage1_loading"]
            np.testing.assert_allclose(group["time_local_s"][:], .1 * TIME, rtol=0, atol=1e-12)
            np.testing.assert_allclose(group["radial_coordinate_m"][:], r * RADIUS, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(group["pressure_Pa_t"][[0, 49]], [0.0, peak])
            values = np.stack([group[key][:] * factor for key, factor in (
                ("u_m_t", 1000.0), ("w_m_t", 1000.0), ("T_K_t", 1.0), ("xi_t", 1.0),
            )])
            if values.shape != (4, 50, 51) or not np.isfinite(values).all():
                raise ValueError(f"Invalid loading field for {condition}")
            np.testing.assert_array_equal(values[:, 0], np.broadcast_to(np.asarray(INITIAL)[:, None], (4, 51)))
            controls.append([r, h, peak, material])
            fields.append(values)
            case_ids.append(case["case_id"])
            fingerprints.add(case["physics_implementation_fingerprint_sha256"])
    if len(fingerprints) != 1:
        raise ValueError("Loading benchmark expects a single recorded solver version")
    controls = np.asarray(controls, dtype=np.float64)
    split = stratified_split(ids, controls)
    return {
        "controls": controls, "fields": np.stack(fields), "split": split,
        "condition_ids": np.asarray(ids), "case_ids": np.asarray(case_ids),
    }, {
        "archive": ARCHIVE, "archive_manifest_sha256": sha256(dataset / "artifact_manifest.json"),
        "archive_source_sha256": manifest["source_content_sha256"], "physics_fingerprint": next(iter(fingerprints)),
        "paired_comparison_sha256": sha256(PAIR_RESULT),
    }


def training_arrays(data):
    train = data["split"] == "train"
    controls = data["controls"][train]
    coarse = data["fields"][train][:, :, TI][:, :, :, RI]
    change = coarse - np.asarray(INITIAL)[None, :, None, None]
    mean = controls.mean(axis=0)
    scale = controls.std(axis=0)
    field_scale = np.sqrt(np.mean(change[:, :, 1:] ** 2, axis=(0, 2, 3)))
    if not np.all(np.r_[scale, field_scale] > 0):
        raise ValueError("Train-only scales must be positive")
    return controls, change / field_scale[None, :, None, None], mean, scale, field_scale


def train_neural(kind, seed, training):
    controls, change, mean, scale, field_scale = training
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(controls, dtype=torch.float32)
    y = torch.as_tensor(change[:, :, 1:], dtype=torch.float32)
    times = torch.as_tensor(TIME[TI[1:]], dtype=torch.float32)
    positions = torch.as_tensor(RADIUS[RI], dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = NeuralLoadingOperator(kind).float()
        model.set_scaling(mean, scale, field_scale)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
        loss_history = []
        start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            losses = []
            for inds in np.array_split(rng.permutation(len(x)), len(x) // BATCH_SIZE):
                optimizer.zero_grad(set_to_none=True)
                loss = (model.normalized_change(x[inds], times, positions) - y[inds]).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss for {kind}, seed {seed}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch in (1, 4, 8, 12, 16, 20, EPOCHS):
                loss_history.append({"epoch": epoch, "normalized_training_mse": float(np.mean(losses))})
    return model.eval(), {
        "seed": seed, "training_seconds": time.perf_counter() - start, "loss": loss_history,
        "parameters": sum(value.numel() for value in model.parameters()),
        "updates": EPOCHS * (len(x) // BATCH_SIZE),
    }


def predict(model, controls):
    if model is None:
        return np.broadcast_to(np.asarray(INITIAL)[None, :, None, None], (len(controls), 4, 50, 51)).copy()
    if isinstance(model, PODRidgeLoading):
        return model.predict(controls, TIME, RADIUS)
    times, positions = [torch.as_tensor(grid, dtype=torch.float32) for grid in (TIME, RADIUS)]
    with torch.no_grad():
        return torch.cat([
            model(torch.as_tensor(controls[i:i + BATCH_SIZE], dtype=torch.float32), times, positions)
            for i in range(0, len(controls), BATCH_SIZE)
        ]).numpy()


def score(prediction, truth, field_scale, case_ids, controls):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite or misaligned loading prediction")
    np.testing.assert_array_equal(prediction[:, :, 0], truth[:, :, 0])
    error = prediction[:, :, 1:] - truth[:, :, 1:]
    rmse = np.sqrt(np.mean(error ** 2, axis=(0, 2, 3)))
    normalized_sq = (error / field_scale[None, :, None, None]) ** 2
    case_norm = np.sqrt(normalized_sq.mean(axis=(1, 2, 3)))
    abs_error = np.abs(error)
    phase = prediction[:, 3, 1:]
    case_table = []
    for i, name in enumerate(case_ids):
        case_table.append({
            "case_id": str(name), "normalized_rmse": float(case_norm[i]),
            "rmse": dict(zip(FIELDS, np.sqrt(np.mean(error[i] ** 2, axis=(1, 2))).tolist())),
        })
    groups = {}
    for radius in sorted(set(controls[:, 0])):
        for material in (0, 1):
            indices = (controls[:, 0] == radius) & (controls[:, 3] == material)
            label = f"{'recAs' if material else 'recA'}_{round(1000 * radius)}mm"
            if indices.any():
                groups[label] = float(np.sqrt(normalized_sq[indices].mean()))
    return {
        "cases": len(truth), "rmse": dict(zip(FIELDS, rmse.tolist())),
        "mae": dict(zip(FIELDS, abs_error.mean(axis=(0, 2, 3)).tolist())),
        "max_abs": dict(zip(FIELDS, abs_error.max(axis=(0, 2, 3)).tolist())),
        "normalized_pooled_rmse": float(np.sqrt(normalized_sq.mean())),
        "normalized_rmse_by_case_p95": float(np.quantile(case_norm, .95)),
        "worst_normalized_case": case_table[int(np.argmax(case_norm))],
        "normalized_rmse_by_material_radius": groups,
        "mean_abs_peak_w_error_mm": float(np.abs(prediction[:, 1].max(axis=(1, 2)) - truth[:, 1].max(axis=(1, 2))).mean()),
        "phase_outside_bounds_fraction_tol_1e-6": float(np.mean((phase < -1e-6) | (phase > 1 + 1e-6))),
        "max_abs_edge_w_error_mm": float(abs_error[:, 1, :, -1].max()),
        "initial_field_exact": True,
        "per_case": case_table,
    }


def average(values):
    return {"mean": float(np.mean(values)), "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def aggregate(records):
    output = {}
    for method in METHODS:
        selected = [record for record in records if record["method"] == method]
        output[method] = {}
        for split in ("validation", "test"):
            output[method][split] = {
                "rmse": {field: average([row[split]["rmse"][field] for row in selected]) for field in FIELDS},
                "normalized_pooled_rmse": average([row[split]["normalized_pooled_rmse"] for row in selected]),
                "phase_outside_bounds_fraction_tol_1e-6": average(
                    [row[split]["phase_outside_bounds_fraction_tol_1e-6"] for row in selected]),
            }
    return output


def run(dataset, out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError(out)
    data, source = snapshot(dataset)
    training = training_arrays(data)
    train_x, train_change, mean, scale, field_scale = training
    counts = {name: int(np.sum(data["split"] == name)) for name in ("train", "validation", "test")}
    if counts != {"train": 140, "validation": 30, "test": 30}:
        raise ValueError("Registered split sizes changed")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "loading_data.npz", **data)
    write(out / "protocol.json", {
        "benchmark": "First-cycle radial loading fields on the historical development archive",
        **source,
        "written_before_fitting": True,
        "model_selection": "Fixed final weights, no validation checkpoint selection or test tuning",
        "evaluation_scope": "Fixed case split within a previously studied numerical archive, across seen material-radius groups",
        "case_counts": counts,
        "condition_splits": {
            name: data["condition_ids"][data["split"] == name].tolist() for name in counts
        },
        "split_key": SPLIT_KEY, "stratification": "3 validation and 3 test per material-radius group",
        "loading_time_seconds": .1, "initial_by_field": list(INITIAL),
        "controls": ["radius_m", "thickness_m", "peak_pressure_Pa", "recAs_indicator"],
        "fields": list(FIELDS),
        "native_grid": [50, 51],
        "train_time_indices": TI.tolist(), "train_radius_indices": RI.tolist(),
        "control_mean": mean.tolist(), "control_scale": scale.tolist(), "field_scale": field_scale.tolist(),
        "objective": "Unit-balanced MSE of departures from fixed starting fields, scaled by train-only future coarse-grid RMS",
        "neural": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "seeds": list(SEEDS),
                   "optimizer": "AdamW", "lr": .002, "weight_decay": 1e-4,
                   "gradient_clip_norm": 3.0, "precision": "float32 CPU, one thread",
                   "batch_schedule": "One permutation per epoch shared across methods at each seed, every train case and future coarse point"},
        "pod_ridge": {"rank_each_channel": POD_RANK, "alpha": 1.0, "fit": "Deterministic full-coarse-grid train-only SVD and fixed Ridge"},
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCE_FILES},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
        "fresh_solver_calls": 0,
    })

    begin = time.perf_counter()
    pod = PODRidgeLoading.fit(train_x, train_change, TIME[TI], RADIUS[RI], mean, scale, field_scale, POD_RANK)
    fits = [
        ("persistence", None, {"seed": None, "updates": 0, "training_seconds": 0., "parameters": 0}, None),
        ("pod_ridge", pod, {"seed": None, "updates": 0, "training_seconds": time.perf_counter() - begin,
                           "variance_retained": pod.parameters["training_centered_variance_fraction"].tolist()}, "pod_ridge.npz"),
    ]
    pod.save(out / "pod_ridge.npz")
    for method in NEURAL_METHODS:
        for seed in SEEDS:
            model, fit = train_neural(method, seed, training)
            name = f"{method}_{seed}.pt"
            model.save(out / name)
            fits.append((method, model, fit, name))

    records, predictions = [], {}
    for method, model, fit, checkpoint in fits:
        run_id = method if fit["seed"] is None else f"{method}_{fit['seed']}"
        record = {"method": method, "run_id": run_id, "checkpoint": checkpoint, **fit}
        for split in ("validation", "test"):
            use = data["split"] == split
            start = time.perf_counter()
            values = predict(model, data["controls"][use])
            seconds = time.perf_counter() - start
            record[split] = score(values, data["fields"][use], field_scale, data["case_ids"][use], data["controls"][use])
            record[split]["prediction_seconds"] = seconds
            if split == "test":
                predictions[run_id] = values
        records.append(record)

    np.savez_compressed(out / "test_predictions.npz", **predictions)
    write(out / "runs.json", records)
    whole = {
        "status": "completed_loading_field_benchmark", "case_counts": counts,
        "model_runs": len(records), "neural_updates_total": sum(row["updates"] for row in records),
        "scores": aggregate(records), "fresh_solver_calls": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    write(out / "summary.json", whole)
    write(out / "manifest.json", {"sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return whole


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = run(args.dataset, args.output)
    print(json.dumps({
        "status": result["status"], "model_runs": result["model_runs"],
        "runtime_seconds": result["runtime_seconds"], "neural_updates_total": result["neural_updates_total"],
        "test_rmse": {method: row["test"]["rmse"] for method, row in result["scores"].items()},
        "fresh_solver_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
