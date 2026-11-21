"""Fixed existing63 geometry/material/protocol/combined holdout learning tests."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from SciML.evaluation.audit_learning_operators import matched_pod
from .learning_coordinate_operator import CoordinateOperator, corpus
from .learning_ensemble import LearningEnsemble, design_matrix, calibrated_intervals
from .learning_scalar_benchmarks import classical_models, score_predictions
from .run_learning_existing import FEATURES, TARGETS, sha, validate_dataset, write_json

SCHEMES = ("geometry", "material", "protocol", "combined")
SEEDS = (20260910, 20260911, 20260912)


def structured_split(metadata, scheme):
    """Only condition/geometry/material/duration metadata; no label-based choices."""
    if scheme not in SCHEMES:
        raise ValueError("unknown structured scheme")
    cols = ["condition_id", "radius_m", "thickness_m", "material_id", "exchange_duration_s"]
    table = metadata[cols].copy()
    if table.condition_id.duplicated().any() or table[cols].isna().any().any():
        raise ValueError("unique complete condition metadata required")
    geometry = table.radius_m == table.radius_m.max()
    material = table.material_id == "recAs"
    protocol = table.exchange_duration_s == table.exchange_duration_s.max()
    held = {"geometry": geometry, "material": material, "protocol": protocol, "combined": geometry | material}[scheme]
    available = table.loc[~held, "condition_id"].astype(str).tolist()
    available.sort(key=lambda case: hashlib.sha256(("structured-20260910:"+case).encode()).hexdigest())
    cal_count, id_count = 13, int(round(.2*len(available)))
    if id_count < 2 or len(available)-cal_count-id_count < 3 or not held.any():
        raise ValueError("insufficient fixed-budget group split")
    groups = {"calibration": available[:cal_count], "id_test": available[cal_count:cal_count+id_count],
              "train": available[cal_count+id_count:]}
    if scheme == "combined":
        for name, mask in (("ood_geometry_only", geometry & ~material), ("ood_material_only", material & ~geometry),
                           ("ood_combined", geometry & material)):
            groups[name] = table.loc[mask, "condition_id"].astype(str).tolist()
    else:
        groups["ood_"+scheme] = table.loc[held, "condition_id"].astype(str).tolist()
    groups = {key: sorted(ids) for key, ids in groups.items()}
    flattened = sum(groups.values(), [])
    if len(flattened) != len(set(flattened)) or set(flattened) != set(table.condition_id.astype(str)):
        raise ValueError("structured split is not a disjoint exhaustive partition")
    return groups


def field_scores(prediction, truth, initial, samples, selected):
    scores = {}
    for stage in (2, 4):
        mask = selected & np.array([sample.stage_index == stage for sample in samples])
        scores[str(stage)] = {}
        for index, field in enumerate(("sma", "htf")):
            target = truth[mask, index, 1:]
            error = prediction[mask, index, 1:]-target
            change = target-initial[mask, index, None]
            if not len(target):
                scores[str(stage)][field] = {"status": "no_observations"}
                continue
            scores[str(stage)][field] = {"stage_sample_count": len(target),
                "future_rmse_K": float(np.sqrt(np.mean(error**2))), "future_mae_K": float(np.abs(error).mean()),
                "future_max_abs_K": float(np.abs(error).max()),
                "relative_l2_absolute_temperature": float(np.linalg.norm(error)/np.linalg.norm(target)),
                "relative_l2_temperature_change": float(np.linalg.norm(error)/np.linalg.norm(change)) if np.linalg.norm(change) else None,
                "initial_max_abs_K": float(np.abs(prediction[mask, index, 0]-initial[mask, index]).max())}
    return scores


def run(dataset, out, scheme):
    if out.exists():
        raise FileExistsError("preserve prior structured evaluation")
    table, quality, _ = validate_dataset(dataset)
    groups = structured_split(table, scheme)
    masks = {key: table.condition_id.isin(ids).to_numpy() for key, ids in groups.items()}
    train, cal = masks["train"], masks["calibration"]
    samples, branch, initial, truth, _ = corpus(dataset)
    fmasks = {key: np.array([s.condition_id in ids for s in samples]) for key, ids in groups.items()}
    stages = np.array([sample.stage_index for sample in samples])
    x, y = design_matrix(table), table[list(TARGETS)].to_numpy(float)
    xs, ys = StandardScaler().fit(x[train]), StandardScaler().fit(y[train])
    sx, sy = xs.transform(x), ys.transform(y)
    baseline = np.broadcast_to(y[train].mean(axis=0), y.shape)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"learning_only": True, "fresh_solver_calls": 0, "scheme": scheme,
        "condition_groups": groups, "group_sizes": {k: len(v) for k, v in groups.items()},
        "dataset_source_content_sha256": quality["source_content_sha256"], "seeds": SEEDS,
        "selection": "input-only largest radius, recAs, longest duration; combined removes union from training/calibration/ID",
        "geometry_note": "R/t=200 across all63; held size pair, not independent R/t variation",
        "calibration": "13 in-domain conditions; 90% simultaneous intervals measured out-of-domain WITHOUT coverage guarantee",
        "scalar": "same fixed Ridge/RF/HistGB as chapter04; five-target 3-member MLP ensemble,300updates; first member is singleMLP",
        "operator": "observed-start DeepONet400updates and matched26x51rank4POD/Ridge; not autonomous whole-cycle forecast",
        "test_history": "all existing63 outputs previously used in development; newly grouped evaluations remain exploratory",
        "target_names": list(TARGETS), "implementation_sha256": sha(Path(__file__)),
        "upstream_implementation_sha256": {name: sha(Path(__file__).with_name(name+".py")) for name in
            ("learning_coordinate_operator", "learning_ensemble", "learning_scalar_benchmarks")}})
    started = time.monotonic()
    pod, parameters = matched_pod(branch, initial, truth, stages, fmasks["train"])
    np.savez_compressed(out/"pod_parameters.npz", **parameters)
    np.savez_compressed(out/"pod_predictions.npz", prediction=pod, sample_ids=np.array([s.sample_id for s in samples]))
    common = {name: field_scores(pod, truth, initial, samples, mask) for name, mask in fmasks.items()}
    persistence = {name: field_scores(np.broadcast_to(initial[:, :, None], truth.shape), truth, initial, samples, mask)
                   for name, mask in fmasks.items()}
    reports = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in SEEDS:
            dest = out/f"s{seed}"
            dest.mkdir()
            predictions = {"train_mean": baseline}
            estimators = classical_models(seed)
            with threadpool_limits(limits=1):
                for name, estimator in estimators.items():
                    estimator.fit(sx[train], sy[train])
                    predictions[name] = ys.inverse_transform(estimator.predict(sx))
            with (dest/"classical.pkl").open("xb") as stream:
                pickle.dump({"estimators": estimators, "x_scaler": xs, "y_scaler": ys,
                             "training_ids": table.case_id[train].tolist()}, stream, protocol=5)
            scalar = LearningEnsemble(members=3, epochs=300, seed=seed).fit(x[train], y[train], table.case_id[train].tolist())
            members = scalar.predict_members(x)
            mean, std = members.mean(axis=0), members.std(axis=0, ddof=1)
            predictions.update({"mlp": members[0], "ensemble3": mean})
            scalar.save(dest/"scalar.pt")
            np.testing.assert_array_equal(members, LearningEnsemble.load(dest/"scalar.pt").predict_members(x))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                operator = CoordinateOperator().double().fit(branch[fmasks["train"]], initial[fmasks["train"]], truth[fmasks["train"]],
                    [sample.sample_id for sample, use in zip(samples, fmasks["train"]) if use], seed=seed)
            fields = operator.predict(branch, initial)
            operator.save(dest/"operator.pt")
            np.testing.assert_array_equal(fields, CoordinateOperator.load(dest/"operator.pt").predict(branch, initial))
            scores = {name: {group: score_predictions(values[mask], y[mask], baseline[mask]) if mask.any() else {"status": "no_observations"}
                            for group, mask in masks.items()} for name, values in predictions.items()}
            uq = {group: calibrated_intervals(y[cal], mean[cal], std[cal], y[mask], mean[mask], std[mask], ys.scale_, .9)[0]
                  for group, mask in masks.items() if group not in ("train", "calibration") and mask.any()}
            report = {"seed": seed, "scalar_scores": scores, "calibrated_interval_diagnostics": uq,
                "operator_scores": {name: field_scores(fields, truth, initial, samples, mask) for name, mask in fmasks.items()},
                "scalar_history": scalar.history, "operator_history": operator.history, "cold_checkpoints_exact": True}
            write_json(dest/"report.json", report)
            np.savez_compressed(dest/"predictions.npz", case_ids=table.case_id.to_numpy(str), sample_ids=np.array([s.sample_id for s in samples]),
                                scalar_members=members, operator=fields, **predictions)
            reports.append(report)
    finally:
        torch.set_num_threads(threads)
    result = {"status": "completed_structured_ood_learning", "scheme": scheme, "learning_only": True, "fresh_solver_calls": 0,
        "group_sizes": {name: len(ids) for name, ids in groups.items()}, "seed_count": len(SEEDS),
        "operator_pod_scores": common, "operator_persistence_scores": persistence,
        "elapsed_seconds": time.monotonic()-started,
        "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    write_json(out/"summary.json", result)
    return {key: result[key] for key in ("status", "scheme", "group_sizes", "seed_count", "elapsed_seconds", "fresh_solver_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.output_root, args.scheme), indent=2))
