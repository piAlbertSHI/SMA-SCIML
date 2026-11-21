"""Matched five-target WP2 strong baselines and target-scaling ablation.

Previously inspected fixed split: exploratory comparison, not fresh confirmation.
No test-based tuning, no solver invocation, no overwrite of historical evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from .learning_ensemble import LearningEnsemble, design_matrix
from .run_learning_existing import FEATURES, TARGETS, sha, validate_dataset, write_json

SEEDS = (20260910, 20260911, 20260912)


def fit_tracked(x, y, ids, cal_x, cal_y, *, seed, epochs=300, members=5, normalize_targets=True):
    """Calibration curves are diagnostic only; no stopping or optimizer feedback."""
    x, y, cal_x, cal_y = (np.asarray(v, float) for v in (x, y, cal_x, cal_y))
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3
        or len(ids) != len(x) or len(set(ids)) != len(ids)
        or cal_x.ndim != 2 or cal_y.ndim != 2 or len(cal_x) != len(cal_y)
        or x.shape[1] != cal_x.shape[1] or y.shape[1] != cal_y.shape[1]
        or not len(cal_x) or not all(np.isfinite(v).all() for v in (x, y, cal_x, cal_y))):
        raise ValueError("invalid finite/shape/unique training or diagnostic data")
    model = LearningEnsemble(members=members, epochs=epochs, seed=seed)
    model.x_scaler = StandardScaler().fit(x)
    model.y_scaler = StandardScaler(with_mean=normalize_targets, with_std=normalize_targets).fit(y)
    reference = StandardScaler().fit(y)
    # Base checkpoint format requires explicit identity parameters in the ablation.
    if not normalize_targets:
        model.y_scaler.mean_ = np.zeros(y.shape[1])
        model.y_scaler.scale_ = np.ones(y.shape[1])
        model.y_scaler.var_ = np.ones(y.shape[1])
    tx = torch.from_numpy(model.x_scaler.transform(x))
    ty = torch.from_numpy(model.y_scaler.transform(y))
    cx = torch.from_numpy(model.x_scaler.transform(cal_x))
    history = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            for member in range(members):
                torch.manual_seed(seed+1009*member)
                network = nn.Sequential(nn.Linear(x.shape[1], 32), nn.SiLU(), nn.Linear(32, 32),
                                        nn.SiLU(), nn.Linear(32, y.shape[1])).double()
                optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=1e-4)
                rows = []
                for epoch in range(epochs+1):
                    if epoch % 50 == 0 or epoch == epochs:
                        with torch.no_grad():
                            train_prediction = model.y_scaler.inverse_transform(network(tx).numpy())
                            cal_prediction = model.y_scaler.inverse_transform(network(cx).numpy())
                            objective = torch.mean((network(tx)-ty)**2).item()
                        rows.append({"updates_completed": epoch, "objective_mse": objective,
                                     "train_standardized_mse": float(np.mean(((train_prediction-y)/reference.scale_)**2)),
                                     "calibration_standardized_mse": float(np.mean(((cal_prediction-cal_y)/reference.scale_)**2)),
                                     "train_target_rmse": np.sqrt(np.mean((train_prediction-y)**2, axis=0)).tolist(),
                                     "calibration_target_rmse": np.sqrt(np.mean((cal_prediction-cal_y)**2, axis=0)).tolist()})
                    if epoch == epochs:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean((network(tx)-ty)**2)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite scalar benchmark loss; diagnose before rerun")
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                network.eval()
                model.networks.append(network)
                history.append({"member": member, "seed": seed+1009*member,
                                "normalize_targets": normalize_targets, "curve": rows})
    finally:
        torch.set_num_threads(threads)
    model.training_ids = tuple(map(str, ids))
    model.history = history
    return model


def classical_models(seed):
    return {"ridge": Ridge(alpha=1.),
            "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                max_features=1., bootstrap=True, n_jobs=1, random_state=seed),
            "hist_gradient_boosting": MultiOutputRegressor(HistGradientBoostingRegressor(
                learning_rate=.05, max_iter=200, max_leaf_nodes=7, min_samples_leaf=3,
                l2_regularization=.1, early_stopping=False, random_state=seed), n_jobs=1)}


def score_predictions(prediction, target, baseline):
    prediction, target, baseline = (np.asarray(a, float) for a in (prediction, target, baseline))
    if (prediction.ndim != 2 or prediction.shape != target.shape or baseline.shape != target.shape
        or not len(target) or not all(np.isfinite(a).all() for a in (prediction, target, baseline))):
        raise ValueError("invalid score arrays")
    mse = np.mean((prediction-target)**2, axis=0)
    base_mse = np.mean((baseline-target)**2, axis=0)
    variance = np.mean((target-target.mean(axis=0))**2, axis=0)
    return {"rmse": np.sqrt(mse).tolist(), "mae": np.mean(np.abs(prediction-target), axis=0).tolist(),
            "max_abs": np.max(np.abs(prediction-target), axis=0).tolist(),
            "r2": [float(1-e/v) if v > 0 else None for e, v in zip(mse, variance)],
            "rmse_skill_vs_train_mean": [float(1-np.sqrt(e/b)) if b > 0 else None for e, b in zip(mse, base_mse)]}


def run(dataset, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve existing scalar experiment")
    table, quality, _ = validate_dataset(dataset)
    original = json.loads(original_protocol.read_text())
    if original["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("dataset/split provenance mismatch")
    masks = {name: table.condition_id.isin(ids).to_numpy() for name, ids in original["condition_splits"].items()}
    if not np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1):
        raise ValueError("splits must form a disjoint partition")
    train, cal = masks["train"], masks["calibration"]
    x, y = design_matrix(table), table[list(TARGETS)].to_numpy(float)
    xs, ys = StandardScaler().fit(x[train]), StandardScaler().fit(y[train])
    sx, sy = xs.transform(x), ys.transform(y)
    baseline = np.broadcast_to(y[train].mean(axis=0), y.shape)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"learning_only": True, "fresh_solver_calls": 0,
                "dataset_source_content_sha256": quality["source_content_sha256"],
                "original_protocol_sha256": sha(original_protocol), "condition_splits": original["condition_splits"],
                "seeds": SEEDS, "feature_names": list(FEATURES)+["material_recA", "material_recAs"],
                "target_names": TARGETS, "members": 5, "epochs": 300,
                "neural": "7->32 SiLU->32 SiLU->5; float64; full batch AdamW lr.002 decay1e-4; grad clip10",
                "classical": {name: str(est) for name, est in classical_models(SEEDS[0]).items()},
                "single_vs_ensemble": "MLP is exactly first member of five-target ensemble, not a different output-head fit",
                "ablation": "Input scaling remains train-only; raw-target first member uses same seed/architecture/updates; target normalization removed",
                "calibration_curve": "Logged every50 completed updates, no early stopping/model selection; no test labels passed to fit",
                "test_history": "Previously inspected split, exploratory fixed-configuration comparison; no new confirmation claim",
                "implementation_sha256": sha(Path(__file__)),
                "ensemble_implementation_sha256": sha(Path(__file__).with_name("learning_ensemble.py"))}
    write_json(out/"protocol.json", protocol)
    started = time.monotonic()
    records = []
    for seed in SEEDS:
        dest = out/f"s{seed}"
        dest.mkdir()
        clocks = {}
        predictions = {"train_mean": baseline.copy()}
        estimators = classical_models(seed)
        with threadpool_limits(limits=1):
            for name, estimator in estimators.items():
                before = time.monotonic()
                estimator.fit(sx[train], sy[train])
                predictions[name] = ys.inverse_transform(estimator.predict(sx))
                clocks[name] = time.monotonic()-before
        with (dest/"classical.pkl").open("xb") as stream:
            pickle.dump({"learning_only": True, "training_ids": table.case_id[train].tolist(),
                         "x_scaler": xs, "y_scaler": ys, "estimators": estimators}, stream, protocol=5)
        before = time.monotonic()
        model = fit_tracked(x[train], y[train], table.case_id[train].tolist(), x[cal], y[cal], seed=seed)
        members = model.predict_members(x)
        predictions["mlp"] = members[0]
        predictions["ensemble5"] = members.mean(axis=0)
        clocks["ensemble5_fit_and_predict"] = time.monotonic()-before
        model.save(dest/"ensemble5.pt")
        np.testing.assert_array_equal(members, LearningEnsemble.load(dest/"ensemble5.pt").predict_members(x))
        before = time.monotonic()
        raw = fit_tracked(x[train], y[train], table.case_id[train].tolist(), x[cal], y[cal],
                          seed=seed, members=2, normalize_targets=False)
        predictions["mlp_raw_targets"] = raw.predict_members(x)[0]
        clocks["raw_target_two_member_fit_and_predict"] = time.monotonic()-before
        raw.save(dest/"raw_targets.pt")
        # Second raw member only satisfies common ensemble serialization; it is
        # retained but NOT averaged or used for the single-network ablation.
        np.testing.assert_array_equal(raw.predict_members(x), LearningEnsemble.load(dest/"raw_targets.pt").predict_members(x))
        report = {"seed": seed, "training_count": int(train.sum()), "calibration_count": int(cal.sum()),
                  "test_count": int(masks["test"].sum()), "model_scores": {}, "elapsed_seconds": clocks,
                  "normalized_history": model.history, "raw_target_history": raw.history,
                  "raw_target_second_member_unused": True, "checkpoint_roundtrip_exact": True}
        for name, prediction in predictions.items():
            report["model_scores"][name] = {part: score_predictions(prediction[mask], y[mask], baseline[mask]) for part, mask in masks.items()}
        np.savez_compressed(dest/"predictions.npz", case_ids=table.case_id.to_numpy(str), targets=y,
                            member_predictions=members, **predictions)
        write_json(dest/"report.json", report)
        records.append(report)
    curves = {}
    for name in records[0]["model_scores"]:
        scores = np.array([r["model_scores"][name]["test"]["rmse"] for r in records])
        curves[name] = {"seed_mean_test_rmse": scores.mean(axis=0).tolist(),
                        "seed_std_test_rmse": scores.std(axis=0, ddof=1).tolist()}
    summary = {"status": "completed_matched_scalar_benchmarks", "learning_only": True,
               "fresh_solver_calls": 0, "seed_count": len(SEEDS), "target_names": TARGETS,
               "comparison": curves, "elapsed_seconds": time.monotonic()-started,
               "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_protocol, args.output_root), indent=2))
