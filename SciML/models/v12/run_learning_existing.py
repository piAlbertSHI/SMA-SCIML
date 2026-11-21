"""Bounded, existing-data WP2-WP4 learning experiment; never publication authority.

Run with ``python -m SciML.models.v12.run_learning_existing --help``.
No MATLAB, additional observations, hyperparameter search or confirmation runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from SciML.models.v11.cycle_proxy import TARGETS
from SciML.models.v11.state_space import LAYOUT, build_transition_corpus
from SciML.models.v11.spatiotemporal_operator import build_operator_corpus
from .neural_state_space_candidate import (
    CausalNeuralStateModel, save_frozen_checkpoint, load_frozen_checkpoint,
)
from .neural_operator_candidate import CausalNeuralPODOperator

SEED = 20260910
FEATURES = ("radius_m", "thickness_m", "pressure_Pa",
            "flush_volume_target_ratio", "exchange_duration_s")
STRICT_COLUMNS = ("run_ok", "physics_ok", "paper_grade_valid",
    "observation_contract_ok", "pump_work_available", "htf_internal_full_field_available",
    "request_config_matches_current", "request_self_contained",
    "export_provenance_available", "export_artifact_hash_matches")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def split_conditions(conditions, seed=SEED):
    """Rank unique physical conditions, not rows; no output-informed selection."""
    ids = sorted(set(map(str, conditions)), key=lambda s: hashlib.sha256(
        f"learning-existing:{seed}:{s}".encode()).hexdigest())
    if len(ids) < 30:
        raise ValueError("learning split requires at least 30 physical conditions")
    n = int(round(.2 * len(ids)))
    return {"train": sorted(ids[2*n:]), "calibration": sorted(ids[n:2*n]),
            "test": sorted(ids[:n])}


def metrics(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or not pred.size or not (
        np.isfinite(pred).all() and np.isfinite(target).all()
    ):
        raise ValueError("metric arrays must match and be nonempty/finite")
    error = pred - target
    return {"rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs": float(np.max(np.abs(error)))}


def comparison(pred, baseline, target):
    model, base = metrics(pred, target), metrics(baseline, target)
    return {"model": model, "baseline": base,
            "rmse_skill": 1-model["rmse"]/base["rmse"] if base["rmse"] else None}


def conformal_radius(residual, scale, alpha=.1):
    """Finite-sample order statistic, max across targets for simultaneous intervals."""
    residual, scale = np.asarray(residual), np.asarray(scale)
    if (residual.ndim != 2 or scale.shape != (residual.shape[1],)
        or not np.isfinite(residual).all() or not np.isfinite(scale).all()
        or np.any(scale <= 0) or not 0 < alpha < 1):
        raise ValueError("invalid conformal inputs")
    scores = np.max(np.abs(residual) / scale, axis=1)
    rank = math.ceil((len(scores)+1)*(1-alpha))
    if not 1 <= rank <= len(scores):
        raise ValueError("too few calibration conditions for finite requested interval")
    return float(np.sort(scores)[rank-1]), rank


def validate_dataset(root):
    manifest = json.loads((root/"artifact_manifest.json").read_text())
    for entry in manifest["artifacts"]:
        if sha(root/entry["path"]) != entry["sha256"]:
            raise ValueError(f"dataset artifact changed: {entry['path']}")
    table = pd.read_parquet(root/"case_index.parquet")
    for column in STRICT_COLUMNS:
        if not table[column].map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v)).all():
            raise ValueError(f"invalid or malformed data eligibility: {column}")
    if table.case_id.duplicated().any() or table.condition_id.duplicated().any():
        raise ValueError("learning corpus must have one case per physical condition")
    if not (table.cycle_count == 1).all():
        raise ValueError("this learning runner only evaluates observed single cycles")
    for field in ("physics_implementation_fingerprint_sha256",
                  "pipeline_implementation_fingerprint_sha256"):
        if table[field].nunique() != 1 or not table[field].str.fullmatch("[a-f0-9]{64}").all():
            raise ValueError("mixed or absent implementation provenance")
    if not np.isfinite(table[list(FEATURES)+list(TARGETS)].to_numpy(float)).all():
        raise ValueError("non-finite scalar data")
    # Existing canonical builders use a documented inactive-path placeholder.
    # Reject ANY other missing side observation before their imputation runs.
    missing = Counter()
    with h5py.File(root/"fields.h5", "r") as handle:
        for row in table.itertuples(index=False):
            cycle = handle[str(row.hdf5_group)+"/cycles/cycle_001"]
            for stage, name in ((2, "stage2_hold_exchange"), (4, "stage4_cooldown_exchange")):
                inactive = 4 if stage == 2 else 2
                allowed = {f"stage{inactive}_path_pipe_T_state_{end}_K" for end in ("in", "out")}
                for key, data in cycle[name]["side_state"].items():
                    if not np.isfinite(np.asarray(data)).all():
                        if key not in allowed:
                            raise ValueError(f"missing active observation: {row.case_id}/{name}/{key}")
                        missing[f"{stage}:{key}"] += 1
    quality = json.loads((root/"quality_report.json").read_text())
    return table, quality, dict(missing)


def scalar_experiment(table, splits, out, epochs):
    design = np.column_stack((table[list(FEATURES)].to_numpy(float),
                              (table.material_id == "recA").to_numpy(float),
                              (table.material_id == "recAs").to_numpy(float)))
    target = table[list(TARGETS)].to_numpy(float)
    masks = {k: table.condition_id.isin(v).to_numpy() for k, v in splits.items()}
    train = masks["train"]
    xs, ys = StandardScaler().fit(design[train]), StandardScaler().fit(target[train])
    x, y = xs.transform(design), ys.transform(target)
    ridge = Ridge(alpha=1.).fit(x[train], y[train])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        network = nn.Sequential(nn.Linear(x.shape[1], 32), nn.SiLU(), nn.Linear(32, 32),
                                nn.SiLU(), nn.Linear(32, y.shape[1])).double()
        optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=1e-4)
        tx, ty = torch.from_numpy(x[train]), torch.from_numpy(y[train])
        first = float(torch.mean((network(tx)-ty)**2).detach())
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((network(tx)-ty)**2)
            if not torch.isfinite(loss):
                raise ValueError("non-finite scalar training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
        network.eval()
        with torch.no_grad():
            prediction = ys.inverse_transform(network(torch.from_numpy(x)).numpy())
            last = float(torch.mean((network(tx)-ty)**2))
    predictions = {"ridge": ys.inverse_transform(ridge.predict(x)), "neural": prediction}
    baseline = np.broadcast_to(target[train].mean(axis=0), target.shape)
    report = {"feature_names": list(FEATURES)+["material_recA", "material_recAs"],
              "target_names": list(TARGETS), "train_only_normalization": True,
              "neural_training": {"epochs": epochs, "hidden": [32, 32], "activation": "SiLU",
                                  "initial_mse": first, "final_mse": last},
              "reliability_classifier": "not_identifiable_all_63_observations_are_valid",
              "interval_limitation": "Calibration exercise only: selected numerical cases are not an iid population; no guaranteed OOD coverage.",
              "models": {}}
    saved = {"case_ids": table.case_id.to_numpy(str), "targets": target, "train_mean": baseline}
    for name, pred in predictions.items():
        result = {part: {t: comparison(pred[mask, j], baseline[mask, j], target[mask, j])
                         for j, t in enumerate(TARGETS)} for part, mask in masks.items()}
        q, rank = conformal_radius(target[masks["calibration"]]-pred[masks["calibration"]], ys.scale_)
        width = q*ys.scale_
        covered = np.abs(target[masks["test"]]-pred[masks["test"]]) <= width
        result["calibrated_intervals"] = {"nominal_simultaneous_coverage": .9,
            "calibration_count": int(masks["calibration"].sum()), "order_statistic_rank": rank,
            "test_simultaneous_coverage": float(covered.all(axis=1).mean()),
            "per_target_test_coverage": dict(zip(TARGETS, covered.mean(axis=0).tolist())),
            "half_width": dict(zip(TARGETS, width.tolist()))}
        report["models"][name] = result
        saved[name] = pred
        saved[name+"_interval_half_width"] = width
    checkpoint = {"learning_only": True, "scalers": (xs, ys), "ridge": ridge,
                  "neural": network, "splits": splits, "epochs": epochs}
    with (out/"wp2_checkpoint.pkl").open("xb") as stream:
        pickle.dump(checkpoint, stream, protocol=5)
    restored = pickle.loads((out/"wp2_checkpoint.pkl").read_bytes())  # just-written trusted local file
    with torch.no_grad():
        restored_prediction = restored["scalers"][1].inverse_transform(
            restored["neural"](torch.from_numpy(restored["scalers"][0].transform(design))).numpy())
    if not np.array_equal(prediction, restored_prediction):
        raise ValueError("scalar checkpoint round trip mismatch")
    np.savez_compressed(out/"wp2_predictions.npz", **saved)
    report["checkpoint_roundtrip_exact"] = True
    return report


def state_scores(pred, base, truth, records):
    # Never pool dimensional temperature, displacement and phase-fraction errors.
    blocks = {"sma_K": LAYOUT.sma_T, "htf_stage2_K": LAYOUT.htf_stage2_T,
              "htf_stage4_K": LAYOUT.htf_stage4_T, "u_m": LAYOUT.u,
              "w_m": LAYOUT.w, "phase_fraction": LAYOUT.xi, "side_K": LAYOUT.side}
    return {str(stage): {name: comparison(pred[mask, slc], base[mask, slc], truth[mask, slc])
                        for name, slc in blocks.items()}
            for stage in range(1, 5)
            for mask in [np.asarray([r.stage_index == stage for r in records])]}


def propagate(model, records, baseline=False):
    """Four-stage diagnostic, initialized ONCE per case; no target resets."""
    result, states = [], {}
    for record in records:
        if record.case_id not in states:
            if record.stage_index != 1:
                raise ValueError("rollout must start at stage 1")
            states[record.case_id] = record.state_in.copy()
        state = states[record.case_id]
        state = (model.predict_stage_mean(state, record.stage_index) if baseline else
                 model.predict(state, record.controls, record.stage_index))
        states[record.case_id] = state
        result.append(state.copy())
    return np.stack(result)


def state_experiment(records, splits, out, epochs, metadata):
    train = [r for r in records if r.condition_id in splits["train"]]
    model = CausalNeuralStateModel(seed=SEED, hidden=32)
    fit = model.fit(train, epochs=epochs)
    path = out/"wp3_checkpoint.pkl"
    digest = save_frozen_checkpoint(path, model, metadata=metadata)
    if sha(path) != digest:
        raise ValueError("WP3 checkpoint file hash mismatch")
    restored = load_frozen_checkpoint(path, expected_metadata=metadata)
    report = {"training": fit, "training_config": model.training_config,
        "parameter_sha256": model.parameter_sha256(), "checkpoint_sha256": digest,
        "initialization_boundary": "Canonical initial state imports exchange-stage initial fields from the observed record. Four-stage propagation is a record-initialized diagnostic, NOT independently verified fully causal or multicycle forecasting.",
        "physics_boundary": "Hard stage masks and admissibility only; Ridge is empirical, not a reduced PDE; no learned energy-conservation guarantee.",
        "evaluations": {}}
    saved = {}
    for part, ids in splits.items():
        subset = [r for r in records if r.condition_id in ids]
        truth = np.stack([r.state_out for r in subset])
        pred, affine = model.predict_records(subset), model.affine_baseline.predict_records(subset)
        mean = np.stack([model.predict_stage_mean(r.state_in, r.stage_index) for r in subset])
        if not np.array_equal(pred, restored.predict_records(subset)):
            raise ValueError("WP3 checkpoint round trip mismatch")
        roll = propagate(model, subset)
        roll_affine = propagate(model.affine_baseline, subset)
        roll_mean = propagate(model, subset, baseline=True)
        report["evaluations"][part] = {"transition_count": len(subset),
            "teacher_forced_vs_stage_mean": state_scores(pred, mean, truth, subset),
            "teacher_forced_vs_affine": state_scores(pred, affine, truth, subset),
            "four_stage_vs_stage_mean": state_scores(roll, roll_mean, truth, subset),
            "four_stage_vs_affine": state_scores(roll, roll_affine, truth, subset),
            "neural_contribution_rms_by_stage": {str(stage): float(np.sqrt(np.mean(
                (pred[np.asarray([r.stage_index == stage for r in subset])] -
                 affine[np.asarray([r.stage_index == stage for r in subset])])**2))) for stage in range(1,5)},
            "rollout_all_finite": bool(np.isfinite(roll).all()),
            "rollout_clamped_edges_exact": bool(np.all(roll[:, [50, 101]] == 0)),
            "rollout_phase_in_bounds": bool(np.all((roll[:, LAYOUT.xi] >= 0) & (roll[:, LAYOUT.xi] <= 1)))}
        for name, values in {"truth": truth, "neural": pred, "affine": affine,
            "stage_mean": mean, "rollout": roll, "rollout_affine": roll_affine,
            "rollout_stage_mean": roll_mean, "sample_ids": np.asarray([r.sample_id for r in subset])}.items():
            saved[part+"_"+name] = values
    np.savez_compressed(out/"wp3_predictions.npz", **saved)
    report["checkpoint_roundtrip_exact"] = True
    return report


def operator_experiment(samples, splits, out, epochs, metadata):
    train = [s for s in samples if s.condition_id in splits["train"]]
    model = CausalNeuralPODOperator(rank=4, hidden=32, seed=SEED)
    fit = model.fit(train, epochs=epochs)
    path = out/"wp4_checkpoint.pkl"
    digest = model.save_checkpoint(path, **metadata)
    if sha(path) != digest:
        raise ValueError("WP4 checkpoint file hash mismatch")
    restored = CausalNeuralPODOperator.from_checkpoint(path,
        expected_dataset_source_content_sha256=metadata["dataset_source_content_sha256"],
        expected_implementation_sha256=metadata["implementation_sha256"])
    report = {"training": fit, "training_config": model.training_config,
        "parameter_sha256": model.parameter_sha256(), "checkpoint_sha256": digest,
        "boundary": "Observed initial fields -> fixed 50x101 stage histories; distinct SMA/HTF; no mesh-independent or multicycle claim.",
        "evaluations": {}}
    saved = {}
    for part, ids in splits.items():
        subset = [s for s in samples if s.condition_id in ids]
        prediction = model.predict(subset)[:2]
        base = model.predict_stage_mean(subset)
        again = restored.predict(subset)[:2]
        result = {}
        for field, pred, mean, roundtrip in zip(("sma", "htf"), prediction, base, again):
            truth = np.stack([getattr(s, field+"_history") for s in subset])
            persist = np.broadcast_to(truth[:, :1], truth.shape)
            if not np.array_equal(pred, roundtrip) or not np.array_equal(pred[:, 0], truth[:, 0]):
                raise ValueError("WP4 checkpoint/initial field mismatch")
            result[field] = {}
            for stage in (2, 4):
                mask = np.asarray([s.stage_index == stage for s in subset])
                profile = lambda a: a - a.mean(axis=-1, keepdims=True)
                result[field][str(stage)] = {
                    "vs_stage_mean": comparison(pred[mask], mean[mask], truth[mask]),
                    "vs_persistence": comparison(pred[mask], persist[mask], truth[mask]),
                    "profile_vs_stage_mean": comparison(profile(pred[mask]), profile(mean[mask]), profile(truth[mask])),
                    "neural_contribution_rms_K": float(np.sqrt(np.mean((pred[mask]-mean[mask])**2)))}
            for name, a in (("truth", truth), ("neural", pred), ("stage_mean", mean)):
                saved[f"{part}_{field}_{name}"] = a
        saved[part+"_sample_ids"] = np.asarray([s.sample_id for s in subset])
        report["evaluations"][part] = result
    np.savez_compressed(out/"wp4_predictions.npz", **saved)
    report["checkpoint_roundtrip_exact"] = True
    return report


def run(root, out, epochs=300):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if out.exists():
        raise FileExistsError("use a fresh output directory; existing evidence is immutable")
    start = time.monotonic()
    table, quality, missing = validate_dataset(root)
    splits = split_conditions(table.condition_id)
    out.mkdir(parents=True, exist_ok=False)
    implementation_files = [Path(__file__), Path(__file__).with_name("neural_state_space_candidate.py"),
        Path(__file__).with_name("neural_operator_candidate.py"), Path(__file__).with_name("state_space_candidate.py"),
        Path(__file__).with_name("temporal_operator_candidate.py"),
        Path(__file__).parents[1]/"v11/state_space.py", Path(__file__).parents[1]/"v11/spatiotemporal_operator.py"]
    source_hashes = {str(p.resolve()): sha(p) for p in implementation_files}
    implementation = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    protocol = {"learning_only": True, "paper_grade_ready": False,
        "seed": SEED, "epochs": epochs, "dataset_root": str(root.resolve()),
        "dataset_source_content_sha256": quality["source_content_sha256"],
        "case_ids": table.case_id.tolist(), "condition_splits": splits,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "selection": "all existing strict-valid same-profile completed cases; not representative random population",
        "implementation_sources": source_hashes,
        "declared_inactive_path_missing_values": missing,
        "missing_policy": "Only non-owning stage path observations may be NaN. WP3 carries inactive state (293K initial placeholder); WP4 encodes inactive side channel as 293K. No active missing data accepted.",
        "test_policy": "Fixed architecture/300-epoch default, no test-driven tuning. Same condition split for all WP models.",
        "temporal_validation": "not performed: only existing one-cycle data"}
    write_json(out/"protocol.json", protocol)  # before fitting or test prediction
    metadata = {"dataset_source_content_sha256": quality["source_content_sha256"],
                "implementation_sha256": implementation}
    records, samples = build_transition_corpus(root), build_operator_corpus(root)
    if len(records) != 4*len(table) or len(samples) != 2*len(table):
        raise ValueError("corpus lost observations")
    for corpus in (records, samples):
        if {r.condition_id for r in corpus} != set(table.condition_id):
            raise ValueError("corpus condition coverage mismatch")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            wp2 = scalar_experiment(table, splits, out, epochs)
            write_json(out/"wp2.json", wp2)
            wp3 = state_experiment(records, splits, out, epochs, metadata)
            write_json(out/"wp3.json", wp3)
            wp4 = operator_experiment(samples, splits, out, epochs, metadata)
            write_json(out/"wp4.json", wp4)
    finally:
        torch.set_num_threads(previous_threads)
    report = {"status": "learning_experiment_completed", "paper_grade_ready": False,
        "case_count": len(table), "split_sizes": protocol["split_sizes"],
        "observed_transition_count": len(records), "operator_stage_count": len(samples),
        "elapsed_seconds": time.monotonic()-start, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "all_checkpoint_roundtrips_exact": all(r["checkpoint_roundtrip_exact"] for r in (wp2, wp3, wp4)),
        "publication_confirmation": "not executed; excluded by revised user scope",
        "positive_skill_required_for_learning_completion": False,
        "artifacts": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.output_root, args.epochs), indent=2, allow_nan=False))
