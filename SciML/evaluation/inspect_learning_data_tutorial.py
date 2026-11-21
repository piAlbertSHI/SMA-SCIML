"""Executable file-to-tensor tutorial and source-owner check on existing63.

No training or solver launch. Does not treat an incomplete220matrix as complete.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform

import h5py
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

from SciML.models.v12.run_learning_existing import FEATURES, STRICT_COLUMNS, validate_dataset, sha, write_json, split_conditions
from SciML.models.v11.cycle_proxy import TARGETS
from SciML.models.v11.spatiotemporal_operator import build_operator_corpus
from SciML.models.v12.temporal_operator_candidate import causal_delta_features
from SciML.models.v12.learning_boundary_state import build_boundaries, features
from SciML.builders.build_dataset_v11_pilot import cycle_payloads


def grouping_counts(ids, masks):
    """Case IDs are used for leakage detection; masks map sample rows to groups."""
    groups = {name: set(np.asarray(ids)[mask]) for name, mask in masks.items()}
    overlap = {a+"/"+b: len(groups[a] & groups[b]) for i, a in enumerate(groups) for b in list(groups)[i+1:]}
    if any(overlap.values()):
        raise ValueError("one physical case appears in more than one split")
    return {name: len(values) for name, values in groups.items()}


def scaling_demo(x, train):
    """Poison copies only: verify train-only stats cannot consume future rows."""
    x, train = np.asarray(x, float), np.asarray(train, bool)
    if x.ndim != 2 or train.shape != (len(x),) or not train.any() or train.all() or not np.isfinite(x).all():
        raise ValueError("finite design and nontrivial training mask required")
    scaler = StandardScaler().fit(x[train])
    poisoned = x.copy()
    poisoned[~train] += 1e6
    safe = StandardScaler().fit(poisoned[train])
    wrong = StandardScaler().fit(poisoned)
    np.testing.assert_array_equal(scaler.mean_, safe.mean_)
    np.testing.assert_array_equal(scaler.scale_, safe.scale_)
    return {"train_only_mean_unchanged_exact": True, "train_only_scale_unchanged_exact": True,
        "wrong_global_mean_max_change": float(np.max(np.abs(wrong.mean_-scaler.mean_))),
        "experiment": "copied holdout features+1e6, illustrative preprocessing leak; source arrays untouched"}


def run(root, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve tutorial audit")
    table, quality, missing = validate_dataset(root)
    protocol = json.loads(original_protocol.read_text())
    if quality["source_content_sha256"] != protocol["dataset_source_content_sha256"]:
        raise ValueError("protocol source mismatch")
    splits = protocol["condition_splits"]
    if splits != split_conditions(table.condition_id, protocol["seed"]):
        raise ValueError("saved condition split differs from deterministic rule")
    masks = {k: table.condition_id.isin(v).to_numpy() for k, v in splits.items()}
    grouping_counts(table.case_id.to_numpy(), masks)
    x = np.column_stack((table[list(FEATURES)].to_numpy(float), (table.material_id == "recA").to_numpy(float), (table.material_id == "recAs").to_numpy(float)))
    y = table[list(TARGETS)].to_numpy(float)
    xs, ys = StandardScaler().fit(x[masks["train"]]), StandardScaler().fit(y[masks["train"]])
    z, yn = xs.transform(x), ys.transform(y)
    np.testing.assert_allclose(ys.inverse_transform(yn), y, rtol=1e-14, atol=1e-12)
    samples = build_operator_corpus(root)
    branch = np.stack([causal_delta_features(s) for s in samples])
    fields = np.stack([np.stack((s.sma_history, s.htf_history)) for s in samples])
    smasks = {k: np.array([s.condition_id in v for s in samples]) for k, v in splits.items()}
    grouping_counts([s.case_id for s in samples], smasks)
    _, _, boundaries, _, _ = build_boundaries(root)
    state = np.stack([features(r.state_in, r.controls) for r in boundaries])
    bmasks = {k: np.array([r.condition_id in v for r in boundaries]) for k, v in splits.items()}
    grouping_counts([r.case_id for r in boundaries], bmasks)
    owners, sample_tree = [], {}
    with h5py.File(root/"fields.h5", "r") as handle:
        for index, row in enumerate(table.itertuples(index=False)):
            path = Path(row.source_artifact)
            req = path.parent/"sciml_case_request.json"
            if sha(path) != row.source_artifact_sha256 or sha(req) != row.source_request_sha256:
                raise ValueError("changed raw case/request")
            raw = loadmat(path, simplify_cells=True)["case_artifact"]
            if raw["case_id"] != row.case_id:
                raise ValueError("case_id source mismatch")
            cycles = cycle_payloads(raw)
            if len(cycles) != 1:
                raise ValueError("current teaching corpus expects exactly one measured cycle")
            cycle = handle[row.hdf5_group+"/cycles/cycle_001"]
            for stage in ("stage1_loading", "stage2_hold_exchange", "stage3_unloading", "stage4_cooldown_exchange"):
                keys = ("u_m_t", "w_m_t", "T_K_t", "xi_t") if stage in ("stage1_loading", "stage3_unloading") else ("sma_strip_T_K_t", "htf_internal_T_K_t")
                for key in keys:
                    source = np.asarray(cycles[0][stage][key])
                    observed = np.asarray(cycle[stage][key])
                    np.testing.assert_array_equal(source, observed)
                    owners.append({"case_id": row.case_id, "stage": stage, "field": key, "shape": list(observed.shape), "raw_equals_hdf5_exact": True})
            if index == 0:
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        value = np.asarray(obj)
                        sample_tree[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "nonfinite_count": int((~np.isfinite(value)).sum())}
                cycle.visititems(visit)
    result = {"status": "completed_file_to_tensor_tutorial_audit", "learning_only": True, "fresh_solver_calls": 0,
        "case_count": len(table), "material_counts": dict(Counter(table.material_id)),
        "split_counts": {k: int(v.sum()) for k, v in masks.items()},
        "split_material_counts": {k: dict(Counter(table.loc[v, "material_id"])) for k, v in masks.items()},
        "state_rows_by_split": {k: int(v.sum()) for k, v in bmasks.items()},
        "field_rows_by_split": {k: int(v.sum()) for k, v in smasks.items()},
        "shapes": {"scalar_input": list(x.shape), "scalar_target": list(y.shape), "boundary_features": list(state.shape),
                   "operator_branch": list(branch.shape), "operator_fields": list(fields.shape)},
        "feature_order": list(FEATURES)+["material_recA", "material_recAs"], "target_order": list(TARGETS),
        "feature_ranges": {k: [float(table[k].min()), float(table[k].max())] for k in FEATURES},
        "radius_thickness_ratio_range": [float((table.radius_m/table.thickness_m).min()), float((table.radius_m/table.thickness_m).max())],
        "strict_flags_true_counts": {k: int(table[k].sum()) for k in STRICT_COLUMNS},
        "raw_to_hdf5_field_comparisons": len(owners), "allowed_inactive_side_missing": missing,
        "scaling_demonstration": scaling_demo(x, masks["train"]),
        "first_case": {"case_id": str(table.iloc[0].case_id), "source_artifact": str(table.iloc[0].source_artifact), "hdf5_group": str(table.iloc[0].hdf5_group),
            "X": x[0].tolist(), "Y": y[0].tolist(), "normalized_X": z[0].tolist(), "normalized_Y": yn[0].tolist()},
        "normalization": {"X_mean": xs.mean_.tolist(), "X_scale": xs.scale_.tolist(), "Y_mean": ys.mean_.tolist(), "Y_scale": ys.scale_.tolist()},
        "old_paper_matrix": {"expected_case_count": quality["expected_case_count"], "expected_matrix_complete": quality["expected_matrix_complete"],
            "confirmation_quality_gate": quality["confirmation_quality_gate"], "interpretation": "unchanged old matrix status, not a learning-size requirement"},
        "dataset_source_content_sha256": quality["source_content_sha256"], "protocol_sha256": sha(original_protocol),
        "environment": {"python": platform.python_version(), **{k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pandas", "h5py", "scikit-learn", "torch", "pyarrow")}},
        "implementation_sha256": sha(Path(__file__))}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"raw_to_hdf5_checks.json", owners)
    write_json(out/"example_hdf5_tree.json", sample_tree)
    np.savez_compressed(out/"tutorial_arrays.npz", X=x, Y=y, normalized_X=z, normalized_Y=yn,
                        train_mask=masks["train"], calibration_mask=masks["calibration"], test_mask=masks["test"])
    result["artifacts"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/"summary.json", result)
    return {k: result[k] for k in ("status", "case_count", "split_counts", "split_material_counts", "state_rows_by_split", "field_rows_by_split", "shapes", "raw_to_hdf5_field_comparisons", "feature_ranges", "environment")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_protocol, args.output_root), indent=2))
