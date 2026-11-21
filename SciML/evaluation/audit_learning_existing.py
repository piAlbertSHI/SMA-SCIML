"""Read-only model/data audit plus new audit/figure artifacts; never train a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from SciML.models.v12.run_learning_existing import (
    TARGETS, LAYOUT, split_conditions, validate_dataset, sha, write_json,
)
from SciML.models.v12.neural_state_space_candidate import load_frozen_checkpoint
from SciML.models.v12.neural_operator_candidate import CausalNeuralPODOperator
from SciML.models.v11.state_space import build_transition_corpus
from SciML.models.v11.spatiotemporal_operator import build_operator_corpus


def check_rmse(pred, truth, recorded):
    actual = float(np.sqrt(np.mean((pred-truth)**2)))
    if not np.isclose(actual, recorded, rtol=1e-12, atol=1e-14):
        raise ValueError("reported RMSE differs from saved predictions")


def audit(result, out):
    if out.exists():
        raise FileExistsError("audit output already exists")
    summary = json.loads((result/"summary.json").read_text())
    for name, digest in summary["artifacts"].items():
        if sha(result/name) != digest:
            raise ValueError(f"changed experiment artifact: {name}")
    protocol = json.loads((result/"protocol.json").read_text())
    for path, digest in protocol["implementation_sources"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"changed learning implementation: {path}")
    root = Path(protocol["dataset_root"])
    table, quality, missing = validate_dataset(root)
    if quality["source_content_sha256"] != protocol["dataset_source_content_sha256"]:
        raise ValueError("dataset source mismatch")
    splits = split_conditions(table.condition_id)
    if splits != protocol["condition_splits"] or table.case_id.tolist() != protocol["case_ids"]:
        raise ValueError("split/data order mismatch")
    train_ids = set(table.loc[table.condition_id.isin(splits["train"]), "case_id"])
    wp2, wp3, wp4 = [json.loads((result/f"wp{i}.json").read_text()) for i in (2,3,4)]
    records, samples = build_transition_corpus(root), build_operator_corpus(root)
    # Pickles are trusted local artifacts only; their SHA was verified above.
    state = load_frozen_checkpoint(result/"wp3_checkpoint.pkl")
    operator = CausalNeuralPODOperator.from_checkpoint(result/"wp4_checkpoint.pkl")
    for model, count in ((state, 4), (operator, 2)):
        if len(model.training_ids) != count*len(train_ids) or {
            value.split(":cycle_")[0] for value in model.training_ids
        } != train_ids:
            raise ValueError("held-out condition in fitted checkpoint, or missing training row")
    wp2_saved = np.load(result/"wp2_predictions.npz", allow_pickle=False)
    if not np.array_equal(wp2_saved["case_ids"], table.case_id.to_numpy(str)):
        raise ValueError("WP2 prediction row mismatch")
    if not np.array_equal(wp2_saved["targets"], table[list(TARGETS)].to_numpy(float)):
        raise ValueError("WP2 target mismatch")
    scalar = pickle.loads((result/"wp2_checkpoint.pkl").read_bytes())
    if scalar["splits"] != splits or scalar["scalers"][0].n_samples_seen_ != len(train_ids):
        raise ValueError("scalar fitting split mismatch")
    for part, ids in splits.items():
        mask = table.condition_id.isin(ids).to_numpy()
        for model in ("neural", "ridge"):
            for j, name in enumerate(TARGETS):
                check_rmse(wp2_saved[model][mask,j], wp2_saved["targets"][mask,j],
                           wp2["models"][model][part][name]["model"]["rmse"])
    state_saved = np.load(result/"wp3_predictions.npz", allow_pickle=False)
    operator_saved = np.load(result/"wp4_predictions.npz", allow_pickle=False)
    for part, ids in splits.items():
        rows = [r for r in records if r.condition_id in ids]
        fields = [s for s in samples if s.condition_id in ids]
        if not np.array_equal(state_saved[part+"_sample_ids"], [r.sample_id for r in rows]):
            raise ValueError("WP3 row mismatch")
        if not np.array_equal(state_saved[part+"_truth"], np.stack([r.state_out for r in rows])):
            raise ValueError("WP3 target mismatch")
        if not np.array_equal(state_saved[part+"_neural"], state.predict_records(rows)):
            raise ValueError("WP3 checkpoint cannot reproduce saved predictions")
        for stage in range(1,5):
            mask = np.asarray([r.stage_index == stage for r in rows])
            for key, metric_key in (("neural", "teacher_forced_vs_stage_mean"),
                                    ("rollout", "four_stage_vs_stage_mean")):
                check_rmse(state_saved[f"{part}_{key}"][mask, LAYOUT.sma_T],
                    state_saved[part+"_truth"][mask, LAYOUT.sma_T],
                    wp3["evaluations"][part][metric_key][str(stage)]["sma_K"]["model"]["rmse"])
        if not np.array_equal(operator_saved[part+"_sample_ids"], [s.sample_id for s in fields]):
            raise ValueError("WP4 row mismatch")
        actual_predictions = operator.predict(fields)[:2]
        for field, actual in zip(("sma", "htf"), actual_predictions):
            truth = np.stack([getattr(s, field+"_history") for s in fields])
            if not np.array_equal(truth, operator_saved[f"{part}_{field}_truth"]):
                raise ValueError("WP4 target mismatch")
            if not np.array_equal(actual, operator_saved[f"{part}_{field}_neural"]):
                raise ValueError("WP4 checkpoint cannot reproduce saved predictions")
            for stage in (2,4):
                mask = np.asarray([s.stage_index == stage for s in fields])
                check_rmse(actual[mask], truth[mask],
                    wp4["evaluations"][part][field][str(stage)]["vs_stage_mean"]["model"]["rmse"])
    for bundle in (wp2_saved, state_saved, operator_saved):
        for name in bundle.files:
            values = bundle[name]
            if values.dtype.kind in "fiu" and not np.isfinite(values).all():
                raise ValueError(f"non-finite saved prediction: {name}")
    material_counts = {part: table[table.condition_id.isin(ids)].material_id.value_counts().to_dict()
                       for part, ids in splits.items()}
    report = {"status": "passed_learning_integrity_audit", "learning_workflow_verified": True,
        "paper_grade_ready": False, "case_count": len(table), "split_sizes": summary["split_sizes"],
        "material_counts": material_counts, "dataset_source_content_sha256": quality["source_content_sha256"],
        "experiment_summary_sha256": sha(result/"summary.json"),
        "checks": ["dataset and experiment artifact hashes", "current implementation hashes",
            "strict-valid same-profile data", "deterministic disjoint condition splits",
            "WP3/WP4 training IDs contain only training conditions", "train-only scalar scaler count",
            "observed targets and prediction row IDs", "saved prediction RMSE independently recomputed",
            "WP3/WP4 checkpoint prediction replay", "finite numeric predictions"],
        "limitations": ["No new MATLAB, Confirm50 or Temporal12 run",
            "All 13 test conditions are recA: no held-out recAs claim",
            "No invalid class: no demonstrated failure classifier",
            "WP3 record-derived initialization, not fully causal multicycle evidence",
            "WP3 neural correction is worse than affine in some stages",
            "63 selected conditions, not 252 independent transition conditions or 126 independent field conditions",
            "13 calibration cases: observed interval coverage is not population reliability"]}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"audit.json", report)
    make_figure(wp2, wp3, wp4, out/"learning_results.png")
    return report


def make_figure(wp2, wp3, wp4, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    x = np.arange(5)
    for name, offset, color in (("ridge", -.18, "#8b98aa"), ("neural", .18, "#007d91")):
        ratios = [1-wp2["models"][name]["test"][t]["rmse_skill"] for t in TARGETS]
        axes[0].bar(x+offset, ratios, .36, label=name, color=color)
    axes[0].set(xticks=x, xticklabels=["Qcool", "Wpump", "COP", "SCP", "delta T"],
                ylabel="Test RMSE / train-mean baseline RMSE", title="WP2 | Scalar prediction")
    axes[0].legend(frameon=False)
    x = np.arange(4)
    for name, offset, color in (("model", .18, "#007d91"), ("baseline", -.18, "#8b98aa")):
        vals = [wp3["evaluations"]["test"]["four_stage_vs_affine"][str(s)]["sma_K"][name]["rmse"] for s in range(1,5)]
        axes[1].bar(x+offset, vals, .36, label="neural hybrid" if name == "model" else "affine", color=color)
    axes[1].set(xticks=x, xticklabels=["Loading", "Hold", "Unloading", "Cooldown"],
                ylabel="SMA temperature RMSE (K)", title="WP3 | Record-initialized propagation")
    axes[1].legend(frameon=False)
    x = np.arange(4)
    pairs = [(f,s) for f in ("sma","htf") for s in ("2","4")]
    for name, offset, color in (("model", .18, "#007d91"), ("baseline", -.18, "#8b98aa")):
        vals = [wp4["evaluations"]["test"][f][s]["vs_stage_mean"][name]["rmse"] for f,s in pairs]
        axes[2].bar(x+offset, vals, .36, label="neural POD" if name == "model" else "stage mean", color=color)
    axes[2].set(xticks=x, xticklabels=["SMA hold", "SMA cool", "HTF hold", "HTF cool"],
                ylabel="Full-field RMSE (K)", title="WP4 | Fixed-grid stage fields")
    axes[2].legend(frameon=False)
    fig.suptitle("Existing-data learning experiment | 63 conditions, 37/13/13 split | 13 recA test conditions", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.result_root, args.output_root), indent=2))
