"""Independent grouped-error/calibration audit and cold checkpoint replay."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from .audit_learning_wp5 import verify_artifacts
from .audit_learning_operators import matched_pod
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator, corpus
from SciML.models.v12.learning_ensemble import LearningEnsemble, design_matrix
from SciML.models.v12.run_learning_existing import TARGETS, sha, validate_dataset, write_json


def run(dataset, campaign, out):
    if out.exists():
        raise FileExistsError("preserve OOD audit")
    table, quality, _ = validate_dataset(dataset)
    samples, branch, initial, truth, _ = corpus(dataset)
    x, y = design_matrix(table), table[list(TARGETS)].to_numpy(float)
    stages = np.array([s.stage_index for s in samples])
    aggregate, verified_sources = [], {}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for scheme in ("geometry", "material", "protocol", "combined"):
            root = campaign/f"ood-{scheme}-r1"
            summary = verify_artifacts(root)
            protocol = json.loads((root/"protocol.json").read_text())
            if protocol["implementation_sha256"] != sha(Path("SciML/models/v12/learning_structured_ood.py")):
                raise ValueError("OOD source changed")
            if protocol["dataset_source_content_sha256"] != quality["source_content_sha256"]:
                raise ValueError("OOD source identity changed")
            for name, digest in protocol["upstream_implementation_sha256"].items():
                if sha(Path("SciML/models/v12")/(name+".py")) != digest:
                    raise ValueError("upstream model changed")
            groups = protocol["condition_groups"]
            flat = sum(groups.values(), [])
            if len(flat) != len(set(flat)) or set(flat) != set(table.condition_id):
                raise ValueError("non-disjoint or incomplete groups")
            g = table.radius_m.eq(table.radius_m.max()).to_numpy()
            m = table.material_id.eq("recAs").to_numpy()
            t = table.exchange_duration_s.eq(table.exchange_duration_s.max()).to_numpy()
            held = {"geometry": g, "material": m, "protocol": t, "combined": g | m}[scheme]
            masks = {name: table.condition_id.isin(ids).to_numpy() for name, ids in groups.items()}
            order = sorted(table.condition_id[~held], key=lambda case: hashlib.sha256(("structured-20260910:"+case).encode()).hexdigest())
            n_id = round(.2*len(order))
            expected = {"calibration": order[:13], "id_test": order[13:13+n_id], "train": order[13+n_id:]}
            for name, ids in expected.items():
                if set(ids) != set(groups[name]):
                    raise ValueError("input-only in-domain split changed")
            expected_ood = {"ood_geometry_only": g & ~m, "ood_material_only": m & ~g, "ood_combined": g & m} if scheme == "combined" else {"ood_"+scheme: held}
            for name, mask in expected_ood.items():
                np.testing.assert_array_equal(mask, masks[name])
            fmasks = {name: np.array([s.condition_id in ids for s in samples]) for name, ids in groups.items()}
            train, cal = masks["train"], masks["calibration"]
            baseline = np.broadcast_to(y[train].mean(axis=0), y.shape)
            # Fresh deterministic POD fit from the exact training labels.
            pod, _ = matched_pod(branch, initial, truth, stages, fmasks["train"])
            stored_pod = np.load(root/"pod_predictions.npz", allow_pickle=False)
            np.testing.assert_array_equal(pod, stored_pod["prediction"])
            records = []
            for seed in (20260910, 20260911, 20260912):
                dest = root/f"s{seed}"
                report = json.loads((dest/"report.json").read_text())
                arrays = np.load(dest/"predictions.npz", allow_pickle=False)
                np.testing.assert_array_equal(arrays["case_ids"], table.case_id.to_numpy(str))
                np.testing.assert_array_equal(arrays["sample_ids"], [s.sample_id for s in samples])
                scalar = LearningEnsemble.load(dest/"scalar.pt")
                if list(scalar.training_ids) != table.case_id[train].tolist():
                    raise ValueError("scalar fit condition leakage")
                np.testing.assert_allclose(scalar.x_scaler.mean_, x[train].mean(axis=0), atol=0, rtol=0)
                np.testing.assert_allclose(scalar.y_scaler.mean_, y[train].mean(axis=0), atol=0, rtol=0)
                members = scalar.predict_members(x)
                np.testing.assert_array_equal(members, arrays["scalar_members"])
                mean, std = members.mean(axis=0), members.std(axis=0, ddof=1)
                np.testing.assert_array_equal(mean, arrays["ensemble3"])
                np.testing.assert_array_equal(members[0], arrays["mlp"])
                with (dest/"classical.pkl").open("rb") as stream:
                    saved = pickle.load(stream)  # only our verified local experiment artifact
                if saved["training_ids"] != table.case_id[train].tolist():
                    raise ValueError("classical fit condition leakage")
                with threadpool_limits(limits=1):
                    for name, estimator in saved["estimators"].items():
                        replay = saved["y_scaler"].inverse_transform(estimator.predict(saved["x_scaler"].transform(x)))
                        np.testing.assert_array_equal(replay, arrays[name])
                operator = CoordinateOperator.load(dest/"operator.pt")
                if operator.training_ids != [s.sample_id for s, use in zip(samples, fmasks["train"]) if use]:
                    raise ValueError("operator fit condition leakage")
                field = operator.predict(branch, initial)
                np.testing.assert_array_equal(field, arrays["operator"])
                np.testing.assert_array_equal(field[:, :, 0], initial)
                for model, parts in report["scalar_scores"].items():
                    for part, values in parts.items():
                        mask = masks[part]
                        if not mask.any():
                            continue
                        error = arrays[model][mask]-y[mask]
                        mse = np.mean(error**2, axis=0)
                        base_mse = np.mean((baseline[mask]-y[mask])**2, axis=0)
                        variance = np.mean((y[mask]-y[mask].mean(axis=0))**2, axis=0)
                        expected_metrics = {"rmse": np.sqrt(mse).tolist(), "mae": np.abs(error).mean(axis=0).tolist(),
                            "max_abs": np.abs(error).max(axis=0).tolist(),
                            "r2": [float(1-a/b) if b > 0 else None for a, b in zip(mse, variance)],
                            "rmse_skill_vs_train_mean": [float(1-np.sqrt(a/b)) if b > 0 else None for a, b in zip(mse, base_mse)]}
                        if expected_metrics != values:
                            raise ValueError("scalar score mismatch")
                for part, mask in fmasks.items():
                    for stage in (2, 4):
                        take = mask & (stages == stage)
                        if not take.any():
                            continue
                        for j, name in enumerate(("sma", "htf")):
                            error = field[take, j, 1:]-truth[take, j, 1:]
                            observed = report["operator_scores"][part][str(stage)][name]
                            checks = {"future_rmse_K": np.sqrt(np.mean(error**2)), "future_mae_K": np.abs(error).mean(),
                                      "future_max_abs_K": np.abs(error).max(),
                                      "relative_l2_absolute_temperature": np.linalg.norm(error)/np.linalg.norm(truth[take, j, 1:]),
                                      "relative_l2_temperature_change": np.linalg.norm(error)/np.linalg.norm(truth[take, j, 1:]-initial[take, j, None])}
                            for key, value in checks.items():
                                if float(value) != observed[key]:
                                    raise ValueError("operator metric mismatch")
                scale = scalar.y_scaler.scale_
                q = np.sort(np.max(np.abs(y[cal]-mean[cal])/np.maximum(std[cal], .01*scale), axis=1))[12]
                for part, interval in report["calibrated_interval_diagnostics"].items():
                    mask = masks[part]
                    width = q*np.maximum(std[mask], .01*scale)
                    inside = np.abs(y[mask]-mean[mask]) <= width
                    checks = {"quantile": float(q), "test_simultaneous_coverage": float(inside.all(axis=1).mean()),
                              "per_target_coverage": inside.mean(axis=0).tolist(), "mean_full_width": (2*width).mean(axis=0).tolist()}
                    if interval["order_statistic_rank"] != 13 or interval["calibration_count"] != 13:
                        raise ValueError("wrong conformal rank")
                    for key, value in checks.items():
                        if value != interval[key]:
                            raise ValueError("interval mismatch")
                records.append(report)
            for part, ids in groups.items():
                if part in ("train", "calibration") or not ids:
                    continue
                models = {}
                for model in records[0]["scalar_scores"]:
                    values = np.array([r["scalar_scores"][model][part]["rmse"] for r in records])
                    models[model] = {"mean_rmse": values.mean(axis=0).tolist(), "std_rmse": values.std(axis=0, ddof=1).tolist()}
                fields = np.array([[r["operator_scores"][part][str(stage)][name]["future_rmse_K"] for stage in (2, 4) for name in ("sma", "htf")] for r in records])
                pod_rmse = [float(np.sqrt(np.mean((pod[fmasks[part] & (stages == stage), j, 1:]-truth[fmasks[part] & (stages == stage), j, 1:])**2))) for stage in (2, 4) for j in (0, 1)]
                aggregate.append({"scheme": scheme, "group": part, "case_count": len(ids), "scalar_models": models,
                    "deeponet_mean_rmse_K": fields.mean(axis=0).tolist(), "deeponet_std_rmse_K": fields.std(axis=0, ddof=1).tolist(),
                    "pod_rmse_K": pod_rmse,
                    "simultaneous90_coverage_by_seed": [r["calibrated_interval_diagnostics"][part]["test_simultaneous_coverage"] for r in records]})
            verified_sources[scheme] = sha(root/"summary.json")
    finally:
        torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"aggregate.json", aggregate)
    figure(aggregate, out)
    result = {"status": "passed_structured_ood_integrity_audit", "learning_only": True, "fresh_solver_calls": 0,
        "scalar_ensemble_checkpoints": 12, "operator_checkpoints": 12, "classical_model_replays": 36,
        "all_cold_predictions_exact": True, "input_only_groups_independently_reconstructed": True,
        "fresh_pod_refits_exact": 4, "metric_and_conformal_reconstruction": True,
        "ood_coverage_guarantee": False, "combined_group_count": 2,
        "implementation_sha256": sha(Path(__file__)), "upstream_summary_hashes": verified_sources,
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return {**result, "compact_results": [{"scheme": r["scheme"], "group": r["group"], "n": r["case_count"],
        "ensemble_rmse": r["scalar_models"]["ensemble3"]["mean_rmse"], "deeponet_rmse": r["deeponet_mean_rmse_K"],
        "coverage90": r["simultaneous90_coverage_by_seed"]} for r in aggregate]}


def figure(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [r["scheme"]+"/"+r["group"].replace("ood_", "") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for model in ("ridge", "random_forest", "hist_gradient_boosting", "ensemble3"):
        axes[0].plot(range(len(rows)), [r["scalar_models"][model]["mean_rmse"][2] for r in rows], marker="o", label=model)
    axes[0].set(ylabel="COP RMSE (3-seed mean)", title="Scalar ID versus structured OOD")
    axes[0].legend(fontsize=8)
    for j, label in enumerate(("S2 SMA", "S2 HTF", "S4 SMA", "S4 HTF")):
        axes[1].plot(range(len(rows)), [r["deeponet_mean_rmse_K"][j] for r in rows], marker="o", label=label)
    axes[1].set(ylabel="DeepONet future-field RMSE (K)", title="Observed-start field prediction, not full-cycle forecast")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(rows)), labels, rotation=65, ha="right", fontsize=8)
    fig.savefig(out/"structured_ood.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.output_root), indent=2))
