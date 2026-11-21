"""Array/checkpoint-bound chapter08--11 review, without fitting or solver calls.

This is a narrow author content review, not a whole-report completion certificate.
Training replay is covered by earlier dedicated audits, not claimed by this run.
"""
import argparse
import json
import math
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.stats import spearmanr
import torch

from .audit_learning_design import area
from .audit_learning_operator_chapter import table_cells
from .audit_learning_scalar_state_chapters import rounded_value
from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_ensemble import LearningEnsemble, ENSEMBLE_TARGETS, design_matrix
from SciML.models.v12.learning_pinn_inverse import HeatInversePINN, predict
from SciML.models.v12.learning_local_thermal_inverse import ThermalNodePINN, ode_prediction
from SciML.models.v12.run_learning_existing import sha, validate_dataset, write_json


def rmse(prediction, truth, axis=0):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or not prediction.size:
        raise ValueError("review refuses empty arrays or broadcasted error")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("review refuses nonfinite error")
    return np.sqrt(np.mean((prediction-truth)**2, axis=axis))


def conformal_order(scores, coverage):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all() or not 0 < coverage < 1:
        raise ValueError("invalid review calibration scores/coverage")
    rank = math.ceil((len(scores)+1)*coverage)
    return rank, None if rank > len(scores) else float(np.sort(scores)[rank-1])


def run(dataset, campaign, chapters, out):
    if out.exists():
        raise FileExistsError("preserve previous inverse/decision content review")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    paths = [next(chapters.glob(f"{i:02d}_*.md")) for i in range(8, 12)]
    texts = {i: p.read_text(encoding="utf-8") for i, p in zip(range(8, 12), paths)}
    roots = {name: campaign/name for name in (
        "wp5-pinn-r1", "wp5-local-thermal-r1", "wp5-local-thermal-r2",
        "wp6-uq-r1", "wp6-active-r1", "wp6-design-r1")}
    summaries = {name: verify_artifacts(root) for name, root in roots.items()}
    source_map = {
        "wp5-pinn-r1": "SciML/models/v12/learning_pinn_inverse.py",
        "wp5-local-thermal-r2": "SciML/models/v12/learning_local_thermal_inverse.py",
        "wp6-uq-r1": "SciML/models/v12/learning_ensemble.py",
        "wp6-active-r1": "SciML/active_learning/learning_pool_loop.py",
        "wp6-design-r1": "SciML/active_learning/learning_design_loop.py"}
    for name, source in source_map.items():
        if read(roots[name]/"protocol.json")["implementation_sha256"] != sha(Path(source)):
            raise ValueError("trained implementation changed: "+source)
    table, quality, _ = validate_dataset(dataset)
    uq_protocol = read(roots["wp6-uq-r1"]/"protocol.json")
    if uq_protocol["source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("dataset content binding differs")
    for name in ("wp5-local-thermal-r2", "wp6-active-r1", "wp6-design-r1"):
        if read(roots[name]/"protocol.json")["dataset_source_content_sha256"] != quality["source_content_sha256"]:
            raise ValueError("experiment dataset binding differs: "+name)
    masks = {k: table.condition_id.isin(ids).to_numpy() for k, ids in uq_protocol["condition_splits"].items()}
    if {k: int(v.sum()) for k, v in masks.items()} != {"train": 37, "calibration": 13, "test": 13}:
        raise ValueError("split changed")
    x, y = design_matrix(table), table[list(ENSEMBLE_TARGETS)].to_numpy(float)
    checked, metrics = [], {}

    def cell(ch, label, col, value, percent=False):
        body = texts[ch]
        if ch == 8 and re.fullmatch(r"s\d+", label):
            body = body.split("## 8.8", 1)[1].split("## 8.9", 1)[0]
        token = table_cells(body, label)[col]
        if percent:
            if not token.endswith("%"):
                raise ValueError("missing percentage unit")
            token, value = token[:-1], value*100
        checked.append({"chapter": ch, "row": label, "column": col, **rounded_value(token, value)})

    def inline(ch, pattern, value):
        matches = re.findall(pattern, texts[ch])
        if len(matches) != 1:
            raise ValueError(f"missing/ambiguous numeric prose: {pattern}")
        checked.append({"chapter": ch, "pattern": pattern, **rounded_value(matches[0], value)})

    # Recompute unseen interior metrics and coefficients from six cold checkpoints.
    data = np.load(roots["wp5-pinn-r1"]/"predictions.npz", allow_pickle=False)
    grid, truth = data["grid"], data["truth"]
    xx, tt = grid[:, :1], grid[:, 1:]
    expected = np.sin(np.pi*xx)*np.exp(-.12*np.pi**2*tt)+.3*np.sin(2*np.pi*xx)*np.exp(-.48*np.pi**2*tt)
    np.testing.assert_allclose(truth, expected, rtol=0, atol=2e-16)
    interior = (grid[:, 0] > 0) & (grid[:, 0] < 1) & (grid[:, 1] > 0)
    for point in data["observation_coordinates"]:
        interior &= ~np.all(np.isclose(grid, point, rtol=0, atol=1e-12), axis=1)
    if int(interior.sum()) != 2438:
        raise ValueError("unobserved grid count changed")
    pin_metrics = {}
    for seed in (410, 411, 412):
        for kind in ("pinn", "data_only"):
            tag = f"s{seed}_{kind}"
            model = HeatInversePINN()
            model.load_state_dict(torch.load(roots["wp5-pinn-r1"]/(tag+".pt"), weights_only=True)["state_dict"])
            prediction = predict(model, grid)
            np.testing.assert_array_equal(prediction, data[tag+"_prediction"])
            error = float(rmse(prediction[interior], truth[interior], axis=None))
            record = read(roots["wp5-pinn-r1"]/(tag+".json"))
            if kind == "pinn":
                a = float(model.diffusivity.detach())
                cell(8, f"s{seed}", 0, a)
                cell(8, f"s{seed}", 1, abs(a-.12)/.12, percent=True)
            elif record["identified_diffusivity"] is not None:
                raise ValueError("data-only coefficient falsely identified")
            cell(8, f"s{seed}", 2 if kind == "pinn" else 3, error)
            with torch.no_grad():
                model.log_diffusivity.fill_(math.log(.12))
            residual = model.residual(torch.tensor(grid[interior][::7], dtype=torch.float64, requires_grad=True)).detach().numpy()
            pin_metrics[tag] = {"interior_rmse": error, "common_a_pde_rms": float(np.sqrt(np.mean(residual**2)))}
    for kind, pattern in (("pinn", r"PDE RMS 为 PINN ([\d.]+)"), ("data_only", r"纯数据 ([\d.]+)。")):
        inline(8, pattern, pin_metrics[f"s410_{kind}"]["common_a_pde_rms"])
    for model, col in ((HeatInversePINN(), 0), (ThermalNodePINN(), 1)):
        cell(8, "网络权重及偏置数", col, sum(p.numel() for p in model.network.parameters()))
        token = table_cells(texts[8], "联合反演参数总数")[col].split("（")[0]
        checked.append({"chapter": 8, "row": "联合反演参数总数", **rounded_value(token, sum(p.numel() for p in model.parameters()))})
    metrics["pinn"] = pin_metrics

    # Bind all six real-data curves to HDF5, cold models and unchanged r1 weights.
    local_root = roots["wp5-local-thermal-r2"]
    curves = np.load(local_root/"curves.npz", allow_pickle=False)
    local_metrics = {}
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for compact in summaries["wp5-local-thermal-r2"]["results"]:
            tag = compact["tag"]
            record = read(local_root/(tag+".json"))
            if any(record[key] != value for key, value in compact.items()):
                raise ValueError("local compact/full record mismatch")
            if sha(local_root/(tag+".pt")) != sha(roots["wp5-local-thermal-r1"]/(tag+".pt")):
                raise ValueError("neural checkpoint changed during classical repair")
            row = table.loc[table.case_id == record["case_id"]].iloc[0]
            if row.condition_id not in uq_protocol["condition_splits"]["train"]:
                raise ValueError("inverse example selected outside training set")
            stage = "stage2_hold_exchange" if record["stage"] == 2 else "stage4_cooldown_exchange"
            group = handle[row.hdf5_group+"/cycles/cycle_001/"+stage]
            values = curves[tag]
            for col, expected in enumerate((np.asarray(group["time_local_s"]).ravel(),
                    np.asarray(group["sma_strip_T_K_t"]).mean(1), np.asarray(group["htf_internal_T_K_t"]).mean(1))):
                np.testing.assert_array_equal(values[:, col], expected)
            payload = torch.load(local_root/(tag+".pt"), weights_only=True)
            model = ThermalNodePINN()
            model.load_state_dict(payload["state_dict"])
            norm = payload["normalization"]
            indices = [0, 5, 11, 17, 23, 29, 35, 39]
            np.testing.assert_equal(norm["temperature_scale_K"], max(float(np.max(np.abs(values[indices, 1]-values[0, 1]))), 1.))
            tau = (values[:, 0]-values[0, 0])/norm["duration_s"]
            with torch.no_grad():
                replay = norm["initial_K"]+norm["temperature_scale_K"]*model(torch.tensor(tau[:, None], dtype=torch.float64)).numpy()[:, 0]
            np.testing.assert_array_equal(replay, values[:, 3])
            classical = ode_prediction(values[:, 0], values[:, 2], values[0, 1], record["classical_rate_per_s"])
            np.testing.assert_array_equal(classical, values[:, 4])
            numbers = [float(torch.exp(model.log_rate).detach()), record["classical_rate_per_s"],
                float(rmse(values[40:, 3], values[40:, 1])), float(rmse(values[40:, 4], values[40:, 1]))]
            for col, value in enumerate(numbers):
                cell(8, tag.replace("_", " "), col, value)
            local_metrics[tag] = numbers
    metrics["local_inverse"] = local_metrics

    # UQ arithmetic is independently rebuilt, not delegated to calibrated_intervals.
    uqroot = roots["wp6-uq-r1"]
    uqdata = np.load(uqroot/"predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(uqdata["targets"], y)
    np.testing.assert_array_equal(uqdata["case_ids"], table.case_id.to_numpy(str))
    ensemble = LearningEnsemble.load(uqroot/"ensemble.pt")
    members = ensemble.predict_members(x)
    np.testing.assert_array_equal(members, uqdata["member_predictions"])
    mean, std = members.mean(0), members.std(0, ddof=1)
    scale = y[masks["train"]].std(0)
    np.testing.assert_array_equal(scale, ensemble.y_scaler.scale_)
    labels = ["冷量 / J", "泵功 / J", "COP", "SCP / W·kg⁻¹", "温差 / K", "COP 分母功 / J"]
    errors = rmse(mean[masks["test"]], y[masks["test"]])
    for j, label in enumerate(labels):
        corr = spearmanr(np.abs(mean[masks["test"], j]-y[masks["test"], j]), std[masks["test"], j]).statistic
        cell(9, label, 0, errors[j])
        cell(9, label, 1, corr)
    cal, test = masks["calibration"], masks["test"]
    scores = np.max(np.abs(y[cal]-mean[cal])/np.maximum(std[cal], .01*scale), axis=1)
    rank90, q = conformal_order(scores, .9)
    rank95, q95 = conformal_order(scores, .95)
    if (rank90, rank95, q95) != (13, 14, None):
        raise ValueError("conformal ranks changed")
    widths = q*np.maximum(std[test], .01*scale)
    np.testing.assert_array_equal(widths, uqdata["width_0.9"])
    if not (np.abs(y[test]-mean[test]) <= widths).all():
        raise ValueError("documented simultaneous coverage changed")
    inline(9, r"k=13，q=([\d.]+)", q)
    inline(9, r"COP 平均区间全宽为 ([\d.]+)", (2*widths).mean(0)[2])
    inline(9, r"冷量为 ([\d.]+) J", (2*widths).mean(0)[0])
    best = min(float(rmse(m[test, 0], y[test, 0])) for m in members)
    inline(9, r"最好单成员 RMSE 约 ([\d.]+) J", best)
    uncertainty = np.mean(std[test]/scale, axis=1)
    point_error = np.mean(((mean[test]-y[test])/scale)**2, axis=1)
    order = np.argsort(uncertainty, kind="stable")
    risk = np.array([point_error[order[:n]].mean() for n in range(1, 14)])
    inline(9, r"诊断误差约 ([\d.]+)", risk[2])
    inline(9, r"全保留时约 ([\d.]+)", risk[-1])
    metrics["uq"] = {"rmse": errors.tolist(), "q90": q, "rank95": rank95, "risk_coverage": risk.tolist()}

    detector = LearningEnsemble.load(uqroot/"material_ood_ensemble.pt")
    ood = (table.material_id == "recAs").to_numpy()
    evaluated = test | ood
    om, os = detector.predict(x[evaluated])
    np.testing.assert_array_equal(om, uqdata["ood_prediction"])
    np.testing.assert_array_equal(os, uqdata["ood_std"])
    np.testing.assert_array_equal(table.case_id[evaluated].to_numpy(str), uqdata["ood_case_ids"])
    labels_ood = ood[evaluated]
    if len(detector.training_ids) != 43 or int(labels_ood.sum()) != 7 or int((~labels_ood).sum()) != 13:
        raise ValueError("material detection population changed")
    disagreement = np.mean(os/detector.y_scaler.scale_, axis=1)
    auc = np.mean([float(a > b)+.5*float(a == b) for a in disagreement[labels_ood] for b in disagreement[~labels_ood]])
    if auc != 1.:
        raise ValueError("documented OOD detection AUC changed")
    iderror = rmse(om[~labels_ood], y[evaluated][~labels_ood])
    ooderror = rmse(om[labels_ood], y[evaluated][labels_ood])
    for j, pattern in ((0, r"冷量 RMSE 约 ([\d.]+)/[\d.]+ J"),
                       (2, r"COP 约 ([\d.]+)/[\d.]+"), (3, r"SCP 约 ([\d.]+)/[\d.]+ W/kg")):
        inline(9, pattern, iderror[j])
        other_pattern = pattern.replace("([\\d.]+)/[\\d.]+", "[\\d.]+/([\\d.]+)")
        inline(9, other_pattern, ooderror[j])
    metrics["uq"]["material_detection_auc"] = float(auc)
    metrics["uq"]["material_id_rmse"] = iderror.tolist()
    metrics["uq"]["material_ood_rmse"] = ooderror.tolist()

    # Nine saved learning curves, with final networks replayed and initial equality.
    active = roots["wp6-active-r1"]
    protocol = read(active/"protocol.json")
    if protocol["ensemble_implementation_sha256"] != sha(Path(source_map["wp6-uq-r1"])):
        raise ValueError("active ensemble implementation changed")
    pool = set(table.case_id[masks["train"]])
    if set(protocol["pool_case_ids"]) != pool or protocol["test_case_ids"] != table.case_id[test].tolist():
        raise ValueError("active membership/order changed")
    learning, initial = {}, {}
    for strategy in ("random", "uncertainty", "composite"):
        seed_errors = []
        for seed in (701, 702, 703):
            root = active/f"s{seed}_{strategy}"
            saved = np.load(root/"test_predictions.npz", allow_pickle=False)
            report = read(root/"report.json")
            np.testing.assert_array_equal(saved["targets"], y[test])
            if seed in initial:
                np.testing.assert_array_equal(initial[seed], saved["predictions"][0])
            initial[seed] = saved["predictions"][0]
            final = LearningEnsemble.load(root/"final_ensemble.pt")
            np.testing.assert_array_equal(final.predict(x[test])[0], saved["predictions"][-1])
            queried = [case for event in report["oracle_queries"] for case in event["case_ids"]]
            if len(queried) != 22 or len(set(queried)) != 22 or not set(queried) <= pool or list(final.training_ids) != queried:
                raise ValueError("query budget/membership mismatch")
            for n in range(2):
                if read(root/f"round_{n}_prequery.json") != report["history"][n]:
                    raise ValueError("prequery history mismatch")
            errors = np.stack([rmse(p, y[test]) for p in saved["predictions"]])
            np.testing.assert_allclose(errors, [r["test_target_rmse"] for r in report["history"]], rtol=1e-12, atol=0)
            seed_errors.append(errors)
        aggregate = np.asarray(seed_errors)
        avg, sd = aggregate.mean(0), aggregate.std(0, ddof=1)
        stored = summaries["wp6-active-r1"]["learning_curves"][strategy]
        np.testing.assert_allclose(avg, stored["seed_mean_rmse"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(sd, stored["seed_std_rmse"], rtol=1e-12, atol=1e-15)
        for j, label in enumerate(labels[:-1]+["分母功 / J"]):
            cell(10, label, ("random", "uncertainty", "composite").index(strategy), avg[-1, j])
        learning[strategy] = {"mean": avg.tolist(), "sample_std": sd.tolist()}
    metrics["active_learning"] = learning

    # Design tables refer to selected outcomes only. Existing audit separately
    # verified all25 raw archives; here reopen selected3 and recompute joint gain.
    design = roots["wp6-design-r1"]
    dp = read(design/"protocol.json")
    proposal = read(design/"proposal_before_reveal.json")
    verification = read(design/"candidate_verification.json")
    reference = read(design/"revealed22_mechanical_reference.json")
    final = LearningEnsemble.load(active/"s701_composite/final_ensemble.pt")
    if sha(active/"s701_composite/final_ensemble.pt") != dp["checkpoint_sha256"]:
        raise ValueError("design checkpoint binding changed")
    if dp["active_summary_sha256"] != sha(active/"summary.json"):
        raise ValueError("design prior evidence changed")
    lookup = table.set_index("case_id", drop=False)
    candidates = sorted(pool-set(final.training_ids))
    if candidates != proposal["candidate_ids"] or len(candidates) != 15:
        raise ValueError("candidate membership mismatch")
    cm, cs = final.predict(x[cal])
    ym = final.y_scaler.scale_
    _, qdesign = conformal_order(np.max(np.abs(y[cal]-cm)/np.maximum(cs, .01*ym), axis=1), .9)
    dm, ds = final.predict(design_matrix(lookup.loc[candidates]))
    lower = dm-qdesign*np.maximum(ds, .01*ym)
    np.testing.assert_allclose(lower, proposal["lower"], rtol=1e-12, atol=1e-12)
    cutoff = np.quantile(np.mean(cs/ym, axis=1), .9)
    valid = (lower[:, [0, 2, 3, 5]] > 0).all(1) & (dm[:, 1] >= 0) & (np.mean(ds/ym, axis=1) <= cutoff)
    front = [i for i in range(15) if valid[i] and not any(valid[j] and
        np.all(lower[j, [2, 3]] >= lower[i, [2, 3]]) and np.any(lower[j, [2, 3]] > lower[i, [2, 3]]) for j in range(15))]
    known = lookup.loc[list(final.training_ids), list(ENSEMBLE_TARGETS)].to_numpy(float)
    points = known[:, [2, 3]]/ym[[2, 3]]
    ref = points.min(0)-1
    gains = [max(0., area(np.vstack((points, p)), ref)-area(points, ref)) for p in lower[:, [2, 3]]/ym[[2, 3]]]
    tie = (lower[:, [2, 3]]/ym[[2, 3]]).sum(1)
    chosen = [candidates[i] for i in sorted(front, key=lambda i: (-gains[i], -tie[i], candidates[i]))[:3]]
    if not valid.all() or len(front) != 4 or any(gains) or chosen != proposal["selected_ids"]:
        raise ValueError("design selection/negative result changed")
    inline(11, r"本次 q=([\d.]+)", qdesign)
    inline(11, r"分歧阈值为 ([\d.]+)", cutoff)
    for record in verification["observations"]:
        row = lookup.loc[record["case_id"]]
        source = Path(row.source_artifact).parent/"run_result.mat"
        if sha(source) != row.source_run_result_sha256 or record["case_id"] not in chosen:
            raise ValueError("design raw source/membership mismatch")
        with h5py.File(source, "r") as handle:
            stress = max(float(np.asarray(handle[f"run_result/cycle_results/{s}/metrics/global_xi_time_sigma_eq_max_MPa"]).item()) for s in ("load_result", "unload_result"))
            wr = float(np.asarray(handle["run_result/cycle_diagnostics/center_deflection_peak_over_radius"]).item())
        numbers = [row.COP, row.SCP_W_per_kg, stress, wr]
        label = record["case_id"].split("_")[2]
        for col, value in enumerate(numbers):
            cell(11, label, col, value)
        if record["verdict"]["engineering_safety"] != "unknown":
            raise ValueError("engineering evidence overstated")
    added = np.array([record["targets"] for record in verification["observations"]])
    gain = area(np.vstack((points, added[:, [2, 3]]/ym[[2, 3]])), ref)-area(points, ref)
    inline(11, r"归一化超体积改善为 \*\*([\d.]+)\*\*", gain)
    for field, pattern in (("sigma_eq_peak_MPa", r"参考最大应力为 ([\d.]+) MPa"),
                           ("center_deflection_peak_over_radius", r"参考最大挠度比为\s+([\d.]+)")):
        maximum = max(record[field] for record in reference["observations"])
        np.testing.assert_equal(maximum, reference["reference"][field])
        inline(11, pattern, maximum)
    metrics["design"] = {"q": qdesign, "cutoff": float(cutoff), "chosen": chosen, "joint_hv_gain": gain}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"numeric_cells.json", checked)
    write_json(out/"independent_metrics.json", metrics)
    result = {"status": "passed_chapters08_11_content_review", "whole_report_complete": False,
        "numeric_cells_checked": len(checked), "cold_pinn_models": 6, "cold_local_inverse_models": 6,
        "cold_uq_members": 10, "cold_active_final_members": 27, "learning_curves_recomputed": 9,
        "raw_local_curves_checked": 6, "raw_design_archives_checked": 3,
        "fresh_training_calls": 0, "fresh_solver_calls": 0,
        "scope": "author full-text/source walkthrough plus array/checkpoint/table arithmetic; no independent reader mastery claim",
        "readability_repairs": ["L-BFGS iteration budget distinguished from actual closure count",
            "Adam-only clipping explicit", "local PINN exact input transform, scale, parameter counts and optimizer contract",
            "acquisition tied-rank percentile formula and hand-worked example"],
        "reviewed_limits": ["manufactured vs effective physical parameter", "known future forcing is conditional reconstruction",
            "ensemble spread is not noise decomposition", "95percent finite rank absent", "OOD detection is not accurate prediction",
            "archived reveal is not new solver sampling", "test outcomes never determine acquisition",
            "zero pessimistic improvement retained", "postselection coverage not guaranteed", "reference maximum not engineering limit"],
        "chapter_sha256": {p.name: sha(p) for p in paths},
        "source_sha256": {p: sha(Path(p)) for p in source_map.values()},
        "upstream_summary_sha256": {name: sha(root/"summary.json") for name, root in roots.items()},
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return {k: result[k] for k in ("status", "whole_report_complete", "numeric_cells_checked", "fresh_training_calls", "fresh_solver_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset-root", "campaign-root", "report-root", "output-root"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.report_root, args.output_root), indent=2))
