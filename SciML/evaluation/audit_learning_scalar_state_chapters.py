"""Source/array-bound content review of learning chapters04 and05; no refitting.

Recomputes displayed error tables from saved predictions, validates dimensions,
training scales and curves, and binds an explicit author readability walkthrough.
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

import numpy as np

from .audit_learning_operator_chapter import table_cells
from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_ensemble import LearningEnsemble, design_matrix
from SciML.models.v12.learning_scalar_benchmarks import classical_models
from SciML.models.v12.learning_boundary_state import build_boundaries, boundary_mask, BoundaryStateModel
from SciML.models.v11.state_space import LAYOUT
from SciML.models.v12.run_learning_existing import sha, write_json, TARGETS, validate_dataset


def rounded_value(text, expected):
    """Allow half the last displayed decimal place, plus floating representation.

    This is a publication-rounding check, not a model or scientific tolerance.
    """
    try:
        decimal = Decimal(text.strip())
    except InvalidOperation as error:
        raise ValueError("not a plain displayed number: "+text) from error
    if not decimal.is_finite() or not np.isfinite(expected):
        raise ValueError("display/source must both be finite")
    shown = float(decimal)
    tolerance = .5*10.**decimal.as_tuple().exponent + 4*np.spacing(abs(float(expected)))
    if abs(shown-float(expected)) > tolerance:
        raise ValueError(f"displayed {text} does not round source {expected}; rounding bound {tolerance}")
    return {"displayed": text, "source_value": float(expected), "rounding_bound": float(tolerance)}


def run(dataset, campaign, chapters, out):
    if out.exists():
        raise FileExistsError("preserve previous scalar/state chapter review")
    ch4, ch5 = (chapters/name for name in ("04_WP2_标量基线与神经训练.md", "05_WP3_状态接口与自主推演.md"))
    text4, text5 = ch4.read_text(encoding="utf-8"), ch5.read_text(encoding="utf-8")
    scalar_root, boundary_root = campaign/"wp2-strong-r1", campaign/"wp3-boundary-r2"
    scalar, boundary = verify_artifacts(scalar_root), verify_artifacts(boundary_root)
    protocols = [json.loads((root/"protocol.json").read_text(encoding="utf-8")) for root in (scalar_root, boundary_root)]
    source_paths = [Path("SciML/models/v12/learning_scalar_benchmarks.py"), Path("SciML/models/v12/learning_boundary_state.py")]
    for protocol, path in zip(protocols, source_paths):
        if sha(path) != protocol["implementation_sha256"]:
            raise ValueError("current model implementation differs from trained protocol")
    table, quality, _ = validate_dataset(dataset)
    for protocol in protocols:
        if protocol["dataset_source_content_sha256"] != quality["source_content_sha256"]:
            raise ValueError("dataset binding differs")
    if protocols[0]["condition_splits"] != protocols[1]["condition_splits"]:
        raise ValueError("WP2/WP3 split mismatch")
    masks = {name: table.condition_id.isin(ids).to_numpy() for name, ids in protocols[0]["condition_splits"].items()}
    if {name: int(mask.sum()) for name, mask in masks.items()} != {"train": 37, "calibration": 13, "test": 13}:
        raise ValueError("documented split counts no longer hold")
    targets = table[list(TARGETS)].to_numpy(float)
    train_scale = targets[masks["train"]].std(axis=0)
    checked = []

    def cell(chapter, row, col, value):
        item = rounded_value(table_cells(chapter, row)[col], value)
        checked.append({"row": row, "column": col, **item})

    for label, value in zip(("Qcool", "Wpump", "COP", "SCP", "deltaT"), train_scale):
        match = re.search(r"^"+label+r":\s*([\d.eE+-]+)", text4, re.M)
        if match is None:
            raise ValueError("missing target scale: "+label)
        checked.append({"row": "scale:"+label, **rounded_value(match.group(1), value)})
    arrays, reports = [], []
    cold_members_checked = 0
    for seed in protocols[0]["seeds"]:
        root = scalar_root/f"s{seed}"
        report = json.loads((root/"report.json").read_text(encoding="utf-8"))
        data = np.load(root/"predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(data["case_ids"], table.case_id.to_numpy(str))
        np.testing.assert_array_equal(data["targets"], targets)
        ensemble = LearningEnsemble.load(root/"ensemble5.pt")
        np.testing.assert_array_equal(ensemble.predict_members(design_matrix(table)), data["member_predictions"])
        cold_members_checked += len(ensemble.networks)
        if sum(p.numel() for p in ensemble.networks[0].parameters()) != 1477:
            raise ValueError("MLP parameter count mismatch")
        reports.append(report)
        arrays.append({name: data[name].copy() for name in scalar["comparison"]})
    scalar_errors = {}
    names = {"训练均值": "train_mean", "Ridge": "ridge", "Random Forest": "random_forest",
        "HistGB": "hist_gradient_boosting", "单 MLP": "mlp", "Ensemble5": "ensemble5", "原单位目标 MLP 消融": "mlp_raw_targets"}
    for label, key in names.items():
        errors = np.array([np.sqrt(np.mean((data[key][masks["test"]]-targets[masks["test"]])**2, axis=0)) for data in arrays])
        mean, std = errors.mean(0), errors.std(0, ddof=1)
        np.testing.assert_allclose(mean, scalar["comparison"][key]["seed_mean_test_rmse"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(std, scalar["comparison"][key]["seed_std_test_rmse"], rtol=1e-12, atol=1e-14)
        for i, value in enumerate(mean):
            cell(text4, label, i, value)
        scalar_errors[key] = {"mean": mean.tolist(), "sample_std": std.tolist()}
    for row in reports[0]["normalized_history"][0]["curve"]:
        for col, key in enumerate(("train_standardized_mse", "calibration_standardized_mse")):
            cell(text4, str(row["updates_completed"]), col, row[key])
    for label, count in (("Linear(7,32)", 7*32+32), ("Linear(32,32)", 32*32+32), ("Linear(32,5)", 32*5+5), ("合计", 1477)):
        cell(text4, label, 2, count)
    classical = classical_models(20260910)
    rf, gb = classical["random_forest"], classical["hist_gradient_boosting"].estimator
    configuration = {"ridge_alpha": classical["ridge"].alpha, "RF_trees": rf.n_estimators,
        "RF_min_leaf": rf.min_samples_leaf, "HistGB_iterations": gb.max_iter, "HistGB_min_leaf": gb.min_samples_leaf,
        "HistGB_max_leaf_nodes": gb.max_leaf_nodes, "HistGB_learning_rate": gb.learning_rate,
        "HistGB_L2": gb.l2_regularization, "HistGB_early_stopping": gb.early_stopping}
    if configuration != {"ridge_alpha": 1., "RF_trees": 200, "RF_min_leaf": 2, "HistGB_iterations": 200,
        "HistGB_min_leaf": 3, "HistGB_max_leaf_nodes": 7, "HistGB_learning_rate": .05, "HistGB_L2": .1, "HistGB_early_stopping": False}:
        raise ValueError("described classical configuration differs")

    # Independently map rows by sample ID and slice the saved state arrays;
    # do not assume flatten order itself establishes stage ownership.
    _, _, records, _, _ = build_boundaries(dataset)
    test_records = [record for record in records if record.condition_id in protocols[1]["condition_splits"]["test"]]
    lookup = {r.sample_id: r for r in test_records}
    state_errors = {name: [] for name in ("autonomous_affine", "autonomous_neural", "teacher_neural", "teacher_affine")}
    state_shape_counts = {}
    for seed in protocols[1]["seeds"]:
        root = boundary_root/f"s{seed}"
        data = np.load(root/"predictions.npz", allow_pickle=False)
        ids = data["test_sample_ids"].tolist()
        if len(set(ids)) != len(ids) or set(ids) != set(lookup):
            raise ValueError("state sample membership differs")
        rows = [lookup[id] for id in ids]
        truth = np.stack([r.state_out for r in rows])
        np.testing.assert_array_equal(data["test_truth"], truth)
        model = BoundaryStateModel.load(root/"model.pt")
        for stage in range(1, 5):
            active = int(boundary_mask(stage).sum())
            state_shape_counts[str(stage)] = active
            if model.networks[stage][-1].out_features != active:
                raise ValueError("state scatter/network width mismatch")
        for name in state_errors:
            values = []
            for stage in range(1, 5):
                mask = np.array([r.stage_index == stage for r in rows])
                error = data["test_"+name][mask, LAYOUT.sma_T]-truth[mask, LAYOUT.sma_T]
                values.append(float(np.sqrt(np.mean(error**2))))
            state_errors[name].append(values)
    performance = text5.split("## 5.9", 1)[1].split("## 5.10", 1)[0]
    for stage in range(1, 5):
        for col, name in enumerate(("autonomous_affine", "autonomous_neural", "teacher_neural")):
            value = np.asarray(state_errors[name]).mean(0)[stage-1]
            cell(performance, f"B{stage}→B{stage+1}", col, value)
        # Same label occurs in two tables: scope by section, never guess a row.
    ownership = text5.split("### 谁能在每个复合过渡中改变", 1)[1].split("## 5.6", 1)[0]
    for stage in range(1, 5):
        cell(ownership, f"B{stage}→B{stage+1}", 1, state_shape_counts[str(stage)])
    for label, count in (("0:51", 51), ("51:102", 51), ("102:153", 51), ("153:204", 51),
                         ("204:305", 101), ("305:406", 101), ("406:414", 8)):
        cell(text5, label, 1, count)
    contract = verify_artifacts(campaign/"wp3-input-contract-r1")
    gaps = contract["stage_start_interface_gap_maxima"]
    for label, value in (("进入阶段 3 的径向位移", gaps["3"]["u_m_t"]), ("进入阶段 3 的挠度", gaps["3"]["w_m_t"]),
        ("进入阶段 3 的相变量", gaps["3"]["xi_t"]), ("进入阶段 4 的 hot sink / hot buffer", gaps["4"]["hot_sink_temp_in"]),
        ("进入阶段 4 的 cold sink pipe 辅助量", gaps["4"]["simscape_cold_sink_pipe_T_state_in_K"])):
        token = table_cells(text5, label)[0].split()[0]
        checked.append({"row": label, **rounded_value(token, value)})
    negative = np.mean(state_errors["teacher_neural"], axis=0)[3] > np.mean(state_errors["teacher_affine"], axis=0)[3]
    if not negative:
        raise ValueError("documented negative teacher-forcing comparison changed")
    old = json.loads((campaign/"wp3-boundary-r1/protocol.json").read_text(encoding="utf-8"))
    if sha(campaign/"source_snapshots/learning_boundary_state_r1.py") != old["implementation_sha256"]:
        raise ValueError("failed residual-scale source snapshot changed")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"numeric_cells.json", checked)
    write_json(out/"independent_metrics.json", {"scalar": scalar_errors, "state_sma_by_seed": state_errors,
        "active_state_widths": state_shape_counts, "classical_configuration": configuration})
    report = {"status": "passed_chapters04_05_content_review", "whole_report_complete": False,
        "numeric_cells_checked": len(checked), "scalar_prediction_families": 7, "scalar_seeds": 3,
        "cold_ensemble_members_checked": cold_members_checked, "state_test_transitions_per_seed": len(test_records),
        "fresh_training_calls": 0, "fresh_solver_calls": 0,
        "review_scope": "author full-text/code walkthrough; displayed tables recomputed/bound to saved arrays, training curves/scales/configuration; not independent human mastery certification",
        "readability_fixes": ["calibration exceeds training only from recorded update50, not initialpoint",
            "single-cycle statement scoped to this chapter, historicalmulticycle pointed to07C",
            "inactive channels carried before projection; out-of-range intervention may be projected",
            "structured materialOOD distinguished from ensemble detection chapter"],
        "walkthrough": ["physicalquestion before X/Y", "units/shape/scaler before loss", "each network layer and parametercount",
            "optimizer pseudocode linked toactualfit", "five-versus-six-output confound explicit", "curves not stopping criterion",
            "newboundary semantics separate oldcanonicalforecast", "Ridge not physicalsolver", "permission versus Markovclosure distinct",
            "failuremechanism units andsource preserved", "teacherforcing versus ownprediction explicit", "negativecomparisons retained",
            "guided-to-independent exercises provide answers andcriteria"],
        "chapter_sha256": {p.name: sha(p) for p in (ch4, ch5)}, "implementation_sha256": sha(Path(__file__)),
        "source_sha256": {str(p): sha(p) for p in source_paths},
        "upstream_summary_sha256": {root.name: sha(root/"summary.json") for root in (scalar_root, boundary_root)},
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", report)
    return {key: report[key] for key in ("status", "whole_report_complete", "numeric_cells_checked", "cold_ensemble_members_checked", "state_test_transitions_per_seed", "fresh_training_calls")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.report_root, args.output_root), indent=2))
