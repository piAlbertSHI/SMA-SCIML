"""Audit request-only inference, measured boundary ownership and fresh retraining."""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v11.state_space import LAYOUT
from SciML.models.v12.learning_boundary_state import build_boundaries, BoundaryStateModel, autonomous, boundary_mask
from SciML.models.v12.run_learning_existing import sha, write_json


def run(dataset, campaign, out):
    if out.exists():
        raise FileExistsError("preserve boundary audit")
    root = campaign/"wp3-boundary-r2"
    summary = verify_artifacts(root)
    old = verify_artifacts(campaign/"wp3-boundary-r1")
    old_protocol = json.loads((campaign/"wp3-boundary-r1/protocol.json").read_text())
    snapshot = campaign/"source_snapshots/learning_boundary_state_r1.py"
    if sha(snapshot) != old_protocol["implementation_sha256"]:
        raise ValueError("failed implementation snapshot not exact")
    protocol = json.loads((root/"protocol.json").read_text())
    if sha(Path("SciML/models/v12/learning_boundary_state.py")) != protocol["implementation_sha256"]:
        raise ValueError("boundary implementation changed")
    table, quality, records, requests, checks = build_boundaries(dataset)
    if quality["source_content_sha256"] != protocol["dataset_source_content_sha256"]:
        raise ValueError("source mismatch")
    if checks != json.loads((root/"boundary_source_checks.json").read_text()):
        raise ValueError("boundary source contract changed")
    lookup = table.set_index("case_id")
    # Independently check the changed target semantics at actual HDF5 boundaries.
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for r in records:
            cycle = handle[str(lookup.loc[r.case_id].hdf5_group)+"/cycles/cycle_001"]
            if r.stage_index == 2:
                for field, slc in (("u_m_t", LAYOUT.u), ("w_m_t", LAYOUT.w), ("xi_t", LAYOUT.xi), ("T_K_t", LAYOUT.sma_T)):
                    np.testing.assert_array_equal(r.state_out[slc], np.asarray(cycle["stage3_unloading/"+field])[0])
            if r.stage_index == 3:
                np.testing.assert_array_equal(r.state_out[LAYOUT.htf_stage4_T], np.asarray(cycle["stage4_cooldown_exchange/htf_internal_T_K_t"])[0])
                for i, name in ((0, "hot_sink_temp_in"), (3, "hot_buffer_temp_in"), (4, "cold_buffer_temp_in"), (6, "stage4_path_pipe_T_state_in_K"), (7, "simscape_cold_sink_pipe_T_state_in_K")):
                    np.testing.assert_equal(r.state_out[LAYOUT.side.start+i], np.asarray(cycle["stage4_cooldown_exchange/side_state/"+name]).item())
            if r.stage_index == 1:
                np.testing.assert_array_equal(r.state_in[LAYOUT.htf_stage4_T], np.full(101, 293.))
    train = [r for r in records if r.condition_id in protocol["condition_splits"]["train"]]
    results = []
    for seed in protocol["seeds"]:
        dest = root/f"s{seed}"
        model = BoundaryStateModel.load(dest/"model.pt")
        retrained = BoundaryStateModel(seed).fit(train)
        if model.training_ids != [r.sample_id for r in train] or model.history != retrained.history:
            raise ValueError("training membership/history mismatch")
        report = json.loads((dest/"report.json").read_text())
        arrays = np.load(dest/"predictions.npz", allow_pickle=False)
        for part, conditions in protocol["condition_splits"].items():
            ids = table.loc[table.condition_id.isin(conditions), "case_id"].tolist()
            rows = [r for r in records if r.condition_id in conditions]
            np.testing.assert_array_equal(arrays[part+"_sample_ids"], [r.sample_id for r in rows])
            target = np.stack([r.state_out for r in rows])
            np.testing.assert_array_equal(arrays[part+"_truth"], target)
            prediction = autonomous(model, requests, ids)
            np.testing.assert_array_equal(prediction, autonomous(retrained, requests, ids))
            # A forecast must not inspect any observed future label attached by a caller.
            poisoned = copy.deepcopy(requests)
            for key in ids:
                poisoned[key]["future_fields"] = np.full((4, 414), np.nan)
            np.testing.assert_array_equal(prediction, autonomous(model, poisoned, ids))
            for neural in (True, False):
                name = "autonomous_neural" if neural else "autonomous_affine"
                actual = autonomous(model, requests, ids, neural)
                np.testing.assert_array_equal(actual, arrays[part+"_"+name])
                for j in range(len(ids)):
                    states = actual[j*4:(j+1)*4]
                    np.testing.assert_array_equal(states[:, LAYOUT.u.start], np.zeros(4))
                    np.testing.assert_array_equal(states[:, LAYOUT.u.stop-1], np.zeros(4))
                    np.testing.assert_array_equal(states[:, LAYOUT.w.stop-1], np.zeros(4))
                    if np.any(states[:, LAYOUT.xi] < 0) or np.any(states[:, LAYOUT.xi] > 1):
                        raise ValueError("phase feasibility violated")
                    for stage in (2, 3, 4):
                        inactive = ~boundary_mask(stage)
                        np.testing.assert_array_equal(states[stage-1, inactive], states[stage-2, inactive])
            for name, scores in report["scores"][part].items():
                prediction = arrays[part+"_"+name]
                for stage in (1, 2, 3, 4):
                    mask = np.array([r.stage_index == stage for r in rows])
                    for block, slc in (("sma_K", LAYOUT.sma_T), ("side_K", LAYOUT.side), ("xi", LAYOUT.xi),
                                       ("u_m", LAYOUT.u), ("w_m", LAYOUT.w), ("htf2_K", LAYOUT.htf_stage2_T), ("htf4_K", LAYOUT.htf_stage4_T)):
                        value = np.sqrt(np.mean((prediction[mask, slc]-target[mask, slc])**2))
                        np.testing.assert_allclose(value, scores[str(stage)][block], rtol=1e-12, atol=1e-14)
        results.append(report)
    for name, stages in summary["test_seed_mean_rmse"].items():
        for stage, blocks in stages.items():
            for block, value in blocks.items():
                np.testing.assert_equal(np.mean([r["scores"]["test"][name][stage][block] for r in results]), value)
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for name, label in (("autonomous_affine", "Request-only affine"), ("autonomous_neural", "Request-only neural"), ("teacher_neural", "Observed-input neural")):
        axes[0].plot(range(1, 5), [summary["test_seed_mean_rmse"][name][str(s)]["sma_K"] for s in (1, 2, 3, 4)], marker="o", label=label)
    axes[0].set(xlabel="Composite transition (stage + measured interface)", ylabel="Test SMA RMSE (K)", xticks=[1, 2, 3, 4], title="Causal-input single-cycle boundary forecast")
    axes[0].legend(fontsize=8)
    for old_result, label in ((old, "r1: one-unit near-zero scale bug"), (summary, "r2: zero correction + center constraint")):
        axes[1].plot(range(1, 5), [old_result["test_seed_mean_rmse"]["autonomous_neural"][str(s)]["u_m"]*1e6 for s in (1, 2, 3, 4)], marker="o", label=label)
    axes[1].set(xlabel="Composite transition", ylabel="Radial displacement RMSE (micrometers)", yscale="log", xticks=[1, 2, 3, 4], title="A unit-scale bug can corrupt rollout")
    axes[1].legend(fontsize=8)
    fig.savefig(out/"boundary_forecast.png", dpi=160)
    plt.close(fig)
    audit = {"status": "passed_boundary_learning_audit", "fresh_solver_calls": 0,
             "case_count": len(table), "transitions": len(records), "request_only_inference": True,
             "three_seed_fresh_fit_exact": True, "boundary_targets_checked_against_raw_fields": True,
             "future_payload_poison_test_invariant": True, "near_zero_residual_bug_evidence_preserved": True,
             "model_is_exact_physical_solver": False, "multicycle_validation": False,
             "test_seed_mean_rmse": summary["test_seed_mean_rmse"],
             "source_summary_sha256": sha(root/"summary.json"), "implementation_sha256": sha(Path(__file__)),
             "figure_sha256": sha(out/"boundary_forecast.png")}
    write_json(out/"summary.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.output_root), indent=2))
