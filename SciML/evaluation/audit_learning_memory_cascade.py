"""Independent assembly and saved-array audit of the directional memory repair."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_boundary_state import BoundaryStateModel, autonomous
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator, corpus
from SciML.models.v12.run_learning_existing import sha, write_json


def assemble_inputs(states, requests, ids, stage, memory=None):
    """Independent explicit layout; do not use production input/memory helpers."""
    branches, initials = [], []
    for i, case in enumerate(ids):
        state = states[i, 0 if stage == 2 else 2]
        radial = state[102:153]
        solid = np.concatenate((radial[1:][::-1], radial))
        if memory is not None:
            solid = solid+memory[i]
        fluid = state[204:305] if stage == 2 else state[305:406]
        side = state[406:414].copy()
        side[6 if stage == 2 else 5] = 293.
        spec = requests[case]["case_spec"]
        control = [spec[key] for key in ("radius_m", "thickness_m", "pressure_Pa",
                                        "flush_volume_target_ratio", "exchange_duration_s")]
        branch = np.concatenate((solid-solid.mean(), fluid-fluid.mean(),
            [solid.mean(), fluid.mean(), stage == 2, stage == 4,
             spec["material_id"] == "recA", spec["material_id"] == "recAs"], control, side))
        branches.append(branch)
        initials.append(np.stack((solid, fluid)))
    return np.stack(branches), np.stack(initials)


def run(dataset, campaign, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve prior memory audit")
    roots = ["wp3-boundary-r2", "wp4-deeponet-r1", "wp4-operator-audit-r1", "wp4-memory-cascade-r2"]
    summaries = {name: verify_artifacts(campaign/name) for name in roots}
    root = campaign/"wp4-memory-cascade-r2"
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["implementation_sha256"] != sha(Path("SciML/models/v12/learning_operator_cascade.py")):
        raise ValueError("changed production implementation")
    old_protocol = json.loads((campaign/"wp4-memory-cascade-r1/protocol.json").read_text())
    if old_protocol["implementation_sha256"] != sha(campaign/"source_snapshots/learning_operator_cascade_r1.py"):
        raise ValueError("failed source snapshot does not match original protocol")
    samples, branch, initial, truth, quality = corpus(dataset)
    original = json.loads(original_protocol.read_text())
    if quality["source_content_sha256"] != protocol["dataset_source_content_sha256"] or quality["source_content_sha256"] != original["dataset_source_content_sha256"]:
        raise ValueError("source/split identity mismatch")
    table = pd.read_parquet(dataset/"case_index.parquet").set_index("case_id")
    ids = [s.case_id for s in samples[::2]]
    requests = {}
    for case in ids:
        path = Path(table.loc[case].source_artifact).parent/"sciml_case_request.json"
        if sha(path) != table.loc[case].source_request_sha256:
            raise ValueError("request source mismatch")
        requests[case] = json.loads(path.read_text())
    data = np.load(root/"predictions.npz", allow_pickle=False)
    prediction, memory = data["prediction"], data["predicted_memory"]
    np.testing.assert_array_equal(data["sample_ids"], [s.sample_id for s in samples])
    expected_memory = (prediction[::2, 0, -1]-prediction[::2, 0, -1, ::-1])/2
    np.testing.assert_array_equal(memory, expected_memory)
    np.testing.assert_array_equal(memory+memory[:, ::-1], np.zeros_like(memory))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        boundary = BoundaryStateModel.load(campaign/"wp3-boundary-r2/s20260910/model.pt")
        operator = CoordinateOperator.load(campaign/"wp4-deeponet-r1/model.pt")
        states = autonomous(boundary, requests, ids).reshape(len(ids), 4, 414)
        b2, i2 = assemble_inputs(states, requests, ids, 2)
        p2 = operator.predict(b2, i2)
        independent_memory = (p2[:, 0, -1]-p2[:, 0, -1, ::-1])/2
        b4, i4 = assemble_inputs(states, requests, ids, 4, independent_memory)
        p4 = operator.predict(b4, i4)
        np.testing.assert_array_equal(p2, prediction[::2])
        np.testing.assert_array_equal(p4, prediction[1::2])
        np.testing.assert_array_equal(p4[:, :, 0], i4)
    finally:
        torch.set_num_threads(threads)
    source_checks = []
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for i, case in enumerate(ids):
            cycle = handle[str(table.loc[case].hdf5_group)+"/cycles/cycle_001"]
            s2 = np.asarray(cycle["stage2_hold_exchange/sma_strip_T_K_t"])[-1]
            s4 = np.asarray(cycle["stage4_cooldown_exchange/sma_strip_T_K_t"])[0]
            radial = np.asarray(cycle["stage4_cooldown_exchange/sma_T_in_51_K"])
            actual_odd = (s2-s2[::-1])/2
            mirror = np.concatenate((radial[1:][::-1], radial))
            # Separate the even and odd projections of the ACTUAL next initial field.
            np.testing.assert_allclose((s4+s4[::-1])/2, mirror, atol=1e-10, rtol=0)
            np.testing.assert_allclose((s4-s4[::-1])/2, actual_odd, atol=1e-10, rtol=0)
            source_checks.append({"case_id": case,
                "identity_max_error_K": float(np.max(np.abs(mirror+actual_odd-s4))),
                "predicted_memory_rmse_K": float(np.sqrt(np.mean((memory[i]-actual_odd)**2)))})
    test = np.array([s.condition_id in original["condition_splits"]["test"] for s in samples])
    stages = np.array([s.stage_index for s in samples])
    metrics = {}
    for part, conditions in original["condition_splits"].items():
        metrics[part] = {}
        for stage in (2, 4):
            selected = np.array([s.condition_id in conditions for s in samples]) & (stages == stage)
            metrics[part][str(stage)] = {}
            for j, field in enumerate(("sma", "htf")):
                error = prediction[selected, j]-truth[selected, j]
                values = {"future_rmse_K": float(np.sqrt(np.mean(error[:, 1:]**2))),
                          "initial_max_abs_K": float(np.abs(error[:, 0]).max()),
                          "future_max_abs_K": float(np.abs(error[:, 1:]).max())}
                if values != summaries[roots[-1]]["scores"][part][str(stage)][field]:
                    raise ValueError("saved metrics disagree with arrays")
                metrics[part][str(stage)][field] = values
    prior = np.load(campaign/"wp4-operator-audit-r1/comparison_predictions.npz", allow_pickle=False)
    direct = np.load(campaign/"wp4-deeponet-r1/predictions.npz", allow_pickle=False)["prediction"]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"source_checks.json", source_checks)
    figure(samples, truth, prediction, prior["request_boundary_cascade"], direct, test, out)
    result = {"status": "passed_independent_memory_cascade_audit", "learning_only": True,
        "fresh_solver_calls": 0, "independent_input_assembly_predictions_exact": True,
        "predicted_endpoint_memory_exact": True, "source_identity_cases": len(ids),
        "source_identity_max_error_K": max(v["identity_max_error_K"] for v in source_checks),
        "failed_source_snapshot_verified": True, "test_scores": metrics["test"],
        "scope": "one-cycle request-only boundary-plus-field cascade, not full closed-loop energy or multicycle evidence",
        "implementation_sha256": sha(Path(__file__)),
        "upstream_summary_hashes": {name: sha(campaign/name/"summary.json") for name in roots},
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


def figure(samples, truth, corrected, old, direct, test, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    stages = np.array([s.stage_index for s in samples])
    for label, values in (("Observed-start DeepONet", direct), ("Request cascade: mirror only", old),
                          ("Request cascade: predicted memory", corrected)):
        rmse = [np.sqrt(np.mean((values[test & (stages == stage), j, 1:]-truth[test & (stages == stage), j, 1:])**2))
                for stage in (2, 4) for j in (0, 1)]
        axes[0].plot(range(4), rmse, marker="o", label=label)
    axes[0].set(xticks=range(4), xticklabels=["S2 SMA", "S2 HTF", "S4 SMA", "S4 HTF"],
                ylabel="Future-field RMSE (K)", title="Test cases: input regimes differ")
    axes[0].legend(fontsize=7)
    row = next(i for i, s in enumerate(samples) if test[i] and s.stage_index == 4)
    x = np.linspace(-1, 1, 101)
    for name, values in (("Truth", truth), ("Mirror only", old), ("Predicted memory", corrected)):
        axes[1].plot(x, values[row, 0, 0], label=name)
        axes[2].plot(x, values[row, 0, -1], label=name)
    for ax, title in zip(axes[1:], ("Stage4 SMA initial field", "Stage4 SMA final field")):
        ax.set(xlabel="Strip coordinate / radius", ylabel="Temperature (K)", title=title)
        ax.legend(fontsize=8)
    fig.suptitle("Spatial-memory repair | first test stage4 by fixed corpus order", fontsize=11)
    fig.savefig(out/"memory_repair.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.original_protocol, args.output_root), indent=2))
