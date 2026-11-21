"""Independent selection arithmetic and raw-archive audit of the finite design loop."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_ensemble import LearningEnsemble, ENSEMBLE_TARGETS, design_matrix
from SciML.models.v12.run_learning_existing import sha, validate_dataset, write_json


def area(points, reference):
    """Independent vertical-slab union area, not the production Pareto routine."""
    points = np.asarray(points)
    points = points[(points > reference).all(axis=1)]
    edges = sorted(set([float(reference[0])]+points[:, 0].tolist()))
    return sum((b-a)*max([0.]+[float(y-reference[1]) for x, y in points if x >= b])
               for a, b in zip(edges[:-1], edges[1:]))


def run(dataset, design, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve previous design audit")
    summary = verify_artifacts(design)
    protocol = json.loads((design/"protocol.json").read_text())
    proposal = json.loads((design/"proposal_before_reveal.json").read_text())
    verified = json.loads((design/"candidate_verification.json").read_text())
    reference = json.loads((design/"revealed22_mechanical_reference.json").read_text())
    table, quality, _ = validate_dataset(dataset)
    original = json.loads(original_protocol.read_text())
    assert sha(original_protocol) == protocol["original_protocol_sha256"]
    assert quality["source_content_sha256"] == protocol["dataset_source_content_sha256"]
    checkpoint = Path(protocol["checkpoint_path"])
    assert sha(checkpoint) == protocol["checkpoint_sha256"]
    model = LearningEnsemble.load(checkpoint)
    lookup = table.set_index("case_id", drop=False)
    train = table[table.condition_id.isin(original["condition_splits"]["train"])]
    cal = table[table.condition_id.isin(original["condition_splits"]["calibration"])]
    assert list(model.training_ids) == protocol["training_ids"]
    assert cal.case_id.tolist() == protocol["calibration_ids"]
    candidates = sorted(set(train.case_id)-set(model.training_ids))
    assert candidates == proposal["candidate_ids"] == protocol["candidate_ids"]
    mean, std = model.predict(design_matrix(lookup.loc[candidates]))
    cm, cs = model.predict(design_matrix(cal))
    scores = np.max(np.abs(cal[list(ENSEMBLE_TARGETS)].to_numpy()-cm)/np.maximum(cs, .01*model.y_scaler.scale_), axis=1)
    assert len(scores) == 13 and proposal["calibration_rank"] == 13
    q = float(max(scores))
    lower = mean-q*np.maximum(std, .01*model.y_scaler.scale_)
    for key, values in (("mean", mean), ("std", std), ("lower", lower)):
        np.testing.assert_allclose(proposal[key], values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q, proposal["calibration_quantile"])
    cutoff = np.quantile(np.mean(cs/model.y_scaler.scale_, axis=1), .9)
    valid = (lower[:, [0, 2, 3, 5]] > 0).all(axis=1) & (mean[:, 1] >= 0)
    valid &= np.mean(std/model.y_scaler.scale_, axis=1) <= cutoff
    front = [i for i in range(len(candidates)) if valid[i] and not any(
        valid[j] and np.all(lower[j, [2, 3]] >= lower[i, [2, 3]])
        and np.any(lower[j, [2, 3]] > lower[i, [2, 3]]) for j in range(len(candidates)))]
    known_y = lookup.loc[list(model.training_ids), list(ENSEMBLE_TARGETS)].to_numpy(float)
    points = known_y[:, [2, 3]]/model.y_scaler.scale_[[2, 3]]
    ref = points.min(axis=0)-1
    gains = [max(0., area(np.vstack((points, p)), ref)-area(points, ref))
             for p in lower[:, [2, 3]]/model.y_scaler.scale_[[2, 3]]]
    tie = (lower[:, [2, 3]]/model.y_scaler.scale_[[2, 3]]).sum(axis=1)
    ordered = sorted(front, key=lambda i: (-gains[i], -tie[i], candidates[i]))[:3]
    chosen = [candidates[i] for i in ordered]
    assert chosen == proposal["selected_ids"] == summary["selected_ids"]
    assert [o["case_id"] for o in verified["observations"]] == chosen
    assert [o["case_id"] for o in reference["observations"]] == list(model.training_ids)
    assert not set(chosen) & set(protocol["training_ids"]+protocol["calibration_ids"])
    np.testing.assert_allclose(gains, proposal["pessimistic_hypervolume_gain"], atol=1e-12)
    np.testing.assert_array_equal(valid, proposal["proposal_eligible"])
    # All 25 read archives must bind to dataset provenance, not merely report hashes.
    with h5py.File(dataset/"fields.h5", "r") as fields:
        for record in reference["observations"]+verified["observations"]:
            row = lookup.loc[record["case_id"]]
            path = Path(row.source_artifact).parent/"run_result.mat"
            assert sha(path) == row.source_run_result_sha256 == record["source_sha256"]
            with h5py.File(path, "r") as handle:
                peaks = [float(np.asarray(handle[f"run_result/cycle_results/{stage}/metrics/global_xi_time_sigma_eq_max_MPa"]).item())
                         for stage in ("load_result", "unload_result")]
                wr = float(np.asarray(handle["run_result/cycle_diagnostics/center_deflection_peak_over_radius"]).item())
                for stage in ("hold", "cooldown"):
                    flag = np.asarray(handle[f"run_result/cycle_diagnostics/{stage}_system_energy_audit_ok"]).item()
                    assert flag == 1
            np.testing.assert_equal(max(peaks), record["sigma_eq_peak_MPa"])
            np.testing.assert_equal(wr, record["center_deflection_peak_over_radius"])
            group = fields[str(row.hdf5_group)+"/cycles/cycle_001"]
            phase = np.concatenate([np.asarray(group[s+"/xi_t"]).ravel() for s in ("stage1_loading", "stage3_unloading")])
            assert np.isfinite(phase).all() and 0 <= phase.min() <= phase.max() <= 1
            np.testing.assert_equal([phase.min(), phase.max()], [record["mechanical_phase_min"], record["mechanical_phase_max"]])
            np.testing.assert_array_equal(row[list(ENSEMBLE_TARGETS)].to_numpy(float), record["targets"])
    for key in reference["reference"]:
        assert max(o[key] for o in reference["observations"]) == reference["reference"][key] == verified["reference"][key]
    for record in verified["observations"]:
        verdict = record["verdict"]
        assert verdict["numerical_gates_pass"] and verdict["engineering_safety"] == "unknown"
        assert all(v is None for v in verdict["engineering_constraint_checks"].values())
        for key, flag in verdict["within_revealed_reference_envelope"].items():
            assert flag == (record[key] <= reference["reference"][key])
        i = candidates.index(record["case_id"])
        np.testing.assert_allclose(mean[i]-record["targets"], record["prediction_error"])
    added = np.array([o["targets"] for o in verified["observations"]])
    actual_gain = area(np.vstack((points, added[:, [2, 3]]/model.y_scaler.scale_[[2, 3]])), ref)-area(points, ref)
    np.testing.assert_allclose(actual_gain, summary["actual_normalized_hypervolume_gain"], atol=1e-12)
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.scatter(known_y[:, 2], known_y[:, 3], c="gray", label="22 revealed observations")
    ax.scatter(mean[:, 2], mean[:, 3], marker="x", c="steelblue", label="15 candidate predictions")
    for index, record in enumerate(verified["observations"]):
        i = candidates.index(record["case_id"])
        truth = np.array(record["targets"])
        ax.scatter(truth[2], truth[3], c="darkorange", marker="*", s=140,
                   label="3 archived verifications" if index == 0 else None)
        ax.annotate(record["case_id"].split("_")[2], (truth[2], truth[3]), xytext=(5, 7), textcoords="offset points")
        ax.annotate("", xy=truth[[2, 3]], xytext=mean[i, [2, 3]], arrowprops={"arrowstyle": "->", "color": "darkorange"})
    ax.set(xlabel="COP", ylabel="SCP (W/kg)", title="Fixed-pool design: prediction to archive")
    ax.legend(fontsize=8)
    ax = axes[1]
    ratios = np.array([[o[k]/reference["reference"][k] for k in reference["reference"]] for o in verified["observations"]])
    for j, label in enumerate(("Peak stress / revealed max", "Deflection ratio / revealed max")):
        ax.bar(np.arange(len(chosen))+(j-.5)*.34, ratios[:, j], width=.34, label=label)
    ax.axhline(1, c="gray", linestyle="--")
    ax.set(xticks=np.arange(len(chosen)), xticklabels=[c.split("_")[2] for c in chosen],
           ylabel="Descriptive reference ratio (not a safety limit)", title="Engineering safety remains UNKNOWN", ylim=(0, 1.18))
    ax.legend(fontsize=8)
    fig.savefig(out/"design_verification.png", dpi=160)
    plt.close(fig)
    report = {"status": "passed_independent_design_audit", "selection_replayed": True,
              "archive_hashes_and_mechanical_values_checked": 25, "chosen_ids": chosen,
              "actual_normalized_hypervolume_gain": actual_gain, "fresh_solver_calls": 0,
              "engineering_safety_certified": False, "source_design_summary_sha256": sha(design/"summary.json"),
              "implementation_sha256": sha(Path(__file__)), "figure_sha256": sha(out/"design_verification.png")}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.design_root, args.original_protocol, args.output_root), indent=2))
