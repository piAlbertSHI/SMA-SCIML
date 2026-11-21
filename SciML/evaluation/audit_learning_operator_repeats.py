"""Cold-replay all 12 operator checkpoints and summarize fixed-seed comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator, KINDS, corpus, TIME_INDICES, SPACE_INDICES
from SciML.models.v12.run_learning_existing import sha, write_json


def run(dataset, campaign, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve repeat audit")
    samples, branch, initial, truth, quality = corpus(dataset)
    original = json.loads(original_protocol.read_text())
    if original["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("source/split mismatch")
    masks = {part: np.array([s.condition_id in ids for s in samples]) for part, ids in original["condition_splits"].items()}
    first_roots = dict(zip(KINDS, ("wp4-deeponet-r1", "wp4-fourier-r1", "wp4-no-stage-r1", "wp4-no-ic-r1")))
    stages = np.array([s.stage_index for s in samples])
    records, sources = [], {}
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in (20260910, 20260911, 20260912):
            parent = None if seed == 20260910 else verify_artifacts(campaign/f"wp4-repeat-s{seed}")
            for kind in KINDS:
                root = campaign/(first_roots[kind] if seed == 20260910 else f"wp4-repeat-s{seed}/{kind}")
                summary = verify_artifacts(root)
                protocol = json.loads((root/"protocol.json").read_text())
                if parent and parent["runs"][kind]["summary_sha256"] != sha(root/"summary.json"):
                    raise ValueError("repeat-child manifest mismatch")
                if protocol["implementation_sha256"] != sha(Path("SciML/models/v12/learning_coordinate_operator.py")):
                    raise ValueError("changed model source")
                if protocol["condition_splits"] != original["condition_splits"] or protocol["seed"] != seed or protocol["kind"] != kind:
                    raise ValueError("changed experiment identity")
                if protocol["optimizer_steps"] != 400 or protocol["supervised_time_indices"] != TIME_INDICES.tolist() or protocol["supervised_space_indices"] != SPACE_INDICES.tolist():
                    raise ValueError("unmatched training budget/grid")
                model = CoordinateOperator.load(root/"model.pt")
                if model.training_ids != [s.sample_id for s, selected in zip(samples, masks["train"]) if selected]:
                    raise ValueError("changed training set")
                arrays = np.load(root/"predictions.npz", allow_pickle=False)
                np.testing.assert_array_equal(arrays["sample_ids"], [s.sample_id for s in samples])
                np.testing.assert_array_equal(arrays["truth"], truth)
                prediction = model.predict(branch, initial)
                np.testing.assert_array_equal(prediction, arrays["prediction"])
                for part, mask in masks.items():
                    for stage in (2, 4):
                        select = mask & (stages == stage)
                        for j, field in enumerate(("sma", "htf")):
                            error = prediction[select, j]-truth[select, j]
                            values = {"future_rmse_K": float(np.sqrt(np.mean(error[:, 1:]**2))),
                                      "initial_max_abs_K": float(np.abs(error[:, 0]).max()),
                                      "future_max_abs_K": float(np.abs(error[:, 1:]).max())}
                            if values != summary["scores"][part]["model"][str(stage)][field]:
                                raise ValueError("metrics disagree with raw arrays")
                            if part == "test":
                                records.append({"kind": kind, "seed": seed, "stage": stage, "field": field, **values})
                if kind != "deeponet_no_ic":
                    np.testing.assert_array_equal(prediction[:, :, 0], initial)
                sources[str(root.relative_to(campaign))] = sha(root/"summary.json")
    finally:
        torch.set_num_threads(threads)
    aggregate = []
    for kind in KINDS:
        for stage in (2, 4):
            for field in ("sma", "htf"):
                group = [r for r in records if (r["kind"], r["stage"], r["field"]) == (kind, stage, field)]
                values = [r["future_rmse_K"] for r in group]
                aggregate.append({"kind": kind, "stage": stage, "field": field, "seeds": 3,
                    "mean_future_rmse_K": float(np.mean(values)), "sample_std_future_rmse_K": float(np.std(values, ddof=1)),
                    "worst_test_initial_max_K": max(r["initial_max_abs_K"] for r in group)})
    pod = verify_artifacts(campaign/"wp4-operator-audit-r1")["test_comparison"]["coarse_pod_ridge"]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"per_seed_metrics.json", records)
    write_json(out/"aggregate_metrics.json", aggregate)
    figure(aggregate, pod, out)
    result = {"status": "passed_twelve_checkpoint_operator_repeat_audit", "learning_only": True,
        "fresh_solver_calls": 0, "checkpoint_count": 12, "cold_predictions_exact": True,
        "same_splits_updates_and_supervision_grid": True, "std_definition": "sample std across three model seeds, ddof=1; NOT predictive uncertainty or interval",
        "aggregate": aggregate, "coarse_pod_ridge": pod,
        "source_summary_hashes": sources, "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


def figure(records, pod, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = ("DeepONet", "Spatial Fourier", "DeepONet: no stage", "DeepONet: no hard IC")
    for kind, label in zip(KINDS, labels):
        rows = [r for r in records if r["kind"] == kind]
        ax.errorbar(range(4), [r["mean_future_rmse_K"] for r in rows],
            yerr=[r["sample_std_future_rmse_K"] for r in rows], marker="o", capsize=4, label=label)
    ax.plot(range(4), [pod[str(s)][f]["future_rmse_K"] for s in (2, 4) for f in ("sma", "htf")],
            marker="s", linestyle="--", label="Matched coarse POD/Ridge (deterministic)")
    ax.set(xticks=range(4), xticklabels=["S2 SMA", "S2 HTF", "S4 SMA", "S4 HTF"],
           ylabel="Test future-field RMSE (K)", title="Fixed 400-update comparisons: 3-seed mean +/- sample SD")
    ax.legend(fontsize=8)
    fig.savefig(out/"operator_three_seeds.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.original_protocol, args.output_root), indent=2))
