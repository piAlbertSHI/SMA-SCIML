"""Independent indexing/feature assembly and fresh historical operator refits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_learning_wp5 import verify_artifacts
from .audit_learning_operators import matched_pod
from SciML.models.v12.learning_historical_rollout import corpus, HistoricalModel, rollout
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator
from SciML.models.v12.temporal_operator_candidate import causal_delta_features
from SciML.models.v11.spatiotemporal_operator import build_operator_corpus
from SciML.models.v12.run_learning_existing import sha, write_json


def run(dataset, campaign, out):
    if out.exists():
        raise FileExistsError("preserve historical field audit")
    root = campaign/"historical-fields-r1"
    summary = verify_artifacts(root)
    verify_artifacts(campaign/"historical-rollout-r1")
    protocol = json.loads((root/"protocol.json").read_text())
    if sha(campaign/"historical-rollout-r1/summary.json") != protocol["boundary_summary_sha256"]:
        raise ValueError("boundary artifact changed")
    for file, key in (("learning_historical_fields.py", "implementation_sha256"), ("learning_coordinate_operator.py", "operator_source_sha256")):
        if sha(Path("SciML/models/v12")/file) != protocol[key]:
            raise ValueError("historical field source changed")
    table, _, starts, fps, _ = corpus(dataset)
    if fps != protocol["fingerprints"]:
        raise ValueError("different field provenance")
    samples = build_operator_corpus(dataset)
    branch = np.stack([causal_delta_features(s) for s in samples])
    truth = np.stack([np.stack((s.sma_history, s.htf_history)) for s in samples])
    initial = truth[:, :, 0].copy()
    train = np.array([s.case_id in protocol["groups"]["train"] for s in samples])
    test = ~train
    fixed = np.array([use and starts[s.case_id]["cycles"] == 3 for s, use in zip(samples, test)])
    pod, _ = matched_pod(branch, initial, truth, np.array([s.stage_index for s in samples]), train)
    np.testing.assert_array_equal(pod, np.load(root/"pod_predictions.npz", allow_pickle=False)["prediction"])
    reports = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in (20260910, 20260911, 20260912):
            boundary = HistoricalModel.load(campaign/f"historical-rollout-r1/s{seed}/model.pt")
            state = {case: rollout(boundary, **starts[case]) for case in table.case_id}
            branches, initials = [], []
            for sample in samples:
                # Four composite stages + one restart between cycles. Do not
                # call production cascade_inputs or field_inputs here.
                index = 5*(sample.cycle_index-1)+(0 if sample.stage_index == 2 else 2)
                z = state[sample.case_id][index]
                solid = np.concatenate((z[103:153][::-1], z[102:153]))+z[414:515]
                fluid = z[204:305] if sample.stage_index == 2 else z[305:406]
                side = z[406:414].copy()
                side[6 if sample.stage_index == 2 else 5] = 293.
                c = starts[sample.case_id]["control"]
                values = np.concatenate((solid-solid.mean(), fluid-fluid.mean(), [solid.mean(), fluid.mean(),
                    sample.stage_index == 2, sample.stage_index == 4], c[5:7], c[:5], side))
                branches.append(values); initials.append(np.stack((solid, fluid)))
            cb, ci = np.stack(branches), np.stack(initials)
            dest = root/f"s{seed}"
            arrays = np.load(dest/"predictions.npz", allow_pickle=False)
            report = json.loads((dest/"report.json").read_text())
            np.testing.assert_array_equal(arrays["sample_ids"], [s.sample_id for s in samples])
            np.testing.assert_array_equal(cb, arrays["forecast_branch"])
            np.testing.assert_array_equal(ci, arrays["forecast_initial"])
            model = CoordinateOperator.load(dest/"operator.pt")
            if model.training_ids != [s.sample_id for s, use in zip(samples, train) if use]:
                raise ValueError("operator training case leakage")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                fresh = CoordinateOperator().double().fit(branch[train], initial[train], truth[train], model.training_ids, seed=seed)
            for current in (model, fresh):
                np.testing.assert_array_equal(current.predict(cb, ci), arrays["autonomous_boundary"])
                np.testing.assert_array_equal(current.predict(branch, initial), arrays["observed_start"])
            for group, mask in (("test", test), ("fixed_three_cycle_cases", fixed)):
                for mode in ("observed_start", "autonomous_boundary"):
                    for cycle in (1, 2, 3):
                        for stage in (2, 4):
                            active = mask & np.array([s.cycle_index == cycle and s.stage_index == stage for s in samples])
                            if not active.any():
                                continue
                            key = f"cycle{cycle}_stage{stage}"
                            for j, field in enumerate(("sma", "htf")):
                                e = arrays[mode][active, j]-truth[active, j]
                                values = {"case_count": int(active.sum()), "future_rmse_K": float(np.sqrt(np.mean(e[:, 1:]**2))),
                                    "future_max_abs_K": float(np.abs(e[:, 1:]).max()), "initial_max_abs_K": float(np.abs(e[:, 0]).max())}
                                if values != report[group][mode][key][field]:
                                    raise ValueError("historical field metric mismatch")
            reports.append(report)
    finally:
        torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=False)
    figure(reports, summary["fixed_three_cycle_pod_scores"], out)
    result = {"status": "passed_historical_multicycle_field_audit", "learning_only": True, "fresh_solver_calls": 0,
        "independent_boundary_indices_and_features_exact": True, "fresh_three_seed_operator_refits_exact": True,
        "training_field_samples": int(train.sum()), "test_field_samples": int(test.sum()), "fixed_three_cycle_case_count": 2,
        "scope": "one-way boundary-to-field decoder; no field feedback to physical energy or mechanical state",
        "implementation_sha256": sha(Path(__file__)), "upstream_summary_sha256": sha(root/"summary.json"),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


def figure(reports, pod, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, field in zip(axes, ("sma", "htf")):
        for mode in ("observed_start", "autonomous_boundary"):
            values = np.array([[r["fixed_three_cycle_cases"][mode][f"cycle{c}_stage4"][field]["future_rmse_K"] for c in (1, 2, 3)] for r in reports])
            ax.errorbar((1, 2, 3), values.mean(axis=0), yerr=values.std(axis=0, ddof=1), marker="o", capsize=4, label=mode)
        ax.plot((1, 2, 3), [pod[f"cycle{c}_stage4"][field]["future_rmse_K"] for c in (1, 2, 3)], marker="s", label="POD observed-start")
        ax.set(xticks=[1, 2, 3], xlabel="Cycle", ylabel="Stage4 future-field RMSE (K)", title=field.upper()+" | same two test cases")
        ax.legend(fontsize=8)
    fig.savefig(out/"historical_fields.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.output_root), indent=2))
