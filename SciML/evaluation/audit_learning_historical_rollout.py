"""Cold/fresh replay, raw-owner checks and same-case horizon audit."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_historical_rollout import corpus, HistoricalModel, rollout, BLOCKS, MEMORY
from SciML.models.v12.run_learning_existing import sha, write_json


def run(dataset, campaign, out):
    if out.exists():
        raise FileExistsError("preserve historical rollout audit")
    root = campaign/"historical-rollout-r1"
    summary = verify_artifacts(root)
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["implementation_sha256"] != sha(Path("SciML/models/v12/learning_historical_rollout.py")):
        raise ValueError("historical implementation changed")
    table, records, inputs, fps, checks = corpus(dataset)
    if fps != protocol["fingerprints"]:
        raise ValueError("historical profile changed")
    groups = protocol["groups"]
    if set(groups["train"]) & set(groups["test"]) or set(sum(groups.values(), [])) != set(table.case_id):
        raise ValueError("case split invalid")
    three = table.loc[table.cycle_count == 3, "case_id"].tolist()
    if len(three) != 2 or not set(three).issubset(groups["test"]):
        raise ValueError("three-cycle trajectories leaked into train")
    # Independently compare every directly measured mechanical field in targets.
    owners = 0
    by_case = table.set_index("case_id")
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for r in records:
            path = by_case.loc[r.case_id].hdf5_group+f"/cycles/cycle_{r.cycle+(r.stage==5):03}"
            group = handle[path]
            mechanical = group["stage1_loading"] if r.stage in (1, 5) else group["stage3_unloading"]
            if r.stage != 4:
                endpoint = 0 if r.stage in (2, 5) else -1
                for field, slc in (("u_m_t", slice(0, 51)), ("w_m_t", slice(51, 102)), ("xi_t", slice(153, 204))):
                    np.testing.assert_array_equal(r.after[slc], np.asarray(mechanical[field])[endpoint])
                    owners += 1
                if r.stage in (2, 5):
                    np.testing.assert_array_equal(r.after[102:153], np.asarray(mechanical["T_K_t"])[endpoint])
                    owners += 1
            if r.stage in (1, 3, 4):
                exchange = group["stage2_hold_exchange" if r.stage == 1 else "stage4_cooldown_exchange"]
                endpoint = -1 if r.stage == 4 else 0
                field = "sma_T_out_51_K" if r.stage == 4 else "sma_T_in_51_K"
                np.testing.assert_array_equal(r.after[102:153], np.asarray(exchange[field]))
                strip = np.asarray(exchange["sma_strip_T_K_t"])[endpoint]
                np.testing.assert_array_equal(r.after[MEMORY], (strip-strip[::-1])/2)
                owners += 2
    train = [r for r in records if r.case_id in groups["train"]]
    rows = [r for r in records if r.case_id in groups["test"]]
    target = np.stack([r.after for r in rows])
    per_case, projection = [], []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in (20260910, 20260911, 20260912):
            dest = root/f"s{seed}"
            arrays = np.load(dest/"predictions.npz", allow_pickle=False)
            report = json.loads((dest/"report.json").read_text())
            np.testing.assert_array_equal(arrays["test_ids"], [r.sample_id for r in rows])
            np.testing.assert_array_equal(arrays["test_truth"], target)
            loaded = HistoricalModel.load(dest/"model.pt")
            if loaded.training_ids != [r.sample_id for r in train]:
                raise ValueError("wrong training history")
            fresh = HistoricalModel(seed).fit(train, {case: data["control"] for case, data in inputs.items()})
            for model in (loaded, fresh):
                replay = np.concatenate([rollout(model, **inputs[case]) for case in groups["test"]])
                np.testing.assert_array_equal(replay, arrays["test_autonomous_neural"])
            variants = {"autonomous_neural": {}, "autonomous_affine": {"neural": False}, "skip_restart": {"skip_restart": True}, "erase_memory": {"erase_memory": True}}
            for name, kwargs in variants.items():
                prediction = np.concatenate([rollout(loaded, **inputs[case], **kwargs) for case in groups["test"]])
                np.testing.assert_array_equal(prediction, arrays["test_"+name])
                for case in groups["test"]:
                    for cycle in range(1, int(by_case.loc[case].cycle_count)+1):
                        select = np.array([r.case_id == case and r.cycle == cycle and r.stage == 4 for r in rows])
                        values = {block: float(np.sqrt(np.mean((prediction[select, slc]-target[select, slc])**2))) for block, slc in BLOCKS.items()}
                        per_case.append({"seed": seed, "variant": name, "case_id": case, "cycle": cycle, "three_cycle_case": case in three, "rmse": values})
                for cycle in range(1, 4):
                    for stage in range(1, 6):
                        select = np.array([r.cycle == cycle and r.stage == stage for r in rows])
                        if select.any():
                            for block, slc in BLOCKS.items():
                                value = float(np.sqrt(np.mean((prediction[select, slc]-target[select, slc])**2)))
                                if value != report["groups"]["test"][name][f"cycle_{cycle}"][str(stage)][block]:
                                    raise ValueError("historical metric mismatch")
                # Temperature range is a diagnostic, not an output clipping gate.
                temps = np.concatenate((prediction[:, 102:153], prediction[:, 204:414]), axis=1)
                projection.append({"seed": seed, "variant": name, "temperature_min_K": float(temps.min()), "temperature_max_K": float(temps.max()),
                                   "outside_200_400K_count": int(((temps < 200) | (temps > 400)).sum())})
    finally:
        torch.set_num_threads(threads)
    fixed_cohort = {}
    for name in variants:
        fixed_cohort[name] = {}
        for cycle in range(1, 4):
            # First pool the SAME two cases per seed using root-mean-square of
            # equal-size case RMSE, then average seeds; do not mix4/4/2case cohorts.
            values = [np.sqrt(np.mean([r["rmse"]["sma_K"]**2 for r in per_case if r["variant"] == name and r["cycle"] == cycle and r["seed"] == seed and r["three_cycle_case"]])) for seed in (20260910, 20260911, 20260912)]
            fixed_cohort[name][str(cycle)] = {"same_case_count": 2, "mean_sma_rmse_K": float(np.mean(values)), "std_sma_rmse_K": float(np.std(values, ddof=1))}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"case_cycle_metrics.json", per_case)
    write_json(out/"temperature_ranges.json", projection)
    figure(fixed_cohort, out)
    result = {"status": "passed_historical_multicycle_learning_audit", "learning_only": True, "fresh_solver_calls": 0,
        "raw_mechanical_and_strip_owner_comparisons": owners, "fresh_three_seed_refits_exact": True,
        "cold_checkpoint_and_intervention_replays_exact": True, "training_cases": 8, "test_cases": 4,
        "training_transitions": len(train), "test_transitions": len(rows), "fixed_three_cycle_cohort": fixed_cohort,
        "scope": "historical composite boundary states; unobserved restart side state is carried, not full Markov or energy closure",
        "implementation_sha256": sha(Path(__file__)), "upstream_summary_sha256": sha(root/"summary.json"),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


def figure(cohort, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    for name, by_cycle in cohort.items():
        ax.errorbar([1, 2, 3], [by_cycle[str(c)]["mean_sma_rmse_K"] for c in (1, 2, 3)],
            yerr=[by_cycle[str(c)]["std_sma_rmse_K"] for c in (1, 2, 3)], marker="o", capsize=4, label=name)
    ax.set(xticks=[1, 2, 3], xlabel="Cycle", ylabel="Cycle-end SMA RMSE (K)",
           title="Same two held-out 3-cycle cases | 3-seed mean +/- SD")
    ax.legend(fontsize=8)
    fig.savefig(out/"historical_rollout.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.output_root), indent=2))
