"""Inference-only sensitivity audit for the HRO persistent memory path."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .formal_hierarchical import cohorts, infer as formal_infer
from .multicycle_multistep import ROOT, data, score, write
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


SEEDS = (20261121, 20261122, 20261123)
FORMAL = ROOT / "evidence/experiments/v13_formal_v1"
DEFAULT = ROOT / "evidence/local_studies/hro_memory_sensitivity_20261123"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_memory_infer(model, d):
    state_map, field_map = {}, {}
    times = torch.linspace(0, 1, 50, dtype=torch.float64)
    positions = torch.linspace(0, 1, 101, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for index in np.flatnonzero(d["split"] == "test"):
            state = torch.tensor(d["initial"][index:index + 1])
            control = torch.tensor(d["control"][index:index + 1])
            keys, field_keys, states, fields = [], [], [], []
            for cycle in range(1, int(d["cycles"][index]) + 1):
                stages = ([5] if cycle > 1 else []) + [1, 2, 3, 4]
                for stage in stages:
                    decode = stage in (2, 4)
                    result = model.step(
                        state,
                        control,
                        stage,
                        memory=None,
                        times=times if decode else None,
                        positions=positions if decode else None,
                    )
                    state = result["state"]
                    states.append(state[0].numpy())
                    keys.append((cycle - 1 if stage == 5 else cycle, stage))
                    if decode:
                        fields.append(result["field"][0].numpy())
                        field_keys.append((cycle, stage))
            for key, value in zip(keys, states):
                state_map[(index, *key)] = value
            for key, value in zip(field_keys, fields):
                field_map[(index, *key)] = value
    test = d["split"][d["case_index"]] == "test"
    state_keys = list(zip(d["case_index"][test], d["cycle"][test], d["stage"][test]))
    field_keys = list(zip(d["field_case_index"], d["field_cycle"], d["field_stage"]))
    if set(state_keys) != set(state_map) or set(field_keys) != set(field_map):
        raise ValueError("Memory sensitivity rollout keys do not match the sealed evaluation index")
    arrays = {
        "state": np.stack([state_map[key] for key in state_keys]),
        "field": np.stack([field_map[key] for key in field_keys]),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Memory sensitivity rollout contains non-finite values")
    return arrays


def run(output):
    if output.exists():
        raise FileExistsError("Preserve previous HRO sensitivity audits and choose a new directory")
    d, provenance = data()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    replay_checks = []
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for seed in SEEDS:
            checkpoint = FORMAL / f"s{seed}/model.pt"
            model = HierarchicalStateOperator.load(checkpoint)
            full = formal_infer(model, d)
            no_memory = _no_memory_infer(model, d)
            # The saved formal predictions are the full-memory reference.
            with np.load(FORMAL / f"s{seed}/test_predictions.npz", allow_pickle=False) as saved:
                for key in full:
                    np.testing.assert_allclose(saved[key], full[key], rtol=0, atol=1e-8)
                    replay_checks.append(float(np.max(np.abs(saved[key] - full[key]))))
            for mode, arrays in (("persistent_memory", full), ("memory_reset_each_step", no_memory)):
                rows.append({
                    "seed": seed,
                    "mode": mode,
                    "all_test": cohorts(arrays, d)["all_test"],
                    "fixed_three_cycle": cohorts(arrays, d)["fixed_three_cycle"],
                    "per_three_cycle_case": cohorts(arrays, d)["per_three_cycle_case"],
                })
            np.savez_compressed(output / f"s{seed}_memory_reset_predictions.npz", **no_memory)

    persistent = [row for row in rows if row["mode"] == "persistent_memory"]
    reset = [row for row in rows if row["mode"] == "memory_reset_each_step"]
    fixed = {}
    for category in ("state", "field"):
        fixed[category] = {}
        keys = persistent[0]["fixed_three_cycle"][category]
        for key in keys:
            fixed[category][key] = {}
            for metric in keys[key]:
                if metric == "case_count":
                    continue
                p = np.asarray([row["fixed_three_cycle"][category][key][metric] for row in persistent])
                r = np.asarray([row["fixed_three_cycle"][category][key][metric] for row in reset])
                fixed[category][key][metric] = {
                    "persistent_memory_mean": float(p.mean()),
                    "memory_reset_mean": float(r.mean()),
                    "reset_minus_persistent": float((r - p).mean()),
                    "persistent_seed_values": p.tolist(),
                    "reset_seed_values": r.tolist(),
                }
    summary = {
        "status": "verified_hro_memory_sensitivity",
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "data_snapshot_sha256": provenance["snapshot_sha256"],
        "seeds": list(SEEDS),
        "modes": {
            "persistent_memory": "archived formal HRO checkpoint replay",
            "memory_reset_each_step": "same checkpoint, memory argument reset to zero at every stage",
        },
        "fixed_three_cycle_comparison": fixed,
        "checkpoint_replay_max_abs_difference": max(replay_checks, default=0.0),
        "interpretation": {
            "causal_claim": False,
            "weights_changed": False,
            "memory_reset_is_inference_ablation": True,
            "test_cohort_is_historical_and_previously_examined": True,
        },
        "artifacts": {
            path.name: sha(path)
            for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    write(output / "rows.json", rows)
    summary["artifacts"] = {
        path.name: sha(path)
        for path in sorted(output.iterdir()) if path.is_file()
    }
    write(output / "summary.json", summary)
    write(output / "manifest.json", {
        "sha256": {
            path.name: sha(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["status"],
        "fresh_training_calls": result["fresh_training_calls"],
        "fresh_solver_calls": result["fresh_solver_calls"],
        "checkpoint_replay_max_abs_difference": result["checkpoint_replay_max_abs_difference"],
        "c3s4_state_sma_reset_minus_persistent": result["fixed_three_cycle_comparison"]["state"]["c3s4"]["sma_K"]["reset_minus_persistent"],
        "c3s4_field_htf_reset_minus_persistent": result["fixed_three_cycle_comparison"]["field"]["c3s4"]["htf_K"]["reset_minus_persistent"],
    }, indent=2))


if __name__ == "__main__":
    main()
