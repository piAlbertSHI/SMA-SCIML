"""Audit archived HRO performance by material configuration without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .formal_hierarchical import infer as hro_infer
from .multicycle_multistep import ROOT, data, score
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator


SEEDS = (20261121, 20261122, 20261123)
OLD_SEEDS = (20260910, 20260911, 20260912)
FORMAL = ROOT / "evidence/experiments/v13_formal_v1"
OLD = ROOT / "evidence/experiments/multicycle_multistep_v1"
DEFAULT = ROOT / "evidence/local_studies/hro_material_split_20261123"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")


def aggregate(rows):
    result = {}
    for material, material_rows in rows.items():
        result[material] = {}
        for model in ("hro", "residual_cascade_one_step"):
            result[material][model] = {}
            for category in ("state", "field"):
                result[material][model][category] = {}
                keys = material_rows[0][model][category]
                for key in keys:
                    result[material][model][category][key] = {}
                    for metric in keys[key]:
                        if metric == "case_count":
                            continue
                        values = [row[model][category][key][metric] for row in material_rows]
                        result[material][model][category][key][metric] = {
                            "mean": float(np.mean(values)),
                            "sd": float(np.std(values, ddof=1)),
                            "seed_values": values,
                        }
    return result


def run(output):
    if output.exists():
        raise FileExistsError("Preserve previous material audits and choose a new directory")
    d, provenance = data()
    test_cases = np.flatnonzero(d["split"] == "test")
    groups = {
        "recA": np.asarray([index for index in test_cases if "_reca_" in str(d["case_ids"][index])], dtype=int),
        "recAS": np.asarray([index for index in test_cases if "_recas_" in str(d["case_ids"][index])], dtype=int),
    }
    if any(len(indices) == 0 for indices in groups.values()) or sum(len(indices) for indices in groups.values()) != len(test_cases):
        raise ValueError("Material split does not partition all historical test cases")
    rows = {material: [] for material in groups}
    checkpoints = {}
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for seed, old_seed in zip(SEEDS, OLD_SEEDS):
            model_path = FORMAL / f"s{seed}/model.pt"
            old_path = OLD / f"s{old_seed}/archived_one_step.npz"
            checkpoints[str(model_path.relative_to(ROOT))] = sha(model_path)
            checkpoints[str(old_path.relative_to(ROOT))] = sha(old_path)
            hro_arrays = hro_infer(HierarchicalStateOperator.load(model_path), d)
            with np.load(old_path, allow_pickle=False) as source:
                residual_arrays = {key: source[key] for key in ("state", "field")}
            for material, indices in groups.items():
                rows[material].append({
                    "seed": seed,
                    "case_ids": d["case_ids"][indices].tolist(),
                    "hro": score(hro_arrays, d, indices),
                    "residual_cascade_one_step": score(residual_arrays, d, indices),
                })

    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "verified_hro_material_split_audit",
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "data_snapshot_sha256": provenance["snapshot_sha256"],
        "test_case_ids_by_material": {material: d["case_ids"][indices].tolist() for material, indices in groups.items()},
        "seed_rows": rows,
        "aggregate": aggregate(rows),
        "interpretation": {
            "material_split_is_case_level": True,
            "historical_test_cohort_previously_examined": True,
            "independent_unseen_material_confirmation": False,
            "model_selection_or_retraining": False,
        },
        "checkpoint_sha256": checkpoints,
    }
    write(output / "summary.json", summary)
    write(output / "manifest.json", {
        "sha256": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}
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
        "materials": {material: len(cases) for material, cases in result["test_case_ids_by_material"].items()},
        "c3s4_hro_sma_by_material": {
            material: result["aggregate"][material]["hro"]["state"]["c3s4"]["sma_K"]
            for material in result["aggregate"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
