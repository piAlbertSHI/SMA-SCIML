"""Audit HRO field-entry errors against future-field errors without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .multicycle_multistep import ROOT, data


SEEDS = (20261121, 20261122, 20261123)
OLD_SEEDS = (20260910, 20260911, 20260912)
FORMAL = ROOT / "evidence/experiments/v13_formal_v1"
OLD = ROOT / "evidence/experiments/multicycle_multistep_v1"
DEFAULT = ROOT / "evidence/local_studies/hro_field_entry_consistency_20261123"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")


def _rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def _field_rows(d, predictions, baseline_predictions, seed, model_name):
    truth = d["field_truth"]
    rows = []
    for material in ("recA", "recAS"):
        for cycle in (1, 2, 3):
            for stage in (2, 4):
                use = (
                    (d["field_cycle"] == cycle)
                    & (d["field_stage"] == stage)
                    & np.asarray([material.lower() in str(d["case_ids"][index]).lower() for index in d["field_case_index"]])
                )
                if not use.any():
                    continue
                pred = predictions[use]
                target = truth[use]
                entry = pred[:, :, 0] - target[:, :, 0]
                future = pred[:, :, 1:] - target[:, :, 1:]
                for channel, name in enumerate(("sma", "htf")):
                    rows.append({
                        "seed": seed,
                        "model": model_name,
                        "material": material,
                        "cycle": cycle,
                        "stage": stage,
                        "channel": name,
                        "case_count": int(use.sum()),
                        "entry_rmse_K": _rmse(entry[:, channel]),
                        "future_rmse_K": _rmse(future[:, channel]),
                        "entry_max_abs_K": float(np.max(np.abs(entry[:, channel]))),
                    })
    return rows


def run(output):
    if output.exists():
        raise FileExistsError("Preserve previous field-entry audits and choose a new directory")
    d, provenance = data()
    rows = []
    checkpoint_hashes = {}
    for seed, old_seed in zip(SEEDS, OLD_SEEDS):
        hro_path = FORMAL / f"s{seed}/test_predictions.npz"
        old_path = OLD / f"s{old_seed}/archived_one_step.npz"
        checkpoint_hashes[str(hro_path.relative_to(ROOT))] = sha(hro_path)
        checkpoint_hashes[str(old_path.relative_to(ROOT))] = sha(old_path)
        with np.load(hro_path, allow_pickle=False) as hro, np.load(old_path, allow_pickle=False) as old:
            hro_field = hro["field"]
            old_field = old["field"]
            if hro_field.shape != d["field_truth"].shape or old_field.shape != d["field_truth"].shape:
                raise ValueError("Saved field prediction shape mismatch")
            if "forecast_initial" in old:
                np.testing.assert_allclose(old["forecast_initial"], old_field[:, :, 0], rtol=0, atol=1e-8)
            rows.extend(_field_rows(d, hro_field, old_field, seed, "hro"))
            rows.extend(_field_rows(d, old_field, old_field, old_seed, "residual_cascade_one_step"))

    selected = [
        row for row in rows
        if row["cycle"] == 3 and row["stage"] == 4 and row["channel"] in ("sma", "htf")
    ]
    grouped = {}
    for row in rows:
        key = (row["model"], row["material"], row["cycle"], row["stage"], row["channel"])
        grouped.setdefault(key, []).append(row)
    aggregate = {}
    for key, values in grouped.items():
        aggregate["|".join(map(str, key))] = {
            "seed_count": len(values),
            "entry_rmse_mean_K": float(np.mean([item["entry_rmse_K"] for item in values])),
            "future_rmse_mean_K": float(np.mean([item["future_rmse_K"] for item in values])),
            "entry_max_abs_mean_K": float(np.mean([item["entry_max_abs_K"] for item in values])),
            "rows": values,
        }
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "verified_hro_field_entry_consistency",
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "data_snapshot_sha256": provenance["snapshot_sha256"],
        "rows": rows,
        "aggregate": aggregate,
        "cycle3_stage4_selected": selected,
        "interpretation": {
            "entry_error_is_interface_and_initial_state_diagnostic": True,
            "entry_error_is_formal_model_error_lower_bound": False,
            "future_field_rmse_excludes_entry_frame": True,
            "historical_test_cohort_previously_examined": True,
        },
        "checkpoint_sha256": checkpoint_hashes,
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
        "cycle3_stage4": result["cycle3_stage4_selected"],
    }, indent=2))


if __name__ == "__main__":
    main()
