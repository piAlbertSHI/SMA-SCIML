"""Inventory existing trajectories without pooling incompatible solver profiles."""
import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from SciML.models.v12.run_learning_existing import STRICT_COLUMNS, sha, write_json, validate_dataset

FP = ("physics_implementation_fingerprint_sha256", "pipeline_implementation_fingerprint_sha256")
STAGES = ("stage1_loading", "stage2_hold_exchange", "stage3_unloading", "stage4_cooldown_exchange")


def run(datasets, reference, out):
    if out.exists():
        raise FileExistsError("preserve prior trajectory inventory")
    ref, quality, _ = validate_dataset(reference)
    profile = tuple(str(ref.iloc[0][key]) for key in FP)
    reports, multi, unique_sources = [], [], {}
    digest_cache = {}

    def cached_sha(path):
        key = str(path.resolve())
        if key not in digest_cache:
            digest_cache[key] = sha(path)
        return digest_cache[key]

    for root in sorted(datasets.iterdir()):
        if not root.is_dir() or not (root/"quality_report.json").exists():
            continue
        if not (root/"cycle_index.parquet").exists():
            reports.append({"dataset": root.name, "status": "legacy_schema_without_cycle_index",
                            "eligible_for_current_fullfield_rollout": False})
            continue
        table = pd.read_parquet(root/"case_index.parquet")
        cycles = pd.read_parquet(root/"cycle_index.parquet")
        manifest = json.loads((root/"artifact_manifest.json").read_text())
        errors = []
        for artifact in manifest["artifacts"]:
            path = root/artifact["path"]
            if not path.is_file() or cached_sha(path) != artifact["sha256"]:
                errors.append("artifact_mismatch:"+artifact["path"])
        if table.case_id.duplicated().any() or cycles.cycle_sample_id.duplicated().any():
            errors.append("duplicate_case_or_cycle_id")
        dataset_profiles = sorted({tuple(str(row[key]) for key in FP) for _, row in table.iterrows()})
        same = table.apply(lambda row: tuple(str(row[key]) for key in FP) == profile, axis=1)
        strict = table.apply(lambda row: all(isinstance(row.get(key), (bool, np.bool_)) and bool(row[key]) for key in STRICT_COLUMNS), axis=1)
        multi_ids = sorted(case for case, rows in cycles.groupby("case_id") if rows.cycle_index.max() > 1)
        report = {"dataset": root.name, "status": "artifact_integrity_passed" if not errors else "invalid_artifacts",
            "artifact_errors": errors, "case_count": len(table), "observed_cycle_rows": len(cycles),
            "cycle_index_counts": {str(k): int(v) for k, v in cycles.cycle_index.value_counts().sort_index().items()},
            "multi_cycle_case_count": len(multi_ids), "strict_flag_case_count": int(strict.sum()),
            "same_implementation_case_count": int(same.sum()),
            "strict_same_implementation_case_count": int((same & strict).sum()),
            "implementation_pairs": [dict(zip(FP, values)) for values in dataset_profiles],
            "index_sha256": {name: cached_sha(root/name) for name in ("case_index.parquet", "cycle_index.parquet")},
            "manifest_sha256": cached_sha(root/"artifact_manifest.json")}
        reports.append(report)
        for _, row in table.iterrows():
            source_id = str(row.source_artifact_semantic_sha256)
            unique_sources.setdefault(source_id, []).append({"dataset": root.name, "case_id": row.case_id,
                "same_implementation": bool(tuple(str(row[key]) for key in FP) == profile),
                "strict_flags": bool(strict.loc[row.name]), "artifact_integrity": not errors})
        if not multi_ids:
            continue
        lookup = table.set_index("case_id")
        with h5py.File(root/"fields.h5", "r") as handle:
            for case in multi_ids:
                row = lookup.loc[case]
                observed = cycles[cycles.case_id == case].sort_values("cycle_index")
                indices = observed.cycle_index.to_numpy(int)
                issues = list(errors)
                if not np.array_equal(indices, np.arange(1, len(indices)+1)):
                    issues.append("noncontiguous_cycles")
                if int(row.cycle_count) != len(indices):
                    issues.append("requested_vs_observed_cycle_count_mismatch")
                source = Path(row.source_artifact)
                source_match = source.is_file() and cached_sha(source) == row.source_artifact_sha256
                request = source.parent/"sciml_case_request.json"
                request_match = request.is_file() and cached_sha(request) == row.source_request_sha256
                if not source_match:
                    issues.append("raw_source_missing_or_changed")
                if not request_match:
                    issues.append("raw_request_missing_or_changed")
                for cycle_row in observed.itertuples(index=False):
                    group_path = str(cycle_row.hdf5_cycle_group)
                    if group_path not in handle:
                        issues.append("missing_hdf5_cycle:"+group_path)
                        continue
                    group = handle[group_path]
                    for stage in STAGES:
                        if stage not in group:
                            issues.append("missing_stage:"+group_path+"/"+stage)
                            continue
                        names = ("sma_strip_T_K_t", "htf_internal_T_K_t") if "exchange" in stage else ("u_m_t", "w_m_t", "T_K_t", "xi_t")
                        for name in names:
                            if name not in group[stage]:
                                issues.append("missing_field:"+group_path+"/"+stage+"/"+name)
                                continue
                            values = np.asarray(group[stage][name])
                            expected_nodes = 101 if "exchange" in stage else 51
                            if values.ndim != 2 or values.shape[-1] != expected_nodes or not np.isfinite(values).all():
                                issues.append("invalid_field:"+group_path+"/"+stage+"/"+name)
                flags = {key: bool(row.get(key)) if isinstance(row.get(key), (bool, np.bool_)) else None for key in STRICT_COLUMNS}
                same_profile = tuple(str(row[key]) for key in FP) == profile
                multi.append({"dataset": root.name, "case_id": case, "source_semantic_sha256": str(row.source_artifact_semantic_sha256),
                    "condition_id": str(row.condition_id), "observed_cycles": indices.tolist(),
                    "source_artifact": str(source), "raw_source_hash_matches": bool(source_match),
                    "raw_request_hash_matches": bool(request_match), "integrity_issues": issues,
                    "declared_flags": flags, "same_implementation_as_existing63": same_profile,
                    "candidate_for_separate_historical_evaluation": not issues and all(value is True for value in flags.values()),
                    "candidate_for_current_profile_multicycle_evaluation": same_profile and not issues and all(value is True for value in flags.values()),
                    "fingerprints": {key: str(row[key]) for key in FP}})
    historical = {row["source_semantic_sha256"] for row in multi if row["candidate_for_separate_historical_evaluation"]}
    current = {row["source_semantic_sha256"] for row in multi if row["candidate_for_current_profile_multicycle_evaluation"]}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"dataset_inventory.json", reports)
    write_json(out/"multicycle_cases.json", multi)
    write_json(out/"source_deduplication.json", unique_sources)
    summary = {"status": "completed_existing_trajectory_inventory", "learning_only": True, "fresh_solver_calls": 0,
        "dataset_count": len(reports), "datasets_with_integrity_errors": [r["dataset"] for r in reports if r["status"] == "invalid_artifacts"],
        "multicycle_dataset_case_records": len(multi), "unique_historical_multicycle_candidates": len(historical),
        "unique_current_profile_multicycle_candidates": len(current),
        "reference_fingerprints": dict(zip(FP, profile)), "reference_source_sha256": quality["source_content_sha256"],
        "scope": "Inventory hashes/fullfield shape-finiteness and recorded gates, not renewed solver physics validation; profile equality necessary but not sufficient for autonomous restart compatibility",
        "next_action": "Historical-profile trajectories require separately identified training/evaluation; never pool with existing63 or fabricate next-cycle targets",
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.datasets_root, args.reference_dataset, args.output_root), indent=2))
