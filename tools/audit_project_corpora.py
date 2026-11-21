"""Inventory the local solver corpora without running the solver or fitting models."""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import h5py


CORPORA = {
    "historical_development": "v1.1-paper220",
    "historical_confirmation": "v1.2-confirm50",
    "historical_temporal": "v1.2-confirm-temporal12",
    "single_cycle_learning": "v12-learning-existing63-20260910",
}
MECHANICAL = ("stage1_loading", "stage3_unloading")
EXCHANGE = ("stage2_hold_exchange", "stage4_cooldown_exchange")
MECHANICAL_FIELDS = ("u_m_t", "w_m_t", "T_K_t", "xi_t")
EXCHANGE_FIELDS = ("sma_strip_T_K_t", "htf_internal_T_K_t")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def recorded_true(row, key):
    return row[key].lower() == "true"


def check_manifest(root):
    manifest_path = root / "artifact_manifest.json"
    manifest = read_json(manifest_path)
    verified = set()
    for row in manifest["artifacts"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or row["path"] in verified:
            raise ValueError(f"Invalid artifact entry in {root.name}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Archived artifact changed in {root.name}: {row['path']}")
        verified.add(row["path"])
    required = {"case_index.csv", "cycle_index.csv", "fields.h5", "quality_report.json"}
    if not required <= verified:
        raise ValueError(f"Required artifacts missing from manifest in {root.name}")
    return manifest, len(verified)


def field_shape_count(handle, cycles, stages, fields, shape):
    complete = 0
    for row in cycles:
        group = handle[row["hdf5_cycle_group"]]
        for stage in stages:
            datasets = [group.get(f"{stage}/{field}") for field in fields]
            if all(isinstance(dataset, h5py.Dataset) and dataset.shape == shape for dataset in datasets):
                complete += 1
    return {"segments_checked": len(cycles) * len(stages), "matching_shape": complete}


def audit_corpus(root):
    manifest, artifact_count = check_manifest(root)
    quality = read_json(root / "quality_report.json")
    cases = read_rows(root / "case_index.csv")
    cycles = read_rows(root / "cycle_index.csv")
    if (len(cases), len(cycles), manifest["source_content_sha256"]) != (
        quality["case_count"], quality["cycle_sample_count"], quality["source_content_sha256"]
    ):
        raise ValueError(f"Archive inventories disagree in {root.name}")

    valid = [row for row in cases if recorded_true(row, "physics_ok")
             and recorded_true(row, "paper_grade_valid")]
    case_ids = {row["case_id"] for row in cases}
    valid_case_ids = {row["case_id"] for row in valid}
    if len(case_ids) != len(cases) or not {row["case_id"] for row in cycles} <= case_ids:
        raise ValueError(f"Invalid case/cycle identities in {root.name}")
    if len(valid) != quality["paper_grade_valid_count"]:
        raise ValueError(f"Recorded validity flags disagree in {root.name}")
    valid_cycles = [row for row in cycles if row["case_id"] in valid_case_ids]
    with h5py.File(root / "fields.h5", "r") as handle:
        mechanical = field_shape_count(handle, valid_cycles, MECHANICAL, MECHANICAL_FIELDS, (50, 51))
        exchange = field_shape_count(handle, valid_cycles, EXCHANGE, EXCHANGE_FIELDS, (50, 101))

    geometries = sorted({(float(row["radius_m"]), float(row["thickness_m"])) for row in valid})
    condition_ids = {row["condition_id"] for row in valid}
    exported_hashes = {row["source_artifact_semantic_sha256"] for row in valid} - {""}
    profile = {
        "directory": root.name,
        "artifact_manifest_sha256": sha256(root / "artifact_manifest.json"),
        "source_content_sha256": manifest["source_content_sha256"],
        "manifest_artifacts_checked": artifact_count,
        "case_rows": len(cases),
        "cycle_rows": len(cycles),
        "passing_case_rows": len(valid),
        "passing_distinct_condition_ids": len(condition_ids),
        "cycle_rows_of_passing_cases": len(valid_cycles),
        "passing_case_horizons": dict(sorted(Counter(int(row["cycle_count"]) for row in valid).items())),
        "passing_case_materials": dict(sorted(Counter(row["material_id"] for row in valid).items())),
        "geometries_m": [list(pair) for pair in geometries],
        "radius_thickness_ratios": sorted({round(radius / thickness, 9) for radius, thickness in geometries}),
        "mechanical_u_w_temperature_phase_50x51": mechanical,
        "exchange_sma_fluid_50x101": exchange,
        "physics_fingerprints": dict(sorted(Counter(
            row["physics_implementation_fingerprint_sha256"] for row in cases
        ).items())),
        "pipeline_fingerprints": dict(sorted(Counter(
            row["pipeline_implementation_fingerprint_sha256"] for row in cases
        ).items())),
        "recorded_matrix_complete": quality["expected_matrix_complete"],
        "recorded_dataset_paper_grade_gate": quality["paper_grade_gate"],
    }
    return profile, (valid_case_ids, condition_ids, exported_hashes)


def audit_loading_archive(root):
    scalars = read_rows(root / "scalar_cases.csv")
    failures = read_rows(root / "failure_events.csv")
    paths = ("scalar_cases.csv", "operator_samples.csv", "failure_events.csv")
    return {
        "directory": root.name,
        "loading_scalar_rows": len(scalars),
        "recorded_design_valid": sum(recorded_true(row, "design_valid") for row in scalars),
        "operator_checkpoint_rows": len(read_rows(root / "operator_samples.csv")),
        "failure_event_rows": len(failures),
        "index_sha256": {name: sha256(root / name) for name in paths},
    }


def run(solver_repo):
    datasets = solver_repo / "SciML/datasets"
    profiles = {}
    identities = {}
    for name, directory in CORPORA.items():
        profiles[name], identities[name] = audit_corpus(datasets / directory)

    overlaps = []
    for left, right in itertools.combinations(CORPORA, 2):
        a, b = identities[left], identities[right]
        overlap = [len(x & y) for x, y in zip(a, b)]
        overlaps.append({
            "corpora": [left, right],
            "same_case_ids": overlap[0],
            "same_condition_ids": overlap[1],
            "same_export_semantic_sha256": overlap[2],
        })

    results = solver_repo / "SciML/results"
    historic_report = results / "readiness_v11_paper220.json"
    latest_strict_report = results / "readiness_v12.json"
    historic = read_json(historic_report)
    strict = read_json(latest_strict_report)
    return {
        "status": "completed_local_corpus_inventory",
        "audit_source_sha256": sha256(Path(__file__)),
        "loading_archive": audit_loading_archive(datasets / "v1.0"),
        "cycle_corpora": profiles,
        "exact_overlap_of_passing_cases": overlaps,
        "recorded_reports": {
            "historical_development": {
                "sha256": sha256(historic_report),
                "engineering_passed": historic["engineering_gate"]["passed"],
                "project_paper_grade_passed": historic["paper_grade_gate"]["passed"],
                "blocking_reasons": historic["paper_grade_gate"]["blocking_reasons"],
            },
            "subsequent_confirmation_attempt": {
                "sha256": sha256(latest_strict_report),
                "passed_at_report_time": strict["passed"],
                "physics_fingerprint_consistent_at_report_time": strict["physics_fingerprint_consistent"],
            },
        },
        "scope": "Byte-integrity checks and field shapes, with eligibility inherited from archive flags",
        "overlap_scope": "Exact registered condition IDs, excluding partial or near matches",
        "fresh_solver_calls": 0,
        "training_runs": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Use a fresh audit output path")
    result = run(args.solver_repo)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
