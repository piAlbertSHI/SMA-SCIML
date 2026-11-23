"""Aggregate and verify the current WP2-WP4 evidence boundary."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "evidence/local_studies/wp2_wp4_readiness_20261123"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")


def verify_manifest(root):
    manifest = read_json(root / "manifest.json")
    checked = 0
    for name, expected in manifest.get("sha256", {}).items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Evidence manifest mismatch: {root / name}")
        checked += 1
    return checked


def run(output):
    if Path(output).exists():
        raise FileExistsError("Preserve previous readiness audits and choose a new directory")
    evidence = {
        "wp2_static": ROOT / "evidence/local_studies/wp2_scalar_reliability_20261123/audit.json",
        "wp2_full_cycle": ROOT / "evidence/local_studies/wp2_full_cycle_scalar_20261123/audit.json",
        "wp3_contract": ROOT / "evidence/local_studies/wp3_state_field_contract_20261123/audit.json",
        "wp3_gaps": ROOT / "evidence/local_studies/wp3_interface_gaps_20261123/audit.json",
        "wp4_energy": ROOT / "evidence/archive/field-energy-information-r1/summary.json",
        "hro_spatial": ROOT / "evidence/local_studies/hro_spatial_supervision_20261123/summary.json",
        "hro_spatial_confirm": ROOT / "evidence/experiments/hro_spatial_ablation_jitter_fluid2_confirm_20261123/summary.json",
        "hro_memory": ROOT / "evidence/local_studies/hro_memory_sensitivity_20261123/summary.json",
        "hro_material": ROOT / "evidence/local_studies/hro_material_split_20261123/summary.json",
        "hro_entry": ROOT / "evidence/local_studies/hro_field_entry_consistency_20261123_r2/summary.json",
    }
    records = {name: read_json(path) for name, path in evidence.items()}
    expected_status = {
        "wp2_static": "verified_archived_wp2_scalar_reliability",
        "wp2_full_cycle": "verified_wp2_full_cycle_scalar_contract",
        "wp3_contract": "verified_wp3_state_field_contract",
        "wp3_gaps": "verified_wp3_interface_gap_distribution",
        "wp4_energy": "completed_field_energy_information_audit",
        "hro_spatial": "completed_hro_spatial_supervision_ablation",
        "hro_spatial_confirm": "completed_hro_spatial_ablation",
        "hro_memory": "verified_hro_memory_sensitivity",
        "hro_material": "verified_hro_material_split_audit",
        "hro_entry": "verified_hro_field_entry_consistency",
    }
    for name, status in expected_status.items():
        if records[name].get("status") != status:
            raise ValueError(f"Unexpected {name} status: {records[name].get('status')}")

    no_solver = {}
    for name, record in records.items():
        calls = record.get("fresh_solver_calls", record.get("solver_calls", 0))
        if calls != 0:
            raise ValueError(f"Fresh solver calls recorded in {name}: {calls}")
        no_solver[name] = calls
    bundle_manifest = read_json(ROOT / "evidence/bundle_manifest.json")
    required_bound = [
        "docs/WP2_SCALAR_RELIABILITY.md",
        "docs/WP3_STATE_FIELD_CONTRACT.md",
        "docs/WP4_FIELD_ENERGY_CONTRACT.md",
        "docs/HRO_SPATIAL_SUPERVISION.md",
        "docs/HRO_MEMORY_SENSITIVITY.md",
        "docs/HRO_MATERIAL_SPLIT.md",
        "docs/HRO_FIELD_ENTRY_CONSISTENCY.md",
    ]
    missing_bound = [name for name in required_bound if name not in bundle_manifest["files"]]
    if missing_bound:
        raise ValueError(f"Required documentation is outside the curated manifest: {missing_bound}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    readiness = {
        "status": "verified_wp2_wp4_current_evidence_boundary",
        "fresh_training_calls": 0,
        "fresh_solver_calls": 0,
        "evidence_records": {
            name: {
                "status": records[name]["status"],
                "source_sha256": sha(path),
            }
            for name, path in evidence.items()
        },
        "curated_documentation_bound": required_bound,
        "fresh_solver_calls_by_record": no_solver,
        "verified_open_boundaries": [
            "The three-seed 200-update query-jitter confirmation is complete. Its HTF field RMSE remains modestly above the coordinate-HRO reference, so the recipe is retained as a state-priority candidate.",
            "Independent failure-label reliability and new-design confirmation remain unavailable in the archived scalar contract.",
            "Port-level energy closure, hidden storage ownership and restart maps remain open.",
            "Unseen-material and independent-grid generalization remain unverified.",
        ],
        "interpretation": "The current WP2-WP4 archive and low-cost model-improvement evidence is internally reconciled. Open boundaries are explicit and are not promoted to completed claims.",
    }
    write(output / "readiness.json", readiness)
    write(output / "manifest.json", {
        "sha256": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}
    })
    return readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["status"],
        "evidence_records": len(result["evidence_records"]),
        "documentation_bound": len(result["curated_documentation_bound"]),
        "fresh_training_calls": result["fresh_training_calls"],
        "fresh_solver_calls": result["fresh_solver_calls"],
        "open_boundaries": len(result["verified_open_boundaries"]),
    }, indent=2))


if __name__ == "__main__":
    main()
