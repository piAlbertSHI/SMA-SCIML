"""Seal the reviewed learning report to current evidence; never train or solve.

Human/author semantic judgment is recorded separately, not inferred from hashes.
This gate checks its coverage plus current chapter, source, test and artifact bindings.
"""
import argparse
import json
from pathlib import Path
import re

from .audit_learning_wp5 import verify_artifacts
from .learning_reproduction_inventory import EXPERIMENTS
from .inspect_learning_report_contract import inspect_chapter
from SciML.models.v12.run_learning_existing import sha, validate_dataset, write_json


REVIEWS = {
    "intro-chapters-review-r1": ("audit_learning_intro_chapters", None),
    "scalar-state-chapters-review-r1": ("audit_learning_scalar_state_chapters", None),
    "wp4-chapter-review-r2": ("audit_learning_operator_chapter", "06_"),
    "physics-chapters-review-r3": ("audit_learning_physics_chapters", None),
    "ood-chapter-review-r2": ("audit_learning_ood_chapter", "07B_"),
    "historical-chapter-review-r2": ("audit_learning_historical_chapter", "07C_"),
    "inverse-decision-chapters-review-r3": ("audit_learning_inverse_decision_chapters", None),
    "engineering-chapters-review-r3": ("audit_learning_engineering_chapters", None),
}
SCIENCE_AUDITS = (
    "wp2-strong-audit-r1/summary.json", "wp3-boundary-audit-r1/summary.json",
    "wp4-operator-audit-r1/summary.json", "wp4-repeat-audit-r1/summary.json",
    "wp4-memory-audit-r1/summary.json", "ood-audit-r1/summary.json",
    "material-ood-diagnostic-r1/summary.json", "historical-rollout-audit-r1/summary.json",
    "historical-fields-audit-r1/summary.json", "staged-physics-audit-r1/summary.json",
    "wp5-audit-r1/audit.json", "wp6-uq-active-audit-r1/audit.json",
    "wp6-design-audit-r1/summary.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_digests(root, mapping):
    require(isinstance(mapping, dict) and bool(mapping), "empty digest map cannot prove coverage")
    for name, expected in mapping.items():
        require(sha(root/name) == expected, "changed bound file: "+str(root/name))


def requirement_ids(body):
    ids = re.findall(r"^\| (R\d{2}) \|", body, re.M)
    require(len(ids) == 16 and set(ids) == {f"R{i:02d}" for i in range(16)},
            "review requires exactly one row for each R00--R15")
    for line in body.splitlines():
        if re.match(r"^\| R\d{2} \|", line):
            require("满足学习交付" in line, "unresolved learning requirement: "+line)
    return ids


def run(dataset, campaign, chapters, inventory, regression, review_record, out):
    require(not out.exists(), "preserve previous final audit")
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    manual = review_record.read_text(encoding="utf-8")
    ids = requirement_ids(manual)
    require("不是独立人类读者测评" in manual and "不等于论文级" in manual, "missing review scope")
    chapters_now = {p.name: sha(p) for p in sorted(chapters.glob("*.md"))}
    require(len(chapters_now) == 19, "unexpected report chapter inventory")
    preflight = verify_artifacts(inventory)
    require(preflight["chapter_sha256"] == chapters_now, "preflight does not bind current report")
    require(preflight["actual_cli_help_checked"], "actual CLI help not checked")
    for module, digest in preflight["documented_module_sha256"].items():
        require(sha(Path(module.replace(".", "/")+".py")) == digest, "documented module changed: "+module)

    # Reinspect links and executable-snippet syntax now, rather than trust an old count.
    structure = [inspect_chapter(chapters/name) for name in chapters_now]
    for chapter in structure:
        require(not chapter["fence_markers"] % 2, "unclosed code fence")
        require(all(not link["local"] or link["exists"] for link in chapter["links"]), "broken report link")
        require(all(s.get("syntax_valid") is not False for s in chapter["snippets"]), "invalid Python snippet")

    table, quality, _ = validate_dataset(dataset)
    require(len(table) == 63 and quality["source_content_sha256"] == preflight["dataset_source_content_sha256"], "main data binding changed")
    # Historical data is verified separately, never concatenated with current63.
    historical = Path("SciML/datasets/v1.2-confirm-temporal12")
    manifest = read(historical/"artifact_manifest.json")
    verify_digests(historical, {v["path"]: v["sha256"] for v in manifest["artifacts"]})

    bound, review_sources, evidence = set(), {}, {}
    for name, (module, prefix) in REVIEWS.items():
        root = campaign/name
        report = read(root/"summary.json")
        require(report["status"].startswith("passed_"), "failed local content review: "+name)
        if report.get("artifacts"):
            verify_digests(root, report["artifacts"])
        digest = report["chapter_sha256"]
        mapping = {next(p for p in chapters_now if p.startswith(prefix)): digest} if prefix else digest
        verify_digests(chapters, mapping)
        bound.update(mapping)
        path = Path("SciML/evaluation")/(module+".py")
        require(sha(path) == report["implementation_sha256"], "local reviewer changed: "+name)
        review_sources[str(path)] = sha(path)
        for upstream, expected in report.get("source_summary_hashes", {}).items():
            require(sha(campaign/upstream/"summary.json") == expected, "review upstream changed")
        for upstream, expected in report.get("upstream_summary_sha256", {}).items():
            require(sha(campaign/upstream/"summary.json") == expected, "numeric review upstream changed")
        evidence[name+"/summary.json"] = sha(root/"summary.json")
    required_bound = {name for name in chapters_now if name != "README.md" and not name.startswith("15_")}
    require(bound == required_bound, "local review coverage must equal all17 content chapters")

    # Verify each original learning experiment's complete saved-artifact manifest.
    for name in EXPERIMENTS:
        verify_artifacts(campaign/name)
        evidence[name+"/summary.json"] = sha(campaign/name/"summary.json")
    for relative in SCIENCE_AUDITS:
        record = read(campaign/relative)
        if relative == "material-ood-diagnostic-r1/summary.json":
            require(record["status"] == "completed_material_representation_diagnostic", "material diagnosis incomplete")
            require(record["unseen_material_cases_rejected"] == 7 and
                all(all(v == 0 for v in row["max_data_gradient_material_weight_by_member"]) for row in record["diagnostics"]),
                "material failure mechanism no longer supported")
        else:
            require(record["status"].startswith("passed_"), "upstream experiment audit not passed: "+relative)
        if record.get("artifacts"):
            verify_digests((campaign/relative).parent, record["artifacts"])
        evidence[relative] = sha(campaign/relative)

    # Match actual trained primary sources, not just their current file existence.
    names = {
        "wp2-strong-r1": "learning_scalar_benchmarks", "wp3-boundary-r2": "learning_boundary_state",
        "wp4-memory-cascade-r2": "learning_operator_cascade", "wp5-pinn-r1": "learning_pinn_inverse",
        "wp5-local-thermal-r2": "learning_local_thermal_inverse", "wp6-uq-r1": "learning_ensemble",
        "historical-rollout-r1": "learning_historical_rollout", "historical-fields-r1": "learning_historical_fields",
        "staged-physics-r1": "learning_staged_physics"}
    names.update({name: "learning_coordinate_operator" for name in ("wp4-deeponet-r1", "wp4-fourier-r1", "wp4-no-stage-r1", "wp4-no-ic-r1")})
    names.update({f"wp4-repeat-s{seed}": "learning_operator_repeats" for seed in (20260911, 20260912)})
    names.update({f"ood-{scheme}-r1": "learning_structured_ood" for scheme in ("geometry", "material", "protocol", "combined")})
    paths = {name: Path("SciML/models/v12")/(module+".py") for name, module in names.items()}
    paths.update({"wp6-active-r1": Path("SciML/active_learning/learning_pool_loop.py"),
                  "wp6-design-r1": Path("SciML/active_learning/learning_design_loop.py")})
    trained_sources = {}
    for name, path in paths.items():
        protocol = read(campaign/name/"protocol.json")
        require(sha(path) == protocol["implementation_sha256"], "trained source changed: "+name)
        trained_sources[str(path)] = sha(path)
        for upstream, digest in protocol.get("upstream_implementation_sha256", {}).items():
            require(sha(Path("SciML/models/v12")/(upstream+".py")) == digest, "trained dependency changed")
        if "ensemble_implementation_sha256" in protocol:
            require(sha(Path("SciML/models/v12/learning_ensemble.py")) == protocol["ensemble_implementation_sha256"], "ensemble dependency changed")
    for name, key, path in (
        ("wp3-boundary-r2", "initial_contract_sha256", "SciML/evaluation/inspect_learning_state_contract.py"),
        ("wp4-memory-cascade-r2", "physical_source_sha256", "AER_OperatorSplit_Bed.m"),
        ("historical-fields-r1", "operator_source_sha256", "SciML/models/v12/learning_coordinate_operator.py")):
        require(sha(Path(path)) == read(campaign/name/"protocol.json")[key], "shared source changed")

    tests = verify_artifacts(regression)
    require(tests["status"] == "passed_python_repository_regression" and tests["test_and_physics_sources_unchanged"], "regression not passed")
    require(tests["failures"] == tests["errors"] == tests["skipped"] == 0, "regression has unpassed tests")
    integrity = read(regression/"source_integrity.json")
    verify_digests(Path("."), integrity["tests_after"])
    verify_digests(Path("."), integrity["physics_after"])
    require(set(integrity["tests_after"]) == {str(p) for p in Path("SciML/tests").glob("test_*.py")}, "new/removed tests after regression")
    cases = read(regression/"terminal_and_cases.json")
    require(len(cases["test_ids"]) == tests["test_count"] == tests["discovered_test_count"], "regression count mismatch")
    exercises = verify_artifacts(campaign/"learning-exercises-r1")
    require(exercises.get("reader_mastery_verified") is False, "reference run must not certify the reader")
    design = read(campaign/"wp6-design-r1/summary.json")
    require(design["fresh_solver_calls"] == 0 and design["engineering_safety_certified"] is False, "design scope changed")
    verification = read(campaign/"wp6-design-r1/candidate_verification.json")
    require(len(verification["observations"]) == 3 and all(v["verdict"]["engineering_safety"] == "unknown" for v in verification["observations"]), "design safety unknown not retained")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"chapter_inventory.json", structure)
    write_json(out/"evidence_bindings.json", {"upstream_records": evidence, "trained_sources": trained_sources,
        "reviewer_sources": review_sources, "test_summary": {str(regression/"summary.json"): sha(regression/"summary.json")}})
    result = {"status": "passed_learning_report_final_audit", "whole_report_complete": True,
        "learning_delivery_requirements": ids, "requirement_count": len(ids),
        "chapter_files": len(chapters_now), "chapter_characters": sum(v["characters"] for v in structure),
        "chapter_sha256": chapters_now, "chapters_with_local_content_binding": len(bound),
        "index_and_synthesis_review": "author full-text cross-chapter walkthrough in hash-bound review record",
        "review_record_sha256": sha(review_record), "review_record": str(review_record),
        "documented_commands_checked": preflight["documented_commands"],
        "documented_cli_modules_checked": preflight["unique_task_modules"],
        "reproduction_inventory_sha256": sha(inventory/"summary.json"), "python_tests_passed": tests["test_count"],
        "scientific_audit_records_bound": len(SCIENCE_AUDITS), "learning_experiment_roots_verified": len(EXPERIMENTS),
        "fresh_project_training_calls": 0, "fresh_solver_calls": 0,
        "reader_mastery_verified": False, "independent_human_comprehension_test": False,
        "paper_readiness_certified": False, "engineering_safety_certified": False,
        "limitations": ["existing-data teaching closure, not completion of excluded new-data research claims",
            "original successful experiments plus recorded fresh replays; CLI recipes were not all run again under user replay directory names",
            "full-text review is author judgment plus executable evidence, not proof reading alone creates mastery"],
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return {k: result[k] for k in ("status", "whole_report_complete", "requirement_count", "chapter_files",
        "chapter_characters", "python_tests_passed", "documented_commands_checked", "fresh_solver_calls", "reader_mastery_verified")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset-root", "campaign-root", "report-root", "inventory-root", "regression-root", "review-record", "output-root"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.report_root, args.inventory_root,
        args.regression_root, args.review_record, args.output_root), indent=2))
