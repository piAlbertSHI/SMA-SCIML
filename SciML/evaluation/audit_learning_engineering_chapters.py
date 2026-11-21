"""Narrow review of chapters12-14: evidence table, versions, examples and links.

Not a final whole-report audit or independent human comprehension assessment.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from .audit_learning_operator_chapter import table_cells
from .audit_learning_wp5 import verify_artifacts
from .learning_exercises import reference_checks
from SciML.models.v12.run_learning_existing import sha, write_json


def run(campaign, report_root, inventory, out):
    if out.exists():
        raise FileExistsError("preserve previous engineering chapter review")
    paths = [report_root/name for name in ("12_统一复现手册与代码索引.md", "13_故障定位与失败证据.md", "14_分级练习与独立扩展.md")]
    bodies = [p.read_text(encoding="utf-8") for p in paths]
    preflight = verify_artifacts(inventory)
    # Bind this review to the exact command-preflight chapter versions.
    for path in paths:
        if preflight["chapter_sha256"].get(path.name) != sha(path):
            raise ValueError("preflight must be rerun after chapter edits: "+path.name)
    if not preflight["actual_cli_help_checked"]:
        raise ValueError("actual help check is required")
    versions = {}
    for name in ("numpy", "scipy", "pandas", "h5py", "scikit-learn", "torch", "pyarrow", "matplotlib", "threadpoolctl"):
        shown = table_cells(bodies[0], name)[0]
        if shown != preflight["environment"][name]:
            raise ValueError("documented environment differs: "+name)
        versions[name] = shown
    initial = verify_artifacts(campaign/"wp3-input-contract-r1")
    memory = verify_artifacts(campaign/"wp4-memory-cascade-r2")
    material = verify_artifacts(campaign/"material-ood-diagnostic-r1")
    uq = verify_artifacts(campaign/"wp6-uq-r1")
    design = verify_artifacts(campaign/"wp6-design-r1")
    proposal = json.loads((campaign/"wp6-design-r1/proposal_before_reveal.json").read_text())
    gradients = [v for d in material["diagnostics"] for v in d["max_data_gradient_material_weight_by_member"]]
    sensitivity = initial["initialization_sensitivity"]["test"]["1"]["sma_K"]
    cases = [
        ("未来初态受影响工况", initial["cases_with_stage4_future_initial_difference"], 0.),
        ("未来 HTF 初态最大差", initial["initial_record_vs_request_max_abs"]["htf4_K"], 5e-8),
        ("旧输入阶段1误差", sensitivity["record_initial_rmse"], 5e-7),
        ("请求输入阶段1误差", sensitivity["request_initial_rmse"], 5e-7),
        ("纯镜像入口最大差", memory["mirror_only_source_max_error_K"], 5e-7),
        ("修复串接阶段4SMA误差", memory["scores"]["test"]["4"]["sma"]["future_rmse_K"], 5e-7),
        ("修复串接阶段4HTF误差", memory["scores"]["test"]["4"]["htf"]["future_rmse_K"], 5e-7),
        ("跨批浮点最大差", memory["stage2_cross_batch_comparison"]["max_abs_K"], 5e-20),
        ("材料列零数据梯度网络", sum(v == 0 for v in gradients), 0.),
        ("95%校准目标秩", uq["intervals"][1]["order_statistic_rank"], 0.),
        ("悲观增益为零的候选", sum(v == 0 for v in proposal["pessimistic_hypervolume_gain"]), 0.),
    ]
    evidence = []
    for label, value, tolerance in cases:
        shown = float(table_cells(bodies[1], label)[0])
        if abs(shown-value) > tolerance:
            raise ValueError("failure evidence table mismatch: "+label)
        evidence.append({"label": label, "displayed": shown, "source_value": value, "rounding_tolerance": tolerance})
    if design["engineering_safety_certified"] or uq["intervals"][1]["finite_interval_available"]:
        raise ValueError("unknown/unbounded narrative contradicted")
    old = campaign/"source_snapshots/learning_boundary_state_r1.py"
    current = Path("SciML/models/v12/learning_boundary_state.py")
    if "np.where(scale > 1e-10, scale, 1.)" not in old.read_text(encoding="utf-8") or "np.where(scale > 1e-10, scale, 0.)" not in current.read_text(encoding="utf-8"):
        raise ValueError("documented residual fix source no longer matches")
    baseline = verify_artifacts(campaign/"wp5-local-thermal-r1")
    row = next(v for v in baseline["results"] if v["tag"] == "case2_stage2")
    if abs(row["classical_rate_per_s"]-.5) > .001 or abs(row["effective_rate_per_s"]-1.25) > .01:
        raise ValueError("documented old inverse discrepancy changed")
    old_audit = json.loads((campaign/"wp5-audit-r1/audit.json").read_text())
    if len(old_audit["local_neural_checkpoints_unchanged"]) != 6:
        raise ValueError("old inverse audit checkpoint count changed")

    verify_artifacts(campaign/"learning-exercises-r1")
    saved = json.loads((campaign/"learning-exercises-r1/reference_answers.json").read_text())
    fresh = reference_checks()
    # Floating calculus checks use their stated analytical tolerance; all other
    # instructional results are deterministic arrays/integers and match exactly.
    for key in saved:
        if key != "E06" and saved[key] != fresh[key]:
            raise ValueError("reference exercise changed: "+key)
    for key in ("normalized_pde_max", "normalized_energy_max"):
        if not (saved["E06"][key] <= 1e-12 and fresh["E06"][key] <= 1e-12):
            raise ValueError("analytic exercise residual failed")
    for i in range(1, 9):
        if f"### E{i:02d}：" not in bodies[2]:
            raise ValueError("missing exercise problem statement")
    if len(re.findall(r"^### B[1-3]：", bodies[2], re.M)) != 3 or len(re.findall(r"^### C[1-5]：", bodies[2], re.M)) != 5:
        raise ValueError("graded independent/advanced task coverage differs")
    links = re.findall(r"\]\(<([^>]+)>\)", "\n".join(bodies))
    for link in links:
        path = re.sub(r":\d+$", "", link)
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"evidence_checks.json", evidence)
    write_json(out/"fresh_reference_answers.json", fresh)
    source_paths = [campaign/name for name in ("wp3-input-contract-r1/summary.json", "wp4-memory-cascade-r2/summary.json",
        "material-ood-diagnostic-r1/summary.json", "wp6-uq-r1/summary.json", "wp6-design-r1/summary.json",
        "wp5-local-thermal-r1/summary.json", "wp5-audit-r1/audit.json", "learning-exercises-r1/summary.json")]
    result = {"status": "passed_chapters12_14_narrow_review", "whole_report_complete": False,
        "environment_cells_checked": len(versions), "failure_table_cells_checked": len(evidence),
        "fresh_reference_exercises": len(fresh), "independent_tasks": 3, "advanced_task_specs": 5,
        "local_links_checked": len(links), "fresh_solver_calls": 0, "project_training_calls": 0,
        "readability_scope": "author walkthrough: questions before hints/answers, explicit inputs and expected outputs, executable examples and independent submission requirements; no human comprehension certification",
        "technical_scope": "evidence-to-text consistency and fresh tiny exercise checks; no new full model refits or full scientific review",
        "walkthrough": ["five reproducibility layers distinguished", "environment is observed not an install-lock guarantee",
            "fixed-name upstream dependencies and historical failure audits explicit", "numerical tolerance differs from physical gate",
            "OOD information support distinguished from software failure", "eight small questions have hand-checkable answers",
            "three independent coding tasks and five advanced specifications have acceptance criteria", "learner mastery not inferred from reference success"],
        "chapter_sha256": {p.name: sha(p) for p in paths}, "inventory_summary_sha256": sha(inventory/"summary.json"),
        "source_evidence_sha256": {str(p): sha(p) for p in source_paths},
        "residual_fix_source_sha256": {str(p): sha(p) for p in (old, current)},
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return {key: result[key] for key in ("status", "whole_report_complete", "environment_cells_checked", "failure_table_cells_checked", "fresh_reference_exercises", "independent_tasks", "advanced_task_specs", "local_links_checked")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--inventory-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign_root, args.report_root, args.inventory_root, args.output_root), indent=2))
