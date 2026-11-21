"""Chapter07B narrow evidence-table/link audit, not full report completion."""
import argparse
import json
from pathlib import Path
import re

from .audit_learning_wp5 import verify_artifacts
from .audit_learning_operator_chapter import table_cells
from SciML.models.v12.run_learning_existing import sha, write_json


def run(campaign, chapter, out):
    if out.exists():
        raise FileExistsError("preserve OOD chapter review")
    body = chapter.read_text(encoding="utf-8")
    verify_artifacts(campaign/"ood-audit-r1")
    diagnostic = verify_artifacts(campaign/"material-ood-diagnostic-r1")
    rows = json.loads((campaign/"ood-audit-r1/aggregate.json").read_text())
    suffixes = {"id_test": "ID", "ood_geometry": "尺寸OOD", "ood_material": "材料OOD",
        "ood_protocol": "时长OOD", "ood_geometry_only": "仅尺寸OOD", "ood_material_only": "仅材料OOD", "ood_combined": "双重OOD"}
    cells = 0
    for row in rows:
        label = row["scheme"]+" / "+suffixes[row["group"]]
        scalar, field = row["scalar_models"]["ensemble3"]["mean_rmse"], row["deeponet_mean_rmse_K"]
        expected = [str(row["case_count"]), f"{scalar[0]:.3f}", f"{scalar[2]:.3f}", f"{field[2]:.3f}", f"{field[3]:.3f}"]
        if table_cells(body, label) != expected:
            raise ValueError("OOD table mismatch: "+label)
        cells += len(expected)
    for scheme in ("geometry", "material", "protocol", "combined"):
        root = campaign/f"ood-{scheme}-r1"
        summary = verify_artifacts(root)
        groups = summary["group_sizes"]
        label = scheme
        observed = table_cells(body, label)[1:]  # skip description of held input group
        expected = [str(groups[key]) for key in ("train", "calibration", "id_test")]
        expected.append("14+5+2" if scheme == "combined" else str(groups["ood_"+scheme]))
        if observed != expected:
            raise ValueError("split-count table mismatch")
        cells += 4
    links = re.findall(r"\]\(<([^>]+)>\)", body)
    for link in links:
        path = re.sub(r":\d+$", "", link)
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if diagnostic["unseen_material_cases_rejected"] != 7 or any(any(v != 0 for v in r["max_data_gradient_material_weight_by_member"]) for r in diagnostic["diagnostics"]):
        raise ValueError("material diagnostic narrative no longer supported")
    commands = re.findall(r"^python .*", body, re.M)
    if len(commands) != 7:
        raise ValueError("unexpected command inventory")
    for command in commands:
        words = command.split()
        module = words[words.index("-m")+1]
        if module != "unittest":
            source = Path(module.replace(".", "/")+".py").read_text(encoding="utf-8")
            for word in words:
                if word.startswith("--") and word not in source:
                    raise ValueError("unknown documented CLI flag")
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "passed_chapter07B_narrow_evidence_review", "whole_report_complete": False,
        "checked_metric_and_split_cells": cells, "local_links_checked": len(links),
        "nine_zero_gradient_networks_verified": True, "documented_commands": len(commands),
        "command_scope": "static module/flag check; equivalent training/audit runs completed, new output-directory recipes not rerun here",
        "readability_checks": ["ID/OOD defined before splits", "R/t constant limitation stated before results",
            "Tensor/model budgets and ensemble3 versus ensemble5 distinction explicit", "Both error denominators explained",
            "Observed-start versus autonomous task separated", "Zero-gradient proof paired with actual diagnostic",
            "Abstention does not claim accuracy repair", "Two-case combined group limitation explicit",
            "Exercises include answers and independent-task acceptance criteria"],
        "chapter_sha256": sha(chapter), "implementation_sha256": sha(Path(__file__)),
        "source_summary_hashes": {name: sha(campaign/name/"summary.json") for name in ("ood-audit-r1", "material-ood-diagnostic-r1")}}
    write_json(out/"summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--chapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign_root, args.chapter, args.output_root), ensure_ascii=False, indent=2))
