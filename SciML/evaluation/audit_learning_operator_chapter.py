"""Narrow chapter06 table/link/snippet check; not the final whole-report audit."""
import argparse
import json
from pathlib import Path
import re

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.run_learning_existing import sha, write_json


def table_cells(body, label):
    rows = [line for line in body.splitlines() if line.startswith("| "+label+" |")]
    if len(rows) != 1:
        raise ValueError(f"missing or ambiguous row {label}")
    return [cell.strip() for cell in rows[0].strip("|").split("|")][1:]


def run(campaign, chapter, out):
    if out.exists():
        raise FileExistsError("preserve chapter review")
    body = chapter.read_text(encoding="utf-8")
    repeats = verify_artifacts(campaign/"wp4-repeat-audit-r1")
    memory = verify_artifacts(campaign/"wp4-memory-cascade-r2")
    first = verify_artifacts(campaign/"wp4-operator-audit-r1")
    labels = {"deeponet": "DeepONet", "fourier": "空间 Fourier",
              "deeponet_no_stage": "DeepONet 去掉阶段标记", "deeponet_no_ic": "DeepONet 去掉硬 IC"}
    checked_cells = 0
    for kind, label in labels.items():
        actual = table_cells(body, label)
        rows = [r for r in repeats["aggregate"] if r["kind"] == kind]
        expected = [f'{r["mean_future_rmse_K"]:.3f}±{r["sample_std_future_rmse_K"]:.3f}' for r in rows]
        if actual != expected:
            raise ValueError(f"three-seed table mismatch: {label}")
        checked_cells += len(expected)
    for label, metrics, decimals in (
        ("匹配粗监督 POD/Ridge", repeats["coarse_pod_ridge"], 3),
        ("真实阶段入口 DeepONet", first["test_comparison"]["deeponet"], 6),
        ("WP3请求级联，仅径向镜像", first["test_comparison"]["request_boundary_to_deeponet"], 6),
        ("WP3请求级联，加预测空间记忆", memory["scores"]["test"], 6)):
        expected = [format(metrics[str(stage)][field]["future_rmse_K"], f".{decimals}f")
                    for stage in (2, 4) for field in ("sma", "htf")]
        if table_cells(body, label) != expected:
            raise ValueError(f"comparison table mismatch: {label}")
        checked_cells += 4
    direct = verify_artifacts(campaign/"wp4-deeponet-r1")
    persistence = direct["scores"]["test"]["persistence"]
    expected = [f'{persistence[str(stage)][field]["future_rmse_K"]:.3f}' for stage in (2, 4) for field in ("sma", "htf")]
    if table_cells(body, "入口场保持不变") != expected:
        raise ValueError("persistence row mismatch")
    checked_cells += 4
    links = re.findall(r"\]\(<([^>]+)>\)", body)
    for link in links:
        local = re.sub(r":\d+$", "", link)
        if re.match(r"^/[A-Za-z]:/", local):
            local = local[1:]
        if not Path(local).is_file():
            raise FileNotFoundError(local)
    snippets = re.findall(r"```python\n(.*?)\n```", body, flags=re.S)
    if len(snippets) != 1:
        raise ValueError("reinspect chapter executable snippets before changing audit")
    # This is the inspected, self-contained NumPy even/odd reference solution.
    exec(compile(snippets[0], str(chapter)+":exercise_B", "exec"), {})
    commands = re.findall(r"^python .*", body, flags=re.M)
    if len(commands) != 6:
        raise ValueError("unexpected command inventory")
    for command in commands:
        tokens = command.split()
        module = tokens[tokens.index("-m")+1]
        if module != "unittest":
            source = Path(module.replace(".", "/")+".py").read_text(encoding="utf-8")
            for token in tokens:
                if token.startswith("--") and token not in source:
                    raise ValueError(f"documented option absent from module: {token}")
    review = {"status": "passed_chapter06_tables_links_and_reference_snippet", "whole_report_complete": False,
        "checked_metric_cells": checked_cells, "local_link_count": len(links), "numpy_reference_snippets_executed": 1,
        "command_count": len(commands), "command_check_scope": "module/option static inventory; equivalent experiment commands already run; these new output-directory recipes not rerun here",
        "readability_review": ["Separate observed-start and request-only tasks before mathematics",
            "Define branch/trunk/IC/FFT/POD, units, slices and shapes before recipes",
            "Separate three-seed variability from predictive confidence intervals",
            "Explain actual failure, preserved evidence and roundoff-only comparison",
            "Fix ambiguous solid-subscript versus stage symbol and Fourier typo",
            "Provide guided answers and explicit independent-exercise acceptance criteria"],
        "limitations_retained": ["Temperature-only outputs", "finite-dimensional control parameters",
            "No new solver mesh truth", "Not full energy closure", "Not multicycle validation",
            "Test previously inspected", "Single-seed spatial-memory repair", "Final whole-report dual review still pending"],
        "chapter_sha256": sha(chapter), "implementation_sha256": sha(Path(__file__)),
        "source_summary_hashes": {name: sha(campaign/name/"summary.json") for name in
            ("wp4-repeat-audit-r1", "wp4-memory-cascade-r2", "wp4-operator-audit-r1", "wp4-deeponet-r1")}}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"summary.json", review)
    return review


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--chapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign_root, args.chapter, args.output_root), ensure_ascii=False, indent=2))
