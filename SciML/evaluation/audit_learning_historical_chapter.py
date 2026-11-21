"""Narrow chapter07C evidence, executable exercise and interface-test review."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import re
import unittest

import numpy as np

from .audit_learning_operator_chapter import table_cells
from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.run_learning_existing import sha, write_json


def run(campaign, chapter, out):
    if out.exists():
        raise FileExistsError("preserve historical chapter audit")
    body = chapter.read_text(encoding="utf-8")
    names = ("historical-rollout-r1", "historical-rollout-audit-r1",
             "historical-fields-r1", "historical-fields-audit-r1")
    evidence = {name: verify_artifacts(campaign/name) for name in names}
    boundary = evidence["historical-rollout-audit-r1"]
    fields = evidence["historical-fields-r1"]
    cells = 0
    for name, values in boundary["fixed_three_cycle_cohort"].items():
        expected = [f"{values[str(c)]['mean_sma_rmse_K']:.6f}" for c in (1, 2, 3)]
        if table_cells(body, name) != expected:
            raise ValueError("boundary metric table mismatch: "+name)
        cells += 3
    means = fields["fixed_three_cycle_seed_mean_rmse_K"]
    for key in means["observed_start"]:
        numbers = [means[mode][key][field] for mode in ("observed_start", "autonomous_boundary") for field in ("sma", "htf")]
        numbers += [fields["fixed_three_cycle_pod_scores"][key][field]["future_rmse_K"] for field in ("sma", "htf")]
        if table_cells(body, key) != [f"{v:.6f}" for v in numbers]:
            raise ValueError("field metric table mismatch: "+key)
        cells += len(numbers)
    for label, value in (("训练工况", boundary["training_cases"]), ("测试工况", boundary["test_cases"]),
                         ("训练边界转移", boundary["training_transitions"]), ("测试边界转移", boundary["test_transitions"]),
                         ("训练换热场样本", evidence["historical-fields-audit-r1"]["training_field_samples"]),
                         ("测试换热场样本", evidence["historical-fields-audit-r1"]["test_field_samples"])):
        if table_cells(body, label) != [str(value)]:
            raise ValueError("count table mismatch: "+label)
        cells += 1
    protocol = json.loads((campaign/"historical-rollout-r1/protocol.json").read_text())
    for fingerprint in protocol["fingerprints"].values():
        if fingerprint not in body:
            raise ValueError("missing source fingerprint")
    checks = json.loads((campaign/"historical-rollout-r1/source_checks.json").read_text())
    if len(checks) != 12 or max(c["field_reconstruction_max_K"] for c in checks) > 6e-14:
        raise ValueError("spatial reconstruction narrative unsupported")
    change = checks[0]["restart_mechanics_max_changes"][0]
    np.testing.assert_allclose([change[k] for k in ("sma_K", "xi", "u_m", "w_m")],
                               [0., .215862, 3.13e-5, 1.706e-3], rtol=.0003, atol=1e-8)
    links = re.findall(r"\]\(<([^>]+)>\)", body)
    for link in links:
        path = re.sub(r":\d+$", "", link)
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    exercises = re.findall(r"```python\n(# exercise: historical_interfaces\n.*?)\n```", body, re.S)
    if len(exercises) != 1:
        raise ValueError("expected one complete executable reference exercise")
    # Executes only the local, explicitly identified chapter reference exercise.
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(exercises[0], str(chapter)+":exercise", "exec"), {})
    if output.getvalue().strip() != "historical interface exercises passed":
        raise ValueError("reference exercise did not finish")
    commands = re.findall(r"^python .*", body, re.M)
    if len(commands) != 5:
        raise ValueError("unexpected CLI recipe count")
    for command in commands:
        words = command.split()
        module = words[words.index("-m")+1]
        if module != "unittest":
            tree = ast.parse(Path(module.replace(".", "/")+".py").read_text(encoding="utf-8"))
            flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                     for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            if any(word.startswith("--") and word not in flags for word in words):
                raise ValueError("unknown documented CLI flag")
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName("SciML.tests.test_learning_historical_rollout")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if result.testsRun != 5 or not result.wasSuccessful():
        raise ValueError("historical interface tests failed: "+stream.getvalue())
    # These checks establish explicit disclosure, NOT comprehension by a reader.
    disclosures = ["不是已证明充分的物理 Markov 状态", "不是完整耦合神经物理求解器",
                   "不是去掉组件后重新训练", "神经模型在第三循环明显不如线性基线",
                   "不是全时空场误差", "不会启动 MATLAB", "当前 63 例"]
    for text in disclosures:
        if text not in body:
            raise ValueError("missing scope disclosure: "+text)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"interface_tests.json", {"tests_run": result.testsRun, "passed": True, "output": stream.getvalue()})
    report = {"status": "passed_chapter07C_narrow_evidence_review", "whole_report_complete": False,
        "checked_metric_and_count_cells": cells, "local_links_and_images_checked": len(links),
        "fingerprints_and_entry_reconstruction_checked": True, "restart_worked_example_checked": True,
        "reference_exercise_executed": True, "interface_tests_passed": result.testsRun,
        "documented_cli_recipes": len(commands),
        "command_scope": "AST module/flag inventory only; corresponding original runs are audited separately; new output paths not rerun",
        "readability_scope": "author walkthrough plus executable worked example; no independent human comprehension test or whole-report review",
        "walkthrough": ["Rollout introduced before formulas", "State table gives slices and units",
            "Five interfaces map targets to source owners", "Training versus inference permissions explicit",
            "Fixed two-case cohort avoids4/4/2 composition confound", "Boundary versus field RMSE separated",
            "Observed and predicted starts distinguish information budgets", "Negative comparisons retained",
            "Guided and independent exercises include expected answers and acceptance criteria"],
        "chapter_sha256": sha(chapter), "implementation_sha256": sha(Path(__file__)),
        "source_summary_hashes": {name: sha(campaign/name/"summary.json") for name in names},
        "artifacts": {"interface_tests.json": sha(out/"interface_tests.json")}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--chapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign_root, args.chapter, args.output_root), ensure_ascii=False, indent=2))
