"""Narrow chapters01-03 review: source labels, tables and executable exercises."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .audit_learning_wp5 import verify_artifacts
from .audit_learning_operator_chapter import table_cells
from SciML.models.v12.run_learning_existing import sha, write_json, FEATURES
from SciML.models.v11.cycle_proxy import TARGETS


def run(dataset, campaign, report_root, out):
    if out.exists():
        raise FileExistsError("preserve introduction chapter audit")
    tutorial = verify_artifacts(campaign/"data-tutorial-r1")
    table = pd.read_parquet(dataset/"case_index.parquet")
    fields_checked = 0
    for row in table.itertuples(index=False):
        source = Path(row.source_artifact)
        request_path = source.parent/"sciml_case_request.json"
        if sha(source) != row.source_artifact_sha256 or sha(request_path) != row.source_request_sha256:
            raise ValueError("source changed since tutorial")
        request = json.loads(request_path.read_text())
        raw = loadmat(source, simplify_cells=True)["case_artifact"]
        for key in FEATURES:
            if float(request["case_spec"][key]) != float(getattr(row, key)):
                raise ValueError("input column differs from request")
            fields_checked += 1
        for key in TARGETS:
            if float(raw["performance"][key]) != float(getattr(row, key)):
                raise ValueError("label column differs from raw MAT")
            fields_checked += 1
    files = [report_root/name for name in ("01_项目全景与学习路线.md", "02_数值数据与可追溯契约.md", "03_从文件到训练张量.md")]
    bodies = [p.read_text(encoding="utf-8") for p in files]
    chapter3 = bodies[2]
    cells = 0
    for label, key in (("标量输入 X", "scalar_input"), ("标量标签 Y", "scalar_target"),
                       ("边界模型输入", "boundary_features"), ("场模型 branch", "operator_branch"), ("场模型标签", "operator_fields")):
        if table_cells(chapter3, label)[0] != "×".join(map(str, tutorial["shapes"][key])):
            raise ValueError("shape table mismatch")
        cells += 1
    for group in ("train", "calibration", "test"):
        expected = [tutorial["split_counts"][group], tutorial["split_material_counts"][group].get("recA", 0),
                    tutorial["split_material_counts"][group].get("recAs", 0), tutorial["state_rows_by_split"][group], tutorial["field_rows_by_split"][group]]
        if table_cells(chapter3, group) != list(map(str, expected)):
            raise ValueError("split material table mismatch")
        cells += 5
    if tutorial["raw_to_hdf5_field_comparisons"] != 756 or "756" not in bodies[1]:
        raise ValueError("source comparison count mismatch")
    if tutorial["old_paper_matrix"]["expected_matrix_complete"]:
        raise ValueError("old matrix incompleteness narrative changed")
    links = re.findall(r"\]\(<([^>]+)>\)", "\n".join(bodies))
    for path in links:
        path = re.sub(r":\d+$", "", path)
        if re.match(r"^/[A-Za-z]:/", path):
            path = path[1:]
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    snippets = re.findall(r"```python\n(# exercise: .*?)\n```", chapter3, re.S)
    if len(snippets) != 4:
        raise ValueError("expected four complete exercises")
    exercises = []
    for snippet in snippets:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(snippet, str(files[2])+":exercise", "exec"), {})
        exercises.append({"name": snippet.splitlines()[0], "output": stream.getvalue().strip(), "passed": True})
    stream = io.StringIO()
    tests = unittest.defaultTestLoader.loadTestsFromName("SciML.tests.test_learning_data_tutorial")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    if not result.wasSuccessful() or result.testsRun != 2:
        raise ValueError("data tutorial unit tests failed: "+stream.getvalue())
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"exercises_and_tests.json", {"exercises": exercises, "unit_tests": stream.getvalue()})
    report = {"status": "passed_chapters01_03_narrow_review", "whole_report_complete": False,
        "independent_request_and_raw_label_values_checked": fields_checked, "shape_and_split_cells_checked": cells,
        "local_links_checked": len(links), "chapter_exercises_executed": len(exercises), "new_unit_tests_passed": result.testsRun,
        "readability_scope": "author walkthrough and runnable exercises; no independent human comprehension assessment",
        "walkthrough": ["physical problem before ML equations", "X/state/output tasks distinguished",
            "historical plan versus learning scope explicit", "sourcehashes do not prove physics",
            "756arrays comparison scope not exaggerated", "ID grouping before normalization",
            "unseen material absent from original test disclosed", "shape and physical owner distinguished",
            "rawCOP boundary disclosed before fitting", "oldfutureinitialization entry warned against",
            "exercises demonstrate actual leakage rejection, scaling isolation and broadcasting failure"],
        "chapter_sha256": {p.name: sha(p) for p in files}, "tutorial_summary_sha256": sha(campaign/"data-tutorial-r1/summary.json"),
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {"exercises_and_tests.json": sha(out/"exercises_and_tests.json")}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.report_root, args.output_root), indent=2))
