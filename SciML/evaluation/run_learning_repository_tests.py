"""Run all discovered repository Python unit tests and retain terminal evidence.

Uses existing/fabricated test data, never launches a scientific campaign.
Legacy subprocess test contracts were inspected: validation-only, bundle/plan,
or temporary fake MATLAB/worker. This is not a real MATLAB integration test.
"""
import argparse
import io
import json
from pathlib import Path
import time
import unittest
import subprocess

from SciML.models.v12.run_learning_existing import sha, write_json


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def run(repo, out):
    repo, out = repo.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("preserve previous whole-suite test evidence")
    test_root = repo/"SciML/tests"
    test_files = sorted(test_root.glob("test_*.py"))
    file_list = subprocess.run(["rg", "--files", "-g", "test_*.py"], cwd=repo, check=True,
        capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    elsewhere = [repo/p for p in file_list if (repo/p).parent != test_root]
    # Record rather than silently imply that this discovers other languages or
    # generated package copies. Current repository Python tests reside in SciML/tests.
    physics = sorted(set(repo.glob("*.m")) | set((repo/"SciML/builders").glob("*.m")))
    physics.append(repo/"nixunhuan_solver101_internal.slx")
    before = {str(p.relative_to(repo)): sha(p) for p in physics}
    source_before = {str(p.relative_to(repo)): sha(p) for p in test_files}
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(test_root), pattern="test_*.py")
    ids = list(flatten(suite))
    transcript = io.StringIO()
    result = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    after = {str(p.relative_to(repo)): sha(p) for p in physics}
    source_after = {str(p.relative_to(repo)): sha(p) for p in test_files}
    unchanged = before == after and source_before == source_after
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"terminal_and_cases.json", {"test_ids": ids, "output": transcript.getvalue(),
        "failures": [{"id": test.id(), "traceback": trace} for test, trace in result.failures],
        "errors": [{"id": test.id(), "traceback": trace} for test, trace in result.errors],
        "skipped": [{"id": test.id(), "reason": reason} for test, reason in result.skipped]})
    write_json(out/"source_integrity.json", {"physics_before": before, "physics_after": after,
        "tests_before": source_before, "tests_after": source_after})
    report = {"status": "passed_python_repository_regression" if result.wasSuccessful() and unchanged else "failed_python_repository_regression",
        "test_count": result.testsRun, "discovered_test_count": len(ids), "test_module_files": len(test_files),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "test_and_physics_sources_unchanged": unchanged, "elapsed_seconds": time.perf_counter()-started,
        "scope": "all test_*.py discovered in SciML/tests; not MATLAB integration, scientific campaign, model confirmation, or reader comprehension certification",
        "other_test_files_outside_discovery": [str(p.relative_to(repo)) for p in elsewhere],
        "legacy_subprocess_safety_review": "fake MATLAB/worker and ValidateConfigurationOnly/ValidateAuthorizationOnly/SkipMatlabCheck/PlanOnly inspected before execution; no new solver campaign requested",
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.repo, args.output_root)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["status"].startswith("passed_") else 1)
