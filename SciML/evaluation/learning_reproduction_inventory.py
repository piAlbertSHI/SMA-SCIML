"""Read-only learning preflight and documented-CLI inventory; never train/solve.

Writes a NEW inventory directory. Optional --check-help invokes only --help in
this same process; never evaluates documented command text as shell input.
"""
import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import re
import runpy
import shlex
import sys

from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12.run_learning_existing import sha, write_json, validate_dataset

EXPERIMENTS = ("wp2-strong-r1", "wp3-boundary-r2", "wp4-deeponet-r1", "wp4-fourier-r1", "wp4-no-stage-r1", "wp4-no-ic-r1",
    "wp4-repeat-s20260911", "wp4-repeat-s20260912", "wp4-memory-cascade-r2", "wp5-pinn-r1", "wp5-local-thermal-r2",
    "wp6-uq-r1", "wp6-active-r1", "wp6-design-r1", "ood-geometry-r1", "ood-material-r1", "ood-protocol-r1", "ood-combined-r1",
    "historical-rollout-r1", "historical-fields-r1", "staged-physics-r1", "field-energy-information-r1", "data-tutorial-r1")


def parse_recipe(command):
    words = shlex.split(command, posix=True)
    if not words or words[0] != "python" or "-m" not in words:
        raise ValueError("expected an explicit Python module recipe")
    index = words.index("-m")
    if index+1 >= len(words) or words[index+1].startswith("-"):
        raise ValueError("missing module after -m")
    module = words[index+1]
    if module != "unittest" and not module.startswith(("SciML.models.v12.learning", "SciML.active_learning.learning", "SciML.evaluation.")):
        raise ValueError("out-of-scope documented module")
    options = {}
    for i in range(index+2, len(words)):
        if words[i].startswith("--"):
            option, equal, value = words[i].partition("=")
            if option in options:
                raise ValueError("duplicate documented option: "+option)
            options[option] = value if equal else (words[i+1] if i+1 < len(words) and not words[i+1].startswith("--") else None)
    return module, options


def verify_help_options(options, help_output):
    """Compare exact option tokens, not --seed accidentally matching --seeds."""
    available = set(re.findall(r"(?<![\w-])--[a-zA-Z][\w-]*", help_output))
    for option in options:
        if option not in available:
            raise ValueError("documented option absent from actual argparse help: "+option)


def read_help(module):
    if module == "unittest":
        raise ValueError("unit tests are not launched by this inventory")
    stdout, stderr = io.StringIO(), io.StringIO()
    previous = sys.argv
    try:
        sys.argv = [module, "--help"]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_module(module, run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise RuntimeError("--help failed for "+module) from error
            else:
                raise RuntimeError("expected argparse --help exit before executing task")
    finally:
        sys.argv = previous
    return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def run(dataset, campaign, chapters, out, check_help=False):
    if out.exists():
        raise FileExistsError("preserve previous reproduction inventory")
    table, quality, missing = validate_dataset(dataset)
    originals = dataset.parents[1]/"results/learning-existing63-20260910-r1/protocol.json"
    protocol = json.loads(originals.read_text())
    if protocol["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("initial protocol differs from dataset")
    experiments = []
    for name in EXPERIMENTS:
        summary = verify_artifacts(campaign/name)
        experiments.append({"name": name, "status": summary["status"], "summary_sha256": sha(campaign/name/"summary.json"),
            "artifact_count": len(summary["artifacts"]), "recorded_elapsed_seconds": summary.get("elapsed_seconds"),
            "scope": "artifact integrity only; no fresh fitting in this preflight"})
    recipes, modules, help_text = [], {}, {}
    for chapter in sorted(chapters.glob("*.md")):
        for line in chapter.read_text(encoding="utf-8").splitlines():
            if not line.startswith("python "):
                continue
            module, options = parse_recipe(line)
            if module != "unittest":
                path = Path(module.replace(".", "/")+".py")
                if not path.is_file():
                    raise FileNotFoundError(path)
                modules[module] = sha(path)
            files = {}
            for key, value in options.items():
                if value is not None and (key.endswith("-root") or key in ("--original-protocol", "--original-result", "--repo", "--chapter")):
                    files[key] = {"path": value, "exists": Path(value).exists(),
                                  "role": "new output, existence means rerun refused" if key == "--output-root" else "input dependency"}
            recipes.append({"chapter": chapter.name, "command": line, "module": module, "options": options, "paths": files})
    if check_help:
        for module in sorted(modules):
            help_text[module] = read_help(module)
        for recipe in recipes:
            if recipe["module"] != "unittest":
                verify_help_options(recipe["options"], help_text[recipe["module"]]["stdout"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"recipes.json", recipes)
    write_json(out/"experiment_integrity.json", experiments)
    write_json(out/"cli_help.json", help_text)
    result = {"status": "completed_learning_reproduction_preflight", "learning_only": True,
        "fresh_solver_calls": 0, "training_calls": 0, "case_count": len(table),
        "experiment_roots_verified": len(experiments), "documented_commands": len(recipes), "unique_task_modules": len(modules),
        "actual_cli_help_checked": check_help, "scope": "source/artifact/CLI/precondition inventory, not whole-report audit or new execution of documented training recipes",
        "environment": {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "h5py", "scikit-learn", "torch", "pyarrow", "matplotlib", "threadpoolctl")}},
        "dataset_source_content_sha256": quality["source_content_sha256"], "initial_protocol_sha256": sha(originals),
        "documented_module_sha256": modules, "chapter_sha256": {p.name: sha(p) for p in sorted(chapters.glob("*.md"))},
        "implementation_sha256": sha(Path(__file__)), "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return {k: result[k] for k in ("status", "training_calls", "fresh_solver_calls", "experiment_roots_verified", "documented_commands", "unique_task_modules", "actual_cli_help_checked", "environment")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check-help", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.report_root, args.output_root, args.check_help), indent=2))
