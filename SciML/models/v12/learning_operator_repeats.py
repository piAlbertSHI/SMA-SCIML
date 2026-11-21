"""Fixed-budget additional seeds for all four existing operator variants."""
import argparse
import json
from pathlib import Path

from .learning_coordinate_operator import KINDS, run as train_operator
from .run_learning_existing import sha, write_json


def run(dataset, protocol, out, seed):
    if seed not in (20260911, 20260912):
        raise ValueError("predeclared additional seeds only")
    if out.exists():
        raise FileExistsError("do not overwrite a previous repeat")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"seed": seed, "variants": list(KINDS),
        "optimizer_updates_per_variant": 400, "hyperparameter_selection": False,
        "source_protocol_sha256": sha(protocol), "fresh_solver_calls": 0,
        "implementation_sha256": sha(Path(__file__))})
    reports = {}
    for kind in KINDS:
        result = train_operator(dataset, protocol, out/kind, kind, seed)
        reports[kind] = {"test_scores": result["scores"]["test"]["model"],
                         "elapsed_seconds": result["elapsed_seconds"],
                         "summary_sha256": sha(out/kind/"summary.json")}
    result = {"status": "completed_fixed_seed_operator_repeat", "seed": seed,
        "learning_only": True, "fresh_solver_calls": 0, "runs": reports,
        "artifacts": {"protocol.json": sha(out/"protocol.json")}}
    write_json(out/"summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_protocol, args.output_root, args.seed), indent=2))
