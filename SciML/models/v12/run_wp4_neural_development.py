"""Develop the genuinely cycle-invariant WP4 candidate, never old prior gains."""

import argparse
import json
from pathlib import Path

from .run_wp4_temporal_development import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--output", type=Path,
        default=Path("SciML/results/development/wp4_v12_neural_candidate.json"))
    parser.add_argument("--checkpoint", type=Path,
        default=Path("SciML/results/development/wp4_v12_neural_candidate.pkl"))
    parser.add_argument("--freeze-manifest", type=Path,
        default=Path("SciML/results/development/wp4_v12_neural_freeze_manifest.json"))
    args = parser.parse_args()
    result = run(args.dataset_root, args.output, args.epochs, args.checkpoint,
                 args.freeze_manifest, candidate_kind="cycle_invariant_neural")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
