"""Run the canonical leakage-audited WP3 stage-transition prototype."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .corpus import OperatorCorpus
from .training import fit_torch_from_corpus


SCIML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = SCIML_ROOT / "configs" / "wp3_hybrid.json"
DEFAULT_DATASET = SCIML_ROOT / "datasets" / "v1.0"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    dataset_root: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    output = config["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / str(output["checkpoint"])
    corpus = OperatorCorpus(dataset_root)
    corpus.assert_group_disjoint()
    corpus.assert_protocol_declarations_match_rows()
    _, result = fit_torch_from_corpus(
        corpus,
        hidden_sizes=tuple(int(value) for value in training["hidden_sizes"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        seed=int(training["seed"]),
        checkpoint_path=checkpoint,
    )
    report = {
        "work_package": "WP3",
        "status": "trained_small_data_stage4_prototype",
        "claim_boundary": config["claim_boundary"],
        "result": asdict(result),
        "checkpoint_provenance": str(checkpoint) + ".provenance.json",
    }
    report_path = output_dir / str(output["report"])
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.config, args.dataset_root, args.output_dir)
    result = report["result"]
    print(json.dumps({
        "report_path": report["report_path"],
        "checkpoint_path": result["checkpoint_path"],
        "parameter_sha256": result["parameter_sha256"],
        "train_mae": result["train_metrics"]["mae"],
        "test_mae": result["held_out_metrics"]["mae"],
        "heuristic_test_mae": result["held_out_heuristic_metrics"]["mae"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
