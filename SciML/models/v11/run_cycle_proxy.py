"""Run the v1.1 full-cycle scalar proxy with an honest readiness report."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .cycle_proxy import PREREGISTERED_RIDGE_ALPHA, fit_cycle_proxy


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve(), Path(__file__).with_name("cycle_proxy.py"))):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_cycle_proxy(args.dataset_root)
    quality = json.loads(
        (args.dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )
    payload = {
        "work_package": "WP2-v1.1",
        "dataset_source_content_sha256": quality["source_content_sha256"],
        "implementation_sha256": _implementation_sha256(),
        "training_config": {
            "seed": 20260820,
            "regressor": "standardized_ridge",
            "ridge_alpha": PREREGISTERED_RIDGE_ALPHA,
            "alpha_selection": "frozen_on_pre-Paper220_engineering_development_corpus",
        },
        "paper_grade_requirements": {
            "minimum_valid_cycles": 200,
            "minimum_independent_conditions": 200,
            "reliability_group_oof_required": True,
            "material_and_geometry_group_holdouts_required": True,
            "minimum_pooled_group_holdout_cycles": 40,
            "positive_condition_holdout_skill_required_for_all_targets": True,
            "positive_skill_over_train_mean_required_for_all_targets": True,
            "positive_skill_required_in_every_structured_holdout_fold": True,
        },
        **asdict(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
