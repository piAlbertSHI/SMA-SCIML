"""Strict Confirm50 evaluator for the pre-frozen WP2-v1.2 candidate.

This module has no fitting path.  It verifies the freeze manifest, development
report, Confirm50 configuration, training corpus hashes, and exact checkpoint
before loading the frozen model.  Confirmation observations are used only for
predictions and registered gates; they can never select features, parameters,
or estimators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cycle_proxy import TARGETS, _metrics, load_index
from .cycle_proxy_v12_development import load_frozen_checkpoint
from .run_cycle_proxy_v12_development import verify_confirmation_freeze


SCHEMA_VERSION = "sciml_wp2_v12_confirm50_evaluation_v1"
WORK_PACKAGE = "WP2-v1.2-confirmation"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_global_freeze_binding(
    global_freeze_manifest: Path,
    local_freeze: Mapping[str, Any],
    development_output: Path,
    confirmation_config: Path,
    *,
    candidate_name: str = "wp2",
) -> dict[str, Any]:
    """Bind this evaluator to the repository-wide confirmation freeze.

    ``local_freeze`` is deliberately still verified by
    :func:`verify_confirmation_freeze`: it contains the WP2 checkpoint
    metadata needed to load the model.  The repository-wide manifest is a
    separate provenance authority for the preregistered configuration and the
    candidate report.  Keeping the two roles separate lets an older
    work-package freeze retain its model metadata while preventing a stale
    local config hash from being treated as the current confirmation protocol.
    """

    if not global_freeze_manifest.is_file():
        raise FileNotFoundError(
            f"global confirmation freeze is missing: {global_freeze_manifest}"
        )
    global_payload = json.loads(
        global_freeze_manifest.read_text(encoding="utf-8-sig")
    )
    if global_payload.get("schema_version") != "sciml_v12_confirmation_freeze_v1":
        raise ValueError("unsupported global v1.2 confirmation freeze manifest")
    if global_payload.get("status") != "frozen_before_confirmation":
        raise ValueError("global confirmation freeze is not in the frozen state")
    artifacts = global_payload.get("artifact_bindings", {})
    config_binding = artifacts.get("confirm50_config")
    if not isinstance(config_binding, Mapping):
        raise ValueError("global freeze lacks the Confirm50 config binding")
    config_hash = _sha256(confirmation_config)
    if config_hash != str(config_binding.get("sha256", "")).lower():
        raise RuntimeError("Confirm50 config differs from the global freeze")

    local_candidate = local_freeze.get("candidate", {})
    global_candidate = global_payload.get("candidate_bindings", {}).get(candidate_name)
    if not isinstance(global_candidate, Mapping):
        raise ValueError(f"global freeze lacks the {candidate_name} candidate binding")
    for key in ("implementation_sha256", "parameter_sha256"):
        if local_candidate.get(key) != global_candidate.get(key):
            raise RuntimeError(f"{candidate_name} candidate {key} differs from global freeze")
    report_hash = _sha256(development_output)
    if report_hash != str(global_candidate.get("report_sha256", "")).lower():
        raise RuntimeError(f"{candidate_name} development report differs from global freeze")
    return {
        "path": str(global_freeze_manifest.resolve()),
        "sha256": _sha256(global_freeze_manifest),
        "config_sha256": config_hash,
        "candidate": dict(global_candidate),
    }


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            Path(__file__).resolve(),
            Path(__file__).with_name("cycle_proxy_v12_development.py"),
            Path(__file__).with_name("run_cycle_proxy_v12_development.py"),
        )
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _geometry_groups(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            f"R{radius:.12g}_t{thickness:.12g}"
            for radius, thickness in frame[["radius_m", "thickness_m"]].to_numpy(
                float
            )
        ]
    )


def _fixed_baseline_group_report(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    training_means: Mapping[str, float],
    groups: np.ndarray,
    split_name: str,
    *,
    require_each_fold_positive: bool,
) -> dict[str, Any]:
    """Score a frozen predictor; this function never fits or transforms state."""

    values = np.asarray([str(value) for value in groups])
    observed_groups = sorted(set(values))
    folds: list[dict[str, Any]] = []
    fold_failures: list[str] = []
    for group in observed_groups:
        indices = np.flatnonzero(values == group)
        skills: dict[str, float] = {}
        metrics: dict[str, Any] = {}
        baseline_metrics: dict[str, Any] = {}
        for target in TARGETS:
            truth = frame[target].to_numpy(float)[indices]
            prediction = np.asarray(predictions[target], dtype=float)[indices]
            baseline = np.full(len(indices), float(training_means[target]))
            metrics[target] = _metrics(truth, prediction)
            baseline_metrics[target] = _metrics(truth, baseline)
            denominator = baseline_metrics[target]["rmse"]
            skills[target] = (
                float(1.0 - metrics[target]["rmse"] / denominator)
                if denominator > 0.0
                else 0.0
            )
            if require_each_fold_positive and skills[target] <= 0.0:
                fold_failures.append(
                    f"{split_name}:{group}:{target}={skills[target]:.12g}"
                )
        folds.append(
            {
                "held_out_group": group,
                "case_count": len(indices),
                "target_metrics": metrics,
                "frozen_training_mean_baseline_metrics": baseline_metrics,
                "skill_vs_frozen_training_mean_baseline": skills,
            }
        )

    pooled_metrics: dict[str, Any] = {}
    pooled_baselines: dict[str, Any] = {}
    pooled_skills: dict[str, float] = {}
    pooled_failures: list[str] = []
    for target in TARGETS:
        truth = frame[target].to_numpy(float)
        prediction = np.asarray(predictions[target], dtype=float)
        baseline = np.full(len(frame), float(training_means[target]))
        pooled_metrics[target] = _metrics(truth, prediction)
        pooled_baselines[target] = _metrics(truth, baseline)
        denominator = pooled_baselines[target]["rmse"]
        pooled_skills[target] = (
            float(1.0 - pooled_metrics[target]["rmse"] / denominator)
            if denominator > 0.0
            else 0.0
        )
        if pooled_skills[target] <= 0.0:
            pooled_failures.append(
                f"{split_name}:pooled:{target}={pooled_skills[target]:.12g}"
            )

    failures = [*pooled_failures, *fold_failures]
    return {
        "status": "evaluated_all_groups_with_frozen_checkpoint",
        "split_semantics": (
            f"frozen-model {split_name} grouping; no Confirm50 refit or tuning"
        ),
        "observed_groups": observed_groups,
        "fold_count": len(observed_groups),
        "case_count": len(frame),
        "require_each_fold_positive": require_each_fold_positive,
        "folds": folds,
        "pooled": {
            "case_count": len(frame),
            "target_metrics": pooled_metrics,
            "frozen_training_mean_baseline_metrics": pooled_baselines,
            "skill_vs_frozen_training_mean_baseline": pooled_skills,
        },
        "nonpositive_registered_skills": failures,
        "passed": not failures,
    }


def _confirmation_frame(
    confirmation_dataset_root: Path, confirmation_config: Path
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    config = json.loads(confirmation_config.read_text(encoding="utf-8"))
    configured_ids = [str(case["case_id"]) for case in config.get("cases", [])]
    frame = load_index(confirmation_dataset_root).copy()
    frame["case_id"] = frame["case_id"].astype(str)
    blockers: list[str] = []
    if len(configured_ids) != 50 or len(set(configured_ids)) != 50:
        raise ValueError("Confirm50 freeze does not contain 50 unique case IDs")
    observed_counts = frame["case_id"].value_counts().to_dict()
    configured_set = set(configured_ids)
    missing = sorted(configured_set - set(observed_counts))
    extra = sorted(set(observed_counts) - configured_set)
    repeated = sorted(
        case_id
        for case_id, count in observed_counts.items()
        if case_id in configured_set and int(count) != 1
    )
    if missing:
        blockers.append(f"{len(missing)} configured Confirm50 cases are missing")
    if extra:
        blockers.append(f"{len(extra)} unregistered cases entered Confirm50")
    if repeated:
        blockers.append(f"{len(repeated)} Confirm50 case IDs are not one-row cases")
    # Reindex in configuration order.  Missing/repeated records cannot be
    # meaningfully scored and are returned as an explicit incomplete frame.
    if missing or repeated:
        return frame.iloc[0:0].copy(), config, blockers
    indexed = frame.set_index("case_id", drop=False)
    ordered = indexed.loc[configured_ids].reset_index(drop=True)
    if ordered["condition_id"].astype(str).nunique() != 50:
        blockers.append("Confirm50 does not contain 50 independent conditions")
    if not ordered["physics_ok"].astype(bool).all():
        blockers.append("one or more Confirm50 cases are not physics-valid")
    if not ordered["pump_work_available"].astype(bool).all():
        blockers.append("one or more Confirm50 cases lack complete pump work")
    if ordered[list(TARGETS)].isna().any().any():
        blockers.append("one or more Confirm50 targets are missing")
    return ordered, config, blockers


def evaluate_confirmation(
    training_dataset_root: Path,
    confirmation_dataset_root: Path,
    confirmation_config: Path,
    development_output: Path,
    checkpoint: Path,
    freeze_manifest: Path,
    output: Path,
    global_freeze_manifest: Path | None = None,
) -> dict[str, Any]:
    """Evaluate untouched Confirm50 with the verified frozen checkpoint only."""

    freeze_checks = verify_confirmation_freeze(
        training_dataset_root,
        confirmation_config,
        development_output,
        checkpoint,
        freeze_manifest,
    )
    freeze = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    global_binding = _verify_global_freeze_binding(
        global_freeze_manifest or freeze_manifest,
        freeze,
        development_output,
        confirmation_config,
    )
    fold_audit = freeze["development_evidence"]["fold_audit"]
    if not fold_audit.get("development_freeze_ready", False):
        raise ValueError("WP2-v1.2 development candidate was not freeze-ready")
    candidate = freeze["candidate"]
    expected_metadata = {
        "dataset_source_content_sha256": freeze["training_corpus"][
            "dataset_source_content_sha256"
        ],
        "implementation_sha256": candidate["implementation_sha256"],
        "confirmation_config_sha256": freeze["confirmation_protocol"][
            "configuration_sha256"
        ],
    }
    model = load_frozen_checkpoint(checkpoint, expected_metadata=expected_metadata)
    repeated_model = load_frozen_checkpoint(
        checkpoint, expected_metadata=expected_metadata
    )

    frame, config, blockers = _confirmation_frame(
        confirmation_dataset_root, confirmation_config
    )
    quality = json.loads(
        (confirmation_dataset_root / "quality_report.json").read_text(
            encoding="utf-8"
        )
    )
    confirmation_source_hash = str(quality.get("source_content_sha256", ""))
    if len(confirmation_source_hash) != 64:
        raise ValueError("Confirm50 quality report lacks a source-content hash")

    gate_reports: dict[str, Any] = {}
    checkpoint_roundtrip_verified = False
    if not frame.empty and not blockers:
        predictions = model.predict(frame)
        repeated_predictions = repeated_model.predict(frame)
        checkpoint_roundtrip_verified = (
            model.parameter_sha256() == repeated_model.parameter_sha256()
            and model.parameter_sha256() == candidate["parameter_sha256"]
            and all(
                np.array_equal(predictions[target], repeated_predictions[target])
                for target in TARGETS
            )
        )
        if not checkpoint_roundtrip_verified:
            blockers.append("checkpoint round-trip changed parameters or predictions")
        training = load_index(training_dataset_root)
        training = training[
            training["physics_ok"].astype(bool)
            & training["pump_work_available"].astype(bool)
        ].dropna(subset=list(TARGETS))
        training_means = {
            target: float(training[target].mean()) for target in TARGETS
        }
        gate_reports = {
            "condition_holdout": _fixed_baseline_group_report(
                frame,
                predictions,
                training_means,
                frame["condition_id"].astype(str).to_numpy(),
                "condition",
                require_each_fold_positive=False,
            ),
            "material_group_holdout": _fixed_baseline_group_report(
                frame,
                predictions,
                training_means,
                frame["material_id"].astype(str).to_numpy(),
                "material",
                require_each_fold_positive=True,
            ),
            "geometry_group_holdout": _fixed_baseline_group_report(
                frame,
                predictions,
                training_means,
                _geometry_groups(frame),
                "geometry",
                require_each_fold_positive=True,
            ),
        }
        for split_name, report in gate_reports.items():
            blockers.extend(report["nonpositive_registered_skills"])
            if split_name == "material_group_holdout" and report["fold_count"] != 2:
                blockers.append("Confirm50 does not cover both registered materials")
            if split_name == "geometry_group_holdout" and report["fold_count"] != 5:
                blockers.append("Confirm50 does not cover all five registered geometries")
    else:
        blockers.append("Confirm50 is incomplete and cannot be scored")

    paper_grade_ready = (
        not blockers
        and checkpoint_roundtrip_verified
        and int(freeze["training_corpus"]["condition_count"]) >= 200
        and len(frame) == 50
        and all(report["passed"] for report in gate_reports.values())
    )
    if int(freeze["training_corpus"]["condition_count"]) < 200:
        blockers.append("frozen training corpus has fewer than 200 conditions")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "confirmatory": True,
        "readiness_eligible": True,
        "paper_grade_ready": paper_grade_ready,
        # The local WP2 freeze is verified above for checkpoint loading, but
        # reports are bound to the repository-wide immutable freeze consumed by
        # readiness_v12.py.
        "freeze_manifest_sha256": global_binding["sha256"],
        "local_freeze_manifest_sha256": _sha256(freeze_manifest),
        "implementation_sha256": implementation_sha256(),
        "parameter_sha256": model.parameter_sha256(),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_roundtrip_verified": checkpoint_roundtrip_verified,
        "confirmation_dataset_source_content_sha256": confirmation_source_hash,
        "confirmation_bindings": {
            "confirm50_source_content_sha256": confirmation_source_hash,
        },
        "confirmation_config_sha256": global_binding["config_sha256"],
        "confirmation_case_count": len(frame),
        "confirmation_independent_condition_count": (
            int(frame["condition_id"].astype(str).nunique())
            if not frame.empty
            else 0
        ),
        "training_condition_count": int(freeze["training_corpus"]["condition_count"]),
        "frozen_model_loaded_without_refit": True,
        "confirmation_results_used_for_selection_or_tuning": False,
        "freeze_verification": freeze_checks,
        "global_freeze_binding": global_binding,
        "registered_gates": gate_reports,
        "blocking_reasons": sorted(set(blockers)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dataset-root", type=Path, required=True)
    parser.add_argument("--confirmation-dataset-root", type=Path, required=True)
    parser.add_argument("--confirmation-config", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--global-freeze-manifest",
        type=Path,
        default=Path("SciML/results/v12_confirmation_freeze.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_confirmation(
        args.training_dataset_root,
        args.confirmation_dataset_root,
        args.confirmation_config,
        args.development_output,
        args.checkpoint,
        args.freeze_manifest,
        args.output,
        args.global_freeze_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
