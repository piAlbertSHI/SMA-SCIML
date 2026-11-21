"""Train and audit the v1.1 four-stage hybrid state-space pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .state_space import (
    LAYOUT,
    PREREGISTERED_RESIDUAL_GAIN,
    HybridStageModel,
    TransitionRecord,
    build_transition_corpus,
    continuity_audit,
)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve(), Path(__file__).with_name("state_space.py"))):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metric(prediction: np.ndarray, target: np.ndarray, slc: slice | None = None) -> dict[str, float]:
    error = prediction - target
    if slc is not None:
        error = error[..., slc]
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def _rmse_skill(model: dict[str, float], baseline: dict[str, float]) -> float:
    denominator = float(baseline["rmse"])
    return float(1.0 - float(model["rmse"]) / denominator) if denominator > 0.0 else 0.0


def _nonpositive_structured_skills(report: dict[str, object]) -> list[str]:
    failures: list[str] = []
    pooled = report.get("pooled", {})
    if isinstance(pooled, dict):
        skills = pooled.get("skill_vs_stage_mean_baseline", {})
        if isinstance(skills, dict):
            for target, skill in skills.items():
                if float(skill) <= 0.0:
                    failures.append(f"pooled:{target}={float(skill):.6g}")
    folds = report.get("folds", [])
    if isinstance(folds, list):
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            group = str(fold.get("held_out_group", "unknown"))
            skills = fold.get("skill_vs_stage_mean_baseline", {})
            if isinstance(skills, dict):
                for target, skill in skills.items():
                    if float(skill) <= 0.0:
                        failures.append(f"{group}:{target}={float(skill):.6g}")
    return failures


def _deduplicated_cycle1(records: list[TransitionRecord]) -> tuple[list[TransitionRecord], int]:
    selected = []
    seen = set()
    for record in records:
        if record.cycle_index != 1:
            continue
        key = (record.condition_id, record.stage_index)
        if key not in seen:
            selected.append(record)
            seen.add(key)
    raw_count = sum(record.cycle_index == 1 for record in records)
    return selected, raw_count - len(selected)


def _condition_partition(
    cycle1: list[TransitionRecord], all_records: list[TransitionRecord]
) -> tuple[list[TransitionRecord], list[TransitionRecord], list[str]]:
    conditions = sorted({record.condition_id for record in cycle1})
    if len(conditions) < 8:
        return cycle1, [], []
    temporal_conditions = {
        record.condition_id for record in all_records if record.cycle_index >= 2
    }
    candidates = [condition for condition in conditions if condition not in temporal_conditions]
    test_count = max(2, int(round(0.2 * len(conditions))))
    ranked = sorted(
        candidates,
        key=lambda value: hashlib.sha256(f"20260820:{value}".encode()).hexdigest(),
    )
    held_out = set(ranked[: min(test_count, len(ranked))])
    return (
        [record for record in cycle1 if record.condition_id not in held_out],
        [record for record in cycle1 if record.condition_id in held_out],
        sorted(held_out),
    )


def _autonomous_multicycle_rollout(
    model: HybridStageModel,
    records: list[TransitionRecord],
) -> tuple[
    list[TransitionRecord],
    np.ndarray,
    np.ndarray,
    list[float],
    list[dict[str, object]],
]:
    """Roll each multi-cycle case independently from its cycle-1 initial state.

    Predictions for cycle 2/3 therefore contain the accumulated cycle-1 model
    error.  State is never carried between different cases or conditions.
    """

    grouped: dict[str, list[TransitionRecord]] = {}
    for record in records:
        grouped.setdefault(record.case_id, []).append(record)

    evaluated: list[TransitionRecord] = []
    predictions: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    trust_scores: list[float] = []
    drift: list[dict[str, object]] = []
    for case_id in sorted(grouped):
        ordered = sorted(
            grouped[case_id], key=lambda record: (record.cycle_index, record.stage_index)
        )
        if max(record.cycle_index for record in ordered) < 2:
            continue
        state = ordered[0].state_in.copy()
        baseline_state = ordered[0].state_in.copy()
        cycle1_reference: dict[int, np.ndarray] = {}
        for record in ordered:
            trust = model.trust_score(state, record.controls, record.stage_index)
            state = model.predict(state, record.controls, record.stage_index)
            baseline_state = model.predict_stage_mean(
                baseline_state, record.stage_index
            )
            if record.cycle_index == 1:
                cycle1_reference[record.stage_index] = record.state_out
                continue
            if record.stage_index not in cycle1_reference:
                raise RuntimeError(
                    f"{case_id} lacks a cycle-1 reference for stage {record.stage_index}"
                )
            evaluated.append(record)
            predictions.append(state.copy())
            baselines.append(baseline_state.copy())
            trust_scores.append(trust)
            reference = cycle1_reference[record.stage_index]
            drift.append(
                {
                    "case_id": case_id,
                    "condition_id": record.condition_id,
                    "cycle_index": int(record.cycle_index),
                    "stage_index": int(record.stage_index),
                    "observed_l2_from_cycle1": float(
                        np.linalg.norm(record.state_out - reference)
                    ),
                    "predicted_l2_from_cycle1": float(
                        np.linalg.norm(state - reference)
                    ),
                }
            )

    if not evaluated:
        return (
            [],
            np.empty((0, LAYOUT.size)),
            np.empty((0, LAYOUT.size)),
            [],
            [],
        )
    return (
        evaluated,
        np.stack(predictions),
        np.stack(baselines),
        trust_scores,
        drift,
    )


def _blocking_reasons(
    records: list[TransitionRecord],
    condition_test: list[TransitionRecord],
    held_out_conditions: list[str],
    rollout_records: list[TransitionRecord],
) -> list[str]:
    reasons = []
    if len(records) < 200:
        reasons.append(
            f"Only {len(records)} observed transitions are available; "
            "the paper-grade minimum is 200."
        )
    if not condition_test:
        reasons.append(
            "No registered cross-condition holdout is available; at least eight "
            "independent conditions are required to activate it."
        )
    elif len(condition_test) < 40:
        reasons.append(
            "Cross-condition evaluation has only "
            f"{len(held_out_conditions)} held-out conditions and "
            f"{len(condition_test)} transitions; the paper-grade minimum is 40 transitions."
        )
    temporal_conditions = {record.condition_id for record in rollout_records}
    if len(temporal_conditions) < 10:
        reasons.append(
            f"Cycle-2/3 temporal rollout contains {len(rollout_records)} transitions from "
            f"only {len(temporal_conditions)} physical condition(s); the paper-grade "
            "minimum is 10 temporal conditions."
        )
    return reasons


def _structured_group_ood(
    cycle1: list[TransitionRecord],
    group_attribute: str,
    epochs: int,
) -> dict[str, object]:
    groups = sorted({str(getattr(record, group_attribute)) for record in cycle1})
    required_group_count = 2 if group_attribute == "material_id" else 3
    split_name = "material" if group_attribute == "material_id" else "geometry"
    if len(groups) < required_group_count:
        return {
            "status": f"unavailable_fewer_than_{required_group_count}_{split_name}_groups",
            "group_attribute": group_attribute,
            "observed_groups": groups,
            "fold_count": 0,
        }

    folds = []
    pooled_prediction = []
    pooled_baseline = []
    pooled_target = []
    for group in groups:
        train = [record for record in cycle1 if str(getattr(record, group_attribute)) != group]
        test = [record for record in cycle1 if str(getattr(record, group_attribute)) == group]
        train_conditions = {record.condition_id for record in train}
        test_conditions = {record.condition_id for record in test}
        if len(train_conditions) < 3 or len(test_conditions) < 2:
            continue
        fold_model = HybridStageModel(seed=20260820, hidden=128)
        fold_training = fold_model.fit(train, epochs=epochs)
        prediction = fold_model.predict_records(test)
        baseline = fold_model.predict_stage_mean_records(test)
        target = np.stack([record.state_out for record in test])
        model_all = _metric(prediction, target)
        model_sma = _metric(prediction, target, LAYOUT.sma_T)
        baseline_all = _metric(baseline, target)
        baseline_sma = _metric(baseline, target, LAYOUT.sma_T)
        folds.append(
            {
                "held_out_group": group,
                "train_condition_count": len(train_conditions),
                "test_condition_count": len(test_conditions),
                "test_transition_count": len(test),
                "all_state": model_all,
                "sma_temperature": model_sma,
                "stage_mean_baseline": {
                    "all_state": baseline_all,
                    "sma_temperature": baseline_sma,
                },
                "skill_vs_stage_mean_baseline": {
                    "all_state": _rmse_skill(model_all, baseline_all),
                    "sma_temperature": _rmse_skill(model_sma, baseline_sma),
                },
                "training": fold_training,
                "parameter_sha256": fold_model.parameter_sha256(),
            }
        )
        pooled_prediction.append(prediction)
        pooled_baseline.append(baseline)
        pooled_target.append(target)

    if not folds:
        return {
            "status": "unavailable_insufficient_conditions_per_group",
            "group_attribute": group_attribute,
            "observed_groups": groups,
            "minimum_train_conditions_per_fold": 3,
            "minimum_test_conditions_per_fold": 2,
            "fold_count": 0,
        }

    prediction = np.concatenate(pooled_prediction)
    baseline = np.concatenate(pooled_baseline)
    target = np.concatenate(pooled_target)
    model_all = _metric(prediction, target)
    model_sma = _metric(prediction, target, LAYOUT.sma_T)
    baseline_all = _metric(baseline, target)
    baseline_sma = _metric(baseline, target, LAYOUT.sma_T)
    evaluated_groups = [str(fold["held_out_group"]) for fold in folds]
    skipped_groups = [group for group in groups if group not in evaluated_groups]
    return {
        "status": "evaluated_all_groups" if not skipped_groups else "partial_evaluation",
        "group_attribute": group_attribute,
        "split_semantics": f"Leave-one-{split_name}-group-out refit; no held-out group enters training.",
        "observed_groups": groups,
        "fold_count": len(folds),
        "evaluated_groups": evaluated_groups,
        "skipped_groups": skipped_groups,
        "folds": folds,
        "pooled": {
            "all_state": model_all,
            "sma_temperature": model_sma,
            "stage_mean_baseline": {
                "all_state": baseline_all,
                "sma_temperature": baseline_sma,
            },
            "skill_vs_stage_mean_baseline": {
                "all_state": _rmse_skill(model_all, baseline_all),
                "sma_temperature": _rmse_skill(model_sma, baseline_sma),
            },
            "transition_count": int(target.shape[0]),
        },
    }


def run(dataset_root: Path, output: Path, checkpoint: Path, epochs: int) -> dict[str, object]:
    records = build_transition_corpus(dataset_root)
    dataset_source_content_sha256 = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )["source_content_sha256"]
    implementation_sha256 = _implementation_sha256()
    cycle1_unique, duplicate_count = _deduplicated_cycle1(records)
    train, condition_test, held_out_conditions = _condition_partition(cycle1_unique, records)
    # Keep the deterministic condition holdout genuinely unseen while fitting
    # the serialized/deployment state model on the complete cycle-1 corpus.
    # The latter is the registered 200-condition evidence set; using the
    # 80/20 evaluation subset as the checkpoint would unnecessarily discard
    # valid training evidence and distort the temporal rollout.
    condition_model = HybridStageModel(seed=20260820, hidden=128)
    condition_training = condition_model.fit(train, epochs=epochs)
    model = HybridStageModel(seed=20260820, hidden=128)
    train_metrics = model.fit(cycle1_unique, epochs=epochs)
    (
        rollout_records,
        rollout_prediction_array,
        rollout_baseline_array,
        trust_scores,
        rollout_drift,
    ) = _autonomous_multicycle_rollout(
        model,
        records,
    )
    structured_ood = {
        "material_group_holdout": _structured_group_ood(
            cycle1_unique, "material_id", epochs
        ),
        "geometry_group_holdout": _structured_group_ood(
            cycle1_unique, "geometry_id", epochs
        ),
    }
    if condition_test:
        condition_prediction = condition_model.predict_records(condition_test)
        condition_baseline = condition_model.predict_stage_mean_records(condition_test)
        condition_target = np.stack([record.state_out for record in condition_test])
        condition_all = _metric(condition_prediction, condition_target)
        condition_sma = _metric(condition_prediction, condition_target, LAYOUT.sma_T)
        baseline_all = _metric(condition_baseline, condition_target)
        baseline_sma = _metric(condition_baseline, condition_target, LAYOUT.sma_T)
        condition_metrics: dict[str, object] = {
            "split_semantics": (
                "Deterministic physical-condition holdout; material and geometry "
                "levels may remain represented in training. This is not a strict "
                "unseen-material or unseen-geometry split."
            ),
            "all_state": condition_all,
            "sma_temperature": condition_sma,
            "stage_mean_baseline": {
                "all_state": baseline_all,
                "sma_temperature": baseline_sma,
            },
            "skill_vs_stage_mean_baseline": {
                "all_state": _rmse_skill(condition_all, baseline_all),
                "sma_temperature": _rmse_skill(condition_sma, baseline_sma),
            },
            "held_out_condition_ids": held_out_conditions,
            "transition_count": len(condition_test),
            "training_transition_count": len(train),
            "training_condition_count": len(
                {record.condition_id for record in train}
            ),
            "training": condition_training,
            "parameter_sha256": condition_model.parameter_sha256(),
        }
    else:
        condition_metrics = {
            "status": "unavailable_fewer_than_eight_conditions",
            "split_semantics": (
                "Condition holdout activates at eight independent conditions; strict "
                "material/geometry group holdouts require a larger registered matrix."
            ),
            "held_out_condition_ids": [],
            "transition_count": 0,
        }
    if not rollout_records:
        raise RuntimeError("WP3 requires at least one observed multi-cycle case")
    teacher_prediction = model.predict_records(rollout_records)
    teacher_baseline = model.predict_stage_mean_records(rollout_records)
    target = np.stack([record.state_out for record in rollout_records])
    rollout_all_metrics = _metric(rollout_prediction_array, target)
    rollout_sma_metrics = _metric(rollout_prediction_array, target, LAYOUT.sma_T)
    rollout_stage2_htf_metrics = _metric(
        rollout_prediction_array, target, LAYOUT.htf_stage2_T
    )
    rollout_stage4_htf_metrics = _metric(
        rollout_prediction_array, target, LAYOUT.htf_stage4_T
    )
    rollout_baseline_all = _metric(rollout_baseline_array, target)
    rollout_baseline_sma = _metric(rollout_baseline_array, target, LAYOUT.sma_T)
    rollout_baseline_stage2_htf = _metric(
        rollout_baseline_array, target, LAYOUT.htf_stage2_T
    )
    rollout_baseline_stage4_htf = _metric(
        rollout_baseline_array, target, LAYOUT.htf_stage4_T
    )
    rollout_skill = {
        "all_state": _rmse_skill(rollout_all_metrics, rollout_baseline_all),
        "sma_temperature": _rmse_skill(rollout_sma_metrics, rollout_baseline_sma),
        "stage2_htf_temperature": _rmse_skill(
            rollout_stage2_htf_metrics, rollout_baseline_stage2_htf
        ),
        "stage4_htf_temperature": _rmse_skill(
            rollout_stage4_htf_metrics, rollout_baseline_stage4_htf
        ),
    }

    constraints = {
        "finite": bool(np.isfinite(rollout_prediction_array).all()),
        "xi_in_unit_interval": bool(
            np.all((rollout_prediction_array[:, LAYOUT.xi] >= 0.0) & (rollout_prediction_array[:, LAYOUT.xi] <= 1.0))
        ),
        "clamped_u_edge_max_abs_m": float(np.max(np.abs(rollout_prediction_array[:, LAYOUT.u.stop - 1]))),
        "clamped_w_edge_max_abs_m": float(np.max(np.abs(rollout_prediction_array[:, LAYOUT.w.stop - 1]))),
    }

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    assert model.model is not None
    torch.save(
        {
            "checkpoint_schema_version": "sciml_wp3_checkpoint_v2",
            "state_dict": model.model.state_dict(),
            "feature_mean": model.feature_mean,
            "feature_scale": model.feature_scale,
            "residual_scale": model.residual_scale,
            "stage_baseline": model.stage_baseline,
            "stage_trust_radius": model.stage_trust_radius,
            "stage_training_features": model.stage_training_features,
            "residual_gain": model.residual_gain,
            "training_ids": model.training_ids,
            "parameter_sha256": model.parameter_sha256(),
            "dataset_source_content_sha256": dataset_source_content_sha256,
            "implementation_sha256": implementation_sha256,
            "training_config": {
                "seed": 20260820,
                "hidden": 128,
                "epochs": epochs,
                "residual_gain": PREREGISTERED_RESIDUAL_GAIN,
                "gain_selection": "frozen_on_pre-Paper220_engineering_development_corpus",
            },
        },
        checkpoint,
    )

    blocking_reasons = _blocking_reasons(
        records, condition_test, held_out_conditions, rollout_records
    )
    for split_name, split_report in structured_ood.items():
        status = str(split_report["status"])
        if status != "evaluated_all_groups":
            blocking_reasons.append(
                f"{split_name} is {status}; skipped groups are "
                f"{split_report.get('skipped_groups', split_report.get('observed_groups', []))}."
            )
        elif int(split_report["pooled"]["transition_count"]) < 40:
            blocking_reasons.append(
                f"{split_name} has only {split_report['pooled']['transition_count']} "
                "pooled held-out transitions."
            )
        else:
            failures = _nonpositive_structured_skills(split_report)
            if failures:
                blocking_reasons.append(
                    f"{split_name} lacks positive skill in every held-out group/state block: "
                    + ", ".join(failures)
                    + "."
                )

    cycle1_condition_count = len({record.condition_id for record in cycle1_unique})
    temporal_condition_count = len({record.condition_id for record in rollout_records})
    structured_ready = all(
        report.get("status") == "evaluated_all_groups"
        and int(report.get("pooled", {}).get("transition_count", 0)) >= 40
        and not _nonpositive_structured_skills(report)
        for report in structured_ood.values()
    )
    condition_skill_ready = bool(
        condition_test
        and all(
            float(skill) > 0.0
            for skill in condition_metrics.get("skill_vs_stage_mean_baseline", {}).values()
        )
    )
    temporal_skill_ready = all(float(skill) > 0.0 for skill in rollout_skill.values())
    paper_grade_ready = (
        len(records) >= 200
        and cycle1_condition_count >= 200
        and len(condition_test) >= 40
        and temporal_condition_count >= 10
        and structured_ready
        and condition_skill_ready
        and temporal_skill_ready
        and all(bool(value) for value in constraints.values() if isinstance(value, (bool, np.bool_)))
    )
    if condition_test and not condition_skill_ready:
        blocking_reasons.append(
            "Cross-condition neural residual does not have positive skill over the stage-mean baseline."
        )
    if rollout_records and not temporal_skill_ready:
        blocking_reasons.append(
            "Autonomous temporal rollout does not have positive skill over the stage-mean baseline for every state block."
        )

    report: dict[str, object] = {
        "work_package": "WP3-v1.1",
        "dataset_source_content_sha256": dataset_source_content_sha256,
        "implementation_sha256": implementation_sha256,
        "training_config": {
            "seed": 20260820,
            "hidden": 128,
            "epochs": epochs,
            "residual_gain": PREREGISTERED_RESIDUAL_GAIN,
            "gain_selection": "frozen_on_pre-Paper220_engineering_development_corpus",
        },
        "paper_grade_requirements": {
            "minimum_cycle1_conditions": 200,
            "minimum_observed_transitions": 200,
            "minimum_condition_holdout_transitions": 40,
            "minimum_temporal_conditions": 10,
            "minimum_pooled_structured_holdout_transitions": 40,
            "positive_skill_over_stage_mean_required": True,
            "positive_skill_required_in_every_structured_holdout_fold": True,
        },
        "model": "physics_constrained_stage_mean_plus_residual_mlp_with_ood_trust_gate",
        "baseline_semantics": (
            "The stage-mean delta is train-fitted and empirical; physics constraints "
            "enter through state ownership, stage masks, Markov continuity and projection. "
            "The neural residual gain is frozen on the pre-Paper220 Engineering "
            "development corpus and is not tuned on Temporal12."
        ),
        "canonical_state_dimension": LAYOUT.size,
        "observed_transition_count": len(records),
        "cycle1_condition_count": cycle1_condition_count,
        "temporal_condition_count": temporal_condition_count,
        "train_transition_count": len(cycle1_unique),
        "train_condition_count": len(
            {record.condition_id for record in cycle1_unique}
        ),
        "condition_evaluation_train_transition_count": len(train),
        "excluded_duplicate_cycle1_transitions": duplicate_count,
        "temporal_rollout_transition_count": len(rollout_records),
        "training": train_metrics,
        "cross_condition_evaluation": condition_metrics,
        "structured_group_ood": structured_ood,
        "teacher_forced_temporal": {
            "all_state": _metric(teacher_prediction, target),
            "sma_temperature": _metric(teacher_prediction, target, LAYOUT.sma_T),
            "stage_mean_baseline": {
                "all_state": _metric(teacher_baseline, target),
                "sma_temperature": _metric(teacher_baseline, target, LAYOUT.sma_T),
            },
        },
        "autonomous_multicycle_rollout": {
            "case_count": len({record.case_id for record in rollout_records}),
            "semantics": (
                "Each case is rolled independently from its observed cycle-1 initial "
                "state through all cycle-1 stages before cycle-2/3 scoring; no state "
                "is transferred across cases."
            ),
            "all_state": rollout_all_metrics,
            "sma_temperature": rollout_sma_metrics,
            "stage2_htf_temperature": rollout_stage2_htf_metrics,
            "stage4_htf_temperature": rollout_stage4_htf_metrics,
            "neural_trust_scores": trust_scores,
            "stage_mean_baseline": {
                "all_state": rollout_baseline_all,
                "sma_temperature": rollout_baseline_sma,
                "stage2_htf_temperature": rollout_baseline_stage2_htf,
                "stage4_htf_temperature": rollout_baseline_stage4_htf,
            },
            "skill_vs_stage_mean_baseline": rollout_skill,
        },
        "cycle_drift_l2_by_stage": rollout_drift,
        "constraints": constraints,
        "observation_continuity": continuity_audit(records),
        "checkpoint": str(checkpoint.resolve()),
        "parameter_sha256": model.parameter_sha256(),
        "paper_grade_ready": paper_grade_ready,
        "blocking_reasons": blocking_reasons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("SciML/datasets/v1.1-pilot-r3"))
    parser.add_argument("--output", type=Path, default=Path("SciML/results/wp3_v11_r3.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("SciML/models/v11/results/wp3_v11_r3.pt"))
    parser.add_argument("--epochs", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.output, args.checkpoint, args.epochs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
