"""Train and audit the v1.1 dual-field POD--DeepONet pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .spatiotemporal_operator import (
    PREREGISTERED_TEMPORAL_HTF_MEAN_GAIN,
    PREREGISTERED_TEMPORAL_SMA_MEAN_GAIN,
    PODDeepONet,
    OperatorSample,
    build_operator_corpus,
)


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (Path(__file__).resolve(), Path(__file__).with_name("spatiotemporal_operator.py"))
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "max_abs_K": float(np.max(np.abs(error))),
    }


def _rmse_skill(model: dict[str, float], baseline: dict[str, float]) -> float:
    denominator = float(baseline["rmse_K"])
    return float(1.0 - float(model["rmse_K"]) / denominator) if denominator > 0.0 else 0.0


def _nonpositive_structured_skills(report: dict[str, object]) -> list[str]:
    failures: list[str] = []
    pooled = report.get("pooled", {})
    if isinstance(pooled, dict):
        skills = pooled.get("skill_vs_stage_mean_baseline", {})
        if isinstance(skills, dict):
            for field, skill in skills.items():
                if float(skill) <= 0.0:
                    failures.append(f"pooled:{field}={float(skill):.6g}")
    folds = report.get("folds", [])
    if isinstance(folds, list):
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            group = str(fold.get("held_out_group", "unknown"))
            skills = fold.get("skill_vs_stage_mean_baseline", {})
            if isinstance(skills, dict):
                for field, skill in skills.items():
                    if float(skill) <= 0.0:
                        failures.append(f"{group}:{field}={float(skill):.6g}")
    return failures


def _cycle1_unique(samples: list[OperatorSample]) -> tuple[list[OperatorSample], int]:
    selected = []
    seen = set()
    for sample in samples:
        if sample.cycle_index != 1:
            continue
        key = (sample.condition_id, sample.stage_index)
        if key not in seen:
            selected.append(sample)
            seen.add(key)
    raw = sum(sample.cycle_index == 1 for sample in samples)
    return selected, raw - len(selected)


def _condition_partition(
    cycle1: list[OperatorSample], all_samples: list[OperatorSample]
) -> tuple[list[OperatorSample], list[OperatorSample], list[str]]:
    conditions = sorted({sample.condition_id for sample in cycle1})
    if len(conditions) < 8:
        return cycle1, [], []
    temporal_conditions = {
        sample.condition_id for sample in all_samples if sample.cycle_index >= 2
    }
    candidates = [condition for condition in conditions if condition not in temporal_conditions]
    count = max(2, int(round(0.2 * len(conditions))))
    ranked = sorted(
        candidates,
        key=lambda value: hashlib.sha256(f"20260820:{value}".encode()).hexdigest(),
    )
    held_out = set(ranked[: min(count, len(ranked))])
    return (
        [sample for sample in cycle1 if sample.condition_id not in held_out],
        [sample for sample in cycle1 if sample.condition_id in held_out],
        sorted(held_out),
    )


def _blocking_reasons(
    cycle1: list[OperatorSample],
    condition_test: list[OperatorSample],
    held_out_conditions: list[str],
    temporal_test: list[OperatorSample],
    temporal_trust: np.ndarray,
) -> list[str]:
    cycle1_conditions = {sample.condition_id for sample in cycle1}
    temporal_conditions = {sample.condition_id for sample in temporal_test}
    reasons = []
    if len(cycle1_conditions) < 200:
        reasons.append(
            f"Only {len(cycle1_conditions)} independent cycle-1 conditions "
            f"({len(cycle1)} stage samples) are available; the paper-grade "
            "case minimum is 200."
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
            f"{len(condition_test)} stage samples; the paper-grade minimum is 40 samples."
        )
    if len(temporal_conditions) < 10:
        reasons.append(
            f"Cycle-2/3 temporal operator evaluation contains {len(temporal_test)} samples "
            f"from only {len(temporal_conditions)} physical condition(s); the paper-grade "
            "minimum is 10 temporal conditions."
        )
    return reasons


def _structured_group_ood(
    cycle1: list[OperatorSample],
    group_attribute: str,
    epochs: int,
) -> dict[str, object]:
    groups = sorted({str(getattr(sample, group_attribute)) for sample in cycle1})
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
    pooled_sma = []
    pooled_htf = []
    pooled_baseline_sma = []
    pooled_baseline_htf = []
    pooled_true_sma = []
    pooled_true_htf = []
    for group in groups:
        train = [sample for sample in cycle1 if str(getattr(sample, group_attribute)) != group]
        test = [sample for sample in cycle1 if str(getattr(sample, group_attribute)) == group]
        train_conditions = {sample.condition_id for sample in train}
        test_conditions = {sample.condition_id for sample in test}
        if len(train_conditions) < 3 or len(test_conditions) < 2:
            continue
        fold_model = PODDeepONet(rank=4, hidden=128, seed=20260820)
        fold_training = fold_model.fit(train, epochs=epochs)
        prediction_sma, prediction_htf, trust = fold_model.predict(test)
        baseline_sma, baseline_htf = fold_model.predict_stage_mean(test)
        true_sma = np.stack([sample.sma_history for sample in test])
        true_htf = np.stack([sample.htf_history for sample in test])
        model_sma = _metrics(prediction_sma, true_sma)
        model_htf = _metrics(prediction_htf, true_htf)
        mean_sma = _metrics(baseline_sma, true_sma)
        mean_htf = _metrics(baseline_htf, true_htf)
        folds.append(
            {
                "held_out_group": group,
                "train_condition_count": len(train_conditions),
                "test_condition_count": len(test_conditions),
                "test_sample_count": len(test),
                "sma": model_sma,
                "htf": model_htf,
                "stage_mean_baseline": {"sma": mean_sma, "htf": mean_htf},
                "skill_vs_stage_mean_baseline": {
                    "sma": _rmse_skill(model_sma, mean_sma),
                    "htf": _rmse_skill(model_htf, mean_htf),
                },
                "neural_trust_scores": trust.tolist(),
                "training": fold_training,
                "parameter_sha256": fold_model.parameter_sha256(),
            }
        )
        pooled_sma.append(prediction_sma)
        pooled_htf.append(prediction_htf)
        pooled_baseline_sma.append(baseline_sma)
        pooled_baseline_htf.append(baseline_htf)
        pooled_true_sma.append(true_sma)
        pooled_true_htf.append(true_htf)

    if not folds:
        return {
            "status": "unavailable_insufficient_conditions_per_group",
            "group_attribute": group_attribute,
            "observed_groups": groups,
            "minimum_train_conditions_per_fold": 3,
            "minimum_test_conditions_per_fold": 2,
            "fold_count": 0,
        }

    prediction_sma = np.concatenate(pooled_sma)
    prediction_htf = np.concatenate(pooled_htf)
    baseline_sma = np.concatenate(pooled_baseline_sma)
    baseline_htf = np.concatenate(pooled_baseline_htf)
    true_sma = np.concatenate(pooled_true_sma)
    true_htf = np.concatenate(pooled_true_htf)
    model_sma = _metrics(prediction_sma, true_sma)
    model_htf = _metrics(prediction_htf, true_htf)
    mean_sma = _metrics(baseline_sma, true_sma)
    mean_htf = _metrics(baseline_htf, true_htf)
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
            "sma": model_sma,
            "htf": model_htf,
            "stage_mean_baseline": {"sma": mean_sma, "htf": mean_htf},
            "skill_vs_stage_mean_baseline": {
                "sma": _rmse_skill(model_sma, mean_sma),
                "htf": _rmse_skill(model_htf, mean_htf),
            },
            "sample_count": int(true_sma.shape[0]),
        },
    }


def _persistence(samples: list[OperatorSample], field: str) -> np.ndarray:
    return np.stack(
        [np.broadcast_to(getattr(sample, field)[0], (50, 101)) for sample in samples]
    )


def run(dataset_root: Path, output: Path, checkpoint: Path, epochs: int) -> dict[str, object]:
    samples = build_operator_corpus(dataset_root)
    dataset_source_content_sha256 = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )["source_content_sha256"]
    implementation_sha256 = _implementation_sha256()
    cycle1_unique, duplicates = _cycle1_unique(samples)
    train, condition_test, held_out_conditions = _condition_partition(cycle1_unique, samples)
    test = [sample for sample in samples if sample.cycle_index >= 2]
    # The deterministic condition holdout is evaluated with a model that never
    # sees those conditions.  A separate deployment/checkpoint model is then
    # fitted on the complete cycle-1 corpus.  Keeping these roles separate
    # avoids under-reporting the registered 200-condition training corpus while
    # preserving a genuinely unseen-condition evaluation.
    condition_model = PODDeepONet(rank=4, hidden=128, seed=20260820)
    condition_training = condition_model.fit(train, epochs=epochs)
    structured_ood = {
        "material_group_holdout": _structured_group_ood(
            cycle1_unique, "material_id", epochs
        ),
        "geometry_group_holdout": _structured_group_ood(
            cycle1_unique, "geometry_id", epochs
        ),
    }
    if condition_test:
        condition_sma, condition_htf, condition_trust = condition_model.predict(
            condition_test
        )
        condition_baseline_sma, condition_baseline_htf = (
            condition_model.predict_stage_mean(condition_test)
        )
        condition_true_sma = np.stack([sample.sma_history for sample in condition_test])
        condition_true_htf = np.stack([sample.htf_history for sample in condition_test])
        condition_sma_metrics = _metrics(condition_sma, condition_true_sma)
        condition_htf_metrics = _metrics(condition_htf, condition_true_htf)
        condition_baseline_sma_metrics = _metrics(
            condition_baseline_sma, condition_true_sma
        )
        condition_baseline_htf_metrics = _metrics(
            condition_baseline_htf, condition_true_htf
        )
        condition_evaluation: dict[str, object] = {
            "split_semantics": (
                "Deterministic physical-condition holdout; material and geometry "
                "levels may remain represented in training. This is not a strict "
                "unseen-material or unseen-geometry split."
            ),
            "sma": condition_sma_metrics,
            "htf": condition_htf_metrics,
            "stage_mean_baseline": {
                "sma": condition_baseline_sma_metrics,
                "htf": condition_baseline_htf_metrics,
            },
            "skill_vs_stage_mean_baseline": {
                "sma": _rmse_skill(condition_sma_metrics, condition_baseline_sma_metrics),
                "htf": _rmse_skill(condition_htf_metrics, condition_baseline_htf_metrics),
            },
            "neural_trust_scores": condition_trust.tolist(),
            "held_out_condition_ids": held_out_conditions,
            "sample_count": len(condition_test),
            "training_sample_count": len(train),
            "training_condition_count": len(
                {sample.condition_id for sample in train}
            ),
            "training": condition_training,
            "parameter_sha256": condition_model.parameter_sha256(),
        }
    else:
        condition_evaluation = {
            "status": "unavailable_fewer_than_eight_conditions",
            "split_semantics": (
                "Condition holdout activates at eight independent conditions; strict "
                "material/geometry group holdouts require a larger registered matrix."
            ),
            "held_out_condition_ids": [],
            "sample_count": 0,
        }
    model = PODDeepONet(rank=4, hidden=128, seed=20260820)
    training = model.fit(cycle1_unique, epochs=epochs)
    predicted_sma, predicted_htf, trust = model.predict(test)
    stage_mean_sma, stage_mean_htf = model.predict_stage_mean(test)
    true_sma = np.stack([sample.sma_history for sample in test])
    true_htf = np.stack([sample.htf_history for sample in test])
    persistence_sma = _persistence(test, "sma_history")
    persistence_htf = _persistence(test, "htf_history")
    temporal_sma_metrics = _metrics(predicted_sma, true_sma)
    temporal_htf_metrics = _metrics(predicted_htf, true_htf)
    persistence_sma_metrics = _metrics(persistence_sma, true_sma)
    persistence_htf_metrics = _metrics(persistence_htf, true_htf)
    stage_mean_sma_metrics = _metrics(stage_mean_sma, true_sma)
    stage_mean_htf_metrics = _metrics(stage_mean_htf, true_htf)

    by_stage: dict[str, object] = {}
    for stage in (2, 4):
        selected = np.asarray([sample.stage_index == stage for sample in test])
        by_stage[f"stage_{stage}"] = {
            "sma": _metrics(predicted_sma[selected], true_sma[selected]),
            "htf": _metrics(predicted_htf[selected], true_htf[selected]),
        }

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    assert model.model is not None
    torch.save(
        {
            "checkpoint_schema_version": "sciml_wp4_checkpoint_v2",
            "state_dict": model.model.state_dict(),
            "branch_mean": model.branch_mean,
            "branch_scale": model.branch_scale,
            "sma_pod_basis": model.sma_basis,
            "htf_pod_basis": model.htf_basis,
            "stage_mean": model.stage_mean,
            "stage_training_branch": model.stage_training_branch,
            "stage_trust_radius": model.stage_trust_radius,
            "coefficient_mean": model.coefficient_mean,
            "coefficient_scale": model.coefficient_scale,
            "temporal_sma_mean_gain": model.temporal_sma_mean_gain,
            "temporal_htf_mean_gain": model.temporal_htf_mean_gain,
            "training_ids": model.training_ids,
            "parameter_sha256": model.parameter_sha256(),
            "dataset_source_content_sha256": dataset_source_content_sha256,
            "implementation_sha256": implementation_sha256,
            "training_config": {
                "seed": 20260820,
                "rank": 4,
                "hidden": 128,
                "epochs": epochs,
                "temporal_sma_mean_gain": PREREGISTERED_TEMPORAL_SMA_MEAN_GAIN,
                "temporal_htf_mean_gain": PREREGISTERED_TEMPORAL_HTF_MEAN_GAIN,
                "gain_selection": "frozen_on_pre-Paper220_engineering_development_corpus",
            },
        },
        checkpoint,
    )

    blocking_reasons = _blocking_reasons(
        cycle1_unique, condition_test, held_out_conditions, test, trust
    )
    for split_name, split_report in structured_ood.items():
        status = str(split_report["status"])
        if status != "evaluated_all_groups":
            blocking_reasons.append(
                f"{split_name} is {status}; skipped groups are "
                f"{split_report.get('skipped_groups', split_report.get('observed_groups', []))}."
            )
        elif int(split_report["pooled"]["sample_count"]) < 20:
            blocking_reasons.append(
                f"{split_name} has only {split_report['pooled']['sample_count']} "
                "pooled held-out stage samples."
            )
        else:
            failures = _nonpositive_structured_skills(split_report)
            if failures:
                blocking_reasons.append(
                    f"{split_name} lacks positive skill in every held-out group/field: "
                    + ", ".join(failures)
                    + "."
                )

    cycle1_condition_count = len({sample.condition_id for sample in cycle1_unique})
    temporal_condition_count = len({sample.condition_id for sample in test})
    structured_ready = all(
        report.get("status") == "evaluated_all_groups"
        and int(report.get("pooled", {}).get("sample_count", 0)) >= 20
        and not _nonpositive_structured_skills(report)
        for report in structured_ood.values()
    )
    condition_skill_ready = bool(
        condition_test
        and all(
            float(skill) > 0.0
            for skill in condition_evaluation.get(
                "skill_vs_stage_mean_baseline", {}
            ).values()
        )
    )
    temporal_skill_vs_persistence = {
        "sma": _rmse_skill(temporal_sma_metrics, persistence_sma_metrics),
        "htf": _rmse_skill(temporal_htf_metrics, persistence_htf_metrics),
    }
    temporal_skill_vs_stage_mean = {
        "sma": _rmse_skill(temporal_sma_metrics, stage_mean_sma_metrics),
        "htf": _rmse_skill(temporal_htf_metrics, stage_mean_htf_metrics),
    }
    temporal_skill_ready = all(
        float(skill) > 0.0
        for skill in (
            *temporal_skill_vs_persistence.values(),
            *temporal_skill_vs_stage_mean.values(),
        )
    )
    hard_constraints_ready = bool(
        np.isfinite(predicted_sma).all()
        and np.isfinite(predicted_htf).all()
        and max(
            np.max(np.abs(predicted_sma[:, 0] - true_sma[:, 0])),
            np.max(np.abs(predicted_htf[:, 0] - true_htf[:, 0])),
        )
        == 0.0
    )
    paper_grade_ready = (
        cycle1_condition_count >= 200
        and len(condition_test) >= 40
        and temporal_condition_count >= 10
        and structured_ready
        and condition_skill_ready
        and temporal_skill_ready
        and hard_constraints_ready
    )
    if condition_test and not condition_skill_ready:
        blocking_reasons.append(
            "Cross-condition neural operator does not have positive skill over the stage-mean baseline for both fields."
        )
    if test and not temporal_skill_ready:
        blocking_reasons.append(
            "Temporal neural operator does not have positive skill over both persistence and stage-mean baselines for SMA and HTF."
        )

    report: dict[str, object] = {
        "work_package": "WP4-v1.1",
        "dataset_source_content_sha256": dataset_source_content_sha256,
        "implementation_sha256": implementation_sha256,
        "training_config": {
            "seed": 20260820,
            "rank": 4,
            "hidden": 128,
            "epochs": epochs,
            "temporal_sma_mean_gain": PREREGISTERED_TEMPORAL_SMA_MEAN_GAIN,
            "temporal_htf_mean_gain": PREREGISTERED_TEMPORAL_HTF_MEAN_GAIN,
            "gain_selection": "frozen_on_pre-Paper220_engineering_development_corpus",
        },
        "paper_grade_requirements": {
            "minimum_cycle1_conditions": 200,
            "minimum_condition_holdout_samples": 40,
            "minimum_temporal_conditions": 10,
            "minimum_pooled_structured_holdout_samples": 20,
            "positive_skill_over_stage_mean_required": True,
            "positive_skill_required_in_every_structured_holdout_fold": True,
            "positive_temporal_skill_over_persistence_required": True,
        },
        "model": "dual_field_train_only_POD_DeepONet_with_frozen_temporal_mean_gains",
        "baseline_semantics": (
            "The stage-mean trajectory is fitted on training samples and is an "
            "empirical baseline, not a reduced-order PDE solver. Temporal model "
            "predictions use pre-Paper220 development gains, while the reported "
            "stage-mean comparator remains the unscaled train-fitted trajectory."
        ),
        "field_shape": [50, 101],
        "operator_sample_count": len(samples),
        "cycle1_condition_count": cycle1_condition_count,
        "temporal_condition_count": temporal_condition_count,
        "train_sample_count": len(cycle1_unique),
        "train_condition_count": len(
            {sample.condition_id for sample in cycle1_unique}
        ),
        "condition_evaluation_train_sample_count": len(train),
        "temporal_operator_sample_count": len(test),
        "excluded_duplicate_cycle1_samples": duplicates,
        "training": training,
        "cross_condition_evaluation": condition_evaluation,
        "structured_group_ood": structured_ood,
        "temporal_cycle2_3_operator_evaluation": {
            "input_semantics": (
                "Each sample is conditioned on the observed cycle-2/3 start-of-stage "
                "SMA field, HTF field and causal side-state. This evaluates temporal "
                "generalization of the stage operator; it is not an autonomous full-cycle "
                "rollout, which is evaluated separately by WP3."
            ),
            "sma": temporal_sma_metrics,
            "htf": temporal_htf_metrics,
            "persistence_sma": persistence_sma_metrics,
            "persistence_htf": persistence_htf_metrics,
            "stage_mean_sma": stage_mean_sma_metrics,
            "stage_mean_htf": stage_mean_htf_metrics,
            "skill_vs_persistence": {
                **temporal_skill_vs_persistence,
            },
            "skill_vs_stage_mean": {
                **temporal_skill_vs_stage_mean,
            },
            "neural_trust_scores": trust.tolist(),
            "by_stage": by_stage,
        },
        "hard_constraints": {
            "finite": bool(np.isfinite(predicted_sma).all() and np.isfinite(predicted_htf).all()),
            "initial_condition_exact_max_abs_K": float(
                max(
                    np.max(np.abs(predicted_sma[:, 0] - true_sma[:, 0])),
                    np.max(np.abs(predicted_htf[:, 0] - true_htf[:, 0])),
                )
            ),
            "sma_htf_outputs_are_distinct": bool(not np.array_equal(predicted_sma, predicted_htf)),
            "temperature_range_K": [
                float(min(predicted_sma.min(), predicted_htf.min())),
                float(max(predicted_sma.max(), predicted_htf.max())),
            ],
        },
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
    parser.add_argument("--output", type=Path, default=Path("SciML/results/wp4_v11_r3.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("SciML/models/v11/results/wp4_v11_r3.pt"))
    parser.add_argument("--epochs", type=int, default=1500)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.output, args.checkpoint, args.epochs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
