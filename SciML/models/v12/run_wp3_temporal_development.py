"""Run the non-confirmatory WP3-v1.2 autonomous temporal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from SciML.models.v11.run_wp3_state_space import (
    LAYOUT,
    _autonomous_multicycle_rollout,
    _condition_partition,
    _deduplicated_cycle1,
    _metric,
    _rmse_skill,
)
from SciML.models.v11.state_space import build_transition_corpus
from .neural_state_space_candidate import (
    CausalNeuralStateModel,
    load_frozen_checkpoint,
    save_frozen_checkpoint,
)


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_name("state_space_candidate.py"),
        Path(__file__).with_name("neural_state_space_candidate.py"),
        Path(__file__).with_name("run_wp3_confirmation.py"),
        Path(__file__).parents[1] / "v11" / "state_space.py",
        Path(__file__).parents[1] / "v11" / "run_wp3_state_space.py",
    ):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


SLICES = {
    "all_state": None,
    "sma_temperature": LAYOUT.sma_T,
    "stage2_htf_temperature": LAYOUT.htf_stage2_T,
    "stage4_htf_temperature": LAYOUT.htf_stage4_T,
}


def _skills(
    prediction: np.ndarray, baseline: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    return {
        name: _rmse_skill(_metric(prediction, target, slc), _metric(baseline, target, slc))
        for name, slc in SLICES.items()
    }


def _temporal_condition_cv(records: list, epochs: int = 1200) -> dict[str, object]:
    """Leave each observed temporal condition wholly outside model fitting."""

    temporal_conditions = sorted(
        {record.condition_id for record in records if record.cycle_index >= 2}
    )
    folds: list[dict[str, object]] = []
    pooled_prediction = []
    pooled_baseline = []
    pooled_target = []
    for condition_id in temporal_conditions:
        train = [record for record in records if record.condition_id != condition_id]
        test = [record for record in records if record.condition_id == condition_id]
        model = CausalNeuralStateModel(seed=20260828)
        model.fit(train, epochs=epochs, allow_development_temporal_labels=True)
        evaluated, prediction, baseline, _, _ = _autonomous_multicycle_rollout(model, test)
        affine_evaluated, affine_prediction, _, _, _ = _autonomous_multicycle_rollout(
            model.affine_baseline, test)
        if [r.sample_id for r in affine_evaluated] != [r.sample_id for r in evaluated]:
            raise RuntimeError("affine ablation temporal rows differ")
        if not evaluated:
            raise RuntimeError(f"temporal CV condition {condition_id} has no rollout")
        target = np.stack([record.state_out for record in evaluated])
        fold_skills = _skills(prediction, baseline, target)
        folds.append(
            {
                "held_out_condition_id": condition_id,
                "training_condition_count": len({record.condition_id for record in train}),
                "evaluated_transition_count": len(evaluated),
                "skill_vs_stage_mean": fold_skills,
                "skill_vs_affine_ablation": _skills(prediction, affine_prediction, target),
                "all_blocks_positive": all(value > 0.0 for value in fold_skills.values()),
                "parameter_sha256": model.parameter_sha256(),
            }
        )
        pooled_prediction.append(prediction)
        pooled_baseline.append(baseline)
        pooled_target.append(target)
    pooled_skills = _skills(
        np.concatenate(pooled_prediction),
        np.concatenate(pooled_baseline),
        np.concatenate(pooled_target),
    )
    minima = {
        name: min(float(fold["skill_vs_stage_mean"][name]) for fold in folds)
        for name in SLICES
    }
    return {
        "split_semantics": (
            "Leave-one-temporal-physical-condition-out refit. All cycle-1/2/3 "
            "transitions of the held condition are excluded from fitting, and "
            "cycle-2/3 predictions are autonomous from its cycle-1 initial state."
        ),
        "fold_count": len(folds),
        "folds": folds,
        "pooled_skill_vs_stage_mean": pooled_skills,
        "minimum_fold_skill_vs_stage_mean": minima,
        "all_folds_all_blocks_positive": all(
            bool(fold["all_blocks_positive"]) for fold in folds
        ),
    }


def _teacher_forced_evaluation(
    model: CausalNeuralStateModel, records: list
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    prediction = model.predict_records(records)
    baseline = np.stack(
        [model.predict_stage_mean(record.state_in, record.stage_index) for record in records]
    )
    target = np.stack([record.state_out for record in records])
    model_metrics = {
        "all_state": _metric(prediction, target),
        "sma_temperature": _metric(prediction, target, LAYOUT.sma_T),
    }
    baseline_metrics = {
        "all_state": _metric(baseline, target),
        "sma_temperature": _metric(baseline, target, LAYOUT.sma_T),
    }
    skills = {
        name: _rmse_skill(model_metrics[name], baseline_metrics[name])
        for name in model_metrics
    }
    affine_prediction = model.affine_baseline.predict_records(records)
    affine_metrics = {
        "all_state": _metric(affine_prediction, target),
        "sma_temperature": _metric(affine_prediction, target, LAYOUT.sma_T),
    }
    return prediction, baseline, target, {
        **model_metrics,
        "stage_mean_baseline": baseline_metrics,
        "skill_vs_stage_mean": skills,
        "affine_ablation": affine_metrics,
        "skill_vs_affine_ablation": {
            name: _rmse_skill(model_metrics[name], affine_metrics[name]) for name in model_metrics
        },
        "all_required_skills_positive": all(value > 0.0 for value in skills.values()),
    }


def _condition_holdout(records: list, cycle1: list, epochs: int = 1200) -> dict[str, object]:
    _, test, held_out = _condition_partition(cycle1, records)
    held_out_set = set(held_out)
    train = [record for record in records if record.condition_id not in held_out_set]
    model = CausalNeuralStateModel(seed=20260828)
    model.fit(train, epochs=epochs, allow_development_temporal_labels=True)
    _, _, _, metrics = _teacher_forced_evaluation(model, test)
    return {
        "split_semantics": (
            "Deterministic physical-condition holdout. Every transition of each "
            "held condition is excluded from fitting; evaluation uses its unique "
            "cycle-1 transitions."
        ),
        "held_out_condition_ids": held_out,
        "training_condition_count": len({record.condition_id for record in train}),
        "transition_count": len(test),
        "parameter_sha256": model.parameter_sha256(),
        **metrics,
    }


def _structured_group_ood(
    records: list, cycle1: list, group_attribute: str, epochs: int = 1200
) -> dict[str, object]:
    split_name = "material" if group_attribute == "material_id" else "geometry"
    groups = sorted({str(getattr(record, group_attribute)) for record in cycle1})
    folds: list[dict[str, object]] = []
    pooled_prediction = []
    pooled_baseline = []
    pooled_target = []
    for group in groups:
        train = [
            record for record in records if str(getattr(record, group_attribute)) != group
        ]
        test = [
            record for record in cycle1 if str(getattr(record, group_attribute)) == group
        ]
        model = CausalNeuralStateModel(seed=20260828)
        model.fit(train, epochs=epochs, allow_development_temporal_labels=True)
        prediction, baseline, target, metrics = _teacher_forced_evaluation(model, test)
        folds.append(
            {
                "held_out_group": group,
                "training_condition_count": len({record.condition_id for record in train}),
                "test_condition_count": len({record.condition_id for record in test}),
                "test_transition_count": len(test),
                "parameter_sha256": model.parameter_sha256(),
                **metrics,
            }
        )
        pooled_prediction.append(prediction)
        pooled_baseline.append(baseline)
        pooled_target.append(target)
    pooled_model = np.concatenate(pooled_prediction)
    pooled_baseline_array = np.concatenate(pooled_baseline)
    pooled_truth = np.concatenate(pooled_target)
    pooled_model_metrics = {
        "all_state": _metric(pooled_model, pooled_truth),
        "sma_temperature": _metric(pooled_model, pooled_truth, LAYOUT.sma_T),
    }
    pooled_baseline_metrics = {
        "all_state": _metric(pooled_baseline_array, pooled_truth),
        "sma_temperature": _metric(
            pooled_baseline_array, pooled_truth, LAYOUT.sma_T
        ),
    }
    pooled_skills = {
        name: _rmse_skill(pooled_model_metrics[name], pooled_baseline_metrics[name])
        for name in pooled_model_metrics
    }
    return {
        "status": "evaluated_all_groups",
        "group_attribute": group_attribute,
        "split_semantics": (
            f"Leave-one-{split_name}-group-out refit. All cycles from the held "
            "group are excluded from fitting; unique cycle-1 transitions are evaluated."
        ),
        "observed_groups": groups,
        "fold_count": len(folds),
        "folds": folds,
        "pooled": {
            **pooled_model_metrics,
            "stage_mean_baseline": pooled_baseline_metrics,
            "skill_vs_stage_mean": pooled_skills,
            "transition_count": int(pooled_truth.shape[0]),
        },
        "all_folds_and_pooled_required_skills_positive": bool(
            all(fold["all_required_skills_positive"] for fold in folds)
            and all(value > 0.0 for value in pooled_skills.values())
        ),
    }
def run(
    dataset_root: Path, output: Path, checkpoint: Path, epochs: int
) -> dict[str, object]:
    if output.exists() or checkpoint.exists():
        raise FileExistsError("development evidence exists; select fresh output/checkpoint paths")
    records = build_transition_corpus(dataset_root)
    cycle1, _ = _deduplicated_cycle1(records)
    if len({r.condition_id for r in cycle1}) < 200:
        raise ValueError("WP3 development requires at least 200 independent cycle-1 conditions")
    if len({r.condition_id for r in records if r.cycle_index >= 2}) < 10:
        raise ValueError("WP3 development requires at least 10 genuine temporal conditions")
    model = CausalNeuralStateModel(seed=20260828)
    training = model.fit(
        records, epochs=epochs, allow_development_temporal_labels=True
    )
    evaluated, prediction, baseline, trust, _ = _autonomous_multicycle_rollout(
        model, records
    )
    target = np.stack([record.state_out for record in evaluated])
    skills = _skills(prediction, baseline, target)
    _, affine_prediction, _, _, _ = _autonomous_multicycle_rollout(model.affine_baseline, records)
    temporal_condition_cv = _temporal_condition_cv(records, epochs=epochs)
    condition_holdout = _condition_holdout(records, cycle1, epochs=epochs)
    structured_ood = {
        "material_group_holdout": _structured_group_ood(
            records, cycle1, "material_id", epochs=epochs
        ),
        "geometry_group_holdout": _structured_group_ood(
            records, cycle1, "geometry_id", epochs=epochs
        ),
    }
    quality = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )
    implementation_hash = implementation_sha256()
    checkpoint_metadata = {
        "dataset_source_content_sha256": str(quality["source_content_sha256"]),
        "implementation_sha256": implementation_hash,
    }
    checkpoint_hash = save_frozen_checkpoint(
        checkpoint, model, metadata=checkpoint_metadata
    )
    restored = load_frozen_checkpoint(
        checkpoint, expected_metadata=checkpoint_metadata
    )
    if restored.parameter_sha256() != model.parameter_sha256():
        raise RuntimeError("WP3-v1.2 checkpoint changed fitted parameters")
    restored_prediction = restored.predict_records(evaluated)
    teacher_prediction = model.predict_records(evaluated)
    if not np.array_equal(restored_prediction, teacher_prediction):
        raise RuntimeError("WP3-v1.2 checkpoint changed model predictions")
    physical_constraints = {
        "finite": bool(np.isfinite(prediction).all()),
        "xi_in_unit_interval": bool(
            np.all(
                (prediction[:, LAYOUT.xi] >= 0.0)
                & (prediction[:, LAYOUT.xi] <= 1.0)
            )
        ),
        "clamped_u_edge_max_abs_m": float(
            np.max(np.abs(prediction[:, LAYOUT.u.stop - 1]))
        ),
        "clamped_w_edge_max_abs_m": float(
            np.max(np.abs(prediction[:, LAYOUT.w.stop - 1]))
        ),
    }
    development_checks = {
        "condition_holdout_minimum_40_transitions": int(
            condition_holdout["transition_count"]
        )
        >= 40,
        "condition_holdout_all_required_skills_positive": bool(
            condition_holdout["all_required_skills_positive"]
        ),
        "material_ood_all_folds_and_pooled_required_skills_positive": bool(
            structured_ood["material_group_holdout"][
                "all_folds_and_pooled_required_skills_positive"
            ]
        ),
        "geometry_ood_all_folds_and_pooled_required_skills_positive": bool(
            structured_ood["geometry_group_holdout"][
                "all_folds_and_pooled_required_skills_positive"
            ]
        ),
        "temporal_cv_at_least_10_conditions": int(
            temporal_condition_cv["fold_count"]
        )
        >= 10,
        "temporal_cv_all_folds_all_blocks_positive": bool(
            temporal_condition_cv["all_folds_all_blocks_positive"]
        ),
        "physical_constraints_pass": bool(
            physical_constraints["finite"]
            and physical_constraints["xi_in_unit_interval"]
            and physical_constraints["clamped_u_edge_max_abs_m"] == 0.0
            and physical_constraints["clamped_w_edge_max_abs_m"] == 0.0
        ),
        "checkpoint_roundtrip_exact": True,
    }
    failed_development_checks = sorted(
        name for name, passed in development_checks.items() if not passed
    )
    report: dict[str, object] = {
        "schema_version": "sciml_wp3_v12_temporal_development_v2",
        "work_package": "WP3-v1.2-development-only",
        "confirmatory": False,
        "readiness_eligible": False,
        "paper_grade_ready": False,
        "replaces_v11_gate": False,
        "evidence_boundary": (
            "Paper220 and Temporal12 are explicitly development/training evidence for "
            "this revised v1.2 model. In-sample temporal diagnostics are not confirmation; "
            "the already frozen, untouched Confirm-Temporal12 matrix is required before "
            "any paper-grade claim."
        ),
        "dataset_source_content_sha256": quality["source_content_sha256"],
        "implementation_sha256": implementation_hash,
        "parameter_sha256": model.parameter_sha256(),
        "checkpoint": {
            "file": checkpoint.name,
            "file_sha256": checkpoint_hash,
            "roundtrip_parameter_sha256_match": True,
            "roundtrip_prediction_exact_match": True,
            "evidence_status": "candidate_frozen_before_confirm_temporal12",
        },
        "training_policy": {
            "allowed_cycle_indices": [1, 2, 3],
            "temporal_label_training_count": sum(r.cycle_index >= 2 for r in records),
            "cycle1_condition_count": len({r.condition_id for r in cycle1}),
            "cycle1_transition_count": len(cycle1),
            "development_temporal_condition_count": len(
                {r.condition_id for r in records if r.cycle_index >= 2}
            ),
            "fit_method": "stage_local_ridge_plus_nonlinear_residual_adamw",
            "neural_hidden_layers": 2,
            "nonlinear_training_config": model.training_config,
            "affine_baseline_is_neural_model": False,
            "physics_boundary": "hard_stage_ownership_and_admissibility_not_a_reduced_PDE_solver",
            "epochs_per_stage_and_every_refit": epochs,
            "seed": 20260828,
        },
        "training": training,
        "condition_holdout": condition_holdout,
        "structured_ood": structured_ood,
        "temporal_condition_cross_validation": temporal_condition_cv,
        "development_freeze_audit": {
            "checks": development_checks,
            "failed_checks": failed_development_checks,
            "development_freeze_ready": not failed_development_checks,
            "claim_boundary": (
                "Development freeze readiness does not constitute confirmation or "
                "paper-grade readiness. Confirm-Temporal12 remains untouched and required."
            ),
        },
        "temporal_diagnostic": {
            "condition_count": len({r.condition_id for r in evaluated}),
            "transition_count": len(evaluated),
            "skill_vs_stage_mean": skills,
            "skill_vs_affine_ablation": _skills(prediction, affine_prediction, target),
            "physical_constraints": physical_constraints,
            "trust_nonzero_count": int(np.count_nonzero(np.asarray(trust) > 0.0)),
            "trust_min": float(np.min(trust)),
            "trust_median": float(np.median(trust)),
            "trust_max": float(np.max(trust)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SciML/results/development/wp3_v12_neural_temporal_candidate.json"),
    )
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("SciML/results/development/wp3_v12_neural_confirmation_candidate.pkl"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.dataset_root, args.output, args.checkpoint, args.epochs),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
