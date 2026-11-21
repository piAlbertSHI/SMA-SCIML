"""Run the non-confirmatory v1.2 temporal-operator development diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from SciML.models.v11.spatiotemporal_operator import (
    OperatorSample,
    build_operator_corpus,
)
from .temporal_operator_candidate import (
    CHECKPOINT_SCHEMA_VERSION,
    TEMPORAL_HTF_PRIOR_GAINS,
    TEMPORAL_SMA_PRIOR_GAINS,
    CausalDeltaPODOperator,
)
from .neural_operator_candidate import (
    CHECKPOINT_SCHEMA_VERSION as NEURAL_CHECKPOINT_SCHEMA_VERSION,
    INFERENCE_POLICY,
    CausalNeuralPODOperator,
)


CONFIRM50_CONFIG_SHA256 = (
    "fb51d6c2be265d8277e8db1b39a436a3e42afc3facdf6741b0b7ed8067c61bdd"
)
CONFIRM_TEMPORAL12_CONFIG_SHA256 = (
    "db5f7a61d0fb9fe995934c09c536866365098fe0003843214070fa4245aefc11"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            Path(__file__).resolve(),
            Path(__file__).with_name("temporal_operator_candidate.py"),
            Path(__file__).with_name("neural_operator_candidate.py"),
            Path(__file__).with_name("run_wp4_neural_development.py"),
            Path(__file__).with_name("run_wp4_confirmatory.py"),
            Path(__file__).parents[1] / "v11" / "spatiotemporal_operator.py",
        )
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _cycle1_unique(samples: list[OperatorSample]) -> list[OperatorSample]:
    selected = []
    seen = set()
    for sample in samples:
        if sample.cycle_index != 1:
            continue
        key = (sample.condition_id, sample.stage_index)
        if key not in seen:
            seen.add(key)
            selected.append(sample)
    return selected


def _condition_partition(
    cycle1: list[OperatorSample], all_samples: list[OperatorSample]
) -> tuple[list[OperatorSample], list[OperatorSample], list[str]]:
    """Return the deterministic 80/20 development condition holdout."""

    conditions = sorted({sample.condition_id for sample in cycle1})
    if len(conditions) < 8:
        return cycle1, [], []
    temporal_conditions = {
        sample.condition_id for sample in all_samples if sample.cycle_index >= 2
    }
    candidates = [
        condition for condition in conditions if condition not in temporal_conditions
    ]
    count = max(2, int(round(0.2 * len(conditions))))
    ranked = sorted(
        candidates,
        key=lambda value: hashlib.sha256(
            f"20260828:{value}".encode("utf-8")
        ).hexdigest(),
    )
    held_out = set(ranked[: min(count, len(ranked))])
    return (
        [sample for sample in cycle1 if sample.condition_id not in held_out],
        [sample for sample in cycle1 if sample.condition_id in held_out],
        sorted(held_out),
    )


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "max_abs_K": float(np.max(np.abs(error))),
    }


def _skill(model: dict[str, float], baseline: dict[str, float]) -> float:
    denominator = float(baseline["rmse_K"])
    return float(1.0 - float(model["rmse_K"]) / denominator) if denominator else 0.0


def _profile(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=-1, keepdims=True)


def _temporal_field_evaluation(
    prediction: np.ndarray,
    target: np.ndarray,
    stage_mean: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, object]:
    model = _metrics(prediction, target)
    mean = _metrics(stage_mean, target)
    persist = _metrics(persistence, target)
    return {
        "model": model,
        "stage_mean": mean,
        "persistence": persist,
        "skill_vs_stage_mean": _skill(model, mean),
        "skill_vs_persistence": _skill(model, persist),
    }


def _temporal_slice_evaluation(
    predicted_sma: np.ndarray,
    predicted_htf: np.ndarray,
    true_sma: np.ndarray,
    true_htf: np.ndarray,
    baseline_sma: np.ndarray,
    baseline_htf: np.ndarray,
    persistence_sma: np.ndarray,
    persistence_htf: np.ndarray,
) -> dict[str, object]:
    full_field = {
        "sma": _temporal_field_evaluation(
            predicted_sma, true_sma, baseline_sma, persistence_sma
        ),
        "htf": _temporal_field_evaluation(
            predicted_htf, true_htf, baseline_htf, persistence_htf
        ),
    }
    spatial_profile = {
        "sma": _temporal_field_evaluation(
            _profile(predicted_sma),
            _profile(true_sma),
            _profile(baseline_sma),
            _profile(persistence_sma),
        ),
        "htf": _temporal_field_evaluation(
            _profile(predicted_htf),
            _profile(true_htf),
            _profile(baseline_htf),
            _profile(persistence_htf),
        ),
    }
    return {"full_field": full_field, "spatial_profile": spatial_profile}


def _finite_positive(value: object) -> bool:
    """Missing, malformed and non-finite scores are not scientific evidence."""

    if isinstance(value, (bool, str)) or not np.isscalar(value):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite(number) and number > 0.0)


def _field_skills_positive(skills: object) -> bool:
    return isinstance(skills, dict) and all(
        _finite_positive(skills.get(field)) for field in ("sma", "htf")
    )


def _all_temporal_slice_skills_positive(report: dict[str, object]) -> bool:
    if not isinstance(report, dict):
        return False
    for block_name in ("full_field", "spatial_profile"):
        block = report.get(block_name, {})
        if not isinstance(block, dict):
            return False
        for field in ("sma", "htf"):
            values = block.get(field, {})
            if not isinstance(values, dict):
                return False
            for name in ("skill_vs_stage_mean", "skill_vs_persistence"):
                if not _finite_positive(values.get(name)):
                    return False
    return True


def _nonpositive_structured_skills(report: dict[str, object]) -> list[str]:
    """Return every non-positive structured-OOD field skill.

    The development audit deliberately applies the same strict ``> 0`` rule as
    the registered v1.1 WP4 gate.  In particular, a positive pooled score may
    not conceal a failed material or geometry fold.
    """

    failures: list[str] = []
    if not isinstance(report, dict):
        return ["report:missing_or_malformed"]

    def check_fields(label: str, block: object) -> None:
        skills = block.get("skill_vs_stage_mean_baseline") if isinstance(block, dict) else None
        for field in ("sma", "htf"):
            value = skills.get(field) if isinstance(skills, dict) else None
            if not _finite_positive(value):
                try:
                    rendered = f"{float(value):.6g}"
                except (TypeError, ValueError, OverflowError):
                    rendered = "missing_or_invalid"
                failures.append(f"{label}:{field}={rendered}")

    check_fields("pooled", report.get("pooled"))
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        return [*failures, "folds:missing_or_empty"]
    observed = report.get("observed_groups")
    if (not isinstance(observed, list) or not observed
            or not all(isinstance(group, str) and group for group in observed)
            or len(set(observed)) != len(observed)):
        failures.append("observed_groups:missing_or_invalid")
        observed = []
    evaluated: list[str] = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            failures.append(f"fold_{index}:malformed")
            continue
        group = fold.get("held_out_group")
        if not isinstance(group, str) or not group:
            failures.append(f"fold_{index}:missing_group")
            group = f"fold_{index}"
        evaluated.append(group)
        check_fields(group, fold)
    if (len(set(evaluated)) != len(evaluated)
            or sorted(evaluated) != sorted(observed)
            or type(report.get("fold_count")) is not int
            or report.get("fold_count") != len(folds)):
        failures.append("folds:incomplete_or_duplicate_group_coverage")
    return failures


def _structured_group_ood(
    cycle1: list[OperatorSample], group_attribute: str, epochs: int,
    *, model_class=CausalDeltaPODOperator,
) -> dict[str, object]:
    """Refit once per held-out material/geometry and pool predictions.

    Both the candidate and its stage-mean comparator are fitted exclusively on
    the training side of each fold.  The function is development evidence only;
    it never reads the untouched v1.2 confirmation matrices.
    """

    groups = sorted({str(getattr(sample, group_attribute)) for sample in cycle1})
    split_name = "material" if group_attribute == "material_id" else "geometry"
    required_group_count = 2 if split_name == "material" else 3
    if len(groups) < required_group_count:
        return {
            "status": f"unavailable_fewer_than_{required_group_count}_{split_name}_groups",
            "group_attribute": group_attribute,
            "observed_groups": groups,
            "fold_count": 0,
        }

    folds: list[dict[str, object]] = []
    pooled: dict[str, list[np.ndarray]] = {
        "predicted_sma": [],
        "predicted_htf": [],
        "baseline_sma": [],
        "baseline_htf": [],
        "true_sma": [],
        "true_htf": [],
    }
    for group in groups:
        train = [
            sample
            for sample in cycle1
            if str(getattr(sample, group_attribute)) != group
        ]
        test = [
            sample
            for sample in cycle1
            if str(getattr(sample, group_attribute)) == group
        ]
        train_conditions = {sample.condition_id for sample in train}
        test_conditions = {sample.condition_id for sample in test}
        if len(train_conditions) < 3 or len(test_conditions) < 2:
            continue
        model = model_class(rank=4, hidden=128, seed=20260828)
        training = model.fit(train, epochs=epochs)
        predicted_sma, predicted_htf, trust = model.predict(test)
        baseline_sma, baseline_htf = model.predict_stage_mean(test)
        true_sma = np.stack([sample.sma_history for sample in test])
        true_htf = np.stack([sample.htf_history for sample in test])
        model_sma = _metrics(predicted_sma, true_sma)
        model_htf = _metrics(predicted_htf, true_htf)
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
                    "sma": _skill(model_sma, mean_sma),
                    "htf": _skill(model_htf, mean_htf),
                },
                "neural_trust_scores": trust.tolist(),
                "training": training,
                "parameter_sha256": model.parameter_sha256(),
            }
        )
        for name, value in (
            ("predicted_sma", predicted_sma),
            ("predicted_htf", predicted_htf),
            ("baseline_sma", baseline_sma),
            ("baseline_htf", baseline_htf),
            ("true_sma", true_sma),
            ("true_htf", true_htf),
        ):
            pooled[name].append(value)

    evaluated_groups = [str(fold["held_out_group"]) for fold in folds]
    skipped_groups = [group for group in groups if group not in evaluated_groups]
    if not folds:
        return {
            "status": "unavailable_insufficient_conditions_per_group",
            "group_attribute": group_attribute,
            "observed_groups": groups,
            "fold_count": 0,
        }

    values = {name: np.concatenate(parts) for name, parts in pooled.items()}
    model_sma = _metrics(values["predicted_sma"], values["true_sma"])
    model_htf = _metrics(values["predicted_htf"], values["true_htf"])
    mean_sma = _metrics(values["baseline_sma"], values["true_sma"])
    mean_htf = _metrics(values["baseline_htf"], values["true_htf"])
    return {
        "status": "evaluated_all_groups" if not skipped_groups else "partial_evaluation",
        "group_attribute": group_attribute,
        "split_semantics": (
            f"Leave-one-{split_name}-group-out refit; no held-out group enters "
            "candidate or stage-mean-baseline fitting."
        ),
        "observed_groups": groups,
        "evaluated_groups": evaluated_groups,
        "skipped_groups": skipped_groups,
        "fold_count": len(folds),
        "folds": folds,
        "pooled": {
            "sma": model_sma,
            "htf": model_htf,
            "stage_mean_baseline": {"sma": mean_sma, "htf": mean_htf},
            "skill_vs_stage_mean_baseline": {
                "sma": _skill(model_sma, mean_sma),
                "htf": _skill(model_htf, mean_htf),
            },
            "sample_count": int(values["true_sma"].shape[0]),
        },
    }


def _persistence(samples: list[OperatorSample], field: str) -> np.ndarray:
    return np.stack(
        [np.broadcast_to(getattr(sample, field)[0], (50, 101)) for sample in samples]
    )


def _condition_holdout(
    cycle1: list[OperatorSample], all_samples: list[OperatorSample], epochs: int,
    *, model_class=CausalDeltaPODOperator,
) -> dict[str, object]:
    train, test, held_out = _condition_partition(cycle1, all_samples)
    if not test:
        return {
            "status": "unavailable_fewer_than_eight_conditions",
            "held_out_condition_ids": [],
            "sample_count": 0,
        }
    model = model_class(rank=4, hidden=128, seed=20260828)
    training = model.fit(train, epochs=epochs)
    predicted_sma, predicted_htf, trust = model.predict(test)
    baseline_sma, baseline_htf = model.predict_stage_mean(test)
    true_sma = np.stack([sample.sma_history for sample in test])
    true_htf = np.stack([sample.htf_history for sample in test])
    model_sma = _metrics(predicted_sma, true_sma)
    model_htf = _metrics(predicted_htf, true_htf)
    mean_sma = _metrics(baseline_sma, true_sma)
    mean_htf = _metrics(baseline_htf, true_htf)
    return {
        "status": "evaluated",
        "split_semantics": (
            "Deterministic physical-condition holdout excluding the already observed "
            "Temporal12 parent conditions; no held-out condition enters candidate or "
            "stage-mean-baseline fitting."
        ),
        "held_out_condition_ids": held_out,
        "train_condition_count": len({sample.condition_id for sample in train}),
        "test_condition_count": len({sample.condition_id for sample in test}),
        "sample_count": len(test),
        "sma": model_sma,
        "htf": model_htf,
        "stage_mean_baseline": {"sma": mean_sma, "htf": mean_htf},
        "skill_vs_stage_mean_baseline": {
            "sma": _skill(model_sma, mean_sma),
            "htf": _skill(model_htf, mean_htf),
        },
        "neural_trust_scores": trust.tolist(),
        "training": training,
        "parameter_sha256": model.parameter_sha256(),
    }


def _supplemental_temporal_diagnostic(
    model: CausalDeltaPODOperator, samples: list[OperatorSample]
) -> dict[str, object]:
    temporal = [sample for sample in samples if sample.cycle_index >= 2]
    if not temporal:
        raise RuntimeError("supplemental temporal development evidence is empty")
    predicted_sma, predicted_htf, trust = model.predict(temporal)
    baseline_sma, baseline_htf = model.predict_stage_mean(temporal)
    persistence_sma = _persistence(temporal, "sma_history")
    persistence_htf = _persistence(temporal, "htf_history")
    true_sma = np.stack([sample.sma_history for sample in temporal])
    true_htf = np.stack([sample.htf_history for sample in temporal])
    slices = _temporal_slice_evaluation(
        predicted_sma,
        predicted_htf,
        true_sma,
        true_htf,
        baseline_sma,
        baseline_htf,
        persistence_sma,
        persistence_htf,
    )
    by_stage: dict[str, object] = {}
    for stage in (2, 4):
        selected = np.asarray([sample.stage_index == stage for sample in temporal])
        by_stage[f"stage_{stage}"] = {
            "sample_count": int(np.count_nonzero(selected)),
            **_temporal_slice_evaluation(
                predicted_sma[selected],
                predicted_htf[selected],
                true_sma[selected],
                true_htf[selected],
                baseline_sma[selected],
                baseline_htf[selected],
                persistence_sma[selected],
                persistence_htf[selected],
            ),
        }
    strictly_positive = bool(
        len({sample.condition_id for sample in temporal}) >= 10
        and _all_temporal_slice_skills_positive(slices)
        and all(
            _all_temporal_slice_skills_positive(value)
            for value in by_stage.values()
        )
    )
    return {
        "condition_count": len({sample.condition_id for sample in temporal}),
        "sample_count": len(temporal),
        **slices,
        "by_stage": by_stage,
        "trust_scores": trust.tolist(),
        "finite": bool(
            np.isfinite(predicted_sma).all() and np.isfinite(predicted_htf).all()
        ),
        "initial_condition_exact_max_abs_K": float(
            max(
                np.max(np.abs(predicted_sma[:, 0] - true_sma[:, 0])),
                np.max(np.abs(predicted_htf[:, 0] - true_htf[:, 0])),
            )
        ),
        "strictly_positive": strictly_positive,
    }


def run(
    dataset_root: Path,
    output: Path,
    epochs: int,
    checkpoint: Path | None = None,
    freeze_manifest: Path | None = None,
    supplemental_temporal_development_root: Path | None = None,
    *, candidate_kind: str = "historical_prior",
) -> dict[str, object]:
    if candidate_kind not in ("historical_prior", "cycle_invariant_neural"):
        raise ValueError("unknown WP4 candidate kind")
    neural_candidate = candidate_kind == "cycle_invariant_neural"
    model_class = CausalNeuralPODOperator if neural_candidate else CausalDeltaPODOperator
    checkpoint_schema = NEURAL_CHECKPOINT_SCHEMA_VERSION if neural_candidate else CHECKPOINT_SCHEMA_VERSION
    checkpoint = checkpoint or output.with_name(f"{output.stem}.pt")
    destinations = [output, checkpoint, *([freeze_manifest] if freeze_manifest else [])]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("report, checkpoint and freeze must have distinct paths")
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(f"preserve existing WP4 evidence: {destination}")
    samples = build_operator_corpus(dataset_root)
    cycle1 = _cycle1_unique(samples)
    temporal = [sample for sample in samples if sample.cycle_index >= 2]
    if not temporal:
        raise RuntimeError("v1.2 temporal diagnostic requires observed cycle-2/3 samples")
    if neural_candidate and (
        len({sample.condition_id for sample in cycle1}) < 200
        or len({sample.condition_id for sample in temporal}) < 10
    ):
        raise RuntimeError("neural WP4 development requires 200 cycle-1 and 10 real temporal conditions")
    quality = json.loads(
        (dataset_root / "quality_report.json").read_text(encoding="utf-8")
    )

    dataset_source_content_sha256 = str(quality["source_content_sha256"])
    implementation_sha256 = _implementation_sha256()
    training_config: dict[str, object] = {
        "seed": 20260828,
        "rank": 4,
        "hidden": 128,
        "epochs": epochs,
        "allowed_cycle_indices": [1],
        "temporal_sma_prior_gains": TEMPORAL_SMA_PRIOR_GAINS,
        "temporal_htf_prior_gains": TEMPORAL_HTF_PRIOR_GAINS,
        "temporal_prior_gain_source": (
            "Paper220 Temporal12 development-only stagewise mean/profile grid"
        ),
    }
    model = model_class(rank=4, hidden=128, seed=20260828)
    training = model.fit(cycle1, epochs=epochs)
    if neural_candidate:
        training_config = dict(model.training_config)
    supplemental_temporal: dict[str, object] | None = None
    supplemental_temporal_count = 0
    if supplemental_temporal_development_root is not None:
        supplemental_quality = json.loads(
            (
                supplemental_temporal_development_root / "quality_report.json"
            ).read_text(encoding="utf-8")
        )
        supplemental_samples = build_operator_corpus(
            supplemental_temporal_development_root
        )
        supplemental_temporal = {
            "evidence_status": (
                "previously inspected failed confirmation; development-only for "
                "this new candidate and ineligible as its confirmation"
            ),
            "dataset_root": str(supplemental_temporal_development_root.resolve()),
            "source_content_sha256": str(
                supplemental_quality["source_content_sha256"]
            ),
            **_supplemental_temporal_diagnostic(model, supplemental_samples),
        }
        supplemental_temporal_count = int(supplemental_temporal["sample_count"])
    condition_holdout = _condition_holdout(cycle1, samples, epochs, model_class=model_class)
    structured_ood = {
        "material_group_holdout": _structured_group_ood(
            cycle1, "material_id", epochs, model_class=model_class
        ),
        "geometry_group_holdout": _structured_group_ood(
            cycle1, "geometry_id", epochs, model_class=model_class
        ),
    }
    predicted_sma, predicted_htf, trust = model.predict(temporal)
    baseline_sma, baseline_htf = model.predict_stage_mean(temporal)
    persistence_sma = _persistence(temporal, "sma_history")
    persistence_htf = _persistence(temporal, "htf_history")
    true_sma = np.stack([sample.sma_history for sample in temporal])
    true_htf = np.stack([sample.htf_history for sample in temporal])

    model_sma = _metrics(predicted_sma, true_sma)
    model_htf = _metrics(predicted_htf, true_htf)
    mean_sma = _metrics(baseline_sma, true_sma)
    mean_htf = _metrics(baseline_htf, true_htf)
    persist_sma = _metrics(persistence_sma, true_sma)
    persist_htf = _metrics(persistence_htf, true_htf)
    temporal_skill_vs_stage_mean = {
        "sma": _skill(model_sma, mean_sma),
        "htf": _skill(model_htf, mean_htf),
    }
    temporal_skill_vs_persistence = {
        "sma": _skill(model_sma, persist_sma),
        "htf": _skill(model_htf, persist_htf),
    }
    temporal_slices = _temporal_slice_evaluation(
        predicted_sma,
        predicted_htf,
        true_sma,
        true_htf,
        baseline_sma,
        baseline_htf,
        persistence_sma,
        persistence_htf,
    )
    temporal_by_stage: dict[str, object] = {}
    for stage in (2, 4):
        selected = np.asarray([sample.stage_index == stage for sample in temporal])
        temporal_by_stage[f"stage_{stage}"] = {
            "sample_count": int(np.count_nonzero(selected)),
            **_temporal_slice_evaluation(
                predicted_sma[selected],
                predicted_htf[selected],
                true_sma[selected],
                true_htf[selected],
                baseline_sma[selected],
                baseline_htf[selected],
                persistence_sma[selected],
                persistence_htf[selected],
            ),
        }
    condition_ready = bool(
        condition_holdout.get("status") == "evaluated"
        and int(condition_holdout.get("sample_count", 0)) >= 40
        and _field_skills_positive(
            condition_holdout.get("skill_vs_stage_mean_baseline")
        )
    )
    structured_ready = bool(
        all(
            split.get("status") == "evaluated_all_groups"
            and int(split.get("pooled", {}).get("sample_count", 0)) >= 20
            and not _nonpositive_structured_skills(split)
            for split in structured_ood.values()
        )
    )
    temporal_ready = bool(
        len({sample.condition_id for sample in temporal}) >= 10
        and _all_temporal_slice_skills_positive(temporal_slices)
        and all(
            _all_temporal_slice_skills_positive(value)
            for value in temporal_by_stage.values()
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

    checkpoint = checkpoint or output.with_name(f"{output.stem}.pt")
    checkpoint_file_sha256 = model.save_checkpoint(
        checkpoint,
        dataset_source_content_sha256=dataset_source_content_sha256,
        implementation_sha256=implementation_sha256,
        training_config=training_config,
    )
    restored = model_class.from_checkpoint(
        checkpoint,
        expected_dataset_source_content_sha256=dataset_source_content_sha256,
        expected_implementation_sha256=implementation_sha256,
    )
    restored_sma, restored_htf, restored_trust = restored.predict(temporal)
    checkpoint_roundtrip_exact = bool(
        restored.parameter_sha256() == model.parameter_sha256()
        and np.array_equal(restored_sma, predicted_sma)
        and np.array_equal(restored_htf, predicted_htf)
        and np.array_equal(restored_trust, trust)
    )
    development_freeze_ready = bool(
        len({sample.condition_id for sample in cycle1}) >= 200
        and condition_ready
        and structured_ready
        and temporal_ready
        and (
            supplemental_temporal is None
            or bool(supplemental_temporal["strictly_positive"])
        )
        and hard_constraints_ready
        and checkpoint_roundtrip_exact
    )
    blocking_reasons: list[str] = []
    if len({sample.condition_id for sample in cycle1}) < 200:
        blocking_reasons.append("Fewer than 200 independent cycle-1 conditions.")
    if not condition_ready:
        blocking_reasons.append(
            "Condition holdout lacks at least 40 samples with positive SMA and HTF skill."
        )
    if not structured_ready:
        blocking_reasons.append(
            "Material/geometry leave-one-group-out does not have positive pooled and every-fold SMA/HTF skill."
        )
    if not temporal_ready:
        blocking_reasons.append(
            "Development temporal diagnostics do not have positive SMA/HTF skill over both persistence and stage mean."
        )
    if supplemental_temporal is not None and not bool(
        supplemental_temporal["strictly_positive"]
    ):
        blocking_reasons.append(
            "Supplemental prior Temporal12 development evidence lacks positive "
            "full-field/profile skill in pooled or stage slices."
        )
    if not hard_constraints_ready:
        blocking_reasons.append("Finite/exact-initial-condition constraints failed.")
    if not checkpoint_roundtrip_exact:
        blocking_reasons.append("Checkpoint prediction/parameter round-trip is not exact.")
    report: dict[str, object] = {
        "schema_version": "sciml_wp4_v12_temporal_development_v1",
        "work_package": "WP4-v1.2-development-only",
        "confirmatory": False,
        "candidate_kind": candidate_kind,
        "readiness_eligible": False,
        "replaces_v11_gate": False,
        "evidence_boundary": (
            "Paper220 Temporal12 and any explicitly supplied prior failed confirmation "
            "labels are development-only hyperparameter evidence and are never passed "
            "to neural fit. This result is non-confirmatory and requires a new untouched temporal matrix "
            "excluding every previously inspected confirmation condition before any paper-grade "
            "or readiness claim."
        ),
        "dataset_source_content_sha256": dataset_source_content_sha256,
        "implementation_sha256": implementation_sha256,
        "training_policy": {
            "allowed_cycle_indices": [1],
            "temporal_label_training_count": 0,
            "temporal_label_hyperparameter_selection_count": (
                len(temporal) + supplemental_temporal_count
            ),
            "cycle1_condition_count": len({sample.condition_id for sample in cycle1}),
            "cycle1_sample_count": len(cycle1),
            "training_ids_sha256": hashlib.sha256(
                "\n".join(model.training_ids).encode("utf-8")
            ).hexdigest(),
            "seed": 20260828,
            "rank": 4,
            "hidden": 128,
            "epochs": epochs,
        },
        "candidate_design": ({
            "initial_field_coordinates": "absolute_mean_plus_mean_free_profile",
            "cycle_index_feature_used": False,
            "cycle_index_used_in_inference": False,
            "inference_policy": INFERENCE_POLICY,
            "hidden_layers": 2,
            "activation": "SiLU",
            "neural_contribution": "unattenuated on both fields and stages in every cycle",
            "trust_output_semantics": "fixed neural weight 1; not confidence or uncertainty",
            "temporal_prior_gain_source": None,
            "evidence_scope": "fixed-grid stage response from observed causal initial fields; not autonomous coupled-cycle rollout",
            "stage_mean_ablation_retained": True,
        } if neural_candidate else {
            "initial_field_coordinates": "absolute_mean_plus_mean_free_profile",
            "cycle_index_feature_used": False,
            "trust_distance": "dimension_normalized_RMS",
            "trust_scale_source": "cycle1_only_with_fixed_physical_scale_floors",
            "temporal_sma_prior_gains": TEMPORAL_SMA_PRIOR_GAINS,
            "temporal_htf_prior_gains": TEMPORAL_HTF_PRIOR_GAINS,
            "temporal_prior_gain_semantics": ["timewise_field_mean", "mean_free_spatial_profile"],
            "temporal_htf_profile_conditioning": "stage_and_causal_material_id",
            "temporal_stage4_sma_strategy": "cycle1_fitted_neural_correction",
            "temporal_prior_gain_source": (
                "Paper220 Temporal12 development-only stagewise mean/profile grid; "
                "no v1.2 confirmation observation used"
            ),
        }),
        "cycle1_structured_ood": structured_ood,
        "cycle1_condition_holdout": condition_holdout,
        "cycle1_condition_holdout_ready": condition_ready,
        "cycle1_structured_ood_strictly_positive": structured_ready,
        "training": training,
        "parameter_sha256": model.parameter_sha256(),
        "temporal_diagnostic": {
            "condition_count": len({sample.condition_id for sample in temporal}),
            "sample_count": len(temporal),
            "sma": model_sma,
            "htf": model_htf,
            "stage_mean_sma": mean_sma,
            "stage_mean_htf": mean_htf,
            "persistence_sma": persist_sma,
            "persistence_htf": persist_htf,
            "skill_vs_stage_mean": temporal_skill_vs_stage_mean,
            "skill_vs_persistence": temporal_skill_vs_persistence,
            **temporal_slices,
            "by_stage": temporal_by_stage,
            "trust_scores": trust.tolist(),
            "trust_nonzero_count": int(np.count_nonzero(trust > 0.0)),
            "trust_min": float(np.min(trust)),
            "trust_median": float(np.median(trust)),
            "trust_max": float(np.max(trust)),
            "initial_condition_exact_max_abs_K": float(
                max(
                    np.max(np.abs(predicted_sma[:, 0] - true_sma[:, 0])),
                    np.max(np.abs(predicted_htf[:, 0] - true_htf[:, 0])),
                )
            ),
        },
        "supplemental_temporal_development_diagnostic": supplemental_temporal,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "schema_version": checkpoint_schema,
            "file_sha256": checkpoint_file_sha256,
            "dataset_hash_matches": True,
            "implementation_hash_matches": True,
            "parameter_hash_matches": bool(
                restored.parameter_sha256() == model.parameter_sha256()
            ),
            "roundtrip_prediction_exact": checkpoint_roundtrip_exact,
            "verified": checkpoint_roundtrip_exact,
        },
        "development_freeze_requirements": {
            "minimum_cycle1_conditions": 200,
            "minimum_condition_holdout_samples": 40,
            "minimum_temporal_conditions": 10,
            "minimum_pooled_structured_holdout_samples": 20,
            "positive_skill_over_stage_mean_required": True,
            "positive_skill_required_in_every_structured_holdout_fold": True,
            "positive_temporal_skill_over_persistence_required": True,
            "positive_temporal_full_field_and_spatial_profile_skill_in_each_stage_required": True,
            "checkpoint_roundtrip_required": True,
            "untouched_confirmation_required_for_paper_grade": True,
        },
        "development_freeze_ready": development_freeze_ready,
        "paper_grade_ready": False,
        "blocking_reasons": blocking_reasons,
    }
    if freeze_manifest is not None:
        report["freeze_manifest"] = str(freeze_manifest.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if freeze_manifest is not None:
        manifest = {
            "schema_version": "sciml_wp4_v12_freeze_manifest_v1",
            "work_package": "WP4-v1.2-frozen-candidate",
            "candidate_kind": candidate_kind,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "development_freeze_ready": development_freeze_ready,
            "confirmation_results_observed_at_freeze": False,
            "dataset_source_content_sha256": dataset_source_content_sha256,
            "implementation_sha256": implementation_sha256,
            "parameter_sha256": model.parameter_sha256(),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_file_sha256,
            "checkpoint_schema_version": checkpoint_schema,
            "checkpoint_roundtrip_verified": checkpoint_roundtrip_exact,
            "development_report_path": str(output.resolve()),
            "development_report_sha256": _file_sha256(output),
            "training_config": training_config,
            "confirmation_contract": {
                "confirm50_config_sha256": CONFIRM50_CONFIG_SHA256,
                "confirm_temporal12_config_sha256": (
                    CONFIRM_TEMPORAL12_CONFIG_SHA256
                ),
                "minimum_confirm50_physics_valid_conditions": 50,
                "minimum_confirm_temporal_physics_valid_conditions": 12,
                "minimum_confirm_temporal_operator_samples": 28,
                "positive_full_field_skill_over_stage_mean_required": True,
                "positive_full_field_skill_over_persistence_required": True,
                "positive_spatial_profile_skill_over_stage_mean_required": True,
                "positive_spatial_profile_skill_over_persistence_required": True,
                "positive_confirm50_skill_in_every_material_geometry_slice_required": True,
                "confirmation_must_not_refit_or_tune": True,
            },
        }
        freeze_manifest.parent.mkdir(parents=True, exist_ok=True)
        freeze_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SciML/results/development/wp4_v12_temporal_candidate.json"),
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("SciML/models/v12/results/wp4_v12_development.pt"),
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("SciML/results/development/wp4_v12_freeze_manifest.json"),
    )
    parser.add_argument(
        "--supplemental-temporal-development-root", type=Path, default=None
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.dataset_root,
                args.output,
                args.epochs,
                args.checkpoint,
                args.freeze_manifest,
                args.supplemental_temporal_development_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
