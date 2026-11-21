"""Evaluate the frozen WP4-v1.2 checkpoint on untouched confirmation data.

This module has no training path.  It only loads a hash-bound freeze manifest
and checkpoint, then evaluates the preregistered Confirm50 and
Confirm-Temporal12 datasets.  Confirmation observations are never passed to
``fit`` and cannot alter model parameters, trust scales, POD bases, stage
means, gains, or gate definitions.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from SciML.models.v11.spatiotemporal_operator import (
    OperatorSample,
    build_operator_corpus,
)
from SciML.models.v12.run_wp4_temporal_development import (
    _cycle1_unique,
    _all_temporal_slice_skills_positive,
    _finite_positive,
    _file_sha256,
    _implementation_sha256,
    _metrics,
    _persistence,
    _skill,
    _temporal_slice_evaluation,
)
from SciML.models.v12.temporal_operator_candidate import CausalDeltaPODOperator
from SciML.models.v12.neural_operator_candidate import CausalNeuralPODOperator


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a JSON object")
    return payload


def _verify_global_freeze_binding(
    global_freeze_manifest: Path,
    local_freeze: Mapping[str, Any],
    development_report: Path,
    confirm50_config: Path,
    confirm_temporal12_config: Path,
) -> dict[str, Any]:
    """Validate global protocol bindings without replacing local freeze checks.

    The WP4 development manifest is intentionally a work-package-local
    checkpoint authority and historically carried hashes from an earlier
    preregistration.  The repository-wide v1.2 freeze is the sole authority
    for confirmation config hashes and for the report binding consumed by
    readiness.  Both authorities are checked and kept explicit in the output.
    """

    if not global_freeze_manifest.is_file():
        raise FileNotFoundError(
            f"global confirmation freeze is missing: {global_freeze_manifest}"
        )
    global_payload = _read_json(global_freeze_manifest)
    if global_payload.get("schema_version") != "sciml_v12_confirmation_freeze_v1":
        raise ValueError("unsupported global v1.2 confirmation freeze manifest")
    if global_payload.get("status") != "frozen_before_confirmation":
        raise ValueError("global confirmation freeze is not in the frozen state")
    artifacts = global_payload.get("artifact_bindings", {})
    config_hashes: dict[str, str] = {}
    for key, path in (
        ("confirm50_config", confirm50_config),
        ("confirm_temporal12_config", confirm_temporal12_config),
    ):
        binding = artifacts.get(key)
        if not isinstance(binding, Mapping):
            raise ValueError(f"global freeze lacks the {key} binding")
        actual = _file_sha256(path)
        if actual != str(binding.get("sha256", "")).lower():
            raise RuntimeError(f"{key} differs from the global freeze")
        config_hashes[key] = actual

    global_candidate = global_payload.get("candidate_bindings", {}).get("wp4")
    if not isinstance(global_candidate, Mapping):
        raise ValueError("global freeze lacks the wp4 candidate binding")
    for key in ("implementation_sha256", "parameter_sha256"):
        if str(local_freeze.get(key, "")) != str(global_candidate.get(key, "")):
            raise RuntimeError(f"WP4 candidate {key} differs from global freeze")
    report_hash = _file_sha256(development_report)
    if report_hash != str(global_candidate.get("report_sha256", "")).lower():
        raise RuntimeError("WP4 development report differs from global freeze")
    return {
        "path": str(global_freeze_manifest.resolve()),
        "sha256": _file_sha256(global_freeze_manifest),
        "config_sha256": config_hashes,
        "candidate": dict(global_candidate),
    }


def _profile(values: np.ndarray) -> np.ndarray:
    """Return the mean-free 101-node spatial profile at every time."""

    return values - np.mean(values, axis=-1, keepdims=True)


def _field_evaluation(
    prediction: np.ndarray,
    target: np.ndarray,
    stage_mean: np.ndarray,
    persistence: np.ndarray | None = None,
) -> dict[str, object]:
    model = _metrics(prediction, target)
    mean = _metrics(stage_mean, target)
    result: dict[str, object] = {
        "model": model,
        "stage_mean": mean,
        "skill_vs_stage_mean": _skill(model, mean),
    }
    if persistence is not None:
        persist = _metrics(persistence, target)
        result["persistence"] = persist
        result["skill_vs_persistence"] = _skill(model, persist)
    return result


def _confirm50_structured_slices(
    samples: list[OperatorSample],
    predicted_sma: np.ndarray,
    predicted_htf: np.ndarray,
    baseline_sma: np.ndarray,
    baseline_htf: np.ndarray,
) -> dict[str, object]:
    true_sma = np.stack([sample.sma_history for sample in samples])
    true_htf = np.stack([sample.htf_history for sample in samples])

    def evaluate(selected: np.ndarray) -> dict[str, object]:
        return {
            "sample_count": int(np.count_nonzero(selected)),
            "condition_count": len(
                {sample.condition_id for sample, keep in zip(samples, selected) if keep}
            ),
            "sma": _field_evaluation(
                predicted_sma[selected], true_sma[selected], baseline_sma[selected]
            ),
            "htf": _field_evaluation(
                predicted_htf[selected], true_htf[selected], baseline_htf[selected]
            ),
        }

    pooled = evaluate(np.ones(len(samples), dtype=bool))
    splits: dict[str, object] = {"pooled": pooled}
    for split_name, attribute in (
        ("material", "material_id"),
        ("geometry", "geometry_id"),
    ):
        groups: dict[str, object] = {}
        for group in sorted({str(getattr(sample, attribute)) for sample in samples}):
            selected = np.asarray(
                [str(getattr(sample, attribute)) == group for sample in samples]
            )
            groups[group] = evaluate(selected)
        splits[f"by_{split_name}"] = groups
    return splits


def _all_confirm50_skills_positive(report: dict[str, object]) -> bool:
    if not isinstance(report, dict) or not isinstance(report.get("pooled"), dict):
        return False
    blocks: list[dict[str, object]] = [report["pooled"]]
    for key in ("by_material", "by_geometry"):
        groups = report.get(key, {})
        if not isinstance(groups, dict) or not groups:
            return False
        if not all(isinstance(value, dict) for value in groups.values()):
            return False
        blocks.extend(groups.values())
    for block in blocks:
        for field in ("sma", "htf"):
            values = block.get(field, {})
            if not isinstance(values, dict) or not _finite_positive(
                values.get("skill_vs_stage_mean")
            ):
                return False
    return True


def _temporal_evaluation(
    samples: list[OperatorSample],
    predicted_sma: np.ndarray,
    predicted_htf: np.ndarray,
    baseline_sma: np.ndarray,
    baseline_htf: np.ndarray,
) -> dict[str, object]:
    true_sma = np.stack([sample.sma_history for sample in samples])
    true_htf = np.stack([sample.htf_history for sample in samples])
    persistence_sma = _persistence(samples, "sma_history")
    persistence_htf = _persistence(samples, "htf_history")
    full_field = {
        "sma": _field_evaluation(
            predicted_sma, true_sma, baseline_sma, persistence_sma
        ),
        "htf": _field_evaluation(
            predicted_htf, true_htf, baseline_htf, persistence_htf
        ),
    }
    spatial_profile = {
        "sma": _field_evaluation(
            _profile(predicted_sma),
            _profile(true_sma),
            _profile(baseline_sma),
            _profile(persistence_sma),
        ),
        "htf": _field_evaluation(
            _profile(predicted_htf),
            _profile(true_htf),
            _profile(baseline_htf),
            _profile(persistence_htf),
        ),
    }
    by_stage: dict[str, object] = {}
    for stage in (2, 4):
        selected = np.asarray([sample.stage_index == stage for sample in samples])
        by_stage[f"stage_{stage}"] = {
            "sample_count": int(np.count_nonzero(selected)),
            **_temporal_slice_evaluation(
                predicted_sma[selected], predicted_htf[selected],
                true_sma[selected], true_htf[selected],
                baseline_sma[selected], baseline_htf[selected],
                persistence_sma[selected], persistence_htf[selected],
            ),
            "sma": _field_evaluation(
                predicted_sma[selected],
                true_sma[selected],
                baseline_sma[selected],
                persistence_sma[selected],
            ),
            "htf": _field_evaluation(
                predicted_htf[selected],
                true_htf[selected],
                baseline_htf[selected],
                persistence_htf[selected],
            ),
        }
    return {
        "condition_count": len({sample.condition_id for sample in samples}),
        "sample_count": len(samples),
        "full_field": full_field,
        "spatial_profile": spatial_profile,
        "by_stage": by_stage,
        "initial_condition_exact_max_abs_K": float(
            max(
                np.max(np.abs(predicted_sma[:, 0] - true_sma[:, 0])),
                np.max(np.abs(predicted_htf[:, 0] - true_htf[:, 0])),
            )
        ),
        "finite": bool(
            np.isfinite(predicted_sma).all() and np.isfinite(predicted_htf).all()
        ),
    }


def _all_temporal_skills_positive(report: dict[str, object]) -> bool:
    if not _all_temporal_slice_skills_positive(report):
        return False
    stages = report.get("by_stage")
    return isinstance(stages, dict) and all(
        _all_temporal_slice_skills_positive(stages.get(f"stage_{stage}"))
        for stage in (2, 4)
    )


def _dataset_binding(
    root: Path, expected_config_sha256: str
) -> tuple[dict[str, Any], list[str]]:
    quality = _read_json(root / "quality_report.json")
    provenance = _read_json(root / "provenance.json")
    failures: list[str] = []
    if str(provenance.get("expected_config_sha256", "")).lower() != (
        expected_config_sha256.lower()
    ):
        failures.append("expected confirmation config hash mismatch")
    quality_hash = str(quality.get("source_content_sha256", ""))
    if not quality_hash or quality_hash != str(
        provenance.get("source_content_sha256", "")
    ):
        failures.append("quality/provenance source-content hash mismatch")
    if int(quality.get("request_config_mismatch_count", 0)) != 0:
        failures.append("one or more requests do not match the frozen config")
    case_count = int(quality.get("case_count", -1))
    for key, description in (
        ("request_config_match_count", "config-matched request"),
        ("request_self_contained_count", "self-contained request"),
        ("export_provenance_complete_count", "complete export provenance"),
        ("observation_contract_ok_count", "observation-contract-compliant case"),
        ("artifact_complete_case_count", "artifact-complete case"),
        ("run_ok_count", "successful solver case"),
        ("physics_ok_count", "physics-valid case"),
    ):
        if case_count < 0 or int(quality.get(key, -2)) != case_count:
            failures.append(f"not every case is a {description}")
    if not bool(quality.get("expected_matrix_complete", False)):
        failures.append("registered confirmation matrix is incomplete")
    confirmation_gate = quality.get("confirmation_quality_gate", {})
    if not isinstance(confirmation_gate, dict) or confirmation_gate.get("passed") is not True:
        failures.append("confirmation dataset strict-quality gate failed")
    if not bool(quality.get("all_required_fields_finite", False)):
        failures.append("required dataset fields are not all finite")
    if int(quality.get("physics_implementation_fingerprint_count", 0)) != 1:
        failures.append("physics implementation fingerprint is not unique")
    return {"quality": quality, "provenance": provenance}, failures


def _time_is_after_freeze(freeze_created: str, quality_generated: str) -> bool:
    try:
        return datetime.fromisoformat(quality_generated) > datetime.fromisoformat(
            freeze_created
        )
    except (TypeError, ValueError):
        return False


def _checkpoint_binding(
    checkpoint: Path, checkpoint_sha256: str, roundtrip_verified: bool
) -> dict[str, object]:
    """Return the readiness-consumable, file-bound checkpoint fields."""

    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_roundtrip_verified": bool(roundtrip_verified),
    }


def run(
    confirm50_root: Path,
    confirm_temporal12_root: Path,
    freeze_manifest_path: Path,
    output: Path,
    checkpoint_override: Path | None = None,
    development_report_override: Path | None = None,
    global_freeze_manifest: Path | None = None,
    confirm50_config: Path | None = None,
    confirm_temporal12_config: Path | None = None,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"preserve existing WP4 confirmation evidence: {output}")
    manifest = _read_json(freeze_manifest_path)
    candidate_kind = manifest.get("candidate_kind", "historical_prior")
    if candidate_kind not in ("historical_prior", "cycle_invariant_neural"):
        raise ValueError("unknown frozen WP4 candidate kind")
    model_class = CausalNeuralPODOperator if candidate_kind == "cycle_invariant_neural" else CausalDeltaPODOperator
    blockers: list[str] = []
    if manifest.get("schema_version") != "sciml_wp4_v12_freeze_manifest_v1":
        blockers.append("Unsupported or missing WP4-v1.2 freeze manifest schema.")
    if not bool(manifest.get("development_freeze_ready", False)):
        blockers.append("Development candidate was not freeze-ready.")
    if bool(manifest.get("confirmation_results_observed_at_freeze", True)):
        blockers.append("Freeze manifest does not prove pre-confirmation freezing.")
    development_report = development_report_override or Path(
        str(manifest.get("development_report_path", ""))
    )
    if not development_report.is_file():
        blockers.append("Frozen development report is missing.")
    else:
        development_payload = _read_json(development_report)
        if _file_sha256(development_report) != str(
            manifest.get("development_report_sha256", "")
        ):
            blockers.append("Frozen development report file hash mismatch.")
        if not bool(development_payload.get("development_freeze_ready", False)):
            blockers.append("Frozen development report is not freeze-ready.")
        if development_payload.get("candidate_kind", "historical_prior") != candidate_kind:
            blockers.append("Frozen development candidate kind mismatch.")
    global_binding: dict[str, Any] | None = None
    contract = manifest.get("confirmation_contract", {})
    if not isinstance(contract, dict):
        contract = {}
        blockers.append("Freeze manifest confirmation contract is missing.")
    # Config hashes are resolved from the global freeze below.  Do not use the
    # legacy constants/local contract values: those may describe an earlier
    # preregistration and are not the readiness binding.
    expected_contract = {
        "minimum_confirm50_physics_valid_conditions": 50,
        "minimum_confirm_temporal_physics_valid_conditions": 12,
        "minimum_confirm_temporal_operator_samples": 28,
        "positive_full_field_skill_over_stage_mean_required": True,
        "positive_full_field_skill_over_persistence_required": True,
        "positive_spatial_profile_skill_over_stage_mean_required": True,
        "positive_spatial_profile_skill_over_persistence_required": True,
        "positive_confirm50_skill_in_every_material_geometry_slice_required": True,
        "confirmation_must_not_refit_or_tune": True,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            blockers.append(f"Frozen confirmation contract changed: {key}.")

    implementation_sha256 = _implementation_sha256()
    if str(manifest.get("implementation_sha256", "")) != implementation_sha256:
        blockers.append("Frozen candidate implementation hash is stale.")
    checkpoint = checkpoint_override or Path(str(manifest.get("checkpoint_path", "")))
    if not checkpoint.is_file():
        blockers.append("Frozen checkpoint is missing.")
        raise FileNotFoundError(blockers[-1])
    checkpoint_sha256 = _file_sha256(checkpoint)
    if checkpoint_sha256 != str(manifest.get("checkpoint_sha256", "")):
        blockers.append("Frozen checkpoint file hash mismatch.")
    if not bool(manifest.get("checkpoint_roundtrip_verified", False)):
        blockers.append("Freeze manifest lacks a verified checkpoint round-trip.")
    if blockers:
        raise ValueError("WP4 pre-deserialization freeze checks failed: " + "; ".join(blockers))
    model = model_class.from_checkpoint(
        checkpoint,
        expected_dataset_source_content_sha256=str(
            manifest.get("dataset_source_content_sha256", "")
        ),
        expected_implementation_sha256=implementation_sha256,
    )
    parameter_sha256 = model.parameter_sha256()
    if parameter_sha256 != str(manifest.get("parameter_sha256", "")):
        blockers.append("Frozen checkpoint parameter hash mismatch.")

    global_binding = _verify_global_freeze_binding(
        global_freeze_manifest or freeze_manifest_path,
        manifest,
        development_report,
        confirm50_config or Path("SciML/configs/wp1_v12_confirm50.json"),
        confirm_temporal12_config
        or Path("SciML/configs/wp1_v12_confirm_temporal12.json"),
    )
    # The command-line config paths are fixed by the preregistered contract;
    # resolve them from the current working repository for the dataset checks.
    global_config_hashes = global_binding["config_sha256"]
    confirm50_config_hash = str(global_config_hashes["confirm50_config"])
    temporal_config_hash = str(global_config_hashes["confirm_temporal12_config"])
    confirm50_binding, confirm50_failures = _dataset_binding(
        confirm50_root, confirm50_config_hash
    )
    temporal_binding, temporal_failures = _dataset_binding(
        confirm_temporal12_root, temporal_config_hash
    )
    blockers.extend(f"Confirm50: {value}." for value in confirm50_failures)
    blockers.extend(
        f"Confirm-Temporal12: {value}." for value in temporal_failures
    )
    freeze_created = str(manifest.get("created_utc", ""))
    for name, binding in (
        ("Confirm50", confirm50_binding),
        ("Confirm-Temporal12", temporal_binding),
    ):
        if not _time_is_after_freeze(
            freeze_created, str(binding["quality"].get("generated_utc", ""))
        ):
            blockers.append(f"{name} dataset does not postdate the model freeze.")

    confirm50_samples = _cycle1_unique(build_operator_corpus(confirm50_root))
    temporal_all = build_operator_corpus(confirm_temporal12_root)
    temporal_samples = [sample for sample in temporal_all if sample.cycle_index >= 2]
    confirm50_conditions = {sample.condition_id for sample in confirm50_samples}
    temporal_conditions = {sample.condition_id for sample in temporal_samples}
    if len(confirm50_conditions) != int(
        contract.get("minimum_confirm50_physics_valid_conditions", 50)
    ):
        blockers.append("Confirm50 does not contain exactly 50 physics-valid conditions.")
    if len(temporal_conditions) != int(
        contract.get("minimum_confirm_temporal_physics_valid_conditions", 12)
    ):
        blockers.append(
            "Confirm-Temporal12 does not contain exactly 12 physics-valid conditions."
        )
    if len(temporal_samples) != int(
        contract.get("minimum_confirm_temporal_operator_samples", 28)
    ):
        blockers.append(
            "Confirm-Temporal12 does not contain exactly 28 temporal operator samples."
        )

    confirm50_sma, confirm50_htf, confirm50_trust = model.predict(confirm50_samples)
    confirm50_baseline_sma, confirm50_baseline_htf = model.predict_stage_mean(
        confirm50_samples
    )
    confirm50_report = _confirm50_structured_slices(
        confirm50_samples,
        confirm50_sma,
        confirm50_htf,
        confirm50_baseline_sma,
        confirm50_baseline_htf,
    )
    confirm50_ready = _all_confirm50_skills_positive(confirm50_report)
    if not confirm50_ready:
        blockers.append(
            "Confirm50 lacks positive SMA/HTF skill in pooled or every material/geometry slice."
        )

    temporal_sma, temporal_htf, temporal_trust = model.predict(temporal_samples)
    temporal_baseline_sma, temporal_baseline_htf = model.predict_stage_mean(
        temporal_samples
    )
    temporal_report = _temporal_evaluation(
        temporal_samples,
        temporal_sma,
        temporal_htf,
        temporal_baseline_sma,
        temporal_baseline_htf,
    )
    temporal_ready = bool(
        _all_temporal_skills_positive(temporal_report)
        and temporal_report["finite"]
        and float(temporal_report["initial_condition_exact_max_abs_K"]) == 0.0
    )
    if not temporal_ready:
        blockers.append(
            "Confirm-Temporal12 full-field/spatial-profile temporal gates failed."
        )

    restored = model_class.from_checkpoint(
        checkpoint,
        expected_dataset_source_content_sha256=str(
            manifest.get("dataset_source_content_sha256", "")
        ),
        expected_implementation_sha256=implementation_sha256,
    )
    restored_confirm50 = restored.predict(confirm50_samples)
    restored_temporal = restored.predict(temporal_samples)
    checkpoint_roundtrip_verified = bool(
        restored.parameter_sha256() == parameter_sha256
        and all(
            np.array_equal(first, second)
            for first, second in zip(
                (confirm50_sma, confirm50_htf, confirm50_trust),
                restored_confirm50,
            )
        )
        and all(
            np.array_equal(first, second)
            for first, second in zip(
                (temporal_sma, temporal_htf, temporal_trust),
                restored_temporal,
            )
        )
    )
    if not checkpoint_roundtrip_verified:
        blockers.append("Checkpoint round-trip predictions are not exact.")

    provenance_ready = not confirm50_failures and not temporal_failures
    frozen_before_confirmation = all(
        _time_is_after_freeze(
            freeze_created, str(binding["quality"].get("generated_utc", ""))
        )
        for binding in (confirm50_binding, temporal_binding)
    )
    confirmatory = bool(
        manifest.get("development_freeze_ready", False)
        and not manifest.get("confirmation_results_observed_at_freeze", True)
        and provenance_ready
        and frozen_before_confirmation
    )
    paper_grade_ready = bool(
        confirmatory
        and not blockers
        and confirm50_ready
        and temporal_ready
        and checkpoint_roundtrip_verified
    )
    confirm50_source_sha256 = str(
        confirm50_binding["quality"]["source_content_sha256"]
    )
    temporal_source_sha256 = str(
        temporal_binding["quality"]["source_content_sha256"]
    )
    combined_source_sha256 = hashlib.sha256(
        (confirm50_source_sha256 + "\n" + temporal_source_sha256).encode("ascii")
    ).hexdigest()
    report: dict[str, object] = {
        "schema_version": "sciml_wp4_v12_confirmatory_v1",
        "work_package": "WP4-v1.2-confirmatory",
        "candidate_kind": candidate_kind,
        "trust_output_semantics": (
            "fixed neural contribution weight; not confidence or uncertainty"
            if candidate_kind == "cycle_invariant_neural" else "historical distance-based attenuation"
        ),
        "confirmatory": confirmatory,
        "readiness_eligible": paper_grade_ready,
        "paper_grade_ready": paper_grade_ready,
        "freeze_manifest_sha256": global_binding["sha256"],
        "local_freeze_manifest_sha256": _file_sha256(freeze_manifest_path),
        "implementation_sha256": implementation_sha256,
        "parameter_sha256": parameter_sha256,
        "checkpoint": str(checkpoint.resolve()),
        **_checkpoint_binding(
            checkpoint, checkpoint_sha256, checkpoint_roundtrip_verified
        ),
        "confirmation_dataset_source_content_sha256": combined_source_sha256,
        "confirmation_bindings": {
            "confirm50_source_content_sha256": confirm50_source_sha256,
            "confirm_temporal12_source_content_sha256": temporal_source_sha256,
            "confirm50_config_sha256": confirm50_config_hash,
            "confirm_temporal12_config_sha256": temporal_config_hash,
        },
        "global_freeze_binding": global_binding,
        "training_or_tuning_on_confirmation_count": 0,
        "confirm50": {
            "condition_count": len(confirm50_conditions),
            "sample_count": len(confirm50_samples),
            "structured_evaluation": confirm50_report,
            "trust_scores": confirm50_trust.tolist(),
            "strictly_positive": confirm50_ready,
        },
        "confirm_temporal12": {
            **temporal_report,
            "trust_scores": temporal_trust.tolist(),
            "strictly_positive": temporal_ready,
        },
        "blocking_reasons": blockers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm50-root", type=Path, required=True)
    parser.add_argument("--confirm-temporal12-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--global-freeze-manifest",
        type=Path,
        default=Path("SciML/results/v12_confirmation_freeze.json"),
    )
    parser.add_argument(
        "--confirm50-config",
        type=Path,
        default=Path("SciML/configs/wp1_v12_confirm50.json"),
    )
    parser.add_argument(
        "--confirm-temporal12-config",
        type=Path,
        default=Path("SciML/configs/wp1_v12_confirm_temporal12.json"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--development-report", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SciML/results/wp4_v12_confirmation.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.confirm50_root,
                args.confirm_temporal12_root,
                args.freeze_manifest,
                args.output,
                args.checkpoint,
                args.development_report,
                args.global_freeze_manifest,
                args.confirm50_config,
                args.confirm_temporal12_config,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
