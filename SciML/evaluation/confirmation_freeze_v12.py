"""Create and verify the immutable pre-confirmation binding for WP2--WP4 v1.2.

The confirmation matrices must stay untouched until every revised candidate is
frozen.  This module binds the exact configs, development reports, implementation
hashes and fitted parameter hashes before a confirmation solver is launched.
It deliberately does not interpret confirmation results or relax any gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SciML.evaluation.physics_preflight_v12 import require_physics_identity


SCHEMA = "sciml_v12_confirmation_freeze_v1"
DEFAULT_ARTIFACTS = {
    "confirm50_config": Path("SciML/configs/wp1_v12_confirm50.json"),
    "confirm_temporal12_config": Path(
        "SciML/configs/wp1_v12_confirm_temporal12.json"
    ),
    "wp2_development_report": Path(
        "SciML/results/development/wp2_v12_development.json"
    ),
    "wp3_development_report": Path(
        "SciML/results/development/wp3_v12_neural_temporal_candidate.json"
    ),
    "wp4_development_report": Path(
        "SciML/results/development/wp4_v12_neural_candidate.json"
    ),
    "development_quality_report": Path(
        "SciML/datasets/v1.1-paper220/quality_report.json"
    ),
    # Freeze the decision code independently from the fitted-model
    # implementation hashes.  This prevents a post-confirmation gate change
    # even when a work package deliberately keeps evaluator and model hashes
    # separate.
    "wp2_confirmation_evaluator": Path(
        "SciML/models/v11/run_cycle_proxy_v12_confirmation.py"
    ),
    "wp3_confirmation_evaluator": Path(
        "SciML/models/v12/run_wp3_confirmation.py"
    ),
    "wp4_confirmation_evaluator": Path(
        "SciML/models/v12/run_wp4_confirmatory.py"
    ),
    "confirm50_smoke_authorizer": Path(
        "SciML/remote/Create_V12_Confirm50_Smoke_Authorization.py"
    ),
    "confirm_temporal12_authorizer": Path(
        "SciML/remote/Create_V12_Confirm_Temporal_Authorization.py"
    ),
    "final_readiness_auditor": Path("SciML/evaluation/readiness_v12.py"),
    "confirmation_freeze_verifier": Path(
        "SciML/evaluation/confirmation_freeze_v12.py"
    ),
    "physics_identity_preflight": Path("SciML/evaluation/physics_preflight_v12.py"),
    "physics_gate_validator": Path("Build_GlobalXi_Run_Validation_Summary.m"),
    "stage_report_builder": Path("Build_GlobalXi_Stage_Report.m"),
    "case_exporter": Path("SciML/builders/Export_SCIML_V11_Case.m"),
    "observation_validator": Path("SciML/builders/Validate_SCIML_Observation_Contract.m"),
    "shared_solver_launcher": Path("SciML/builders/Run_SCIML_V11_Engineering20_Shards.ps1"),
    "confirmation_dataset_builder": Path(
        "SciML/builders/build_dataset_v11_pilot.py"
    ),
    "confirm50_controller": Path(
        "SciML/builders/Run_SCIML_V12_Confirm50_Controller.ps1"
    ),
    "confirm50_launcher": Path(
        "SciML/builders/Run_SCIML_V12_Confirm50_Shards.ps1"
    ),
    "confirm_temporal12_launcher": Path(
        "SciML/builders/Run_SCIML_V12_Confirm_Temporal12_Shards.ps1"
    ),
}
DEFAULT_CONFIRMATION_OUTPUTS = (
    Path("Output/c"),
    Path("Output/ct"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def _relative(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _semantic_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_confirmation_configs(
    confirm50: dict[str, Any], temporal12: dict[str, Any]
) -> None:
    confirm_cases = confirm50.get("cases", [])
    temporal_cases = temporal12.get("cases", [])
    if confirm50.get("dataset_target_version") != "1.2-confirm50":
        raise ValueError("Confirm50 config has the wrong target version")
    if not isinstance(confirm_cases, list) or len(confirm_cases) != 50:
        raise ValueError("Confirm50 config must contain exactly 50 cases")
    confirm_ids = [str(case.get("case_id", "")) for case in confirm_cases]
    if any(not value for value in confirm_ids) or len(set(confirm_ids)) != 50:
        raise ValueError("Confirm50 case IDs must be nonempty and unique")
    confirm_policy = confirm50.get("matrix_policy", {})
    if confirm_policy.get("confirmation_results_must_not_select_or_tune_model") is not True:
        raise ValueError("Confirm50 config permits retrospective model selection")
    if int(confirm_policy.get("paper220_condition_overlap_count", -1)) != 0:
        raise ValueError("Confirm50 config overlaps Paper220 development conditions")
    if int(confirm_policy.get("previous_confirmation_condition_overlap_count", 0)) != 0:
        raise ValueError("Confirm50 config overlaps prior confirmation conditions")
    for source in confirm_policy.get("excluded_confirmation_sources", []):
        if int(source.get("generated_matrix_overlap_count", -1)) != 0:
            raise ValueError("Confirm50 exclusion source records a physical overlap")

    if temporal12.get("dataset_target_version") != "1.2-confirm-temporal12":
        raise ValueError("Confirm-Temporal12 config has the wrong target version")
    if not isinstance(temporal_cases, list) or len(temporal_cases) != 12:
        raise ValueError("Confirm-Temporal12 config must contain exactly 12 cases")
    parents = [str(case.get("parent_confirm50_case_id", "")) for case in temporal_cases]
    if any(not value for value in parents) or len(set(parents)) != 12:
        raise ValueError("Confirm-Temporal12 must bind 12 unique Confirm50 parents")
    if not set(parents) <= set(confirm_ids):
        raise ValueError("Confirm-Temporal12 references a parent outside Confirm50")
    cycles = sorted(int(case.get("cycle_count", 0)) for case in temporal_cases)
    if cycles != [2] * 10 + [3] * 2:
        raise ValueError("Confirm-Temporal12 cycle distribution is not exact")
    temporal_policy = temporal12.get("matrix_policy", {})
    if temporal_policy.get("selection_uses_observed_outcome") is not False:
        raise ValueError("Confirm-Temporal12 selection is outcome-dependent")
    if temporal_policy.get("confirmation_results_must_not_select_or_tune_model") is not True:
        raise ValueError("Confirm-Temporal12 permits retrospective model selection")
    if temporal_policy.get("parent_config_semantic_sha256") != _semantic_sha256(
        confirm50
    ):
        raise ValueError("Confirm-Temporal12 parent semantic hash is stale")


def _assert_untouched_outputs(repo: Path, outputs: tuple[Path, ...]) -> None:
    observed: list[str] = []
    for output in outputs:
        root = _resolve(repo, output)
        if root.is_dir():
            observed.extend(
                _relative(repo, path)
                for path in root.rglob("*")
                if path.is_file()
            )
    if observed:
        raise RuntimeError(
            "confirmation outputs already contain solver artifacts; the pre-run "
            f"freeze cannot be created retrospectively: {observed[:5]}"
        )


def _candidate_binding(name: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    if report.get("confirmatory") is not False:
        raise ValueError(f"{name} report must be explicitly non-confirmatory")
    if report.get("readiness_eligible") is not False:
        raise ValueError(f"{name} development report cannot be readiness eligible")
    implementation = report.get("implementation_sha256")
    parameter = report.get("parameter_sha256")
    for label, value in (
        ("implementation_sha256", implementation),
        ("parameter_sha256", parameter),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{name} report lacks a frozen {label}")
    return {
        "report_path": path.as_posix(),
        "report_sha256": _sha256(path),
        "schema_version": report.get("schema_version"),
        "implementation_sha256": implementation,
        "parameter_sha256": parameter,
        "confirmatory": False,
        "readiness_eligible": False,
    }


def create_manifest(
    repo: Path,
    output: Path,
    *,
    artifacts: dict[str, Path] | None = None,
    confirmation_outputs: tuple[Path, ...] = DEFAULT_CONFIRMATION_OUTPUTS,
    campaign_id: str = "v12-confirmation",
) -> dict[str, Any]:
    repo = repo.resolve()
    destination = _resolve(repo, output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite an existing freeze: {destination}")
    selected = dict(DEFAULT_ARTIFACTS if artifacts is None else artifacts)
    resolved = {name: _resolve(repo, path).resolve() for name, path in selected.items()}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"freeze inputs are missing: {missing}")
    _assert_untouched_outputs(repo, confirmation_outputs)
    if not confirmation_outputs:
        raise ValueError("confirmation freeze requires versioned output roots")
    confirm50_output = confirmation_outputs[0]
    temporal_output = (
        confirmation_outputs[1]
        if len(confirmation_outputs) > 1
        else confirmation_outputs[0]
    )
    if not campaign_id.strip():
        raise ValueError("confirmation campaign_id must be nonempty")

    quality = _json(resolved["development_quality_report"])
    source_hash = quality.get("source_content_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("development quality report lacks source_content_sha256")
    physics_identity = require_physics_identity(repo, resolved)
    resolved["development_case_index"] = resolved["development_quality_report"].parent / "case_index.csv"
    for index, name in enumerate(sorted(physics_identity["current_physics_sources"])):
        resolved[f"physics_source_{index}"] = repo / name
    _validate_confirmation_configs(
        _json(resolved["confirm50_config"]),
        _json(resolved["confirm_temporal12_config"]),
    )

    candidate_bindings = {}
    for wp in ("wp2", "wp3", "wp4"):
        key = f"{wp}_development_report"
        candidate_bindings[wp] = _candidate_binding(
            wp.upper(),
            resolved[key],
            _json(resolved[key]),
        )
        candidate_bindings[wp]["report_path"] = _relative(repo, resolved[key])

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_confirmation",
        "development_dataset_source_content_sha256": source_hash,
        "physics_identity_binding": physics_identity,
        "artifact_bindings": {
            name: {
                "path": _relative(repo, path),
                "sha256": _sha256(path),
            }
            for name, path in sorted(resolved.items())
        },
        "candidate_bindings": candidate_bindings,
        "confirmation_output_roots": {
            "confirm50": _relative(repo, _resolve(repo, confirm50_output)),
            "confirm_temporal12": _relative(
                repo, _resolve(repo, temporal_output)
            ),
        },
        "confirmation_policy": {
            "confirmation_results_must_not_select_or_tune_model": True,
            "required_independent_confirm50_conditions": 50,
            "required_temporal_conditions": 12,
            "required_cycle_counts": [2, 3],
            "required_gates": [
                "physics",
                "provenance",
                "structured_ood",
                "temporal",
                "checkpoint_hashes",
                "completion",
            ],
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_manifest(repo: Path, manifest_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    path = _resolve(repo, manifest_path)
    manifest = _json(path)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unexpected v1.2 confirmation-freeze schema")
    if manifest.get("status") != "frozen_before_confirmation":
        raise ValueError("confirmation freeze is not in the frozen state")
    if not str(manifest.get("campaign_id", "")).strip():
        raise ValueError("confirmation freeze lacks a campaign_id")
    output_roots = manifest.get("confirmation_output_roots", {})
    if not isinstance(output_roots, dict) or set(output_roots) != {
        "confirm50",
        "confirm_temporal12",
    }:
        raise ValueError("confirmation freeze lacks exact versioned output roots")
    bindings = manifest.get("artifact_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("confirmation freeze has no artifact bindings")
    if not isinstance(manifest.get("physics_identity_binding"), dict):
        raise ValueError("historical freeze lacks pre-confirmation physics identity; new development/freeze required")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid binding for {name}")
        artifact = _resolve(repo, Path(str(binding.get("path", ""))))
        if not artifact.is_file():
            raise FileNotFoundError(f"frozen artifact is missing: {artifact}")
        actual = _sha256(artifact)
        if actual != binding.get("sha256"):
            raise RuntimeError(f"frozen artifact changed after binding: {name}")
    current_identity = require_physics_identity(repo, {
        name: _resolve(repo, Path(str(binding["path"]))) for name, binding in bindings.items()
    })
    if current_identity != manifest["physics_identity_binding"]:
        raise RuntimeError("physics identity changed after binding")
    for wp, binding in manifest.get("candidate_bindings", {}).items():
        report_path = _resolve(repo, Path(str(binding["report_path"])))
        report = _json(report_path)
        if _sha256(report_path) != binding.get("report_sha256"):
            raise RuntimeError(f"{wp} development report changed after binding")
        for key in ("implementation_sha256", "parameter_sha256"):
            if report.get(key) != binding.get(key):
                raise RuntimeError(f"{wp} {key} changed after binding")
        if report.get("confirmatory") is not False or report.get(
            "readiness_eligible"
        ) is not False:
            raise RuntimeError(f"{wp} development evidence boundary changed")
    _validate_confirmation_configs(
        _json(_resolve(repo, Path(str(bindings["confirm50_config"]["path"])))),
        _json(
            _resolve(
                repo,
                Path(str(bindings["confirm_temporal12_config"]["path"])),
            )
        ),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("SciML/results/v12_confirmation_freeze.json"),
    )
    parser.add_argument("--campaign-id", default="v12-confirmation")
    parser.add_argument("--development-root", type=Path, default=Path("SciML/datasets/v1.1-paper220"))
    parser.add_argument("--development-reports-root", type=Path, default=Path("SciML/results/development"))
    parser.add_argument(
        "--confirm50-config",
        type=Path,
        default=DEFAULT_ARTIFACTS["confirm50_config"],
    )
    parser.add_argument(
        "--confirm-temporal12-config",
        type=Path,
        default=DEFAULT_ARTIFACTS["confirm_temporal12_config"],
    )
    parser.add_argument(
        "--confirm50-output", type=Path, default=DEFAULT_CONFIRMATION_OUTPUTS[0]
    )
    parser.add_argument(
        "--confirm-temporal12-output",
        type=Path,
        default=DEFAULT_CONFIRMATION_OUTPUTS[1],
    )
    args = parser.parse_args()
    if args.mode == "create":
        artifacts = dict(DEFAULT_ARTIFACTS)
        artifacts["confirm50_config"] = args.confirm50_config
        artifacts["confirm_temporal12_config"] = args.confirm_temporal12_config
        artifacts["development_quality_report"] = args.development_root / "quality_report.json"
        for wp, filename in (
            ("wp2", "wp2_v12_development.json"),
            ("wp3", "wp3_v12_neural_temporal_candidate.json"),
            ("wp4", "wp4_v12_neural_candidate.json"),
        ):
            artifacts[f"{wp}_development_report"] = args.development_reports_root / filename
        payload = create_manifest(
            args.repo,
            args.manifest,
            artifacts=artifacts,
            confirmation_outputs=(
                args.confirm50_output,
                args.confirm_temporal12_output,
            ),
            campaign_id=args.campaign_id,
        )
    else:
        payload = verify_manifest(args.repo, args.manifest)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
