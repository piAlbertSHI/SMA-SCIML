"""Run the audited WP4 protocol-holdout prototype and ablations.

Example
-------
python -m SciML.models.operators.run_wp4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .data import OperatorBatch, load_protocol_holdout, manifest_coverage
from .deeponet import DeepONetStyleRandomFeatureRegressor
from .metrics import operator_metrics, training_temperature_range
from .spectral import SpectralResidualBaseline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_root(dataset_root: str | Path | None) -> Path:
    default = Path(__file__).resolve().parents[2] / "datasets" / "v1.0"
    return Path(dataset_root).resolve() if dataset_root is not None else default


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "SciML"],
                cwd=repo_root,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        head, dirty = "unavailable", True
    return {
        "git_head": head,
        "git_worktree_dirty_for_sciml": dirty,
        "git_head_semantics": (
            "base HEAD at execution time; when dirty, configuration and per-file "
            "SHA-256 values are the authoritative executable identity"
        ),
    }


def _provenance(config_path: Path, dataset_root: Path) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    code_files = ("__init__.py", "data.py", "deeponet.py", "metrics.py", "run_wp4.py", "spectral.py")
    dataset_files = ("operator_samples.csv", "fields.h5", "split_manifest.json", "provenance.json")
    artifact_manifest = dataset_root.parent / "manifests" / "dataset_v1_artifacts.json"
    artifact_record = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    return {
        "configuration": {
            "file": config_path.name,
            "sha256": _sha256(config_path),
        },
        "dataset": {
            "version": dataset_root.name.removeprefix("v"),
            "artifact_manifest_aggregate_sha256": artifact_record["aggregate_sha256"],
            "files_sha256": {name: _sha256(dataset_root / name) for name in dataset_files},
        },
        "code_sha256": {name: _sha256(package_root / name) for name in code_files},
        **_git_provenance(package_root.parents[2]),
        "environment": {
            # Patch releases vary across the bundled and system interpreters
            # without changing numerical behavior.  Keep the provenance
            # identity stable at the major/minor runtime level so a canonical
            # result remains reproducible across Python 3.12 patch releases.
            "python": ".".join(str(part) for part in platform.python_version_tuple()[:2]),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
    }


def _metrics(
    config: dict[str, Any],
    batch: OperatorBatch,
    prediction: tuple[np.ndarray, np.ndarray],
    temperature_range_k: tuple[float, float],
) -> dict[str, float]:
    settings = config["metrics"]
    return operator_metrics(
        batch,
        *prediction,
        temperature_range_k=temperature_range_k,
        endpoint_tolerance_k=float(settings["endpoint_tolerance_K"]),
        mean_delta_t_balance_tolerance_k=float(
            settings["mean_deltaT_balance_tolerance_K"]
        ),
    )


def _cross_role_duplicate_pairs(
    train: OperatorBatch, test: OperatorBatch
) -> tuple[list[list[str]], list[list[str]]]:
    field_only: list[list[str]] = []
    full_causal: list[list[str]] = []
    for train_index, train_id in enumerate(train.sample_ids):
        for test_index, test_id in enumerate(test.sample_ids):
            fields_equal = (
                np.array_equal(train.sma_in[train_index], test.sma_in[test_index])
                and np.array_equal(train.htf_in[train_index], test.htf_in[test_index])
            )
            if not fields_equal:
                continue
            pair = [train_id, test_id]
            field_only.append(pair)
            causal_equal = (
                np.array_equal(train.topology[train_index], test.topology[test_index])
                and np.array_equal(
                    train.side_input_observed[train_index],
                    test.side_input_observed[test_index],
                )
                and np.array_equal(
                    np.nan_to_num(train.side_input[train_index], nan=0.0),
                    np.nan_to_num(test.side_input[test_index], nan=0.0),
                )
                and train.stage_index[train_index] == test.stage_index[test_index]
                and train.branches[train_index] == test.branches[test_index]
                and train.duration_s[train_index] == test.duration_s[test_index]
            )
            if causal_equal:
                full_causal.append(pair)
    return field_only, full_causal


def run(config_path: str | Path, dataset_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = _dataset_root(dataset_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    batches = load_protocol_holdout(root)
    train, test, golden = (batches[name] for name in ("train", "test", "golden_validation"))
    field_duplicates, full_causal_duplicates = _cross_role_duplicate_pairs(train, test)
    train_ids = set(train.physical_case_ids)
    test_ids = set(test.physical_case_ids)
    golden_ids = set(golden.physical_case_ids)
    if train_ids & test_ids or train_ids & golden_ids or test_ids & golden_ids:
        raise RuntimeError("physical_case_id split leakage detected before fitting")
    metric_settings = config["metrics"]
    train_range = training_temperature_range(
        train, float(metric_settings["train_temperature_range_margin_K"])
    )

    result: dict[str, Any] = {
        "task": config["task"],
        "dataset_version": config["dataset_version"],
        "split_scheme": config["split_scheme"],
        "dataset_audit": {
            "role_counts": {name: len(batch) for name, batch in batches.items()},
            "physical_group_counts": {
                name: len(set(batch.physical_case_ids)) for name, batch in batches.items()
            },
            "group_isolation": "passed",
            "golden_fit_eligible": False,
            "observed_protocols": {
                name: sorted(set(batch.protocol_ids)) for name, batch in batches.items()
            },
            "train_test_protocol_disjoint": not bool(set(train.protocol_ids) & set(test.protocol_ids)),
            "field_only_cross_role_duplicate_pairs": field_duplicates,
            "full_causal_input_cross_role_duplicate_pairs": full_causal_duplicates,
            "no_side_state_ablation_leakage_risk": bool(field_duplicates),
            "content_overlap_interpretation": (
                "Repeated SMA+HTF inlet fields cross train/test, but full causal inputs "
                "including side-state, topology, branch and duration do not. The no-side-state "
                "ablation is diagnostic and not independent generalization evidence."
            ),
            "manifest_coverage": manifest_coverage(root),
            "side_state_causal_contract": {
                "model_inputs": list(train.side_input_labels),
                "input_missing_mask_included": True,
                "audit_only_outputs": list(train.side_output_labels),
                "future_outputs_used_as_features": False,
            },
        },
        "metric_contract": {
            "primary": ["rmse_K", "mae_K", "relative_delta_temperature_residual_l2"],
            "secondary": ["aux_absolute_temperature_relative_l2"],
            "temperature_plausibility_bounds_K": list(train_range),
            "temperature_plausibility_bounds_source": "fit-role inlet and target extrema",
            "cross_grid_interface_metric": "not_computed_no_audited_radial_directional_mapping",
            "mean_deltaT_balance": "unweighted heuristic; not an energy conservation metric",
        },
        "models": {},
        "provenance": _provenance(config_path, root),
        "claim_boundary": config["claim_boundary"],
    }
    base = config["deeponet"]
    for ablation in config["ablations"]:
        model = DeepONetStyleRandomFeatureRegressor(
            latent_dim=int(base["latent_dim"]),
            ridge=float(base["ridge"]),
            seed=int(config["random_seed"]),
            use_side_state=bool(ablation["use_side_state"]),
            stage_specific=bool(ablation["stage_specific"]),
        ).fit(train)
        if set(model.fitted_physical_ids) != train_ids or set(model.fitted_sample_ids) != set(train.sample_ids):
            raise RuntimeError("model fit provenance differs from published train role")
        result["models"][ablation["name"]] = {
            "family": "DeepONet-style random-feature stage-endpoint spatial operator",
            "use_side_state": model.use_side_state,
            "stage_specific": model.stage_specific,
            "trained_stage_ids": list(model.trained_stage_ids),
            "stage_ablation_identifiable": len(model.trained_stage_ids) > 1,
            "input_signature": list(model.input_signature),
            "context_dimension": model.context_dimension,
            "fit_sample_count": len(model.fitted_sample_ids),
            "fit_physical_case_ids": list(model.fitted_physical_ids),
            "train": _metrics(config, train, model.predict(train), train_range),
            "test": _metrics(config, test, model.predict(test), train_range),
            "normalized_train_loss": model.loss(train),
            "normalized_test_loss": model.loss(test),
        }

    spectral_config = config["spectral_baseline"]
    if spectral_config["enabled"]:
        spectral = SpectralResidualBaseline(
            modes=int(spectral_config["modes"]), ridge=float(spectral_config["ridge"])
        ).fit(train)
        if set(spectral.fitted_physical_ids) != train_ids:
            raise RuntimeError("spectral fit provenance differs from published train role")
        result["models"]["spectral_1d"] = {
            "family": spectral_config["scope"],
            "fit_sample_count": len(train),
            "fit_physical_case_ids": list(spectral.fitted_physical_ids),
            "train": _metrics(config, train, spectral.predict(train), train_range),
            "test": _metrics(config, test, spectral.predict(test), train_range),
        }
    return result


def write_report(
    config_path: str | Path,
    output_path: str | Path,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run from source inputs and write canonical deterministic JSON."""

    report = run(config_path, dataset_root)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    operators_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=operators_root.parents[1] / "configs" / "wp4_operator.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=operators_root / "results" / "wp4_operator_results.json"
    )
    args = parser.parse_args()
    report = write_report(args.config, args.output, args.dataset_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
