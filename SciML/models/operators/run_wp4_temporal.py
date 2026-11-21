"""Train and audit the WP4 temporal-probe DeepONet over multiple seeds."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from statistics import mean, stdev
import subprocess
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from .data import load_protocol_holdout
from .temporal_probe import temporal_probe_metrics, train_temporal_probe_operator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "wp4_temporal_probe.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "wp4_temporal_probe_results.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(seed_records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "rmse_K", "mae_K", "final_time_rmse_K", "relative_temperature_change_l2",
        "persistence_rmse_K", "skill_vs_persistence", "temperature_range_violation_rate",
    )
    result: dict[str, Any] = {}
    for role in ("train", "test", "golden_validation"):
        result[role] = {}
        for metric in metrics:
            values = [float(record["metrics"][role][metric]) for record in seed_records]
            result[role][metric + "_mean"] = mean(values)
            result[role][metric + "_std"] = stdev(values) if len(values) > 1 else 0.0
    return result


def _git_provenance() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "SciML"],
                cwd=ROOT.parent,
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


def _generalization_gate(
    config: dict[str, Any], aggregate: dict[str, Any]
) -> dict[str, Any]:
    settings = config["generalization_gate"]
    criteria = {
        "test_skill_vs_persistence": {
            "observed": float(aggregate["test"]["skill_vs_persistence_mean"]),
            "operator": ">",
            "threshold": float(settings["test_skill_vs_persistence_min"]),
        },
        "test_relative_temperature_change_l2": {
            "observed": float(
                aggregate["test"]["relative_temperature_change_l2_mean"]
            ),
            "operator": "<=",
            "threshold": float(settings["test_relative_temperature_change_l2_max"]),
        },
        "golden_skill_vs_persistence": {
            "observed": float(
                aggregate["golden_validation"]["skill_vs_persistence_mean"]
            ),
            "operator": ">",
            "threshold": float(settings["golden_skill_vs_persistence_min"]),
        },
    }
    for item in criteria.values():
        item["passed"] = (
            item["observed"] > item["threshold"]
            if item["operator"] == ">"
            else item["observed"] <= item["threshold"]
        )
    return {
        "policy": settings["policy"],
        "all_criteria_required": True,
        "criteria": criteria,
        "passed": all(bool(item["passed"]) for item in criteria.values()),
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    batches = load_protocol_holdout()
    train = batches["train"]
    train_range = (
        float(np.min(train.temperature_probes_t_K)),
        float(np.max(train.temperature_probes_t_K)),
    )
    training = config["training"]
    model_config = config["model"]
    seed_records: list[dict[str, Any]] = []
    for seed in config["random_seeds"]:
        fitted = train_temporal_probe_operator(
            train,
            seed=int(seed),
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            smoothness_weight=float(training["smoothness_weight"]),
            latent_dim=int(model_config["latent_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
        )
        role_metrics = {
            role: temporal_probe_metrics(
                batch, fitted.predict(batch), temperature_range_k=train_range
            )
            for role, batch in batches.items()
        }
        seed_records.append(
            {
                "seed": int(seed),
                "fit_sample_ids": list(fitted.fitted_sample_ids),
                "fit_physical_case_ids": sorted(set(train.physical_case_ids)),
                "loss_initial": fitted.loss_history[0],
                "loss_final": fitted.loss_history[-1],
                "metrics": role_metrics,
            }
        )
    package = Path(__file__).resolve().parent
    dataset = ROOT / "datasets" / "v1.0"
    code_files = ("data.py", "temporal_probe.py", "run_wp4_temporal.py")
    aggregate = _aggregate(seed_records)
    gate = _generalization_gate(config, aggregate)
    artifact_manifest = dataset.parent / "manifests" / "dataset_v1_artifacts.json"
    artifact_record = json.loads(artifact_manifest.read_text(encoding="utf-8"))
    return {
        "task": config["task"],
        "status": (
            "trained_research_prototype_generalization_gate_passed"
            if gate["passed"]
            else "trained_research_prototype_with_failed_generalization_gate"
        ),
        "dataset_version": "1.0",
        "claim_boundary": config["claim_boundary"],
        "fit_evidence": {
            "samples": len(train),
            "physical_groups": len(set(train.physical_case_ids)),
            "stages": sorted(map(int, np.unique(train.stage_index))),
            "golden_used_for_fit": False,
            "time_samples_per_checkpoint": int(train.time_local_s.shape[1]),
            "temperature_probe_count": int(train.temperature_probes_t_K.shape[2]),
            "realized_mass_flow_trajectory_used_for_fit": False,
        },
        "config": config,
        "seeds": seed_records,
        "aggregate": aggregate,
        "generalization_gate": gate,
        "provenance": {
            "config_sha256": _sha256(config_path),
            "code_sha256": {name: _sha256(package / name) for name in code_files},
            "dataset_sha256": {
                name: _sha256(dataset / name)
                for name in ("fields.h5", "operator_samples.csv", "split_manifest.json")
            },
            "dataset_artifact_manifest_sha256": _sha256(artifact_manifest),
            "dataset_artifact_aggregate_sha256": artifact_record["aggregate_sha256"],
            **_git_provenance(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "h5py": h5py.__version__,
                "device": "cpu",
            },
        },
    }


def write_report(report: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    report = run()
    path = write_report(report)
    print(json.dumps({"output": str(path), "aggregate": report["aggregate"]}, indent=2))
