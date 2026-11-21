"""Leakage-audited full-cycle scalar proxy for dataset v1.1 artifacts.

The implementation is deliberately gate-aware: it can fit a deterministic
ridge baseline when enough physics-valid, pump-complete cases exist, but it
refuses to report paper-grade metrics on a pilot corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "radius_m",
    "thickness_m",
    "pressure_Pa",
    "flush_volume_target_ratio",
    "exchange_duration_s",
    "cycle_index",
)
TARGETS = ("Qcool_J", "Wpump_J", "COP", "SCP_W_per_kg", "deltaT_hot_cold_max_K")
MIN_ENGINEERING_CYCLES = 20
MIN_PAPER_CYCLES = 200
MIN_ENGINEERING_CONDITIONS = 20
MIN_PAPER_CONDITIONS = 200
PREREGISTERED_RIDGE_ALPHA = 10.0


@dataclass(frozen=True)
class CycleProxyResult:
    status: str
    paper_grade_ready: bool
    valid_case_count: int
    valid_cycle_count: int
    unique_cycle_count: int
    physical_condition_count: int
    train_case_count: int
    test_case_count: int
    evaluation_scheme: str
    target_metrics: dict[str, dict[str, float]]
    condition_train_mean_baseline_metrics: dict[str, dict[str, float]]
    condition_skill_vs_train_mean: dict[str, float]
    reliability_status: str
    reliability_class_counts: dict[str, int]
    reliability_evaluation_scheme: str
    reliability_metrics: dict[str, float | int]
    structured_group_ood: dict[str, dict[str, Any]]
    blocking_reasons: tuple[str, ...]


def load_index(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    stem = "cycle_index" if (root / "cycle_index.csv").exists() else "case_index"
    try:
        frame = pd.read_parquet(root / f"{stem}.parquet")
    except (ImportError, OSError):
        frame = pd.read_csv(root / f"{stem}.csv")
    required = set(FEATURES) | set(TARGETS) | {
        "physics_ok",
        "pump_work_available",
        "case_id",
        "condition_id",
        "material_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v1.1 case index lacks required columns: {missing}")
    return frame


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
    }


def _nonpositive_structured_skills(report: dict[str, Any]) -> list[str]:
    """Return every pooled or per-group target that fails positive skill."""

    failures: list[str] = []
    pooled = report.get("pooled", {}).get("skill_vs_train_mean_baseline", {})
    for target, skill in pooled.items():
        if float(skill) <= 0.0:
            failures.append(f"pooled:{target}={float(skill):.6g}")
    for fold in report.get("folds", []):
        group = str(fold.get("held_out_group", "unknown"))
        for target, skill in fold.get("skill_vs_train_mean_baseline", {}).items():
            if float(skill) <= 0.0:
                failures.append(f"{group}:{target}={float(skill):.6g}")
    return failures


def _reliability_baseline(
    frame: pd.DataFrame,
) -> tuple[str, str, dict[str, int], dict[str, float | int]]:
    labels = frame["physics_ok"].astype(bool).astype(int).to_numpy()
    counts = {
        "physics_valid": int(labels.sum()),
        "physics_invalid": int((1 - labels).sum()),
    }
    if not all(counts.values()):
        return "unavailable_single_class_corpus", "none", counts, {}
    group_class_counts = frame.assign(_label=labels).groupby("_label")["condition_id"].nunique()
    if len(group_class_counts) != 2 or int(group_class_counts.min()) < 2:
        return "unavailable_insufficient_class_groups", "none", counts, {}
    numeric_names = [name for name in FEATURES if name in frame.columns]
    numeric = frame[numeric_names].to_numpy(float)
    material = pd.get_dummies(frame["material_id"], prefix="material", dtype=float)
    design = np.column_stack((numeric, material.to_numpy(float)))
    groups = frame["condition_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=20260820)
    probability = np.full(len(frame), np.nan)
    for train_index, test_index in splitter.split(design, labels, groups):
        if len(np.unique(labels[train_index])) != 2:
            return "unavailable_fold_single_class", "none", counts, {}
        scaler = StandardScaler().fit(design[train_index])
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=20260820,
        ).fit(scaler.transform(design[train_index]), labels[train_index])
        probability[test_index] = model.predict_proba(scaler.transform(design[test_index]))[:, 1]
    if not np.isfinite(probability).all():
        raise RuntimeError("Reliability cross-validation left samples without predictions")
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    metrics: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "recall_valid": float(recall),
        "recall_invalid": float(specificity),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return "trained_group_stratified_baseline", "two_fold_stratified_group", counts, metrics


def _structured_group_ood(
    frame: pd.DataFrame,
    design: np.ndarray,
    group_values: np.ndarray,
    group_name: str,
) -> dict[str, Any]:
    groups = sorted({str(value) for value in group_values})
    required_group_count = 2 if group_name == "material" else 3
    if len(groups) < required_group_count:
        return {
            "status": f"unavailable_fewer_than_{required_group_count}_{group_name}_groups",
            "observed_groups": groups,
            "fold_count": 0,
        }

    folds = []
    pooled_indices = []
    pooled_predictions = {target: [] for target in TARGETS}
    pooled_baselines = {target: [] for target in TARGETS}
    values = np.asarray([str(value) for value in group_values])
    for group in groups:
        test_index = np.flatnonzero(values == group)
        train_index = np.flatnonzero(values != group)
        train_conditions = frame.iloc[train_index]["condition_id"].nunique()
        test_conditions = frame.iloc[test_index]["condition_id"].nunique()
        if train_conditions < 3 or test_conditions < 2:
            continue
        scaler = StandardScaler().fit(design[train_index])
        x_train = scaler.transform(design[train_index])
        x_test = scaler.transform(design[test_index])
        fold_metrics = {}
        fold_baselines = {}
        fold_skills = {}
        for target in TARGETS:
            y = frame[target].to_numpy(float)
            prediction = Ridge(alpha=PREREGISTERED_RIDGE_ALPHA).fit(
                x_train, y[train_index]
            ).predict(x_test)
            baseline = np.full(len(test_index), float(np.mean(y[train_index])))
            model_metrics = _metrics(y[test_index], prediction)
            baseline_metrics = _metrics(y[test_index], baseline)
            fold_metrics[target] = model_metrics
            fold_baselines[target] = baseline_metrics
            denominator = baseline_metrics["rmse"]
            fold_skills[target] = (
                float(1.0 - model_metrics["rmse"] / denominator)
                if denominator > 0.0
                else 0.0
            )
            pooled_predictions[target].append(prediction)
            pooled_baselines[target].append(baseline)
        pooled_indices.append(test_index)
        folds.append(
            {
                "held_out_group": group,
                "train_condition_count": int(train_conditions),
                "test_condition_count": int(test_conditions),
                "test_cycle_count": int(len(test_index)),
                "target_metrics": fold_metrics,
                "train_mean_baseline_metrics": fold_baselines,
                "skill_vs_train_mean_baseline": fold_skills,
            }
        )

    if not folds:
        return {
            "status": "unavailable_insufficient_conditions_per_group",
            "observed_groups": groups,
            "minimum_train_conditions_per_fold": 3,
            "minimum_test_conditions_per_fold": 2,
            "fold_count": 0,
        }

    indices = np.concatenate(pooled_indices)
    pooled_metrics = {}
    pooled_baseline_metrics = {}
    pooled_skill = {}
    for target in TARGETS:
        truth = frame[target].to_numpy(float)[indices]
        prediction = np.concatenate(pooled_predictions[target])
        baseline = np.concatenate(pooled_baselines[target])
        model_metrics = _metrics(truth, prediction)
        baseline_metrics = _metrics(truth, baseline)
        pooled_metrics[target] = model_metrics
        pooled_baseline_metrics[target] = baseline_metrics
        pooled_skill[target] = (
            float(1.0 - model_metrics["rmse"] / baseline_metrics["rmse"])
            if baseline_metrics["rmse"] > 0.0
            else 0.0
        )
    evaluated_groups = [str(fold["held_out_group"]) for fold in folds]
    skipped_groups = [group for group in groups if group not in evaluated_groups]
    return {
        "status": "evaluated_all_groups" if not skipped_groups else "partial_evaluation",
        "split_semantics": f"Leave-one-{group_name}-group-out refit; no held-out group enters training.",
        "observed_groups": groups,
        "evaluated_groups": evaluated_groups,
        "skipped_groups": skipped_groups,
        "fold_count": len(folds),
        "folds": folds,
        "pooled": {
            "target_metrics": pooled_metrics,
            "train_mean_baseline_metrics": pooled_baseline_metrics,
            "skill_vs_train_mean_baseline": pooled_skill,
            "cycle_count": int(len(indices)),
        },
    }


def fit_cycle_proxy(dataset_root: str | Path, *, seed: int = 20260820) -> CycleProxyResult:
    frame = load_index(dataset_root)
    valid_case_count = int(frame.loc[frame["physics_ok"].astype(bool), "case_id"].nunique())
    reliability_status, reliability_scheme, class_counts, reliability_metrics = (
        _reliability_baseline(frame)
    )
    valid = frame[frame["physics_ok"].astype(bool) & frame["pump_work_available"].astype(bool)].copy()
    valid = valid.dropna(subset=list(FEATURES) + list(TARGETS)).reset_index(drop=True)
    valid_cycle_count = len(valid)
    # Deterministic simulations may repeat cycle 1 across 1-cycle and 2-cycle
    # runs. Keep one such row for fitting and retain the raw count in the report.
    dedupe_columns = ["condition_id", *FEATURES, *TARGETS]
    unique = valid.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    condition_count = int(unique["condition_id"].nunique())
    blockers: list[str] = []
    if valid_cycle_count < MIN_ENGINEERING_CYCLES:
        blockers.append(
            f"only {valid_cycle_count} valid observed cycles; engineering minimum is {MIN_ENGINEERING_CYCLES}"
        )
    if valid_cycle_count < MIN_PAPER_CYCLES:
        blockers.append(f"paper-grade minimum is {MIN_PAPER_CYCLES} valid observed cycles")
    if condition_count < MIN_ENGINEERING_CONDITIONS:
        blockers.append(
            f"only {condition_count} independent physical conditions; "
            f"engineering minimum is {MIN_ENGINEERING_CONDITIONS}"
        )
    if condition_count < MIN_PAPER_CONDITIONS:
        blockers.append(
            f"paper-grade minimum is {MIN_PAPER_CONDITIONS} independent physical conditions"
        )
    if condition_count < 3:
        blockers.append("fewer than three independent physical conditions")
    if reliability_status != "trained_group_stratified_baseline":
        blockers.append("full-cycle reliability classifier lacks a failed class")
    if len(unique) < 2 or condition_count < 2:
        return CycleProxyResult(
            "insufficient_data",
            False,
            valid_case_count,
            valid_cycle_count,
            len(unique),
            condition_count,
            0,
            0,
            "none",
            {},
            {},
            {},
            reliability_status,
            class_counts,
            reliability_scheme,
            reliability_metrics,
            {},
            tuple(blockers),
        )

    numeric = unique[list(FEATURES)].to_numpy(float)
    materials = pd.get_dummies(unique["material_id"], prefix="material", dtype=float)
    design = np.column_stack([numeric, materials.to_numpy(float)])
    geometry_groups = np.asarray(
        [f"R{radius:.12g}_t{thickness:.12g}" for radius, thickness in unique[["radius_m", "thickness_m"]].to_numpy(float)]
    )
    structured_group_ood = {
        "material_group_holdout": _structured_group_ood(
            unique, design, unique["material_id"].astype(str).to_numpy(), "material"
        ),
        "geometry_group_holdout": _structured_group_ood(
            unique, design, geometry_groups, "geometry"
        ),
    }
    for split_name, split_report in structured_group_ood.items():
        status = str(split_report["status"])
        if status != "evaluated_all_groups":
            blockers.append(
                f"{split_name} is {status}; skipped groups are "
                f"{split_report.get('skipped_groups', split_report.get('observed_groups', []))}"
            )
        elif int(split_report["pooled"]["cycle_count"]) < 40:
            blockers.append(
                f"{split_name} has only {split_report['pooled']['cycle_count']} "
                "pooled held-out cycles"
            )
        else:
            failures = _nonpositive_structured_skills(split_report)
            if failures:
                blockers.append(
                    f"{split_name} lacks positive skill in every held-out group/target: "
                    + ", ".join(failures)
                )
    groups = unique["condition_id"].to_numpy()
    splitter = LeaveOneGroupOut()
    predictions = {target: np.full(len(unique), np.nan) for target in TARGETS}
    baseline_predictions = {target: np.full(len(unique), np.nan) for target in TARGETS}
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    # Seed is retained in the public API/provenance even though LOGO itself is
    # deterministic; it prevents a later stochastic estimator from silently
    # changing the experiment identity.
    _ = seed
    for train_idx, test_idx in splitter.split(design, groups=groups):
        scaler = StandardScaler().fit(design[train_idx])
        x_train = scaler.transform(design[train_idx])
        x_test = scaler.transform(design[test_idx])
        train_sizes.append(len(train_idx))
        test_sizes.append(len(test_idx))
        for target in TARGETS:
            y = unique[target].to_numpy(float)
            model = Ridge(alpha=PREREGISTERED_RIDGE_ALPHA).fit(
                x_train, y[train_idx]
            )
            predictions[target][test_idx] = model.predict(x_test)
            baseline_predictions[target][test_idx] = float(np.mean(y[train_idx]))
    metrics: dict[str, dict[str, float]] = {}
    baseline_metrics: dict[str, dict[str, float]] = {}
    condition_skill: dict[str, float] = {}
    for target in TARGETS:
        y_true = unique[target].to_numpy(float)
        if not np.isfinite(predictions[target]).all():
            raise RuntimeError(f"LOGO did not produce an out-of-fold prediction for {target}")
        metrics[target] = _metrics(y_true, predictions[target])
        baseline_metrics[target] = _metrics(y_true, baseline_predictions[target])
        denominator = baseline_metrics[target]["rmse"]
        condition_skill[target] = (
            float(1.0 - metrics[target]["rmse"] / denominator)
            if denominator > 0.0
            else 0.0
        )
    if any(skill <= 0.0 for skill in condition_skill.values()):
        blockers.append(
            "leave-one-condition-out proxy does not have positive skill over the train-mean baseline for every target"
        )
    structured_ready = all(
        report.get("status") == "evaluated_all_groups"
        and int(report.get("pooled", {}).get("cycle_count", 0)) >= 40
        and not _nonpositive_structured_skills(report)
        for report in structured_group_ood.values()
    )
    paper_ready = (
        valid_cycle_count >= MIN_PAPER_CYCLES
        and condition_count >= MIN_PAPER_CONDITIONS
        and reliability_status == "trained_group_stratified_baseline"
        and structured_ready
        and all(skill > 0.0 for skill in condition_skill.values())
    )
    return CycleProxyResult(
        "trained_pilot_baseline" if blockers else "trained_gate_eligible_baseline",
        paper_ready,
        valid_case_count,
        valid_cycle_count,
        len(unique),
        condition_count,
        max(train_sizes),
        max(test_sizes),
        "leave_one_physical_condition_out",
        metrics,
        baseline_metrics,
        condition_skill,
        reliability_status,
        class_counts,
        reliability_scheme,
        reliability_metrics,
        structured_group_ood,
        tuple(blockers),
    )
