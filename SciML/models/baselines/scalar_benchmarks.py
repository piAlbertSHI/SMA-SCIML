"""Reproducible WP2 design-law scalar baseline subtask.

The runner deliberately consumes the published split manifest instead of making
new random splits.  Every preprocessing and target-scaling object is fitted by
``Pipeline.fit`` on the selected train role only.

This module is intentionally narrower than full-cycle WP2.  Dataset v1.0 does
not contain V*, cycle frequency, exchange time, Qcool, Wpump, COP, SCP, or
deltaT, so this runner must not be represented as a full-cycle surrogate.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


SCALAR_SCHEMES = (
    "iid",
    "geometry_group_holdout",
    "material_group_holdout",
    "combined_ood_holdout",
)
EVALUATION_ROLES = ("validation", "test")
REGRESSION_MODELS = (
    "linear_ridge",
    "quadratic_ridge",
    "sparse_rbf",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
    "deep_ensemble_mlp",
)
CLASSIFICATION_MODELS = (
    "linear_logistic",
    "quadratic_logistic",
    "sparse_rbf_logistic",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
    "deep_ensemble_mlp",
)
SUBTASK_NAME = "WP2 design-law scalar baseline subtask"
FULL_CYCLE_MISSING_FIELDS = (
    "V_star",
    "cycle_frequency",
    "heat_exchange_time",
    "Qcool",
    "Wpump",
    "COP",
    "SCP",
    "deltaT",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head(repo_root: Path) -> str | None:
    """Return the exact source revision without failing outside a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _runtime_provenance(
    *,
    repo_root: Path,
    config_path: Path,
    dataset_root: Path,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Capture source, data, software, hardware, and wall-clock provenance."""
    code_path = Path(__file__).resolve()
    data_files = {
        name: dataset_root / name
        for name in ("scalar_cases.csv", "split_manifest.json", "provenance.json")
        if (dataset_root / name).is_file()
    }
    return {
        "git_head": _git_head(repo_root),
        "hashes": {
            "code_sha256": _sha256(code_path),
            "config_sha256": _sha256(config_path),
            "data_sha256": {name: _sha256(path) for name, path in data_files.items()},
        },
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "hardware": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
        },
        "runtime": {
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "elapsed_seconds": float(elapsed_seconds),
            "executable": sys.executable,
        },
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the WP2 design-law scalar subtask configuration."""
    config = _json(Path(path).resolve())
    required = {
        "dataset_version",
        "random_seeds",
        "split_schemes",
        "features",
        "regression",
        "classification",
        "models",
        "uncertainty",
        "output",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"WP2 config is missing keys: {missing}")
    if str(config["dataset_version"]) != "1.0":
        raise ValueError("WP2 currently supports dataset v1.0 only")
    if config.get("scope", {}).get("name") != SUBTASK_NAME:
        raise ValueError(f"scope.name must be {SUBTASK_NAME!r}")
    if config.get("scope", {}).get("full_cycle_wp2_status") != "incomplete":
        raise ValueError("full-cycle WP2 must remain explicitly marked incomplete")
    declared_missing = set(config.get("scope", {}).get("missing_full_cycle_fields", []))
    if not set(FULL_CYCLE_MISSING_FIELDS).issubset(declared_missing):
        raise ValueError("scope is missing required full-cycle data gaps")
    schemes = tuple(config["split_schemes"])
    unknown = sorted(set(schemes) - set(SCALAR_SCHEMES))
    if unknown:
        raise ValueError(f"Non-scalar or unknown split schemes: {unknown}")
    if config.get("require_all_scalar_schemes", True) and set(schemes) != set(SCALAR_SCHEMES):
        raise ValueError(f"Strict WP2 requires exactly these schemes: {SCALAR_SCHEMES}")
    if not config["random_seeds"] or len(set(config["random_seeds"])) != len(config["random_seeds"]):
        raise ValueError("random_seeds must be a non-empty unique list")
    numeric = config["features"].get("numeric", [])
    categorical = config["features"].get("categorical", [])
    if not numeric or set(numeric) & set(categorical):
        raise ValueError("Features need numeric inputs and disjoint numeric/categorical lists")
    if not config["regression"].get("targets"):
        raise ValueError("At least one regression target is required")
    for name in config["models"]["regression"]:
        if name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {name}")
    for name in config["models"]["classification"]:
        if name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {name}")
    if int(config["uncertainty"]["ensemble_members"]) < 2:
        raise ValueError("deep ensemble requires at least two members")
    level = float(config["uncertainty"]["interval_level"])
    if not 0.0 < level < 1.0:
        raise ValueError("interval_level must lie strictly between zero and one")
    bins = int(config.get("metrics", {}).get("ece_bins", 10))
    if bins < 2:
        raise ValueError("metrics.ece_bins must be at least two")
    return config


def load_scalar_assets(dataset_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Load CSV-first so the benchmark does not require a parquet engine."""
    root = Path(dataset_root).resolve()
    csv_path = root / "scalar_cases.csv"
    manifest_path = root / "split_manifest.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("dataset_root must contain scalar_cases.csv and split_manifest.json")
    return pd.read_csv(csv_path), _json(manifest_path), root


def split_frames(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    scheme: str,
) -> dict[str, pd.DataFrame]:
    """Apply one published split and enforce coverage and group non-leakage."""
    if scheme not in SCALAR_SCHEMES:
        raise ValueError(f"{scheme!r} is not a WP2 scalar scheme")
    try:
        spec = manifest["schemes"][scheme]
    except KeyError as exc:
        raise ValueError(f"Split manifest lacks {scheme}") from exc
    if spec.get("task_family") != "scalar_supervised":
        raise ValueError(f"{scheme} is not marked scalar_supervised")
    assignments = spec["assignments"]
    case_ids = set(frame["case_id"].astype(str))
    assigned_ids = set(assignments)
    if case_ids != assigned_ids:
        raise ValueError(
            f"{scheme} assignment coverage mismatch: "
            f"missing={len(case_ids - assigned_ids)}, extra={len(assigned_ids - case_ids)}"
        )
    role_by_parent: dict[str, set[str]] = defaultdict(set)
    parent = frame.set_index("case_id")["physical_case_id"].astype(str).to_dict()
    for case_id, role in assignments.items():
        role_by_parent[parent[case_id]].add(role)
    leaks = sorted(key for key, roles in role_by_parent.items() if len(roles) != 1)
    if leaks:
        raise ValueError(f"physical_case_id leakage in {scheme}: {leaks[:5]}")
    parts: dict[str, pd.DataFrame] = {}
    for role in ("train", *EVALUATION_ROLES):
        selected = {case_id for case_id, assigned in assignments.items() if assigned == role}
        parts[role] = frame[frame["case_id"].isin(selected)].copy().reset_index(drop=True)
        if parts[role].empty:
            raise ValueError(f"{scheme}/{role} is empty")
    return parts


def _one_hot() -> OneHotEncoder:
    # sklearn <1.2 called the dense-output option ``sparse``.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility branch
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(config: Mapping[str, Any]) -> ColumnTransformer:
    """Create an unfitted transformer; it must only be fitted inside train pipelines."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(config["features"]["numeric"])),
            ("categorical", _one_hot(), list(config["features"]["categorical"])),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _model_parameters(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return config.get("hyperparameters", {}).get(name, {})


def make_estimator(task: str, name: str, seed: int, config: Mapping[str, Any]) -> Any:
    """Build one unfitted point model. Ensemble orchestration is handled separately."""
    if name == "deep_ensemble_mlp":
        name = "mlp"
    p = _model_parameters(config, name)
    steps: list[tuple[str, Any]] = [("preprocess", make_preprocessor(config))]
    if name.startswith("quadratic_"):
        steps.append(("quadratic", PolynomialFeatures(degree=2, include_bias=False)))
    if name.startswith("sparse_rbf"):
        steps.append(
            (
                "kernel",
                Nystroem(
                    kernel="rbf",
                    gamma=float(p.get("gamma", 0.5)),
                    n_components=int(p.get("n_components", 256)),
                    random_state=seed,
                ),
            )
        )

    if task == "regression":
        if name in {"linear_ridge", "quadratic_ridge", "sparse_rbf"}:
            model = Ridge(alpha=float(p.get("alpha", 1.0)))
        elif name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=int(p.get("n_estimators", 200)),
                min_samples_leaf=int(p.get("min_samples_leaf", 1)),
                max_features=p.get("max_features", 1.0),
                n_jobs=int(p.get("n_jobs", -1)),
                random_state=seed,
            )
        elif name == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(
                learning_rate=float(p.get("learning_rate", 0.06)),
                max_iter=int(p.get("max_iter", 250)),
                max_leaf_nodes=int(p.get("max_leaf_nodes", 31)),
                l2_regularization=float(p.get("l2_regularization", 1e-4)),
                random_state=seed,
            )
        elif name == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", [128, 64])),
                alpha=float(p.get("alpha", 1e-4)),
                learning_rate_init=float(p.get("learning_rate_init", 1e-3)),
                early_stopping=True,
                validation_fraction=float(p.get("internal_validation_fraction", 0.1)),
                max_iter=int(p.get("max_iter", 500)),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown regression model: {name}")
        steps.append(("model", model))
        # This target scaler is also fit exclusively on the role=train target.
        return TransformedTargetRegressor(
            regressor=Pipeline(steps), transformer=StandardScaler()
        )

    if task != "classification":
        raise ValueError(f"Unknown task: {task}")
    if name in {"linear_logistic", "quadratic_logistic", "sparse_rbf_logistic"}:
        model = LogisticRegression(
            C=float(p.get("C", 1.0)),
            max_iter=int(p.get("max_iter", 1000)),
            random_state=seed,
        )
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            max_features=p.get("max_features", "sqrt"),
            n_jobs=int(p.get("n_jobs", -1)),
            random_state=seed,
        )
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=float(p.get("learning_rate", 0.06)),
            max_iter=int(p.get("max_iter", 250)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)),
            l2_regularization=float(p.get("l2_regularization", 1e-4)),
            random_state=seed,
        )
    elif name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=tuple(p.get("hidden_layer_sizes", [128, 64])),
            alpha=float(p.get("alpha", 1e-4)),
            learning_rate_init=float(p.get("learning_rate_init", 1e-3)),
            early_stopping=True,
            validation_fraction=float(p.get("internal_validation_fraction", 0.1)),
            max_iter=int(p.get("max_iter", 500)),
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classification model: {name}")
    steps.append(("model", model))
    return Pipeline(steps)


def balanced_sample_weights(y_train: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Compute one train-only balanced weighting rule shared by all classifiers."""
    y_train = np.asarray(y_train, dtype=int)
    classes, counts = np.unique(y_train, return_counts=True)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("Balanced binary weights require both classes 0 and 1")
    total = int(y_train.size)
    weights = {
        str(int(label)): float(total / (len(classes) * int(count)))
        for label, count in zip(classes, counts)
    }
    return np.asarray([weights[str(int(label))] for label in y_train]), weights


def _fit_classifier(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
) -> str:
    """Fit with shared train-only weights whenever the final estimator supports them."""
    final_model = estimator.named_steps["model"]
    if "sample_weight" not in inspect.signature(final_model.fit).parameters:
        estimator.fit(x_train, y_train)
        return "unsupported_unweighted"
    estimator.fit(x_train, y_train, model__sample_weight=sample_weight)
    return "train_only_balanced_sample_weight"


def regression_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    train_target: np.ndarray,
) -> dict[str, float | None]:
    """Compute metrics with a train-derived denominator floor for relative error."""
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    train_target = np.asarray(train_target, dtype=float)
    finite_train = np.abs(train_target[np.isfinite(train_target)])
    reference = float(np.median(finite_train)) if finite_train.size else 0.0
    epsilon = max(reference * 1e-6, np.finfo(float).eps)
    relative = np.abs(prediction - y_true) / np.maximum(np.abs(y_true), epsilon)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)) if y_true.size >= 2 else None,
        "mean_relative_error": float(np.mean(relative)),
        "relative_error_denominator_floor": epsilon,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE for diagnostic, explicitly uncalibrated probability scores."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Keep probability==1 in the last bin.
    assignments = np.minimum(np.digitize(probability, edges[1:-1]), n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(probability[mask])) - float(np.mean(y_true[mask]))
        )
    return float(ece)


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Score majority-valid probabilities and minority-invalid detection separately."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-15, 1.0 - 1e-15)
    two_classes = np.unique(y_true).size == 2
    predicted_valid = (probability >= 0.5).astype(int)
    invalid_true = (y_true == 0).astype(int)
    invalid_probability = 1.0 - probability
    predicted_invalid = (predicted_valid == 0).astype(int)
    nll = float(log_loss(y_true, probability, labels=[0, 1]))
    return {
        "auroc": float(roc_auc_score(y_true, probability)) if two_classes else None,
        "auprc_valid_class": (
            float(average_precision_score(y_true, probability)) if two_classes else None
        ),
        "invalid_auprc": (
            float(average_precision_score(invalid_true, invalid_probability))
            if two_classes
            else None
        ),
        "invalid_precision": float(
            precision_score(invalid_true, predicted_invalid, zero_division=0)
        ),
        "invalid_recall": float(recall_score(invalid_true, predicted_invalid, zero_division=0)),
        "invalid_f1": float(f1_score(invalid_true, predicted_invalid, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_valid)),
        "mcc": float(matthews_corrcoef(y_true, predicted_valid)),
        "brier": float(brier_score_loss(y_true, probability)),
        "nll": nll,
        "log_loss": nll,
        "ece": expected_calibration_error(invalid_true, invalid_probability, n_bins=ece_bins),
        "ece_bins": int(ece_bins),
        "probability_status": "diagnostic_uncalibrated_scores",
        "ranking_metric_status": "computed" if two_classes else "not_computable_single_class",
        "positive_prevalence": float(np.mean(y_true)),
        "invalid_prevalence": float(np.mean(invalid_true)),
        "decision_threshold_valid_probability": 0.5,
    }


def _member_seeds(seed: int, count: int) -> list[int]:
    return [seed * 1000 + member for member in range(count)]


def _regression_predictions(
    name: str,
    seed: int,
    config: Mapping[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:
    if name != "deep_ensemble_mlp":
        estimator = make_estimator("regression", name, seed, config)
        estimator.fit(x_train, y_train)
        return np.asarray(estimator.predict(x_eval), dtype=float), None
    predictions = []
    count = int(config["uncertainty"]["ensemble_members"])
    for member_seed in _member_seeds(seed, count):
        estimator = make_estimator("regression", "mlp", member_seed, config)
        estimator.fit(x_train, y_train)
        predictions.append(np.asarray(estimator.predict(x_eval), dtype=float))
    members = np.vstack(predictions)
    return np.mean(members, axis=0), members


def _classification_predictions(
    name: str,
    seed: int,
    config: Mapping[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    sample_weight, _ = balanced_sample_weights(y_train)
    if name != "deep_ensemble_mlp":
        estimator = make_estimator("classification", name, seed, config)
        weighting_status = _fit_classifier(estimator, x_train, y_train, sample_weight)
        return (
            np.asarray(estimator.predict_proba(x_eval)[:, 1], dtype=float),
            None,
            weighting_status,
        )
    probabilities = []
    weighting_statuses = set()
    count = int(config["uncertainty"]["ensemble_members"])
    for member_seed in _member_seeds(seed, count):
        estimator = make_estimator("classification", "mlp", member_seed, config)
        weighting_statuses.add(_fit_classifier(estimator, x_train, y_train, sample_weight))
        probabilities.append(np.asarray(estimator.predict_proba(x_eval)[:, 1], dtype=float))
    members = np.vstack(probabilities)
    weighting_status = (
        next(iter(weighting_statuses))
        if len(weighting_statuses) == 1
        else "mixed_support_across_ensemble_members"
    )
    return np.mean(members, axis=0), members, weighting_status


def _regression_row(
    *,
    y_true: np.ndarray,
    prediction: np.ndarray,
    y_train: np.ndarray,
    members: np.ndarray | None,
    interval_level: float,
) -> dict[str, Any]:
    row = regression_metrics(y_true, prediction, y_train)
    if members is None:
        row.update(
            {
                "interval_coverage": None,
                "mean_interval_width": None,
                "coverage_status": "not_available_point_model",
            }
        )
        return row
    alpha = 1.0 - interval_level
    lower = np.quantile(members, alpha / 2.0, axis=0)
    upper = np.quantile(members, 1.0 - alpha / 2.0, axis=0)
    row.update(
        {
            "interval_coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
            "mean_interval_width": float(np.mean(upper - lower)),
            "coverage_status": "uncalibrated_member_quantile_interval",
        }
    )
    return row


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = ("scheme", "task", "target", "model", "role")
    metrics = (
        "mae",
        "rmse",
        "r2",
        "mean_relative_error",
        "interval_coverage",
        "mean_interval_width",
        "auroc",
        "auprc_valid_class",
        "invalid_auprc",
        "invalid_precision",
        "invalid_recall",
        "invalid_f1",
        "balanced_accuracy",
        "mcc",
        "brier",
        "nll",
        "log_loss",
        "ece",
        "fit_seconds",
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summary = []
    for group_key, group in sorted(groups.items()):
        item = dict(zip(keys, group_key))
        item["n_seeds"] = len({row["seed"] for row in group})
        item["n_eval"] = int(group[0]["n_eval"])
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else None
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(item)
    return summary


def _write_outputs(
    output_dir: Path,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wp2_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(output_dir / "wp2_metrics_long.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "wp2_metrics_summary.csv", index=False)


def _append_progress(path: Path, event: Mapping[str, Any], *, console: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": _utc_now(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if console:
        details = " ".join(f"{key}={value}" for key, value in event.items())
        print(f"[{SUBTASK_NAME}] {details}", flush=True)


def _checkpoint_fingerprint(config_path: Path, root: Path) -> dict[str, str]:
    return {
        "code_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(config_path),
        "scalar_cases_sha256": _sha256(root / "scalar_cases.csv"),
        "split_manifest_sha256": _sha256(root / "split_manifest.json"),
    }


def _write_checkpoint(
    path: Path,
    *,
    fingerprint: Mapping[str, str],
    completed_schemes: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subtask": SUBTASK_NAME,
        "full_cycle_wp2_status": "incomplete",
        "fingerprint": dict(fingerprint),
        "completed_schemes": list(completed_schemes),
        "rows": list(rows),
        "updated_at_utc": _utc_now(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    *,
    fingerprint: Mapping[str, str],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.is_file():
        return [], []
    payload = _json(path)
    if payload.get("fingerprint") != dict(fingerprint):
        raise ValueError("Resume checkpoint fingerprint does not match code/config/data")
    completed = [str(item) for item in payload.get("completed_schemes", [])]
    rows = [dict(item) for item in payload.get("rows", [])]
    return completed, rows


def run_benchmarks(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    resume: bool | None = None,
) -> dict[str, Any]:
    """Run the design-law subtask and write auditable metrics and checkpoints."""
    run_started = time.perf_counter()
    started_at_utc = _utc_now()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    default_root = Path(__file__).resolve().parents[2] / "datasets" / "v1.0"
    frame, manifest, root = load_scalar_assets(dataset_root or default_root)
    if str(manifest.get("dataset_version")) != "1.0":
        raise ValueError("split manifest is not dataset v1.0")
    features = list(config["features"]["numeric"]) + list(config["features"]["categorical"])
    targets = list(config["regression"]["targets"])
    classification_target = str(config["classification"]["target"])
    missing = sorted(set(features + targets + [classification_target]) - set(frame.columns))
    if missing:
        raise ValueError(f"scalar_cases lacks configured columns: {missing}")
    if frame[features].isna().any().any():
        raise ValueError("Configured deployment features contain missing values")

    destination = Path(output_dir or config["output"]["directory"])
    if not destination.is_absolute():
        destination = (Path(__file__).resolve().parents[2] / destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    execution = config.get("execution", {})
    progress_path = destination / execution.get("progress_file", "wp2_progress.jsonl")
    checkpoint_path = destination / execution.get("checkpoint_file", "wp2_checkpoint.json")
    resume_enabled = bool(execution.get("resume", False) if resume is None else resume)
    console_progress = bool(execution.get("console_progress", True))
    checkpoint_each_scheme = bool(execution.get("checkpoint_each_scheme", True))
    fingerprint = _checkpoint_fingerprint(config_path, root)
    if not resume_enabled:
        progress_path.write_text("", encoding="utf-8")
        # A fresh run must never inherit completed schemes from an older attempt.
        checkpoint_path.unlink(missing_ok=True)
    completed_schemes, rows = (
        _load_checkpoint(checkpoint_path, fingerprint=fingerprint)
        if resume_enabled
        else ([], [])
    )
    if len(completed_schemes) != len(set(completed_schemes)):
        raise ValueError("Resume checkpoint repeats a completed scheme")
    unknown_checkpoint_schemes = set(completed_schemes) - set(config["split_schemes"])
    if unknown_checkpoint_schemes:
        raise ValueError(
            f"Resume checkpoint contains unconfigured schemes: {sorted(unknown_checkpoint_schemes)}"
        )
    _append_progress(
        progress_path,
        {
            "event": "run_started",
            "resume": resume_enabled,
            "restored_schemes": completed_schemes,
            "planned_schemes": config["split_schemes"],
        },
        console=console_progress,
    )
    for scheme in config["split_schemes"]:
        if scheme in completed_schemes:
            _append_progress(
                progress_path,
                {"event": "scheme_skipped_from_checkpoint", "scheme": scheme},
                console=console_progress,
            )
            continue
        scheme_started = time.perf_counter()
        _append_progress(
            progress_path,
            {"event": "scheme_started", "scheme": scheme},
            console=console_progress,
        )
        parts = split_frames(frame, manifest, scheme)
        regression_parts = parts
        if config["regression"].get("valid_designs_only", True):
            regression_parts = {
                role: part[part[classification_target].astype(bool)].reset_index(drop=True)
                for role, part in parts.items()
            }
            if any(part.empty for part in regression_parts.values()):
                raise ValueError(f"{scheme} has an empty valid-only regression role")
        for seed_value in config["random_seeds"]:
            seed = int(seed_value)
            for target in targets:
                x_train = regression_parts["train"][features]
                y_train = regression_parts["train"][target].to_numpy(dtype=float)
                if not np.isfinite(y_train).all():
                    raise ValueError(f"Non-finite training target {target} in {scheme}")
                for model_name in config["models"]["regression"]:
                    start = time.perf_counter()
                    eval_sizes = [len(regression_parts[role]) for role in EVALUATION_ROLES]
                    x_eval = pd.concat(
                        [regression_parts[role][features] for role in EVALUATION_ROLES],
                        ignore_index=True,
                    )
                    prediction_all, members_all = _regression_predictions(
                        model_name, seed, config, x_train, y_train, x_eval
                    )
                    elapsed = time.perf_counter() - start
                    _append_progress(
                        progress_path,
                        {
                            "event": "regressor_fit_completed",
                            "scheme": scheme,
                            "seed": seed,
                            "target": target,
                            "model": model_name,
                            "fit_seconds": elapsed,
                        },
                        console=console_progress,
                    )
                    offset = 0
                    for role, size in zip(EVALUATION_ROLES, eval_sizes):
                        prediction = prediction_all[offset : offset + size]
                        members = (
                            members_all[:, offset : offset + size]
                            if members_all is not None
                            else None
                        )
                        offset += size
                        y_true = regression_parts[role][target].to_numpy(dtype=float)
                        row = {
                            "scheme": scheme,
                            "task": "regression",
                            "target": target,
                            "model": model_name,
                            "seed": seed,
                            "role": role,
                            "n_train": len(x_train),
                            "n_eval": len(y_true),
                            "fit_seconds": elapsed,
                        }
                        row.update(
                            _regression_row(
                                y_true=y_true,
                                prediction=prediction,
                                y_train=y_train,
                                members=members,
                                interval_level=float(config["uncertainty"]["interval_level"]),
                            )
                        )
                        rows.append(row)

            x_train = parts["train"][features]
            y_train = parts["train"][classification_target].astype(int).to_numpy()
            if np.unique(y_train).size != 2:
                raise ValueError(f"{scheme} classification train role has only one class")
            for model_name in config["models"]["classification"]:
                start = time.perf_counter()
                eval_sizes = [len(parts[role]) for role in EVALUATION_ROLES]
                x_eval = pd.concat(
                    [parts[role][features] for role in EVALUATION_ROLES], ignore_index=True
                )
                probability_all, members_all, weighting_status = _classification_predictions(
                    model_name, seed, config, x_train, y_train, x_eval
                )
                elapsed = time.perf_counter() - start
                _, class_weights = balanced_sample_weights(y_train)
                _append_progress(
                    progress_path,
                    {
                        "event": "classifier_fit_completed",
                        "scheme": scheme,
                        "seed": seed,
                        "model": model_name,
                        "fit_seconds": elapsed,
                        "weighting_status": weighting_status,
                    },
                    console=console_progress,
                )
                offset = 0
                for role, size in zip(EVALUATION_ROLES, eval_sizes):
                    probability = probability_all[offset : offset + size]
                    members = (
                        members_all[:, offset : offset + size]
                        if members_all is not None
                        else None
                    )
                    offset += size
                    y_true = parts[role][classification_target].astype(int).to_numpy()
                    row = {
                        "scheme": scheme,
                        "task": "classification",
                        "target": classification_target,
                        "model": model_name,
                        "seed": seed,
                        "role": role,
                        "n_train": len(x_train),
                        "n_eval": len(y_true),
                        "fit_seconds": elapsed,
                        "class_weighting_status": weighting_status,
                        "train_class_weight_invalid": class_weights["0"],
                        "train_class_weight_valid": class_weights["1"],
                        "interval_coverage": None,
                        "mean_interval_width": None,
                        "coverage_status": "not_applicable_classification",
                        "probability_ensemble_std_mean": (
                            float(np.mean(np.std(members, axis=0))) if members is not None else None
                        ),
                        "probability_ensemble_status": (
                            "uncalibrated_member_spread_diagnostic"
                            if members is not None
                            else "not_available_point_model"
                        ),
                    }
                    row.update(
                        classification_metrics(
                            y_true,
                            probability,
                            ece_bins=int(config.get("metrics", {}).get("ece_bins", 10)),
                        )
                    )
                    rows.append(row)

        completed_schemes.append(scheme)
        _append_progress(
            progress_path,
            {
                "event": "scheme_completed",
                "scheme": scheme,
                "scheme_seconds": time.perf_counter() - scheme_started,
                "metric_rows_total": len(rows),
            },
            console=console_progress,
        )
        if checkpoint_each_scheme:
            _write_checkpoint(
                checkpoint_path,
                fingerprint=fingerprint,
                completed_schemes=completed_schemes,
                rows=rows,
            )

    summary = _summary_rows(rows)
    finished_at_utc = _utc_now()
    elapsed_seconds = time.perf_counter() - run_started
    repo_root = Path(__file__).resolve().parents[3]
    payload: dict[str, Any] = {
        "benchmark": SUBTASK_NAME,
        "status": SUBTASK_NAME,
        "subtask_completion_status": "completed",
        "full_cycle_wp2_status": "incomplete",
        "full_cycle_wp2_missing_fields": list(FULL_CYCLE_MISSING_FIELDS),
        "full_cycle_wp2_scope_reason": (
            "V*, frequency, exchange time, Qcool, Wpump, COP, SCP, and deltaT "
            "are absent from dataset v1.0"
        ),
        "dataset_version": "1.0",
        "dataset_root": str(root),
        "provenance": _runtime_provenance(
            repo_root=repo_root,
            config_path=config_path,
            dataset_root=root,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            elapsed_seconds=elapsed_seconds,
        ),
        "methodology": {
            "split_source": "published dataset v1.0 split_manifest.json",
            "preprocessing_fit_scope": "train role only",
            "regression_population": "design_valid=true only",
            "classification_population": "all scalar cases",
            "classification_weighting": "one explicit balanced sample-weight vector computed from each scheme's train role; passed unchanged to every classifier whose fit API supports sample_weight; support status is recorded per result row",
            "minority_class": "design_valid=false (invalid design); invalid-class AUPRC, precision, recall, and F1 are primary reliability diagnostics",
            "probability_metrics": "Brier, NLL/log-loss, and equal-width ECE are diagnostics on uncalibrated scores; they do not establish probability calibration",
            "relative_error": "absolute error divided by max(abs(y), 1e-6 * train median abs target)",
            "ensemble_interval": "uncalibrated central member-quantile interval; not a calibrated guarantee",
            "sparse_kernel_baseline": "Nystroem RBF features followed by Ridge/LogisticRegression; this is a CPU-scalable sparse kernel approximation, not an exact GP posterior",
            "unavailable_inputs": config["features"].get("known_unavailable", []),
        },
        "config": config,
        "results": rows,
        "summary": summary,
    }
    _write_outputs(destination, payload, rows, summary)
    _append_progress(
        progress_path,
        {
            "event": "run_completed",
            "elapsed_seconds": elapsed_seconds,
            "metric_rows_total": len(rows),
        },
        console=console_progress,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[2] / "configs" / "wp2_scalar.json"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="resume completed split schemes from a fingerprint-checked checkpoint",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmarks(
        args.config, args.dataset_root, args.output_dir, resume=args.resume
    )
    print(
        f"{SUBTASK_NAME} complete: {len(payload['results'])} metric rows; "
        "full-cycle WP2 remains incomplete because required cycle/control/performance "
        f"fields are absent; outputs written for dataset {payload['dataset_version']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
