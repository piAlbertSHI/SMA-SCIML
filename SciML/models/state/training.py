"""Leakage-aware fitting and held-out evaluation for the WP3 prototype.

Only the three ``train`` rows of dataset v1.0 may fit parameters or scalers.
The three ``test`` rows are a protocol/duration shift check, not a validation
set for tuning.  Fourteen ``golden_validation`` rows remain audit-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable

import numpy as np

from .corpus import OperatorCorpus, Transition
from .hybrid import Ablation, HybridStateSpaceModel, RidgeResidual, StageControl
from .torch_backend import TORCH_AVAILABLE, TorchResidualMLP, TrainRoleScaler


FIT_ROLE = "train"
EVALUATION_ROLES = ("test",)


@dataclass(frozen=True)
class TrainingAudit:
    estimator: str
    split_scheme: str
    fitted_role: str
    sample_count: int
    physical_group_count: int
    sample_ids: tuple[str, ...]
    physical_case_ids: tuple[str, ...]
    stage_indices: tuple[int, ...]
    unfitted_stage_indices: tuple[int, ...]
    protocol_ids: tuple[str, ...]
    observed_target_values: int
    unobserved_target_values: int
    dataset_manifest_sha256: str
    operator_table_sha256: str
    fields_h5_sha256: str
    dataset_artifact_manifest_sha256: str
    dataset_aggregate_sha256: str
    random_seed: int | None = None


@dataclass(frozen=True)
class HeldOutMetrics:
    role: str
    sample_count: int
    physical_group_count: int
    stage_indices: tuple[int, ...]
    protocol_ids: tuple[str, ...]
    observed_value_count: int
    mae: float
    rmse: float
    p95_absolute_error: float
    max_absolute_error: float
    sma_temperature_mae_k: float
    htf_temperature_mae_k: float
    side_state_mae: float | None
    xi_evaluated: bool
    projection_intervention_count: int
    projection_postcheck_failures: int
    capacity_proxy_targets_scored: int
    capacity_proxy_constraint_failures: int


@dataclass(frozen=True)
class NeuralTrainingResult:
    audit: TrainingAudit
    train_metrics: HeldOutMetrics
    held_out_metrics: HeldOutMetrics
    held_out_heuristic_metrics: HeldOutMetrics
    held_out_mae_improvement_vs_heuristic: float
    final_masked_scaled_mse: float
    epochs: int
    learning_rate: float
    parameter_sha256: str
    checkpoint_path: str | None
    provenance_path: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_identity(corpus: OperatorCorpus) -> dict[str, str]:
    fields_path = corpus.dataset_dir / "fields.h5"
    artifact_manifest_path = (
        corpus.dataset_dir.parent / "manifests" / "dataset_v1_artifacts.json"
    )
    if not fields_path.is_file() or not artifact_manifest_path.is_file():
        raise FileNotFoundError(
            "audited WP3 training requires fields.h5 and dataset_v1_artifacts.json"
        )
    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    aggregate = str(artifact_manifest.get("aggregate_sha256", ""))
    if len(aggregate) != 64 or any(character not in "0123456789abcdef" for character in aggregate):
        raise ValueError("dataset artifact manifest has an invalid aggregate_sha256")
    fields_sha256 = _sha256(fields_path)
    fields_records = [
        item
        for item in artifact_manifest.get("artifacts", [])
        if str(item.get("artifact", "")).replace("\\", "/")
        == "datasets/v1.0/fields.h5"
    ]
    if len(fields_records) != 1 or fields_records[0].get("sha256") != fields_sha256:
        raise ValueError("fields.h5 does not match the dataset artifact manifest")
    return {
        "fields_h5_sha256": fields_sha256,
        "dataset_artifact_manifest_sha256": _sha256(artifact_manifest_path),
        "dataset_aggregate_sha256": aggregate,
    }


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "SciML"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
        dirty = True
    return {
        "git_head": head,
        "git_worktree_dirty_for_sciml": dirty,
        "git_head_semantics": (
            "base HEAD at training time; when dirty, the per-file SHA-256 map is "
            "the authoritative identity of the executed WP3 code"
        ),
    }


def _wp3_source_identity() -> dict[str, Any]:
    state_root = Path(__file__).resolve().parent
    sciml_root = state_root.parents[1]
    paths = [
        state_root / name
        for name in (
            "corpus.py",
            "hybrid.py",
            "layout.py",
            "projection.py",
            "run_wp3.py",
            "torch_backend.py",
            "training.py",
        )
    ]
    config_path = sciml_root / "configs" / "wp3_hybrid.json"
    if not config_path.is_file():
        raise FileNotFoundError("audited WP3 training requires configs/wp3_hybrid.json")
    return {
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "code_sha256": {path.name: _sha256(path) for path in paths},
        **_git_provenance(sciml_root.parent),
    }


def _control(transition: Transition) -> StageControl:
    return StageControl(
        stage_index=transition.stage_index,
        duration_s=transition.duration_s,
        valve_mask=transition.valve_mask,
        side_observed=transition.side_input_observed,
    )


def _transitions(corpus: OperatorCorpus, role: str) -> list[Transition]:
    ids = corpus.split_ids(role)
    if not ids:
        raise ValueError(f"role {role!r} has no operator samples")
    return [corpus.load_transition(sample_id) for sample_id in ids]


def _arrays(
    corpus: OperatorCorpus,
    model: HybridStateSpaceModel,
    role: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Transition]]:
    transitions = _transitions(corpus, role)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for transition in transitions:
        control = _control(transition)
        features.append(model.features(transition.state_in, control))
        targets.append(
            model.training_target(transition.state_in, transition.state_out, control)
        )
        masks.append(transition.target.observed_mask)
    x = np.stack(features)
    y = np.stack(targets)
    mask = np.stack(masks)
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite features in role {role!r}")
    if not np.isfinite(y[mask]).all():
        raise ValueError(f"non-finite observed targets in role {role!r}")
    return x, y, mask, transitions


def _audit(
    corpus: OperatorCorpus,
    transitions: Iterable[Transition],
    mask: np.ndarray,
    estimator: str,
    seed: int | None = None,
) -> TrainingAudit:
    items = tuple(transitions)
    rows = [corpus._row(item.sample_id) for item in items]
    dataset_identity = _dataset_identity(corpus)
    return TrainingAudit(
        estimator=estimator,
        split_scheme="protocol_frequency_holdout",
        fitted_role=FIT_ROLE,
        sample_count=len(items),
        physical_group_count=len({item.physical_case_id for item in items}),
        sample_ids=tuple(item.sample_id for item in items),
        physical_case_ids=tuple(sorted({item.physical_case_id for item in items})),
        stage_indices=tuple(sorted({item.stage_index for item in items})),
        unfitted_stage_indices=tuple(
            stage for stage in (1, 2, 3, 4) if stage not in {item.stage_index for item in items}
        ),
        protocol_ids=tuple(sorted({row["protocol_id"] for row in rows})),
        observed_target_values=int(np.count_nonzero(mask)),
        unobserved_target_values=int(mask.size - np.count_nonzero(mask)),
        dataset_manifest_sha256=_sha256(corpus.dataset_dir / "split_manifest.json"),
        operator_table_sha256=_sha256(corpus.dataset_dir / "operator_samples.csv"),
        **dataset_identity,
        random_seed=seed,
    )


def _masked_error(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    finite = mask & np.isfinite(target) & np.isfinite(prediction)
    count = int(np.count_nonzero(finite))
    if count == 0:
        raise ValueError("metric has no finite observed values")
    error = prediction[finite] - target[finite]
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2))), count


def evaluate_model_on_role(
    corpus: OperatorCorpus,
    model: HybridStateSpaceModel,
    role: str = "test",
) -> HeldOutMetrics:
    """One-step observed-channel metrics; never scores fabricated xi outputs."""
    if role == "golden_validation":
        raise ValueError("golden_validation is audit-only, not a model performance split")
    if role not in ("train", *EVALUATION_ROLES):
        raise ValueError("role must be 'train' or an allowed held-out evaluation role")
    transitions = _transitions(corpus, role)
    predicted: list[np.ndarray] = []
    target: list[np.ndarray] = []
    mask: list[np.ndarray] = []
    interventions = 0
    postcheck_failures = 0
    capacity_targets_scored = 0
    capacity_constraint_failures = 0
    for transition in transitions:
        output, report = model.step(transition.state_in, _control(transition))
        predicted.append(output)
        target.append(transition.state_out)
        mask.append(transition.target.observed_mask)
        interventions += report.intervention_count
        postcheck_failures += int(not report.postcheck_passed)
        if report.capacity_proxy_constraint_satisfied is not None:
            capacity_targets_scored += 1
            capacity_constraint_failures += int(
                not report.capacity_proxy_constraint_satisfied
            )
    predictions = np.stack(predicted)
    targets = np.stack(target)
    observed = np.stack(mask)
    mae, rmse, count = _masked_error(targets, predictions, observed)
    finite = observed & np.isfinite(targets) & np.isfinite(predictions)
    absolute_error = np.abs(predictions[finite] - targets[finite])
    layout = corpus.layout

    def component_mae(component: slice) -> float | None:
        component_mask = observed[:, component]
        if not np.any(component_mask):
            return None
        value, _, _ = _masked_error(
            targets[:, component], predictions[:, component], component_mask
        )
        return value

    rows = [corpus._row(item.sample_id) for item in transitions]
    return HeldOutMetrics(
        role=role,
        sample_count=len(transitions),
        physical_group_count=len({item.physical_case_id for item in transitions}),
        stage_indices=tuple(sorted({item.stage_index for item in transitions})),
        protocol_ids=tuple(sorted({row["protocol_id"] for row in rows})),
        observed_value_count=count,
        mae=mae,
        rmse=rmse,
        p95_absolute_error=float(np.percentile(absolute_error, 95.0)),
        max_absolute_error=float(np.max(absolute_error)),
        sma_temperature_mae_k=float(component_mae(layout.sma_t)),
        htf_temperature_mae_k=float(component_mae(layout.htf_t)),
        side_state_mae=component_mae(layout.side),
        xi_evaluated=bool(np.any(observed[:, layout.xi])),
        projection_intervention_count=interventions,
        projection_postcheck_failures=postcheck_failures,
        capacity_proxy_targets_scored=capacity_targets_scored,
        capacity_proxy_constraint_failures=capacity_constraint_failures,
    )


def fit_ridge_from_corpus(
    corpus: OperatorCorpus,
    *,
    role: str = FIT_ROLE,
    alpha: float = 1.0,
    ablation: Ablation | str = Ablation.HYBRID,
) -> tuple[HybridStateSpaceModel, TrainingAudit]:
    if role != FIT_ROLE:
        raise ValueError("only role='train' may fit; other roles are evaluation-only")
    corpus.assert_group_disjoint()
    template = HybridStateSpaceModel(layout=corpus.layout, ablation=ablation)
    x, y, mask, transitions = _arrays(corpus, template, role)
    residual = RidgeResidual(corpus.layout.size, alpha=alpha).fit(x, y, mask)
    fitted = HybridStateSpaceModel(
        layout=corpus.layout,
        baseline=template.baseline,
        projector=template.projector,
        residual=residual,
        ablation=ablation,
    )
    return fitted, _audit(corpus, transitions, mask, "masked_dual_ridge")


def fit_torch_from_corpus(
    corpus: OperatorCorpus,
    *,
    role: str = FIT_ROLE,
    held_out_role: str = "test",
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    seed: int = 20260819,
    checkpoint_path: str | Path | None = None,
) -> tuple[HybridStateSpaceModel, NeuralTrainingResult]:
    """Fit a deterministic masked PyTorch residual on train rows only.

    This executable path is intentionally small-data and reports the test-role
    protocol shift. It does not claim four-stage or multi-cycle validation.
    """
    if role != FIT_ROLE:
        raise ValueError("only role='train' may fit; other roles are evaluation-only")
    if held_out_role not in EVALUATION_ROLES:
        raise ValueError("held_out_role must be 'test'; golden_validation is audit-only")
    if not TORCH_AVAILABLE:
        raise ModuleNotFoundError("fit_torch_from_corpus requires optional dependency 'torch>=2.2'")
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not hidden_sizes or any(
        isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width <= 0
        for width in hidden_sizes
    ):
        raise ValueError("hidden_sizes must contain positive integers")

    import torch

    corpus.assert_group_disjoint()
    seed = int(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    template = HybridStateSpaceModel(layout=corpus.layout)
    x, y, mask, transitions = _arrays(corpus, template, role)
    scaler = TrainRoleScaler.fit(x, y, mask, role=role)
    network = TorchResidualMLP(
        x.shape[1], corpus.layout.size, hidden_sizes=hidden_sizes, scaler=scaler
    )
    network.train()
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    x_tensor = torch.as_tensor(scaler.transform_features(x), dtype=torch.float32)
    # Replacing unobserved values before tensor construction prevents NaN * 0
    # from poisoning the masked loss.
    scaled_y = scaler.transform_targets(np.where(mask, y, scaler.target_mean))
    y_tensor = torch.as_tensor(scaled_y, dtype=torch.float32)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
    if not bool(mask_tensor.any()):
        raise ValueError("training role has no observed targets")
    final_loss = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = network(x_tensor)
        error = prediction[mask_tensor] - y_tensor[mask_tensor]
        loss = torch.mean(error.square())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite masked neural training loss")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    fitted = HybridStateSpaceModel(
        layout=corpus.layout,
        baseline=template.baseline,
        projector=template.projector,
        residual=network,
    )
    audit = _audit(corpus, transitions, mask, "masked_torch_residual_mlp", seed)
    train_metrics = evaluate_model_on_role(corpus, fitted, "train")
    held_out_metrics = evaluate_model_on_role(corpus, fitted, held_out_role)
    heuristic_reference = HybridStateSpaceModel(
        layout=corpus.layout,
        baseline=template.baseline,
        projector=template.projector,
        ablation=Ablation.HEURISTIC_ONLY,
    )
    held_out_heuristic_metrics = evaluate_model_on_role(
        corpus, heuristic_reference, held_out_role
    )
    held_out_mae_improvement = held_out_heuristic_metrics.mae - held_out_metrics.mae
    checkpoint_text: str | None = None
    provenance_text: str | None = None
    parameter_sha256 = network.parameter_sha256()
    metadata: dict[str, Any] = {
        "claim_boundary": (
            "Fit on three stage-4 samples from one physical run; evaluated on "
            "three stage-4 samples from one held-out physical run."
        ),
        "audit": asdict(audit),
        "train_metrics": asdict(train_metrics),
        "held_out_metrics": asdict(held_out_metrics),
        "held_out_heuristic_metrics": asdict(held_out_heuristic_metrics),
        "held_out_mae_improvement_vs_heuristic": held_out_mae_improvement,
        "scaler": scaler.summary(),
        "architecture": {
            "model_class": "TorchResidualMLP",
            "input_dimension": int(x.shape[1]),
            "output_dimension": int(corpus.layout.size),
            "hidden_sizes": [int(width) for width in hidden_sizes],
            "activation": "SiLU",
            "normalization": "LayerNorm after every hidden activation",
            "residual_output_units": "physical state units after train-only inverse scaling",
        },
        "optimizer": {
            "class": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "final_masked_scaled_mse": final_loss,
        "parameter_sha256": parameter_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "source_identity": _wp3_source_identity(),
        "hyperparameter_provenance": {
            "neural_selection": (
                "Fixed executable prototype settings; test and golden roles are not read "
                "by the fitting loop. No independent preregistration record is claimed."
            ),
            "heuristic_selection": (
                "Hard-coded persistence/exchange reference coefficients are not fitted by "
                "this training path; their historical selection provenance is unavailable, "
                "so only absence of programmatic held-out fitting is claimed."
            ),
        },
    }
    if checkpoint_path is not None:
        destination = Path(checkpoint_path)
        network.save_audited_checkpoint(destination, metadata)
        checkpoint_text = str(destination)
        provenance = destination.with_suffix(destination.suffix + ".provenance.json")
        metadata["checkpoint_sha256"] = _sha256(destination)
        provenance.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        provenance_text = str(provenance)
    result = NeuralTrainingResult(
        audit=audit,
        train_metrics=train_metrics,
        held_out_metrics=held_out_metrics,
        held_out_heuristic_metrics=held_out_heuristic_metrics,
        held_out_mae_improvement_vs_heuristic=held_out_mae_improvement,
        final_masked_scaled_mse=final_loss,
        epochs=epochs,
        learning_rate=learning_rate,
        parameter_sha256=parameter_sha256,
        checkpoint_path=checkpoint_text,
        provenance_path=provenance_text,
    )
    return fitted, result
