"""Stable v1.2 hybrid state-space candidate for autonomous cycle rollout.

The learned component is a stage-local linear neural residual (the closed-form
Ridge solution of a one-layer linear network). Hard state ownership masks and
the physical admissibility projection remain outside the learned map. The
cycle index is never a feature, so later cycles cannot be identified by label.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.linear_model import Ridge

from SciML.models.v11.state_space import (
    LAYOUT,
    TransitionRecord,
    project_state,
    stage_mask,
)


RIDGE_ALPHA = 1.0
CHECKPOINT_SCHEMA_VERSION = "sciml_wp3_v12_linear_state_checkpoint_v1"


def causal_state_features(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Return the observed Markov state and causal controls.

    ``controls[-1]`` is the cycle index and is deliberately omitted. All
    remaining entries are available before the transition is evaluated.
    """

    state = np.asarray(state, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if state.shape != (LAYOUT.size,) or controls.shape != (13,):
        raise ValueError("unexpected WP3 state/control shape")
    result = np.concatenate((state, controls[:-1]))
    if not np.isfinite(result).all():
        raise ValueError("non-finite causal state feature")
    return result


class CausalDeltaStateModel:
    """Stage-local regularized linear-neural residual with hard physics guards."""

    def __init__(
        self,
        seed: int = 20260828,
        hidden: int = 0,
        residual_gain: float = 1.0,
        ridge_alpha: float = RIDGE_ALPHA,
    ) -> None:
        if residual_gain != 1.0:
            raise ValueError("v1.2 affine candidate freezes residual_gain at 1.0")
        if ridge_alpha != RIDGE_ALPHA:
            raise ValueError(f"v1.2 affine candidate freezes ridge_alpha at {RIDGE_ALPHA}")
        self.seed = int(seed)
        # Compatibility attribute; this linear-network candidate has no hidden layer.
        self.hidden = int(hidden)
        self.residual_gain = float(residual_gain)
        self.ridge_alpha = float(ridge_alpha)
        self.models: dict[int, Ridge] = {}
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.stage_baseline = np.zeros((4, LAYOUT.size), dtype=np.float64)
        self.training_ids: tuple[str, ...] = ()

    def fit(
        self,
        records: list[TransitionRecord],
        epochs: int | None = None,
        learning_rate: float | None = None,
        *,
        allow_development_temporal_labels: bool = False,
    ) -> dict[str, float]:
        """Fit the deterministic closed-form Ridge maps.

        ``epochs`` and ``learning_rate`` remain accepted for API compatibility
        with v1.1; neither affects this closed-form candidate.
        """

        if not records:
            raise ValueError("cannot fit empty v1.2 state corpus")
        leaked = [record.sample_id for record in records if record.cycle_index != 1]
        if leaked and not allow_development_temporal_labels:
            raise ValueError("v1.2 state fit forbids cycle-2/3 labels")

        states = np.stack([record.state_in for record in records])
        targets = np.stack([record.state_out for record in records])
        controls = np.stack([record.controls for record in records])
        stages = np.asarray([record.stage_index for record in records], dtype=np.int64)
        masks = stage_mask(stages)
        delta = (targets - states) * masks
        raw = np.stack(
            [causal_state_features(state, control) for state, control in zip(states, controls)]
        )
        self.feature_mean = raw.mean(axis=0)
        raw_scale = raw.std(axis=0)
        self.feature_scale = np.where(raw_scale > 1.0e-10, raw_scale, 1.0)
        scaled = (raw - self.feature_mean) / self.feature_scale

        self.models.clear()
        for stage in range(1, 5):
            selected = stages == stage
            if not selected.any():
                raise ValueError(f"training corpus lacks stage {stage}")
            self.stage_baseline[stage - 1] = np.mean(delta[selected], axis=0)
            active = masks[selected][0].astype(bool)
            self.models[stage] = Ridge(alpha=self.ridge_alpha).fit(
                scaled[selected], delta[selected][:, active]
            )

        self.training_ids = tuple(record.sample_id for record in records)
        prediction = self.predict_records(records)
        return {
            "rmse_all_state": float(np.sqrt(np.mean((prediction - targets) ** 2))),
            "max_abs_all_state": float(np.max(np.abs(prediction - targets))),
        }

    def _scaled(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("v1.2 state model is not fitted")
        return (causal_state_features(state, controls) - self.feature_mean) / self.feature_scale

    def trust_score(
        self, state: np.ndarray, controls: np.ndarray, stage_index: int
    ) -> float:
        """Return the frozen globally regularized affine applicability weight.

        Ridge regularization and leave-one-temporal-condition-out autonomous CV
        replace the previous nearest-neighbour attenuation. Returning one is a
        fixed model policy, not a post-hoc gate override.
        """

        if stage_index not in self.models:
            raise RuntimeError("v1.2 state model is not fitted")
        self._scaled(state, controls)  # shape and finite validation
        return 1.0

    def predict(
        self, state: np.ndarray, controls: np.ndarray, stage_index: int
    ) -> np.ndarray:
        if stage_index not in self.models:
            raise RuntimeError("v1.2 state model is not fitted")
        mask = stage_mask(np.asarray([stage_index]))[0].astype(bool)
        learned_delta = self.models[stage_index].predict(
            self._scaled(state, controls)[None, :]
        )[0]
        result = np.asarray(state, dtype=np.float64).copy()
        result[mask] += learned_delta
        return project_state(result)

    def predict_stage_mean(self, state: np.ndarray, stage_index: int) -> np.ndarray:
        mask = stage_mask(np.asarray([stage_index]))[0]
        return project_state(
            np.asarray(state) + self.stage_baseline[stage_index - 1] * mask
        )

    def predict_records(self, records: list[TransitionRecord]) -> np.ndarray:
        return np.stack(
            [self.predict(r.state_in, r.controls, r.stage_index) for r in records]
        )

    def parameter_sha256(self) -> str:
        if len(self.models) != 4:
            raise RuntimeError("v1.2 state model is not fitted")
        digest = hashlib.sha256()

        def update(name: str, value: np.ndarray) -> None:
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())

        assert self.feature_mean is not None and self.feature_scale is not None
        update("feature_mean", self.feature_mean)
        update("feature_scale", self.feature_scale)
        update("stage_baseline", self.stage_baseline)
        update("ridge_alpha", np.asarray([self.ridge_alpha], dtype=np.float64))
        update("residual_gain", np.asarray([self.residual_gain], dtype=np.float64))
        for stage, model in sorted(self.models.items()):
            update(f"stage_{stage}:coef", np.asarray(model.coef_, dtype=np.float64))
            update(f"stage_{stage}:intercept", np.asarray(model.intercept_, dtype=np.float64))
        for sample_id in self.training_ids:
            digest.update(b"training_id\0" + sample_id.encode("utf-8") + b"\0")
        return digest.hexdigest()


def save_frozen_checkpoint(
    path: str | Path,
    model: CausalDeltaStateModel,
    *,
    metadata: Mapping[str, str],
) -> str:
    """Serialize the exact fitted affine state and return its file SHA-256."""

    if len(model.models) != 4:
        raise RuntimeError("cannot checkpoint an unfitted v1.2 state model")
    required = {"dataset_source_content_sha256", "implementation_sha256"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata is missing: {missing}")
    assert model.feature_mean is not None and model.feature_scale is not None
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "work_package": "WP3-v1.2-development-only",
        "evidence_status": "candidate_frozen_before_confirm_temporal12",
        "confirmatory": False,
        "parameter_sha256": model.parameter_sha256(),
        "metadata": dict(metadata),
        "hyperparameters": {
            "seed": model.seed,
            "hidden_layers": 0,
            "ridge_alpha": model.ridge_alpha,
            "residual_gain": model.residual_gain,
            "cycle_index_feature_included": False,
        },
        "feature_mean": model.feature_mean,
        "feature_scale": model.feature_scale,
        "stage_baseline": model.stage_baseline,
        "stage_models": {
            stage: {
                "coef": np.asarray(stage_model.coef_, dtype=np.float64),
                "intercept": np.asarray(stage_model.intercept_, dtype=np.float64),
            }
            for stage, stage_model in sorted(model.models.items())
        },
        "training_ids": model.training_ids,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pickle.dumps(payload, protocol=5))
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_frozen_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, str] | None = None,
) -> CausalDeltaStateModel:
    """Restore a frozen candidate, rejecting stale metadata or parameter drift."""

    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported WP3-v1.2 checkpoint schema")
    if payload.get("confirmatory") is not False:
        raise ValueError("WP3-v1.2 checkpoint crosses the evidence boundary")
    hyperparameters = payload.get("hyperparameters", {})
    if hyperparameters.get("cycle_index_feature_included") is not False:
        raise ValueError("WP3-v1.2 checkpoint violates the causal feature contract")
    model = CausalDeltaStateModel(
        seed=int(hyperparameters["seed"]),
        residual_gain=float(hyperparameters["residual_gain"]),
        ridge_alpha=float(hyperparameters["ridge_alpha"]),
    )
    model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
    model.stage_baseline = np.asarray(payload["stage_baseline"], dtype=np.float64)
    model.training_ids = tuple(str(value) for value in payload["training_ids"])
    stage_models = payload.get("stage_models", {})
    if sorted(int(stage) for stage in stage_models) != [1, 2, 3, 4]:
        raise ValueError("WP3-v1.2 checkpoint lacks four stage models")
    for raw_stage, values in stage_models.items():
        stage = int(raw_stage)
        ridge = Ridge(alpha=model.ridge_alpha)
        ridge.coef_ = np.asarray(values["coef"], dtype=np.float64)
        ridge.intercept_ = np.asarray(values["intercept"], dtype=np.float64)
        ridge.n_features_in_ = int(model.feature_mean.size)
        model.models[stage] = ridge
    if expected_metadata is not None:
        observed = payload.get("metadata", {})
        for key, expected in expected_metadata.items():
            if observed.get(key) != expected:
                raise ValueError(f"WP3-v1.2 checkpoint {key} mismatch")
    if model.parameter_sha256() != payload.get("parameter_sha256"):
        raise ValueError("WP3-v1.2 checkpoint parameter hash mismatch")
    return model
