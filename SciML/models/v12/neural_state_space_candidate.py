"""Causal, physics-constrained nonlinear WP3 candidate (development only).

A separately identifiable Ridge baseline is corrected by four trainable
two-hidden-layer SiLU networks. Physics enters through stage ownership,
Markov state propagation and admissibility constraints, not through a claim
that the empirical affine baseline solves a reduced-order PDE.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from SciML.models.v11.state_space import LAYOUT, TransitionRecord, project_state, stage_mask
from .state_space_candidate import CausalDeltaStateModel, causal_state_features

CHECKPOINT_SCHEMA_VERSION = "sciml_wp3_v12_nonlinear_state_checkpoint_v1"
DEFAULT_EPOCHS = 1200
DEFAULT_LEARNING_RATE = 0.002


def _network(input_size: int, output_size: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, output_size),
    ).double()


class CausalNeuralStateModel:
    """Affine ablation plus a learned nonlinear residual; no silent fallback."""

    def __init__(self, seed: int = 20260828, hidden: int = 64) -> None:
        if isinstance(hidden, bool) or int(hidden) != hidden or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.seed, self.hidden = int(seed), int(hidden)
        self.affine_baseline = CausalDeltaStateModel(seed=self.seed)
        self.networks: dict[int, nn.Sequential] = {}
        self.residual_scales: dict[int, np.ndarray] = {}
        self.training_config: dict[str, object] = {}
        self.training_ids: tuple[str, ...] = ()

    def _active(self, stage: int) -> np.ndarray:
        if isinstance(stage, bool) or stage not in (1, 2, 3, 4):
            raise ValueError("stage must be 1, 2, 3 or 4")
        return stage_mask(np.asarray([stage]))[0].astype(bool)

    def _features(self, state: np.ndarray, controls: np.ndarray, stage: int) -> np.ndarray:
        self._active(stage)
        raw = causal_state_features(state, controls)
        if not np.array_equal(np.asarray(controls)[:4], np.eye(4)[stage - 1]):
            raise ValueError("stage/control one-hot mismatch")
        if self.affine_baseline.feature_mean is None:
            raise RuntimeError("nonlinear state model is not fitted")
        return (raw - self.affine_baseline.feature_mean) / self.affine_baseline.feature_scale

    def fit(self, records: list[TransitionRecord], epochs: int = DEFAULT_EPOCHS,
            learning_rate: float = DEFAULT_LEARNING_RATE, *,
            allow_development_temporal_labels: bool = False) -> dict[str, object]:
        if isinstance(epochs, bool) or int(epochs) != epochs or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not records or len({r.sample_id for r in records}) != len(records):
            raise ValueError("training records must be nonempty with unique sample IDs")
        self.networks.clear()
        self.residual_scales.clear()
        self.training_ids = ()
        for record in records:
            self._active(record.stage_index)
            if np.asarray(record.state_out).shape != (LAYOUT.size,) or not np.isfinite(record.state_out).all():
                raise ValueError("invalid target state")
        affine_metrics = self.affine_baseline.fit(
            records, allow_development_temporal_labels=allow_development_temporal_labels)
        stages = np.asarray([r.stage_index for r in records])
        features = np.stack([self._features(r.state_in, r.controls, r.stage_index) for r in records])
        targets = np.stack([r.state_out for r in records])
        states = np.stack([r.state_in for r in records])
        stage_losses = {}
        # Small deterministic CPU batches: avoid 24-thread BLAS overhead and
        # restore caller settings/RNG on both success and failure.
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                for stage in range(1, 5):
                    selected = stages == stage
                    active = self._active(stage)
                    x = torch.from_numpy(features[selected])
                    affine = self.affine_baseline.models[stage].predict(features[selected])
                    residual = (targets - states)[selected][:, active] - affine
                    scale = np.sqrt(np.mean(residual ** 2, axis=0))
                    scale = np.where(scale > 1e-10, scale, 1.0)
                    y = torch.from_numpy(residual / scale)
                    network = _network(features.shape[1], int(active.sum()), self.hidden)
                    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
                    initial_loss = float(torch.mean((network(x) - y) ** 2).detach())
                    for _ in range(int(epochs)):
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.mean((network(x) - y) ** 2)
                        if not torch.isfinite(loss):
                            raise RuntimeError("non-finite neural training loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), 10.0, error_if_nonfinite=True)
                        optimizer.step()
                    network.eval()
                    final_loss = float(torch.mean((network(x) - y) ** 2).detach())
                    if not np.isfinite(final_loss):
                        raise RuntimeError("non-finite final neural training loss")
                    self.networks[stage] = network
                    self.residual_scales[stage] = scale
                    stage_losses[str(stage)] = {"initial_normalized_mse": initial_loss,
                                                 "final_normalized_mse": final_loss}
        except Exception:
            self.networks.clear()  # a partially fitted candidate cannot predict
            raise
        finally:
            torch.set_num_threads(previous_threads)
        self.training_ids = tuple(r.sample_id for r in records)
        self.training_config = {"epochs": int(epochs), "learning_rate": float(learning_rate),
            "optimizer": "AdamW", "weight_decay": 1e-4, "gradient_clip_norm": 10.0,
            "hidden": self.hidden, "hidden_layers": 2, "activation": "SiLU", "seed": self.seed,
            "cycle_index_feature_included": False,
            "allow_development_temporal_labels": bool(allow_development_temporal_labels)}
        prediction = self.predict_records(records)
        return {"rmse_all_state": float(np.sqrt(np.mean((prediction - targets) ** 2))),
                "max_abs_all_state": float(np.max(np.abs(prediction - targets))),
                "affine_baseline": affine_metrics, "stage_neural_losses": stage_losses,
                "optimizer_steps": int(epochs) * 4}

    def _require_fitted(self) -> None:
        if sorted(self.networks) != [1, 2, 3, 4] or not self.training_ids:
            raise RuntimeError("nonlinear state model is not fitted")

    def trust_score(self, state: np.ndarray, controls: np.ndarray, stage_index: int) -> float:
        self._require_fitted()
        self._features(state, controls, stage_index)
        return 1.0  # fixed unattenuated residual; no nearest-neighbour fallback

    def predict(self, state: np.ndarray, controls: np.ndarray, stage_index: int) -> np.ndarray:
        self._require_fitted()
        features = self._features(state, controls, stage_index)[None, :]
        active = self._active(stage_index)
        with torch.no_grad():
            correction = self.networks[stage_index](torch.from_numpy(features)).numpy()[0]
        delta = self.affine_baseline.models[stage_index].predict(features)[0]
        delta = delta + correction * self.residual_scales[stage_index]
        if not np.isfinite(delta).all():
            raise RuntimeError("non-finite neural prediction before physical projection")
        result = np.asarray(state, dtype=np.float64).copy()
        result[active] += delta
        return project_state(result)

    def predict_records(self, records: list[TransitionRecord]) -> np.ndarray:
        return np.stack([self.predict(r.state_in, r.controls, r.stage_index) for r in records])

    def predict_stage_mean(self, state: np.ndarray, stage_index: int) -> np.ndarray:
        self._require_fitted()
        self._active(stage_index)
        return self.affine_baseline.predict_stage_mean(state, stage_index)

    def parameter_sha256(self) -> str:
        self._require_fitted()
        digest = hashlib.sha256(self.affine_baseline.parameter_sha256().encode())
        digest.update(json.dumps(self.training_config, sort_keys=True).encode())
        for stage in range(1, 5):
            values = {"residual_scale": self.residual_scales[stage],
                      **{k: v.detach().cpu().numpy() for k, v in self.networks[stage].state_dict().items()}}
            for name, value in sorted(values.items()):
                array = np.ascontiguousarray(value, dtype=np.float64)
                if not np.isfinite(array).all():
                    raise ValueError("non-finite checkpoint parameter")
                digest.update(f"{stage}:{name}:{array.shape}".encode())
                digest.update(array.tobytes())
        return digest.hexdigest()


def save_frozen_checkpoint(path: str | Path, model: CausalNeuralStateModel, *,
                          metadata: Mapping[str, str]) -> str:
    required = {"dataset_source_content_sha256", "implementation_sha256"}
    if required - set(metadata):
        raise ValueError("checkpoint metadata is missing")
    payload = {"checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "confirmatory": False, "metadata": dict(metadata),
        "parameter_sha256": model.parameter_sha256(), "training_config": model.training_config,
        "affine_baseline": model.affine_baseline,
        "residual_scales": model.residual_scales,
        "stage_networks": {k: v.state_dict() for k, v in model.networks.items()}}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Evidence is append-only: callers must name a new candidate checkpoint.
    with destination.open("xb") as stream:
        stream.write(pickle.dumps(payload, protocol=5))
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_frozen_checkpoint(path: str | Path, *,
                          expected_metadata: Mapping[str, str] | None = None) -> CausalNeuralStateModel:
    # Trusted local artifacts only; the confirmation caller checks file SHA
    # against the immutable manifest before deserializing this pickle.
    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported nonlinear WP3 checkpoint schema")
    if payload.get("confirmatory") is not False:
        raise ValueError("checkpoint crosses the development evidence boundary")
    for key, expected in (expected_metadata or {}).items():
        if payload.get("metadata", {}).get(key) != expected:
            raise ValueError(f"WP3 nonlinear checkpoint {key} mismatch")
    config = payload["training_config"]
    if config.get("hidden_layers") != 2 or config.get("activation") != "SiLU" or config.get("cycle_index_feature_included") is not False:
        raise ValueError("checkpoint violates nonlinear causal architecture contract")
    model = CausalNeuralStateModel(seed=config["seed"], hidden=config["hidden"])
    model.affine_baseline = payload["affine_baseline"]
    if not isinstance(model.affine_baseline, CausalDeltaStateModel):
        raise ValueError("invalid affine baseline type")
    model.training_ids = model.affine_baseline.training_ids
    model.training_config = dict(config)
    if sorted(payload["stage_networks"]) != [1, 2, 3, 4]:
        raise ValueError("checkpoint lacks four stage networks")
    with torch.random.fork_rng(devices=[]):
        for stage in range(1, 5):
            count = int(model._active(stage).sum())
            network = _network(LAYOUT.size + 12, count, model.hidden)
            network.load_state_dict(payload["stage_networks"][stage], strict=True)
            network.eval()
            model.networks[stage] = network
            scale = np.asarray(payload["residual_scales"][stage], dtype=np.float64)
            if scale.shape != (count,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("invalid residual scale")
            model.residual_scales[stage] = scale
    if model.parameter_sha256() != payload.get("parameter_sha256"):
        raise ValueError("WP3 nonlinear checkpoint parameter hash mismatch")
    return model
