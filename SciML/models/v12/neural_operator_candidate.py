"""Cycle-invariant neural POD stage operator; separate from the historical prior.

This is a fixed-grid numerical surrogate, not a mesh-independent PDE solver.
Only observed start-of-stage fields and causal controls enter inference. Both
fields and both stages use the same trained neural path on every cycle. There
is no cycle-triggered prior, nearest-neighbour attenuation or fallback.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from SciML.models.v11.spatiotemporal_operator import OperatorSample
from .temporal_operator_candidate import (
    CoefficientMLP,
    FEATURE_LAYOUT,
    _feature_scale_floor,
    causal_delta_features,
)


CHECKPOINT_SCHEMA_VERSION = "sciml_wp4_cycle_invariant_neural_checkpoint_v1"
INFERENCE_POLICY = "neural_pod_all_fields_all_cycles_no_fallback_v1"


class CausalNeuralPODOperator:
    def __init__(self, rank: int = 4, hidden: int = 128, seed: int = 20260828):
        for name, value in (("rank", rank), ("hidden", hidden)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.rank, self.hidden, self.seed = rank, hidden, seed
        self.model: CoefficientMLP | None = None
        self.training_ids: tuple[str, ...] = ()
        self.training_config: dict[str, object] = {}
        self.stage_mean: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.sma_basis: np.ndarray | None = None
        self.htf_basis: np.ndarray | None = None
        self.coefficient_mean: np.ndarray | None = None
        self.coefficient_scale: np.ndarray | None = None

    @staticmethod
    def _features(sample: OperatorSample) -> np.ndarray:
        if sample.stage_index not in (2, 4):
            raise ValueError("operator only owns exchange stages 2 and 4")
        for history in (sample.sma_history, sample.htf_history):
            values = np.asarray(history)
            if values.shape != (50, 101):
                raise ValueError("operator requires 50 x 101 fields")
            initial = values[0]
            if not np.isfinite(initial).all() or np.any((initial < 200) | (initial > 400)):
                raise ValueError("invalid initial field before physical projection")
        result = causal_delta_features(sample)
        expected_stage = np.eye(2)[0 if sample.stage_index == 2 else 1]
        if not np.array_equal(result[FEATURE_LAYOUT.stage], expected_stage):
            raise ValueError("stage metadata disagrees with causal stage")
        expected_material = {"recA": [1., 0.], "recAs": [0., 1.]}.get(sample.material_id)
        if expected_material is None or not np.array_equal(
            result[FEATURE_LAYOUT.material], expected_material
        ):
            raise ValueError("material metadata disagrees with causal material")
        return result

    def fit(self, samples: list[OperatorSample], epochs: int = 1500,
            learning_rate: float = 2e-3) -> dict[str, object]:
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if len(samples) < 4 or {s.stage_index for s in samples} != {2, 4}:
            raise ValueError("training requires at least four samples and both exchange stages")
        if any(s.cycle_index != 1 for s in samples):
            raise ValueError("cycle-1 training forbids Temporal12/cycle-2/3 labels")
        ids = tuple(s.sample_id for s in samples)
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("training sample IDs must be unique and nonempty")
        features = np.stack([self._features(s) for s in samples])
        histories = [np.stack([getattr(s, field) for s in samples])
                     for field in ("sma_history", "htf_history")]
        if not all(np.isfinite(values).all() for values in histories):
            raise ValueError("non-finite training target")
        self.model = None
        self.training_ids = ()
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.maximum(features.std(axis=0), _feature_scale_floor())
        scaled = (features - self.feature_mean) / self.feature_scale
        residuals = [(values - values[:, :1]).reshape(len(samples), -1) for values in histories]
        centered = [np.empty_like(values) for values in residuals]
        self.stage_mean = {}
        for stage in (2, 4):
            selected = np.asarray([s.stage_index == stage for s in samples])
            self.stage_mean[stage] = tuple(values[selected].mean(axis=0) for values in residuals)
            for index in range(2):
                centered[index][selected] = residuals[index][selected] - self.stage_mean[stage][index]
        bases = [np.linalg.svd(values, full_matrices=False)[2][:self.rank] for values in centered]
        self.sma_basis, self.htf_basis = bases
        coefficients = np.concatenate([values @ basis.T for values, basis in zip(centered, bases)], axis=1)
        self.coefficient_mean = coefficients.mean(axis=0)
        deviation = coefficients.std(axis=0)
        self.coefficient_scale = np.where(deviation > 1e-10, deviation, 1.0)
        target = (coefficients - self.coefficient_mean) / self.coefficient_scale
        previous_threads = torch.get_num_threads()
        losses = []
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                network = CoefficientMLP(FEATURE_LAYOUT.size, target.shape[1], self.hidden).double()
                optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-6)
                x, y = torch.from_numpy(scaled), torch.from_numpy(target)
                for epoch in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean((network(x) - y) ** 2)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite neural operator training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                    if epoch in (0, epochs - 1):
                        losses.append(float(loss.detach()))
                network.eval()
                self.model = network
            self.training_ids = ids
            self.training_config = {"rank": self.rank, "hidden": self.hidden, "seed": self.seed,
                "epochs": epochs, "learning_rate": float(learning_rate), "optimizer": "AdamW",
                "weight_decay": 1e-6, "gradient_clip_norm": 10., "hidden_layers": 2,
                "activation": "SiLU", "allowed_cycle_indices": [1],
                "cycle_index_used_in_inference": False, "inference_policy": INFERENCE_POLICY}
            self.parameter_sha256()  # reject non-finite final optimizer state
            predictions = self.predict(samples)
        except Exception:
            self.model = None
            self.training_ids = ()
            raise
        finally:
            torch.set_num_threads(previous_threads)
        return {"sma_train_rmse_K": float(np.sqrt(np.mean((predictions[0] - histories[0]) ** 2))),
                "htf_train_rmse_K": float(np.sqrt(np.mean((predictions[1] - histories[1]) ** 2))),
                "optimizer_steps": epochs, "initial_loss": losses[0], "last_loss": losses[-1],
                "inference_policy": INFERENCE_POLICY}

    def _require_fitted(self):
        if self.model is None or not self.training_ids or set(self.stage_mean) != {2, 4}:
            raise RuntimeError("neural operator is not fitted")

    def _reconstruct(self, samples, coefficients=None):
        self._require_fitted()
        fields = [[], []]
        split = self.sma_basis.shape[0]
        for index, sample in enumerate(samples):
            self._features(sample)
            residuals = list(self.stage_mean[sample.stage_index])
            if coefficients is not None:
                residuals = [residuals[0] + coefficients[index, :split] @ self.sma_basis,
                             residuals[1] + coefficients[index, split:] @ self.htf_basis]
            for field, (history, residual) in enumerate(zip(
                (sample.sma_history, sample.htf_history), residuals
            )):
                prediction = history[0][None, :] + residual.reshape(50, 101)
                if not np.isfinite(prediction).all():
                    raise RuntimeError("non-finite operator prediction before projection")
                prediction = np.clip(prediction, 200., 400.)
                prediction[0] = history[0]
                fields[field].append(prediction)
        return tuple(np.stack(values) for values in fields)

    def predict(self, samples: list[OperatorSample]):
        self._require_fitted()
        features = np.stack([self._features(s) for s in samples])
        scaled = (features - self.feature_mean) / self.feature_scale
        with torch.no_grad():
            coefficients = self.model(torch.from_numpy(scaled)).numpy()
        coefficients = coefficients * self.coefficient_scale + self.coefficient_mean
        sma, htf = self._reconstruct(samples, coefficients)
        # Compatibility third output is a fixed neural contribution weight,
        # never a confidence, uncertainty or out-of-distribution score.
        return sma, htf, np.ones(len(samples))

    def predict_stage_mean(self, samples: list[OperatorSample]):
        return self._reconstruct(samples)

    def parameter_sha256(self):
        self._require_fitted()
        digest = hashlib.sha256(INFERENCE_POLICY.encode())
        digest.update(json.dumps(self.training_config, sort_keys=True).encode())
        arrays = {name: getattr(self, name) for name in (
            "feature_mean", "feature_scale", "sma_basis", "htf_basis",
            "coefficient_mean", "coefficient_scale")}
        arrays.update({f"network:{name}": value.detach().cpu().numpy()
                       for name, value in self.model.state_dict().items()})
        for stage, values in self.stage_mean.items():
            arrays[f"stage:{stage}:sma"], arrays[f"stage:{stage}:htf"] = values
        for name, value in sorted(arrays.items()):
            array = np.ascontiguousarray(value)
            if not np.isfinite(array).all():
                raise ValueError("non-finite neural operator checkpoint parameter")
            digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
        digest.update(json.dumps(self.training_ids).encode())
        return digest.hexdigest()

    def save_checkpoint(self, path, *, dataset_source_content_sha256,
                        implementation_sha256, training_config=None):
        if training_config is not None and any(
            self.training_config.get(key) != value for key, value in training_config.items()
        ):
            raise ValueError("supplied training configuration does not match actual training")
        payload = {"checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                   "candidate": self, "parameter_sha256": self.parameter_sha256(),
                   "dataset_source_content_sha256": dataset_source_content_sha256,
                   "implementation_sha256": implementation_sha256}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            pickle.dump(payload, stream, protocol=5)
        return hashlib.sha256(destination.read_bytes()).hexdigest()

    @classmethod
    def from_checkpoint(cls, path, *, expected_dataset_source_content_sha256=None,
                        expected_implementation_sha256=None):
        # Trusted local artifacts only. Confirmation must verify the file SHA
        # against its immutable freeze before any pickle deserialization.
        payload = pickle.loads(Path(path).read_bytes())
        if not isinstance(payload, dict) or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported cycle-invariant WP4 checkpoint schema")
        for key, expected in (("dataset_source_content_sha256", expected_dataset_source_content_sha256),
                              ("implementation_sha256", expected_implementation_sha256)):
            if expected is not None and payload.get(key) != expected:
                raise ValueError(f"WP4 checkpoint {key} mismatch")
        model = payload.get("candidate")
        if type(model) is not cls or model.training_config.get("inference_policy") != INFERENCE_POLICY:
            raise ValueError("WP4 checkpoint violates cycle-invariant neural policy")
        if model.parameter_sha256() != payload.get("parameter_sha256"):
            raise ValueError("WP4 checkpoint parameter hash mismatch")
        return model
