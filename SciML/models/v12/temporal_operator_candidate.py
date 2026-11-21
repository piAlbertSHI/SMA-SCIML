"""Leakage-resistant v1.2 temporal stage-operator candidate.

The v1.1 operator standardizes every node of the observed start-of-stage HTF
field independently.  Cycle-1 HTF fields are almost spatially constant, so a
causal cycle-2 mean-temperature shift can be divided by a near-zero training
standard deviation at 101 coordinates.  The resulting Euclidean distance is
thousands of trust radii even though the physical shift is only a few kelvin.

This development candidate represents each initial field as one absolute mean
plus a mean-free spatial profile.  Fixed physical scale floors prevent a
near-constant cycle-1 coordinate from becoming infinitely informative.  Trust
uses a dimension-normalized RMS distance, fitted only from cycle-1 features.
Temporal labels are explicitly rejected by ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from SciML.models.v11.spatiotemporal_operator import (
    OperatorSample,
)


FIELD_POINTS = 101
PROFILE_SCALE_FLOOR_K = 0.5
ABSOLUTE_TEMPERATURE_SCALE_FLOOR_K = 5.0
SIDE_TEMPERATURE_SCALE_FLOOR_K = 5.0
TRUST_RADIUS_FLOOR = 0.35
# Frozen before Paper220 in the v1.1 development protocol and carried forward
# unchanged.  Keep the value local so the v1.2 implementation and parameter
# hashes cover every numerical inference constant.
# These temporal-prior gains were selected on the already-inspected Paper220
# Temporal12 *development* diagnostic, never on either v1.2 confirmation
# matrix.  The selection used a coarse, prereproducible mean/profile grid and
# required positive skill over both persistence and the stage-mean comparator
# in each stage.  Stage 2 SMA is deliberately prior-dominated: development
# evidence showed that its learned coefficient correction did not transport
# across cycles.  Stage 4 SMA retains the learned correction, which was strong
# on that same development split.  HTF uses the causal stage prior in both
# stages because its learned spatial correction was worse than the stage prior
# on development temporal samples.
TEMPORAL_SMA_PRIOR_GAINS = {2: (0.925, 0.85)}
TEMPORAL_HTF_PRIOR_GAINS = {
    2: {"recA": (0.15, 0.75), "recAs": (0.15, 1.20)},
    4: {"recA": (0.0, 0.70), "recAs": (0.0, 1.40)},
}
CHECKPOINT_SCHEMA_VERSION = "sciml_wp4_v12_checkpoint_v2"


@dataclass(frozen=True)
class CausalFeatureLayout:
    sma_profile: slice = slice(0, 101)
    htf_profile: slice = slice(101, 202)
    field_means: slice = slice(202, 204)
    stage: slice = slice(204, 206)
    material: slice = slice(206, 208)
    controls: slice = slice(208, 213)
    side: slice = slice(213, 221)
    size: int = 221


FEATURE_LAYOUT = CausalFeatureLayout()


class CoefficientMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def causal_delta_features(sample: OperatorSample) -> np.ndarray:
    """Return causal, cycle-invariant branch features.

    ``cycle_index`` is deliberately excluded.  It is constant during cycle-1
    fitting and would otherwise be an unlearned temporal extrapolation flag.
    The observed initial fields and side-state already carry the causal thermal
    history required by a Markov stage operator.
    """

    sma0 = np.asarray(sample.sma_history[0], dtype=np.float64)
    htf0 = np.asarray(sample.htf_history[0], dtype=np.float64)
    if sma0.shape != (FIELD_POINTS,) or htf0.shape != (FIELD_POINTS,):
        raise ValueError("v1.2 candidate requires 101-node initial fields")
    sma_mean = float(np.mean(sma0))
    htf_mean = float(np.mean(htf0))

    # v1.1 metadata after the two 101-node initial fields:
    # stage(2), material(2), radius/thickness/pressure/flush/duration/cycle(6),
    # side-state(8).  The cycle coordinate is omitted here.
    metadata = np.asarray(sample.branch[2 * FIELD_POINTS :], dtype=np.float64)
    if metadata.shape != (18,):
        raise ValueError(f"unexpected v1.1 branch metadata shape {metadata.shape}")
    stage = metadata[0:2]
    material = metadata[2:4]
    controls_without_cycle = metadata[4:9]
    side = metadata[10:18]
    result = np.concatenate(
        (
            sma0 - sma_mean,
            htf0 - htf_mean,
            np.asarray([sma_mean, htf_mean]),
            stage,
            material,
            controls_without_cycle,
            side,
        )
    )
    if result.shape != (FEATURE_LAYOUT.size,) or not np.isfinite(result).all():
        raise ValueError("invalid v1.2 causal feature vector")
    return result


def _feature_scale_floor() -> np.ndarray:
    floor = np.ones(FEATURE_LAYOUT.size, dtype=np.float64)
    floor[FEATURE_LAYOUT.sma_profile] = PROFILE_SCALE_FLOOR_K
    floor[FEATURE_LAYOUT.htf_profile] = PROFILE_SCALE_FLOOR_K
    floor[FEATURE_LAYOUT.field_means] = ABSOLUTE_TEMPERATURE_SCALE_FLOOR_K
    floor[FEATURE_LAYOUT.stage] = 0.5
    floor[FEATURE_LAYOUT.material] = 0.5
    # radius [m], thickness [m], pressure [Pa], flush ratio, duration [s]
    floor[FEATURE_LAYOUT.controls] = np.asarray(
        [0.003, 1.5e-5, 1.0e5, 0.1, 0.4], dtype=np.float64
    )
    floor[FEATURE_LAYOUT.side] = SIDE_TEMPERATURE_SCALE_FLOOR_K
    return floor


class CausalDeltaPODOperator:
    """POD coefficient model with cycle-1-only fit and robust causal trust."""

    def __init__(self, rank: int = 4, hidden: int = 128, seed: int = 20260828) -> None:
        self.rank = int(rank)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.temporal_sma_prior_gains = dict(TEMPORAL_SMA_PRIOR_GAINS)
        self.temporal_htf_prior_gains = dict(TEMPORAL_HTF_PRIOR_GAINS)
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.stage_mean: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.stage_training_features: dict[int, np.ndarray] = {}
        self.stage_trust_radius: dict[int, float] = {}
        self.sma_basis: np.ndarray | None = None
        self.htf_basis: np.ndarray | None = None
        self.coefficient_mean: np.ndarray | None = None
        self.coefficient_scale: np.ndarray | None = None
        self.model: CoefficientMLP | None = None
        self.training_ids: tuple[str, ...] = ()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _residual(sample: OperatorSample) -> tuple[np.ndarray, np.ndarray]:
        return (
            (sample.sma_history - sample.sma_history[0]).reshape(-1),
            (sample.htf_history - sample.htf_history[0]).reshape(-1),
        )

    @staticmethod
    def _basis(matrix: np.ndarray, rank: int) -> np.ndarray:
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        return vh[: min(rank, len(vh))]

    def fit(
        self,
        samples: list[OperatorSample],
        epochs: int = 1500,
        learning_rate: float = 2e-3,
    ) -> dict[str, float]:
        if len(samples) < 4:
            raise ValueError("v1.2 candidate requires at least four cycle-1 samples")
        leaked = [sample.sample_id for sample in samples if sample.cycle_index != 1]
        if leaked:
            raise ValueError(
                "v1.2 development fit forbids Temporal12/cycle-2/3 labels: "
                + ", ".join(leaked[:3])
            )

        raw_features = np.stack([causal_delta_features(sample) for sample in samples])
        self.feature_mean = raw_features.mean(axis=0)
        self.feature_scale = np.maximum(raw_features.std(axis=0), _feature_scale_floor())
        scaled = (raw_features - self.feature_mean) / self.feature_scale

        sma_residual = np.stack([self._residual(sample)[0] for sample in samples])
        htf_residual = np.stack([self._residual(sample)[1] for sample in samples])
        centered_sma = np.empty_like(sma_residual)
        centered_htf = np.empty_like(htf_residual)
        for stage in (2, 4):
            selected = np.asarray([sample.stage_index == stage for sample in samples])
            if not selected.any():
                raise ValueError(f"v1.2 training corpus lacks stage {stage}")
            mean_sma = sma_residual[selected].mean(axis=0)
            mean_htf = htf_residual[selected].mean(axis=0)
            self.stage_mean[stage] = (mean_sma, mean_htf)
            centered_sma[selected] = sma_residual[selected] - mean_sma
            centered_htf[selected] = htf_residual[selected] - mean_htf
            reference = scaled[selected]
            self.stage_training_features[stage] = reference
            if len(reference) >= 2:
                distances = np.sqrt(
                    np.mean(
                        (reference[:, None, :] - reference[None, :, :]) ** 2,
                        axis=2,
                    )
                )
                distances[distances == 0.0] = np.nan
                nearest = np.nanmin(distances, axis=1)
                self.stage_trust_radius[stage] = max(
                    float(np.nanmedian(nearest)), TRUST_RADIUS_FLOOR
                )
            else:
                self.stage_trust_radius[stage] = TRUST_RADIUS_FLOOR

        self.sma_basis = self._basis(centered_sma, self.rank)
        self.htf_basis = self._basis(centered_htf, self.rank)
        coefficients = np.concatenate(
            (
                centered_sma @ self.sma_basis.T,
                centered_htf @ self.htf_basis.T,
            ),
            axis=1,
        )
        self.coefficient_mean = coefficients.mean(axis=0)
        self.coefficient_scale = np.where(
            coefficients.std(axis=0) > 1e-10,
            coefficients.std(axis=0),
            1.0,
        )
        target = (coefficients - self.coefficient_mean) / self.coefficient_scale
        x = torch.tensor(scaled, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        self.model = CoefficientMLP(x.shape[1], y.shape[1], self.hidden)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-6
        )
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((self.model(x) - y) ** 2)
            loss.backward()
            optimizer.step()
        self.training_ids = tuple(sample.sample_id for sample in samples)
        predicted_sma, predicted_htf, _ = self.predict(samples)
        true_sma = np.stack([sample.sma_history for sample in samples])
        true_htf = np.stack([sample.htf_history for sample in samples])
        return {
            "sma_train_rmse_K": float(
                np.sqrt(np.mean((predicted_sma - true_sma) ** 2))
            ),
            "htf_train_rmse_K": float(
                np.sqrt(np.mean((predicted_htf - true_htf) ** 2))
            ),
        }

    def _scaled_feature(self, sample: OperatorSample) -> np.ndarray:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("v1.2 candidate is not fitted")
        return (causal_delta_features(sample) - self.feature_mean) / self.feature_scale

    def trust_score(self, sample: OperatorSample) -> float:
        scaled = self._scaled_feature(sample)
        reference = self.stage_training_features[sample.stage_index]
        distance = float(
            np.min(np.sqrt(np.mean((reference - scaled[None, :]) ** 2, axis=1)))
        )
        radius = self.stage_trust_radius[sample.stage_index]
        return float(np.exp(-((distance / radius) ** 2)))

    @staticmethod
    def _scaled_stage_prior(
        residual: np.ndarray, mean_gain: float, profile_gain: float
    ) -> np.ndarray:
        """Scale temporal mean and mean-free spatial dynamics separately."""

        field = np.asarray(residual, dtype=np.float64).reshape(50, FIELD_POINTS)
        mean = np.mean(field, axis=1, keepdims=True)
        profile = field - mean
        return (mean_gain * mean + profile_gain * profile).reshape(-1)

    def predict(
        self, samples: list[OperatorSample]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        required = (
            self.feature_mean,
            self.feature_scale,
            self.sma_basis,
            self.htf_basis,
            self.coefficient_mean,
            self.coefficient_scale,
            self.model,
        )
        if any(value is None for value in required):
            raise RuntimeError("v1.2 candidate is not fitted")
        scaled = np.stack([self._scaled_feature(sample) for sample in samples])
        with torch.no_grad():
            normalized = self.model(torch.tensor(scaled, dtype=torch.float32)).numpy()
        coefficients = normalized * self.coefficient_scale + self.coefficient_mean
        split = self.sma_basis.shape[0]
        predictions_sma = []
        predictions_htf = []
        trusts = []
        for index, sample in enumerate(samples):
            trust = self.trust_score(sample)
            trusts.append(trust)
            mean_sma, mean_htf = self.stage_mean[sample.stage_index]
            sma_correction = coefficients[index, :split] @ self.sma_basis
            htf_correction = coefficients[index, split:] @ self.htf_basis
            if (
                sample.cycle_index >= 2
                and sample.stage_index in self.temporal_sma_prior_gains
            ):
                sma_residual = self._scaled_stage_prior(
                    mean_sma, *self.temporal_sma_prior_gains[sample.stage_index]
                )
            else:
                sma_residual = mean_sma + trust * sma_correction
            if sample.cycle_index >= 2:
                htf_residual = self._scaled_stage_prior(
                    mean_htf,
                    *self.temporal_htf_prior_gains[sample.stage_index][
                        sample.material_id
                    ],
                )
            else:
                htf_residual = mean_htf + trust * htf_correction
            sma = sample.sma_history[0][None, :] + sma_residual.reshape(50, 101)
            htf = sample.htf_history[0][None, :] + htf_residual.reshape(50, 101)
            sma[0] = sample.sma_history[0]
            htf[0] = sample.htf_history[0]
            predictions_sma.append(np.clip(sma, 200.0, 400.0))
            predictions_htf.append(np.clip(htf, 200.0, 400.0))
        return np.stack(predictions_sma), np.stack(predictions_htf), np.asarray(trusts)

    def predict_stage_mean(
        self, samples: list[OperatorSample]
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions_sma = []
        predictions_htf = []
        for sample in samples:
            mean_sma, mean_htf = self.stage_mean[sample.stage_index]
            sma = sample.sma_history[0][None, :] + mean_sma.reshape(50, 101)
            htf = sample.htf_history[0][None, :] + mean_htf.reshape(50, 101)
            sma[0] = sample.sma_history[0]
            htf[0] = sample.htf_history[0]
            predictions_sma.append(np.clip(sma, 200.0, 400.0))
            predictions_htf.append(np.clip(htf, 200.0, 400.0))
        return np.stack(predictions_sma), np.stack(predictions_htf)

    def parameter_sha256(self) -> str:
        if self.model is None:
            raise RuntimeError("v1.2 candidate is not fitted")
        digest = hashlib.sha256()

        def update_array(name: str, value: np.ndarray) -> None:
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())

        for name, tensor in sorted(self.model.state_dict().items()):
            update_array(f"network:{name}", tensor.detach().cpu().numpy())
        for name, value in (
            ("feature_mean", self.feature_mean),
            ("feature_scale", self.feature_scale),
            ("sma_basis", self.sma_basis),
            ("htf_basis", self.htf_basis),
            ("coefficient_mean", self.coefficient_mean),
            ("coefficient_scale", self.coefficient_scale),
        ):
            if value is None:
                raise RuntimeError(f"missing v1.2 parameter {name}")
            update_array(name, value)
        update_array(
            "model_hyperparameters",
            np.asarray([self.rank, self.hidden, self.seed], dtype=np.int64),
        )
        for stage, gains in sorted(self.temporal_sma_prior_gains.items()):
            update_array(
                f"temporal_sma_prior_gains:{stage}",
                np.asarray(gains, dtype=np.float64),
            )
        for stage, material_gains in sorted(
            self.temporal_htf_prior_gains.items()
        ):
            for material, gains in sorted(material_gains.items()):
                update_array(
                    f"temporal_htf_prior_gains:{stage}:{material}",
                    np.asarray(gains, dtype=np.float64),
                )
        for stage in sorted(self.stage_mean):
            update_array(f"stage_mean_sma:{stage}", self.stage_mean[stage][0])
            update_array(f"stage_mean_htf:{stage}", self.stage_mean[stage][1])
            update_array(
                f"stage_training_features:{stage}",
                self.stage_training_features[stage],
            )
            update_array(
                f"stage_trust_radius:{stage}",
                np.asarray([self.stage_trust_radius[stage]]),
            )
        for sample_id in self.training_ids:
            digest.update(b"training_id\0")
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def checkpoint_payload(
        self,
        *,
        dataset_source_content_sha256: str,
        implementation_sha256: str,
        training_config: dict[str, object],
    ) -> dict[str, object]:
        """Return the complete, hash-bound inference state."""

        if self.model is None:
            raise RuntimeError("v1.2 candidate is not fitted")
        required = (
            self.feature_mean,
            self.feature_scale,
            self.sma_basis,
            self.htf_basis,
            self.coefficient_mean,
            self.coefficient_scale,
        )
        if any(value is None for value in required):
            raise RuntimeError("v1.2 checkpoint state is incomplete")
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_dict": self.model.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "sma_pod_basis": self.sma_basis,
            "htf_pod_basis": self.htf_basis,
            "stage_mean": self.stage_mean,
            "stage_training_features": self.stage_training_features,
            "stage_trust_radius": self.stage_trust_radius,
            "coefficient_mean": self.coefficient_mean,
            "coefficient_scale": self.coefficient_scale,
            "training_ids": self.training_ids,
            "temporal_sma_prior_gains": self.temporal_sma_prior_gains,
            "temporal_htf_prior_gains": self.temporal_htf_prior_gains,
            "parameter_sha256": self.parameter_sha256(),
            "dataset_source_content_sha256": str(dataset_source_content_sha256),
            "implementation_sha256": str(implementation_sha256),
            "training_config": dict(training_config),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        dataset_source_content_sha256: str,
        implementation_sha256: str,
        training_config: dict[str, object],
    ) -> str:
        """Save complete inference state and return the file SHA-256."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.checkpoint_payload(
                dataset_source_content_sha256=dataset_source_content_sha256,
                implementation_sha256=implementation_sha256,
                training_config=training_config,
            ),
            destination,
        )
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        expected_dataset_source_content_sha256: str | None = None,
        expected_implementation_sha256: str | None = None,
    ) -> "CausalDeltaPODOperator":
        """Load a checkpoint and reject stale, incomplete or corrupt state."""

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("WP4 v1.2 checkpoint payload is not a mapping")
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported WP4 v1.2 checkpoint schema")
        if (
            expected_dataset_source_content_sha256 is not None
            and str(payload.get("dataset_source_content_sha256", ""))
            != expected_dataset_source_content_sha256
        ):
            raise ValueError("WP4 v1.2 checkpoint dataset hash mismatch")
        if (
            expected_implementation_sha256 is not None
            and str(payload.get("implementation_sha256", ""))
            != expected_implementation_sha256
        ):
            raise ValueError("WP4 v1.2 checkpoint implementation hash mismatch")

        config = payload.get("training_config", {})
        if not isinstance(config, dict):
            raise TypeError("WP4 v1.2 checkpoint training_config is not a mapping")
        model = cls(
            rank=int(config["rank"]),
            hidden=int(config["hidden"]),
            seed=int(config["seed"]),
        )
        state_dict = payload["state_dict"]
        coefficient_mean = np.asarray(payload["coefficient_mean"], dtype=np.float64)
        first_weight = state_dict["network.0.weight"]
        model.model = CoefficientMLP(
            int(first_weight.shape[1]), int(coefficient_mean.size), model.hidden
        )
        model.model.load_state_dict(state_dict)
        model.model.eval()
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        model.sma_basis = np.asarray(payload["sma_pod_basis"], dtype=np.float64)
        model.htf_basis = np.asarray(payload["htf_pod_basis"], dtype=np.float64)
        model.coefficient_mean = coefficient_mean
        model.coefficient_scale = np.asarray(
            payload["coefficient_scale"], dtype=np.float64
        )
        model.stage_mean = {
            int(stage): (
                np.asarray(value[0], dtype=np.float64),
                np.asarray(value[1], dtype=np.float64),
            )
            for stage, value in payload["stage_mean"].items()
        }
        model.stage_training_features = {
            int(stage): np.asarray(value, dtype=np.float64)
            for stage, value in payload["stage_training_features"].items()
        }
        model.stage_trust_radius = {
            int(stage): float(value)
            for stage, value in payload["stage_trust_radius"].items()
        }
        model.training_ids = tuple(str(value) for value in payload["training_ids"])
        model.temporal_sma_prior_gains = {
            int(stage): tuple(float(item) for item in value)
            for stage, value in payload["temporal_sma_prior_gains"].items()
        }
        model.temporal_htf_prior_gains = {
            int(stage): {
                str(material): tuple(float(item) for item in gains)
                for material, gains in value.items()
            }
            for stage, value in payload["temporal_htf_prior_gains"].items()
        }
        if model.parameter_sha256() != str(payload.get("parameter_sha256", "")):
            raise ValueError("WP4 v1.2 checkpoint parameter hash mismatch")
        return model
