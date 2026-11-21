"""POD--DeepONet pilot for distinct SMA and HTF 50 x 101 fields."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


SIDE_INPUT_KEYS = (
    "hot_sink_temp_in",
    "cold_sink_temp_in",
    "chip_temp_in",
    "hot_buffer_temp_in",
    "cold_buffer_temp_in",
    "stage2_path_pipe_T_state_in_K",
    "stage4_path_pipe_T_state_in_K",
    "simscape_cold_sink_pipe_T_state_in_K",
)
PREREGISTERED_TEMPORAL_SMA_MEAN_GAIN = 1.25
PREREGISTERED_TEMPORAL_HTF_MEAN_GAIN = 0.35


@dataclass(frozen=True)
class OperatorSample:
    sample_id: str
    case_id: str
    condition_id: str
    material_id: str
    geometry_id: str
    cycle_index: int
    stage_index: int
    branch: np.ndarray
    sma_history: np.ndarray
    htf_history: np.ndarray


def _table(root: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(root / "cycle_index.parquet")
    except (ImportError, OSError):
        return pd.read_csv(root / "cycle_index.csv")


def _finite_side(group: h5py.Group) -> np.ndarray:
    values = np.asarray([float(np.asarray(group["side_state"][key])) for key in SIDE_INPUT_KEYS])
    return np.where(np.isfinite(values), values, 293.0)


def build_operator_corpus(dataset_root: str | Path) -> list[OperatorSample]:
    root = Path(dataset_root)
    cycles = _table(root)
    cycles = cycles[cycles["physics_ok"].astype(bool)].sort_values(
        ["case_id", "cycle_index"]
    )
    samples: list[OperatorSample] = []
    with h5py.File(root / "fields.h5", "r") as handle:
        for row in cycles.itertuples(index=False):
            cycle = handle[str(row.hdf5_cycle_group)]
            for stage_index, stage_name in (
                (2, "stage2_hold_exchange"),
                (4, "stage4_cooldown_exchange"),
            ):
                stage = cycle[stage_name]
                sma = np.asarray(stage["sma_strip_T_K_t"], dtype=np.float64)
                htf = np.asarray(stage["htf_internal_T_K_t"], dtype=np.float64)
                if sma.shape != (50, 101) or htf.shape != (50, 101):
                    raise ValueError(f"unexpected field shape for {row.cycle_sample_id}/{stage_name}")
                if not np.isfinite(sma).all() or not np.isfinite(htf).all():
                    raise ValueError(f"non-finite field for {row.cycle_sample_id}/{stage_name}")
                stage_onehot = np.eye(2)[0 if stage_index == 2 else 1]
                material = np.array(
                    [float(str(row.material_id) == "recA"), float(str(row.material_id) == "recAs")]
                )
                controls = np.array(
                    [
                        float(row.radius_m),
                        float(row.thickness_m),
                        float(row.pressure_Pa),
                        float(row.flush_volume_target_ratio),
                        float(row.exchange_duration_s),
                        float(row.cycle_index),
                    ]
                )
                branch = np.concatenate((sma[0], htf[0], stage_onehot, material, controls, _finite_side(stage)))
                samples.append(
                    OperatorSample(
                        sample_id=f"{row.cycle_sample_id}:stage_{stage_index}",
                        case_id=str(row.case_id),
                        condition_id=str(row.condition_id),
                        material_id=str(row.material_id),
                        geometry_id=(
                            f"R{float(row.radius_m):.12g}_t{float(row.thickness_m):.12g}"
                        ),
                        cycle_index=int(row.cycle_index),
                        stage_index=stage_index,
                        branch=branch,
                        sma_history=sma,
                        htf_history=htf,
                    )
                )
    return samples


class BranchMLP(nn.Module):
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


class PODDeepONet:
    """Train-only POD trunks with a neural branch coefficient map."""

    def __init__(
        self,
        rank: int = 4,
        hidden: int = 128,
        seed: int = 20260820,
        temporal_sma_mean_gain: float = PREREGISTERED_TEMPORAL_SMA_MEAN_GAIN,
        temporal_htf_mean_gain: float = PREREGISTERED_TEMPORAL_HTF_MEAN_GAIN,
    ) -> None:
        if not 0.0 <= temporal_sma_mean_gain <= 2.0:
            raise ValueError("temporal_sma_mean_gain must be in [0, 2]")
        if not 0.0 <= temporal_htf_mean_gain <= 2.0:
            raise ValueError("temporal_htf_mean_gain must be in [0, 2]")
        self.rank = int(rank)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.temporal_sma_mean_gain = float(temporal_sma_mean_gain)
        self.temporal_htf_mean_gain = float(temporal_htf_mean_gain)
        self.branch_mean: np.ndarray | None = None
        self.branch_scale: np.ndarray | None = None
        self.stage_mean: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.sma_basis: np.ndarray | None = None
        self.htf_basis: np.ndarray | None = None
        self.coefficient_mean: np.ndarray | None = None
        self.coefficient_scale: np.ndarray | None = None
        self.stage_training_branch: dict[int, np.ndarray] = {}
        self.stage_trust_radius: dict[int, float] = {}
        self.model: BranchMLP | None = None
        self.training_ids: tuple[str, ...] = ()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "PODDeepONet":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("checkpoint_schema_version") != "sciml_wp4_checkpoint_v2":
            raise ValueError("unsupported WP4 checkpoint schema")
        config = payload.get("training_config", {})
        state_dict = payload["state_dict"]
        first_weight = state_dict["network.0.weight"]
        sma_basis = np.asarray(payload["sma_pod_basis"], dtype=np.float64)
        coefficient_mean = np.asarray(payload["coefficient_mean"], dtype=np.float64)
        hidden = int(config.get("hidden", first_weight.shape[0]))
        model = cls(
            rank=int(config.get("rank", sma_basis.shape[0])),
            hidden=hidden,
            seed=int(config.get("seed", 20260820)),
            temporal_sma_mean_gain=float(payload["temporal_sma_mean_gain"]),
            temporal_htf_mean_gain=float(payload["temporal_htf_mean_gain"]),
        )
        model.model = BranchMLP(
            int(first_weight.shape[1]), int(coefficient_mean.size), hidden
        )
        model.model.load_state_dict(state_dict)
        model.model.eval()
        model.branch_mean = np.asarray(payload["branch_mean"], dtype=np.float64)
        model.branch_scale = np.asarray(payload["branch_scale"], dtype=np.float64)
        model.sma_basis = sma_basis
        model.htf_basis = np.asarray(payload["htf_pod_basis"], dtype=np.float64)
        model.coefficient_mean = coefficient_mean
        model.coefficient_scale = np.asarray(
            payload["coefficient_scale"], dtype=np.float64
        )
        model.stage_mean = {
            int(stage): (
                np.asarray(values[0], dtype=np.float64),
                np.asarray(values[1], dtype=np.float64),
            )
            for stage, values in payload["stage_mean"].items()
        }
        model.stage_training_branch = {
            int(stage): np.asarray(values, dtype=np.float64)
            for stage, values in payload["stage_training_branch"].items()
        }
        model.stage_trust_radius = {
            int(stage): float(value)
            for stage, value in payload["stage_trust_radius"].items()
        }
        model.training_ids = tuple(str(value) for value in payload["training_ids"])
        if model.parameter_sha256() != str(payload.get("parameter_sha256", "")):
            raise ValueError("WP4 checkpoint parameter hash mismatch")
        return model

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

    def fit(self, samples: list[OperatorSample], epochs: int = 1500, learning_rate: float = 2e-3) -> dict[str, float]:
        if len(samples) < 4:
            raise ValueError("POD--DeepONet pilot requires at least four samples")
        branches = np.stack([sample.branch for sample in samples])
        self.branch_mean = branches.mean(axis=0)
        self.branch_scale = np.where(branches.std(axis=0) > 1e-10, branches.std(axis=0), 1.0)
        scaled_branch = (branches - self.branch_mean) / self.branch_scale

        sma_residual = np.stack([self._residual(sample)[0] for sample in samples])
        htf_residual = np.stack([self._residual(sample)[1] for sample in samples])
        centered_sma = np.empty_like(sma_residual)
        centered_htf = np.empty_like(htf_residual)
        for stage in (2, 4):
            selected = np.asarray([sample.stage_index == stage for sample in samples])
            mean_sma = sma_residual[selected].mean(axis=0)
            mean_htf = htf_residual[selected].mean(axis=0)
            self.stage_mean[stage] = (mean_sma, mean_htf)
            centered_sma[selected] = sma_residual[selected] - mean_sma
            centered_htf[selected] = htf_residual[selected] - mean_htf
            stage_branch = scaled_branch[selected]
            self.stage_training_branch[stage] = stage_branch
            distances = np.sqrt(
                np.sum((stage_branch[:, None, :] - stage_branch[None, :, :]) ** 2, axis=2)
            )
            distances[distances == 0.0] = np.nan
            nearest = np.nanmin(distances, axis=1)
            self.stage_trust_radius[stage] = max(float(np.nanmedian(nearest)), 1.0)

        self.sma_basis = self._basis(centered_sma, self.rank)
        self.htf_basis = self._basis(centered_htf, self.rank)
        sma_coeff = centered_sma @ self.sma_basis.T
        htf_coeff = centered_htf @ self.htf_basis.T
        coefficients = np.concatenate((sma_coeff, htf_coeff), axis=1)
        self.coefficient_mean = coefficients.mean(axis=0)
        self.coefficient_scale = np.where(
            coefficients.std(axis=0) > 1e-10, coefficients.std(axis=0), 1.0
        )
        target = (coefficients - self.coefficient_mean) / self.coefficient_scale
        x = torch.tensor(scaled_branch, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)
        self.model = BranchMLP(x.shape[1], y.shape[1], self.hidden)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
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
            "sma_train_rmse_K": float(np.sqrt(np.mean((predicted_sma - true_sma) ** 2))),
            "htf_train_rmse_K": float(np.sqrt(np.mean((predicted_htf - true_htf) ** 2))),
        }

    def _trust(self, scaled_branch: np.ndarray, stage: int) -> float:
        reference = self.stage_training_branch[stage]
        distance = float(np.min(np.linalg.norm(reference - scaled_branch[None, :], axis=1)))
        return float(np.exp(-((distance / self.stage_trust_radius[stage]) ** 2)))

    def predict(self, samples: list[OperatorSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if any(
            value is None
            for value in (
                self.branch_mean,
                self.branch_scale,
                self.sma_basis,
                self.htf_basis,
                self.coefficient_mean,
                self.coefficient_scale,
                self.model,
            )
        ):
            raise RuntimeError("model is not fitted")
        branches = np.stack([sample.branch for sample in samples])
        scaled = (branches - self.branch_mean) / self.branch_scale
        with torch.no_grad():
            normalized = self.model(torch.tensor(scaled, dtype=torch.float32)).numpy()
        coefficients = normalized * self.coefficient_scale + self.coefficient_mean
        split = self.sma_basis.shape[0]
        predictions_sma = []
        predictions_htf = []
        trusts = []
        for index, sample in enumerate(samples):
            trust = self._trust(scaled[index], sample.stage_index)
            trusts.append(trust)
            mean_sma, mean_htf = self.stage_mean[sample.stage_index]
            temporal = sample.cycle_index >= 2
            sma_mean_gain = self.temporal_sma_mean_gain if temporal else 1.0
            htf_mean_gain = self.temporal_htf_mean_gain if temporal else 1.0
            sma_residual = sma_mean_gain * mean_sma + trust * (
                coefficients[index, :split] @ self.sma_basis
            )
            htf_residual = htf_mean_gain * mean_htf + trust * (
                coefficients[index, split:] @ self.htf_basis
            )
            sma = sample.sma_history[0][None, :] + sma_residual.reshape(50, 101)
            htf = sample.htf_history[0][None, :] + htf_residual.reshape(50, 101)
            sma[0] = sample.sma_history[0]
            htf[0] = sample.htf_history[0]
            predictions_sma.append(np.clip(sma, 200.0, 400.0))
            predictions_htf.append(np.clip(htf, 200.0, 400.0))
        return np.stack(predictions_sma), np.stack(predictions_htf), np.asarray(trusts)

    def predict_stage_mean(self, samples: list[OperatorSample]) -> tuple[np.ndarray, np.ndarray]:
        """Predict with only the train-fitted stage-mean trajectory baseline."""

        if not self.stage_mean:
            raise RuntimeError("model is not fitted")
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
            raise RuntimeError("model is not fitted")
        digest = hashlib.sha256()

        def update_array(name: str, value: np.ndarray) -> None:
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())

        for name, tensor in sorted(self.model.state_dict().items()):
            update_array(f"network:{name}", tensor.detach().cpu().numpy())
        for name, value in (
            ("branch_mean", self.branch_mean),
            ("branch_scale", self.branch_scale),
            ("sma_basis", self.sma_basis),
            ("htf_basis", self.htf_basis),
            ("coefficient_mean", self.coefficient_mean),
            ("coefficient_scale", self.coefficient_scale),
        ):
            if value is None:
                raise RuntimeError(f"model parameter {name} is unavailable")
            update_array(name, value)
        update_array(
            "temporal_mean_gains",
            np.asarray(
                [self.temporal_sma_mean_gain, self.temporal_htf_mean_gain],
                dtype=np.float64,
            ),
        )
        for stage in sorted(self.stage_mean):
            mean_sma, mean_htf = self.stage_mean[stage]
            update_array(f"stage_mean_sma:{stage}", mean_sma)
            update_array(f"stage_mean_htf:{stage}", mean_htf)
            update_array(
                f"stage_training_branch:{stage}", self.stage_training_branch[stage]
            )
            update_array(
                f"stage_trust_radius:{stage}",
                np.asarray([self.stage_trust_radius[stage]], dtype=np.float64),
            )
        for sample_id in self.training_ids:
            digest.update(b"training_id\0")
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()
