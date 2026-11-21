"""Train-only-scaled PyTorch residual MLP; importing remains dependency-safe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None


@dataclass(frozen=True)
class TrainRoleScaler:
    """Feature/target statistics fitted exclusively from fit-role arrays."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    target_observed_count: np.ndarray
    fitted_role: str = "train"

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        observed: np.ndarray,
        *,
        role: str,
    ) -> "TrainRoleScaler":
        if role != "train":
            raise ValueError("scaler fitting is restricted to role='train'")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        mask = np.asarray(observed, dtype=bool)
        if x.ndim != 2 or y.ndim != 2 or y.shape != mask.shape or x.shape[0] != y.shape[0]:
            raise ValueError("invalid scaler feature/target/mask shapes")
        if not np.isfinite(x).all():
            raise ValueError("training features must be finite")
        feature_mean = np.mean(x, axis=0)
        feature_scale = np.std(x, axis=0)
        feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
        count = np.sum(mask & np.isfinite(y), axis=0)
        safe = np.where(mask & np.isfinite(y), y, 0.0)
        target_mean = np.divide(
            np.sum(safe, axis=0), count, out=np.zeros(y.shape[1]), where=count > 0
        )
        centered = np.where(mask & np.isfinite(y), y - target_mean, 0.0)
        variance = np.divide(
            np.sum(centered**2, axis=0), count,
            out=np.ones(y.shape[1]), where=count > 0,
        )
        target_scale = np.sqrt(variance)
        target_scale = np.where((count > 0) & (target_scale > 1e-8), target_scale, 1.0)
        return cls(feature_mean, feature_scale, target_mean, target_scale, count, role)

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.feature_mean) / self.feature_scale

    def transform_targets(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.target_mean) / self.target_scale

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.target_scale + self.target_mean

    def summary(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for array in (
            self.feature_mean, self.feature_scale, self.target_mean,
            self.target_scale, self.target_observed_count,
        ):
            digest.update(np.ascontiguousarray(array).tobytes())
        return {
            "fitted_role": self.fitted_role,
            "feature_dimension": int(self.feature_mean.size),
            "target_dimension": int(self.target_mean.size),
            "observed_target_channels": int(np.count_nonzero(self.target_observed_count)),
            "sha256": digest.hexdigest(),
        }


if TORCH_AVAILABLE:

    class TorchResidualMLP(nn.Module):
        """End-to-end trained MLP returning residuals in physical units."""

        def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_sizes: tuple[int, ...] = (256, 256),
            scaler: TrainRoleScaler | None = None,
        ):
            super().__init__()
            if (
                isinstance(input_size, bool)
                or isinstance(output_size, bool)
                or not isinstance(input_size, (int, np.integer))
                or not isinstance(output_size, (int, np.integer))
                or input_size <= 0
                or output_size <= 0
            ):
                raise ValueError("input_size and output_size must be positive integers")
            if not hidden_sizes or any(
                isinstance(width, bool)
                or not isinstance(width, (int, np.integer))
                or width <= 0
                for width in hidden_sizes
            ):
                raise ValueError("hidden_sizes must contain positive integers")
            widths = (input_size, *hidden_sizes, output_size)
            layers: list[nn.Module] = []
            for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(left, right))
                if index < len(widths) - 2:
                    layers.extend((nn.SiLU(), nn.LayerNorm(right)))
            self.network = nn.Sequential(*layers)
            self.scaler = scaler

        def forward(self, scaled_features):
            return self.network(scaled_features)

        def predict(self, features):
            if self.scaler is None:
                raise RuntimeError("TorchResidualMLP requires a train-role scaler before prediction")
            parameter = next(self.parameters())
            scaled = self.scaler.transform_features(np.asarray(features, dtype=np.float64))
            array = torch.as_tensor(scaled, dtype=parameter.dtype, device=parameter.device)
            self.eval()
            with torch.no_grad():
                result = self(array).detach().cpu().numpy()
            physical = self.scaler.inverse_targets(result)
            # Masked loss supplies no information for channels absent from all
            # train targets (notably xi_out). Their residual contract is exact
            # zero/persistence, never an arbitrary untrained network output.
            physical[..., self.scaler.target_observed_count == 0] = 0.0
            return physical

        def parameter_sha256(self) -> str:
            digest = hashlib.sha256()
            for name, value in sorted(self.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        def save_audited_checkpoint(self, path: str | Path, metadata: dict[str, Any]) -> None:
            if self.scaler is None:
                raise RuntimeError("cannot checkpoint a model without a fitted scaler")
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "scaler": {
                        key: (
                            getattr(self.scaler, key)
                            if key == "fitted_role"
                            else torch.as_tensor(getattr(self.scaler, key)).clone()
                        )
                        for key in (
                            "feature_mean", "feature_scale", "target_mean",
                            "target_scale", "target_observed_count", "fitted_role",
                        )
                    },
                    "metadata": metadata,
                    "parameter_sha256": self.parameter_sha256(),
                },
                destination,
            )

else:

    class TorchResidualMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "TorchResidualMLP requires the optional dependency 'torch>=2.2'; "
                "no neural-training result is claimed in this environment."
            )
