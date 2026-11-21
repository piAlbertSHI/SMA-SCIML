"""End-to-end temporal DeepONet for the 50-point audited probe trajectories.

The model maps causal stage-entry fields, scheduled horizon, topology and
side-state inputs to temperature probe functions queried at normalized time.
The stored mass-flow trajectory is a post-run diagnostic and is never used as
an input. Dataset v1.0 does not
contain time-resolved 51/101-node fields, so this is a temporal *probe* operator,
not the full spatiotemporal field operator planned for paper-grade WP4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from .data import OperatorBatch

try:  # Keep the package importable for dataset-only users.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("WP4 temporal training requires torch; install SciML/requirements-ml.txt")


def _safe_scale(values: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    scale = np.nanstd(values, axis=axis)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def _static_context(batch: OperatorBatch) -> np.ndarray:
    stage = np.zeros((len(batch), 4), dtype=np.float64)
    valid = (batch.stage_index >= 1) & (batch.stage_index <= 4)
    stage[np.arange(len(batch))[valid], batch.stage_index[valid] - 1] = 1.0
    branch = np.column_stack(
        [[value == label for value in batch.branches] for label in ("full", "half1", "half2")]
    ).astype(np.float64)
    return np.concatenate((stage, branch, batch.duration_s[:, None], batch.topology), axis=1)


@dataclass
class TemporalProbeScaler:
    """Train-role-only preprocessing for causal branch inputs and targets."""

    side_median: np.ndarray
    branch_mean: np.ndarray
    branch_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray

    @classmethod
    def fit(cls, train: OperatorBatch) -> "TemporalProbeScaler":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            side_median = np.nanmedian(train.side_input, axis=0)
        side_median = np.where(np.isfinite(side_median), side_median, 0.0)
        raw = cls._raw_branch(train, side_median)
        targets = train.temperature_probes_t_K
        return cls(
            side_median=side_median,
            branch_mean=np.mean(raw, axis=0),
            branch_scale=_safe_scale(raw, axis=0),
            target_mean=np.mean(targets, axis=(0, 1)),
            target_scale=_safe_scale(targets, axis=(0, 1)),
        )

    @staticmethod
    def _raw_branch(batch: OperatorBatch, side_median: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(batch.side_input)
        side = np.where(~missing, batch.side_input, side_median)
        # ``mass_flow_t_kgps`` is a realized Simulink trajectory sampled after
        # execution, not an audited ex-ante command. It must remain audit-only.
        return np.concatenate(
            (
                batch.sma_in,
                batch.htf_in,
                _static_context(batch),
                side,
                missing.astype(np.float64),
            ),
            axis=1,
        )

    def branch(self, batch: OperatorBatch) -> np.ndarray:
        raw = self._raw_branch(batch, self.side_median)
        return (raw - self.branch_mean) / self.branch_scale

    def time(self, batch: OperatorBatch) -> np.ndarray:
        tau = batch.time_local_s / batch.duration_s[:, None]
        return np.stack((tau, tau**2, np.sin(np.pi * tau), np.cos(np.pi * tau)), axis=2)

    def targets(self, batch: OperatorBatch) -> np.ndarray:
        return (batch.temperature_probes_t_K - self.target_mean) / self.target_scale

    def inverse_targets(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * self.target_scale + self.target_mean


if nn is not None:

    class TemporalProbeDeepONet(nn.Module):
        """Multi-output branch/trunk network for temperature probe functions."""

        def __init__(self, branch_dim: int, probes: int, latent_dim: int = 24, hidden_dim: int = 64):
            super().__init__()
            self.probes = int(probes)
            self.latent_dim = int(latent_dim)
            self.branch_net = nn.Sequential(
                nn.Linear(branch_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, probes * latent_dim),
            )
            self.trunk_net = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.bias = nn.Parameter(torch.zeros(probes))

        def forward(self, branch: "torch.Tensor", time_features: "torch.Tensor") -> "torch.Tensor":
            coefficients = self.branch_net(branch).reshape(-1, self.probes, self.latent_dim)
            basis = self.trunk_net(time_features)
            return torch.einsum("bpl,btl->btp", coefficients, basis) + self.bias


@dataclass
class TrainedTemporalProbeOperator:
    model: Any
    scaler: TemporalProbeScaler
    seed: int
    loss_history: list[float]
    fitted_sample_ids: tuple[str, ...]

    def predict(self, batch: OperatorBatch) -> np.ndarray:
        _require_torch()
        self.model.eval()
        with torch.no_grad():
            branch = torch.as_tensor(self.scaler.branch(batch), dtype=torch.float32)
            time = torch.as_tensor(self.scaler.time(batch), dtype=torch.float32)
            normalized = self.model(branch, time).cpu().numpy().astype(np.float64)
        return self.scaler.inverse_targets(normalized)


def train_temporal_probe_operator(
    train: OperatorBatch,
    *,
    seed: int,
    epochs: int = 600,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-5,
    latent_dim: int = 24,
    hidden_dim: int = 64,
    smoothness_weight: float = 1e-3,
) -> TrainedTemporalProbeOperator:
    """Fit one deterministic full-batch model on the published train role only."""

    _require_torch()
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    scaler = TemporalProbeScaler.fit(train)
    branch_np = scaler.branch(train)
    time_np = scaler.time(train)
    target_np = scaler.targets(train)
    branch = torch.as_tensor(branch_np, dtype=torch.float32)
    time = torch.as_tensor(time_np, dtype=torch.float32)
    target = torch.as_tensor(target_np, dtype=torch.float32)
    model = TemporalProbeDeepONet(
        branch_dim=branch.shape[1],
        probes=target.shape[2],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(branch, time)
        data_loss = torch.mean((prediction - target) ** 2)
        second_difference = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        smoothness = torch.mean(second_difference**2)
        loss = data_loss + float(smoothness_weight) * smoothness
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    return TrainedTemporalProbeOperator(
        model=model,
        scaler=scaler,
        seed=int(seed),
        loss_history=history,
        fitted_sample_ids=train.sample_ids,
    )


def temporal_probe_metrics(
    batch: OperatorBatch,
    prediction: np.ndarray,
    *,
    temperature_range_k: tuple[float, float] | None = None,
) -> dict[str, Any]:
    truth = batch.temperature_probes_t_K
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("temporal probe prediction has an invalid shape or non-finite values")
    error = prediction - truth
    initial = truth[:, :1, :]
    true_change = truth - initial
    predicted_change = prediction - initial
    denominator = np.linalg.norm(true_change.reshape(len(batch), -1), axis=1)
    relative = np.linalg.norm((predicted_change - true_change).reshape(len(batch), -1), axis=1)
    relative /= np.maximum(denominator, 1e-8)
    persistence = np.broadcast_to(initial, truth.shape)
    persistence_rmse = float(np.sqrt(np.mean((persistence - truth) ** 2)))
    rmse = float(np.sqrt(np.mean(error**2)))
    per_probe = np.sqrt(np.mean(error**2, axis=(0, 1)))
    if temperature_range_k is None:
        lower = float(np.min(batch.temperature_probes_t_K))
        upper = float(np.max(batch.temperature_probes_t_K))
        range_source = "evaluated_role_observations"
    else:
        lower, upper = map(float, temperature_range_k)
        if not np.isfinite([lower, upper]).all() or not lower < upper:
            raise ValueError("temperature_range_k must be finite and increasing")
        range_source = "fit_role_observations"
    return {
        "rmse_K": rmse,
        "mae_K": float(np.mean(np.abs(error))),
        "final_time_rmse_K": float(np.sqrt(np.mean(error[:, -1, :] ** 2))),
        "relative_temperature_change_l2": float(np.mean(relative)),
        "persistence_rmse_K": persistence_rmse,
        "skill_vs_persistence": float(1.0 - rmse / max(persistence_rmse, 1e-12)),
        "prediction_min_K": float(np.min(prediction)),
        "prediction_max_K": float(np.max(prediction)),
        "plausibility_lower_K": lower,
        "plausibility_upper_K": upper,
        "plausibility_range_source": range_source,
        "temperature_range_violation_rate": float(np.mean((prediction < lower) | (prediction > upper))),
        "per_probe_rmse_K": dict(zip(batch.temperature_probe_labels, map(float, per_probe))),
    }
