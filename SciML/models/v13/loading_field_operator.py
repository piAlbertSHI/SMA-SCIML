"""Stage-one loading surrogates conditioned on known design and pressure controls."""

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F

from SciML.models.v12.learning_coordinate_operator import SpectralLayer, interpolate


INITIAL = (0.0, 0.0, 293.0, 0.0)
FIELDS = ("u_mm", "w_mm", "temperature_K", "phase_fraction")
NEURAL_METHODS = ("mlp", "deeponet", "fourier")


def mlp(in_features, out_features, width, layers):
    blocks = []
    for _ in range(layers):
        blocks.extend((nn.Linear(in_features, width), nn.GELU()))
        in_features = width
    blocks.append(nn.Linear(in_features, out_features))
    return nn.Sequential(*blocks)


def query_coordinates(times, positions):
    tau, radius = torch.meshgrid(times, positions, indexing="ij")
    return torch.stack((2.0 * tau - 1.0, 2.0 * radius - 1.0), dim=-1)


class CoordinateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = mlp(6, 4, width=44, layers=3)

    def forward(self, controls, times, positions):
        coords = query_coordinates(times, positions)
        batch, nt, nr = len(controls), len(times), len(positions)
        local = coords[None].expand(batch, nt, nr, 2)
        conditions = controls[:, None, None].expand(batch, nt, nr, 4)
        return self.network(torch.cat((conditions, local), dim=-1)).permute(0, 3, 1, 2)


class LoadingDeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank = 8
        self.branch = mlp(4, 4 * self.rank, width=32, layers=2)
        self.trunk = mlp(2, 4 * self.rank, width=32, layers=2)
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, controls, times, positions):
        coeff = self.branch(controls).reshape(len(controls), 4, self.rank)
        basis = self.trunk(query_coordinates(times, positions)).reshape(len(times), len(positions), 4, self.rank)
        raw = torch.einsum("bfc,trfc->bftr", coeff, basis) / self.rank ** .5
        return raw + self.bias[None, :, None, None]


class RadialFourierOperator(nn.Module):
    """Spatial spectral layers at each requested time on a uniform radial grid."""

    def __init__(self):
        super().__init__()
        width = 12
        self.conditions = nn.Linear(4, width)
        self.lift = nn.Conv1d(2, width, 1)
        self.layers = nn.ModuleList(SpectralLayer(width, modes=6) for _ in range(2))
        self.project = nn.Sequential(nn.Conv1d(width, 16, 1), nn.GELU(), nn.Conv1d(16, 4, 1))

    def forward(self, controls, times, positions):
        expected = torch.linspace(0, 1, len(positions), dtype=positions.dtype, device=positions.device)
        if len(positions) < 8 or not torch.allclose(positions, expected):
            raise ValueError("Fourier queries require a uniform full radial grid")
        n, nt, nr = len(controls), len(times), len(positions)
        coords = query_coordinates(times, positions).permute(0, 2, 1)
        local = self.lift(coords)[None].expand(n, -1, -1, -1)
        value = (local + self.conditions(controls)[:, None, :, None]).reshape(n * nt, 12, nr)
        value = F.pad(value, (0, nr // 4))
        for layer in self.layers:
            value = F.gelu(layer(value))
        return self.project(value)[..., :nr].reshape(n, nt, 4, nr).permute(0, 2, 1, 3)


class NeuralLoadingOperator(nn.Module):
    SCHEMA = "stage_one_loading_operator"

    def __init__(self, kind):
        super().__init__()
        architectures = {"mlp": CoordinateMLP, "deeponet": LoadingDeepONet, "fourier": RadialFourierOperator}
        if kind not in architectures:
            raise ValueError(f"Unknown loading operator: {kind}")
        self.kind = kind
        self.core = architectures[kind]()
        self.register_buffer("control_mean", torch.zeros(4))
        self.register_buffer("control_scale", torch.ones(4))
        self.register_buffer("field_scale", torch.ones(4))
        self.register_buffer("initial", torch.tensor(INITIAL))

    def set_scaling(self, control_mean, control_scale, field_scale):
        for value, buffer in zip((control_mean, control_scale, field_scale),
                                 (self.control_mean, self.control_scale, self.field_scale)):
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype))

    def normalized_change(self, controls, times, positions):
        if controls.ndim != 2 or controls.shape[1] != 4:
            raise ValueError("Controls must have columns for radius, thickness, peak pressure and material")
        if not all(grid.ndim == 1 and len(grid) and torch.isfinite(grid).all()
                   and torch.all((grid >= 0) & (grid <= 1)) for grid in (times, positions)):
            raise ValueError("Normalized coordinates must be in [0, 1]")
        gate = -torch.expm1(-4.0 * times)
        scaled = (controls - self.control_mean) / self.control_scale
        return self.core(scaled, times, positions) * gate[None, None, :, None]

    def forward(self, controls, times, positions):
        change = self.normalized_change(controls, times, positions)
        return self.initial[None, :, None, None] + change * self.field_scale[None, :, None, None]

    def save(self, path):
        with Path(path).open("xb") as stream:
            torch.save({"schema": self.SCHEMA, "method": self.kind, "weights": self.state_dict()}, stream)

    @classmethod
    def load(cls, path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["schema"] != cls.SCHEMA:
            raise ValueError("Invalid loading operator checkpoint")
        with torch.random.fork_rng(devices=[]):
            model = cls(value["method"]).float()
        model.load_state_dict(value["weights"])
        return model.eval()


class PODRidgeLoading:
    """Fixed low-rank basis of centered loading trajectories with Ridge coefficients."""

    def __init__(self, parameters):
        self.parameters = parameters

    @classmethod
    def fit(cls, controls, normalized_change, times, positions, control_mean, control_scale, field_scale, rank=8):
        x = (controls - control_mean) / control_scale
        flat = normalized_change.reshape(len(x), 4, -1)
        mean = flat.mean(axis=0)
        modes, coefficients, offsets, fractions = [], [], [], []
        for column in range(4):
            centered = flat[:, column] - mean[column]
            _, singular, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:rank]
            ridge = Ridge(alpha=1.0).fit(x, centered @ basis.T)
            modes.append(basis)
            coefficients.append(ridge.coef_)
            offsets.append(ridge.intercept_)
            fractions.append(np.sum(singular[:rank] ** 2) / np.sum(singular ** 2))
        return cls({
            "coordinates_time": times, "coordinates_radius": positions,
            "control_mean": control_mean, "control_scale": control_scale, "field_scale": field_scale,
            "mean_change": mean, "basis": np.stack(modes), "coef": np.stack(coefficients),
            "intercept": np.stack(offsets), "training_centered_variance_fraction": np.array(fractions),
        })

    def predict(self, controls, times, positions):
        p = self.parameters
        x = (controls - p["control_mean"]) / p["control_scale"]
        fields = []
        for i in range(4):
            modal = x @ p["coef"][i].T + p["intercept"][i]
            fields.append(modal @ p["basis"][i] + p["mean_change"][i])
        t, r = p["coordinates_time"], p["coordinates_radius"]
        change = torch.from_numpy(np.stack(fields, axis=1).reshape(len(x), 4, len(t), len(r)))
        change = interpolate(change, torch.from_numpy(r), torch.from_numpy(positions))
        change = interpolate(change.transpose(-1, -2), torch.from_numpy(t), torch.from_numpy(times)).transpose(-1, -2)
        result = change.numpy() * p["field_scale"][None, :, None, None] + np.asarray(INITIAL)[None, :, None, None]
        result[:, :, times == 0] = np.asarray(INITIAL)[None, :, None, None]
        return result

    def save(self, path):
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, **self.parameters)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as source:
            return cls({key: source[key] for key in source.files})
