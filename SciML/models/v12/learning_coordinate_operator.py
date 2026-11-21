"""Matched coarse-label DeepONet/Fourier teaching experiment on existing fields.

Stage-conditional input: observed start-of-stage fields, not whole-cycle forecast.
Continuous output coordinates do not certify new-mesh solver generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from SciML.models.v11.spatiotemporal_operator import build_operator_corpus
from .temporal_operator_candidate import causal_delta_features, _feature_scale_floor
from .run_learning_existing import sha, validate_dataset, write_json

TIME_INDICES = np.r_[np.arange(0, 50, 2), 49]
SPACE_INDICES = np.arange(0, 101, 2)
KINDS = ("deeponet", "fourier", "deeponet_no_stage", "deeponet_no_ic")


def interpolate(values, source, query):
    """Piecewise linear interpolation along final axis; bounded query only."""
    if source.ndim != 1 or query.ndim != 1 or values.shape[-1] != len(source) or len(source) < 2:
        raise ValueError("invalid interpolation shape")
    if not torch.isfinite(query).all() or torch.any(query < source[0]) or torch.any(query > source[-1]):
        raise ValueError("query outside observed initial-profile coordinate support")
    if not torch.all(torch.diff(source) > 0):
        raise ValueError("source coordinates must increase")
    left = torch.searchsorted(source.contiguous(), query.contiguous(), right=True)-1
    left = left.clamp(0, len(source)-2)
    weight = (query-source[left])/(source[left+1]-source[left])
    return values[..., left]*(1-weight)+values[..., left+1]*weight


def mlp(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, output_size))


class MultiBranchDeepONet(nn.Module):
    def __init__(self, latent=24):
        super().__init__()
        self.latent = latent
        self.field_branch = mlp(204, 32)
        self.control_branch = mlp(9, 32)
        self.side_branch = mlp(8, 32)
        self.fusion = nn.Sequential(nn.Tanh(), nn.Linear(96, 2*latent))
        self.trunk = mlp(2, 2*latent)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, branch, initial, times, positions):
        coefficients = self.fusion(torch.cat((self.field_branch(branch[:, :204]),
            self.control_branch(branch[:, 204:213]), self.side_branch(branch[:, 213:])), dim=1))
        tt, xx = torch.meshgrid(times, positions, indexing="ij")
        coordinates = torch.stack((2*xx-1, 2*tt-1), dim=-1).reshape(-1, 2)
        trunk = self.trunk(coordinates).reshape(-1, 2, self.latent)
        coefficients = coefficients.reshape(len(branch), 2, self.latent)
        prediction = torch.einsum("bfl,qfl->bfq", coefficients, trunk)/(self.latent**.5)
        return (prediction+self.bias[None, :, None]).reshape(len(branch), 2, len(times), len(positions))


class SpectralLayer(nn.Module):
    def __init__(self, width=12, modes=8):
        super().__init__()
        self.width, self.modes = width, modes
        # Store complex weights as paired REAL parameters: portable optimizer state.
        self.weight = nn.Parameter(torch.randn(width, width, modes, 2)/width)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, values):
        spectrum = torch.fft.rfft(values, dim=-1, norm="ortho")
        count = min(self.modes, spectrum.shape[-1])
        output = torch.zeros_like(spectrum)
        weight = torch.view_as_complex(self.weight.contiguous())
        output[:, :, :count] = torch.einsum("bik,iok->bok", spectrum[:, :, :count], weight[:, :, :count])
        return torch.fft.irfft(output, n=values.shape[-1], dim=-1, norm="ortho")+self.local(values)


class ConditionalFourier(nn.Module):
    """1-D spatial FNO, time-conditioned; not a full 2-D space-time FNO."""
    def __init__(self, width=12, modes=8):
        super().__init__()
        self.condition = nn.Linear(221, width)
        self.lift = nn.Conv1d(4, width, 1)
        self.layers = nn.ModuleList([SpectralLayer(width, modes), SpectralLayer(width, modes)])
        self.output = nn.Sequential(nn.Conv1d(width, 16, 1), nn.GELU(), nn.Conv1d(16, 2, 1))

    def forward(self, branch, initial, times, positions):
        if len(positions) < 8 or not torch.allclose(positions, torch.linspace(0, 1, len(positions), dtype=positions.dtype, device=positions.device)):
            raise ValueError("Fourier comparison requires an evenly spaced whole-domain grid")
        batch, steps, nodes = len(branch), len(times), len(positions)
        profiles = ((initial-293.)/5.)[:, None].expand(batch, steps, 2, nodes)
        x = positions[None, None, None].expand(batch, steps, 1, nodes)
        t = times[None, :, None, None].expand(batch, steps, 1, nodes)
        local = torch.cat((profiles, x, t), dim=2).reshape(batch*steps, 4, nodes)
        value = self.lift(local)+self.condition(branch)[:, None, :, None].expand(batch, steps, -1, nodes).reshape(batch*steps, -1, nodes)
        # Fixed fraction of zero padding mitigates Fourier wrap-around; it is
        # NOT a physically derived periodic/Dirichlet boundary condition.
        padding = max(1, nodes//4)
        value = F.pad(value, (0, padding))
        for layer in self.layers:
            value = F.gelu(layer(value))
        value = self.output(value)[..., :nodes]
        return value.reshape(batch, steps, 2, nodes).permute(0, 2, 1, 3)


class CoordinateOperator(nn.Module):
    def __init__(self, kind="deeponet"):
        super().__init__()
        if kind not in KINDS:
            raise ValueError("unknown operator variant")
        self.kind = kind
        self.core = ConditionalFourier() if kind == "fourier" else MultiBranchDeepONet()
        self.register_buffer("feature_mean", torch.zeros(221))
        self.register_buffer("feature_scale", torch.ones(221))
        self.register_buffer("field_scale", torch.ones(2))
        self.register_buffer("initial_grid", torch.from_numpy(np.linspace(0., 1., 101)))
        self.training_ids = []
        self.history = []

    def forward(self, branch, initial, times, positions):
        if branch.ndim != 2 or branch.shape[1] != 221 or initial.shape != (len(branch), 2, 101):
            raise ValueError("invalid stage input shapes")
        if not all(torch.isfinite(v).all() for v in (branch, initial, times, positions)):
            raise ValueError("nonfinite operator input")
        if times.ndim != 1 or torch.any(times < 0) or torch.any(times > 1):
            raise ValueError("time query outside single observed stage")
        scaled = (branch-self.feature_mean)/self.feature_scale
        if self.kind == "deeponet_no_stage":
            scaled = torch.cat((scaled[:, :204], torch.zeros_like(scaled[:, 204:206]), scaled[:, 206:]), dim=1)
        start = interpolate(initial, self.initial_grid, positions)
        residual = self.core(scaled, start, times, positions)
        gate = torch.ones_like(times) if self.kind == "deeponet_no_ic" else -torch.expm1(-4*times)
        return start[:, :, None, :]+self.field_scale[None, :, None, None]*gate[None, None, :, None]*residual

    def fit(self, branch, initial, target, ids, *, seed=20260910, steps=400):
        if len(ids) != len(branch) or len(set(ids)) != len(ids) or target.shape != (len(ids), 2, 50, 101):
            raise ValueError("invalid fitting corpus")
        if len(ids) < 4 or type(steps) is not int or steps < 1:
            raise ValueError("invalid training budget")
        self.seed, self.steps = seed, steps
        self.training_ids = []
        branch, initial, target = (np.asarray(v, dtype=np.float64) for v in (branch, initial, target))
        if not all(np.isfinite(v).all() for v in (branch, initial, target)):
            raise ValueError("nonfinite training fields")
        coarse = target[:, :, TIME_INDICES][:, :, :, SPACE_INDICES]
        residual = coarse-initial[:, :, None, SPACE_INDICES]
        self.feature_mean.copy_(torch.from_numpy(branch.mean(axis=0)))
        self.feature_scale.copy_(torch.from_numpy(np.maximum(branch.std(axis=0), _feature_scale_floor())))
        self.field_scale.copy_(torch.from_numpy(np.maximum(np.sqrt(np.mean(residual**2, axis=(0, 2, 3))), .1)))
        bx, init, truth = map(torch.from_numpy, (branch, initial, target))
        grid_x = torch.tensor(SPACE_INDICES/100., dtype=torch.float64)
        rng = np.random.default_rng(seed+77)
        optimizer = torch.optim.AdamW(self.parameters(), lr=.002, weight_decay=1e-5)
        self.history = []
        for step in range(steps):
            batch = rng.choice(len(ids), size=min(16, len(ids)), replace=False)
            time_indices = np.sort(rng.choice(TIME_INDICES, size=6, replace=False))
            query_t = torch.tensor(time_indices/49., dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            prediction = self(bx[batch], init[batch], query_t, grid_x)
            label = truth[batch][:, :, time_indices][:, :, :, SPACE_INDICES]
            loss = torch.mean(((prediction-label)/self.field_scale[None, :, None, None])**2)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite operator loss; diagnose rather than restart")
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step % 50 == 0 or step == steps-1:
                self.history.append({"updates": step+1, "sampled_normalized_mse": float(loss.detach())})
        self.training_ids = list(ids)
        self.eval()
        return self

    def predict(self, branch, initial, times=None, positions=None):
        if not self.training_ids:
            raise RuntimeError("operator is not fitted")
        times = np.linspace(0, 1, 50) if times is None else np.asarray(times)
        positions = np.linspace(0, 1, 101) if positions is None else np.asarray(positions)
        values = []
        with torch.no_grad():
            for start in range(0, len(branch), 8):
                value = self(torch.tensor(branch[start:start+8], dtype=torch.float64),
                             torch.tensor(initial[start:start+8], dtype=torch.float64),
                             torch.tensor(times, dtype=torch.float64), torch.tensor(positions, dtype=torch.float64))
                values.append(value.numpy())
        prediction = np.concatenate(values)
        if not np.isfinite(prediction).all():
            raise RuntimeError("nonfinite field prediction")
        return prediction

    def save(self, path):
        with path.open("xb") as stream:
            torch.save({"schema": "learning_coordinate_operator_v1", "kind": self.kind, "seed": self.seed,
                        "steps": self.steps, "training_ids": self.training_ids, "history": self.history,
                        "state_dict": self.state_dict()}, stream)

    @classmethod
    def load(cls, path):
        payload = torch.load(path, weights_only=True)
        if payload["schema"] != "learning_coordinate_operator_v1":
            raise ValueError("invalid operator checkpoint")
        with torch.random.fork_rng(devices=[]):
            model = cls(payload["kind"]).double()
        model.load_state_dict(payload["state_dict"])
        model.seed, model.steps = payload["seed"], payload["steps"]
        model.training_ids, model.history = payload["training_ids"], payload["history"]
        model.eval()
        return model


def corpus(dataset):
    table, quality, _ = validate_dataset(dataset)
    samples = build_operator_corpus(dataset)
    # Verify that normalized coordinates used below really match exported grids.
    lookup = table.set_index("case_id")
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for sample in samples:
            row = lookup.loc[sample.case_id]
            stage = "stage2_hold_exchange" if sample.stage_index == 2 else "stage4_cooldown_exchange"
            group = handle[str(row.hdf5_group)+"/cycles/cycle_001/"+stage]
            np.testing.assert_allclose(np.asarray(group["sma_strip_coordinate_m"]), np.linspace(-row.radius_m, row.radius_m, 101), atol=1e-12)
            np.testing.assert_allclose(np.asarray(group["time_local_s"]), np.linspace(0, row.exchange_duration_s, 50), atol=1e-12)
    branch = np.stack([causal_delta_features(s) for s in samples])
    truth = np.stack([np.stack((s.sma_history, s.htf_history)) for s in samples])
    initial = truth[:, :, 0].copy()
    return samples, branch, initial, truth, quality


def evaluate(prediction, truth, samples, mask):
    report = {}
    for stage in (2, 4):
        selected = mask & np.array([s.stage_index == stage for s in samples])
        report[str(stage)] = {name: {"future_rmse_K": float(np.sqrt(np.mean((prediction[selected, j, 1:]-truth[selected, j, 1:])**2))),
                                   "initial_max_abs_K": float(np.max(np.abs(prediction[selected, j, 0]-truth[selected, j, 0]))),
                                   "future_max_abs_K": float(np.max(np.abs(prediction[selected, j, 1:]-truth[selected, j, 1:])))}
                             for j, name in enumerate(("sma", "htf"))}
    return report


def run(dataset, original_protocol, out, kind, seed=20260910):
    if out.exists():
        raise FileExistsError("preserve previous operator evidence")
    samples, branch, initial, truth, quality = corpus(dataset)
    original = json.loads(original_protocol.read_text())
    if original["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("dataset/split mismatch")
    masks = {part: np.array([s.condition_id in ids for s in samples]) for part, ids in original["condition_splits"].items()}
    if not np.all(np.stack(list(masks.values())).sum(axis=0) == 1):
        raise ValueError("non-disjoint split")
    train = masks["train"]
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"learning_only": True, "fresh_solver_calls": 0, "kind": kind, "seed": seed, "optimizer_steps": 400,
                "dataset_source_content_sha256": quality["source_content_sha256"], "condition_splits": original["condition_splits"],
                "training_sample_ids": [s.sample_id for s, active in zip(samples, train) if active],
                "supervised_time_indices": TIME_INDICES.tolist(), "supervised_space_indices": SPACE_INDICES.tolist(),
                "native_shape": [50, 101], "training_grid_shape": [26, 51], "batch_cases": 16, "batch_time_queries": 6,
                "input_policy": "Observed start-of-stage fields and side inputs; neither future field values nor realized flow trajectories",
                "time_space_coordinates": "tau=t/exchange_duration; SMA x=(strip_coordinate+R)/(2R); HTF x=(pipe_index-1)/100, distinct physical meanings",
                "fine_grid_evaluation": "Withheld coordinates of existing discretization; not new high-fidelity mesh simulation",
                "test_policy": "Previously inspected fixed condition split, exploratory model development",
                "model_spec": "Multi-branch DeepONet latent24 or two-layer spatial FNO width12 modes8 time-conditioned; float64 AdamW .002 wd1e-5 clip10",
                "initial_condition": "T0+(1-exp(-4tau))*field_scale*network; no-IC ablation usesgate1",
                "physics_boundary": "Hard initial value only; Fourier zero padding is not a physical boundary; no PDE/energy closure claim",
                "implementation_sha256": sha(Path(__file__))}
    write_json(out/"protocol.json", protocol)
    threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = CoordinateOperator(kind).double().fit(branch[train], initial[train], truth[train],
                protocol["training_sample_ids"], seed=seed)
        prediction = model.predict(branch, initial)
        model.save(out/"model.pt")
        replay = CoordinateOperator.load(out/"model.pt").predict(branch, initial)
        np.testing.assert_array_equal(prediction, replay)
        # A query-only grid smoke test has no new truth and cannot be reported as
        # verified physics accuracy. Stored to demonstrate arbitrary output size.
        alternate = model.predict(branch[:1], initial[:1], np.array([0., .137, .731, 1.]), np.linspace(0, 1, 201))
    finally:
        torch.set_num_threads(threads)
    persistence = np.broadcast_to(initial[:, :, None], truth.shape)
    metrics = {part: {"model": evaluate(prediction, truth, samples, mask), "persistence": evaluate(persistence, truth, samples, mask)}
               for part, mask in masks.items()}
    seen = np.zeros((50, 101), dtype=bool)
    seen[np.ix_(TIME_INDICES, SPACE_INDICES)] = True
    unseen = ~seen
    unseen[0] = False  # full initial profile is an input, never an unobserved target
    held_coordinate_rmse = [float(np.sqrt(np.mean((prediction[train, j][:, unseen]-truth[train, j][:, unseen])**2))) for j in (0, 1)]
    np.savez_compressed(out/"predictions.npz", sample_ids=np.array([s.sample_id for s in samples]),
                        prediction=prediction, truth=truth, alternative_grid_prediction=alternate)
    summary = {"status": "completed_coarse_label_operator_learning", "learning_only": True, "fresh_solver_calls": 0,
               "kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "history": model.history,
               "scores": metrics, "train_condition_unseen_coordinate_rmse_K": held_coordinate_rmse,
               "initial_max_abs_K": float(np.max(np.abs(prediction[:, :, 0]-initial))),
               "alternative_grid_shape": list(alternate.shape), "alternative_grid_has_ground_truth": False,
               "checkpoint_roundtrip_exact": True, "elapsed_seconds": time.monotonic()-started,
               "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_protocol, args.output_root, args.kind, args.seed), indent=2))
