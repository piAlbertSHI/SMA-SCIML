"""Manufactured two-cycle heat benchmark for separately exercised physical losses.

Not SMA data, not a learned parametric operator, and not horizon extrapolation.
Eight local segments have sparse labels; interfaces are trained constraints.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .run_learning_existing import sha, write_json


FO = np.array([.05, .08, .03, .06])
DURATION = np.array([.4, .5, .3, .6])
LENGTH = .02
AREA = 1e-4
VOLUMETRIC_CAPACITY = 2e6
TEMPERATURE_SCALE = 5.
ENERGY_SCALE = VOLUMETRIC_CAPACITY*AREA*LENGTH*TEMPERATURE_SCALE
VARIANTS = ("data", "pde", "pde_interfaces", "full")


def exact(coords, blocks):
    x, tau = np.asarray(coords).T
    blocks = np.asarray(blocks, int)
    elapsed = np.array([FO[np.arange(b) % 4].sum() for b in blocks])
    return (np.sin(np.pi*x)*np.exp(-np.pi**2*(elapsed+FO[blocks % 4]*tau)))[:, None]


class StagedHeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1)).double()

    def forward(self, coords, blocks):
        if coords.ndim != 2 or coords.shape[1] != 2 or blocks.shape != (len(coords),):
            raise ValueError("coordinates[N,2], block[N] required")
        if blocks.dtype != torch.long or torch.any((blocks < 0) | (blocks > 7)):
            raise ValueError("block must be integer0..7")
        if not torch.isfinite(coords).all() or torch.any((coords < 0) | (coords > 1)):
            raise ValueError("coordinates outside unit square")
        x, tau = coords[:, :1], coords[:, 1:]
        condition = torch.cat((torch.nn.functional.one_hot(blocks % 4, 4),
                               torch.nn.functional.one_hot(blocks // 4, 2)), dim=1).to(coords.dtype)
        correction = x*(1-x)*self.net(torch.cat((2*coords-1, condition), dim=1))
        # Only the very first initial condition is known; later ICs are NOT
        # filled from the analytic reference. Dirichlet boundaries are hard.
        return torch.where((blocks == 0)[:, None], torch.sin(torch.pi*x)+tau*correction, correction)


def derivatives(model, coords, blocks, second=True):
    if not coords.requires_grad:
        raise ValueError("derivative coordinates require gradients")
    value = model(coords, blocks)
    first = torch.autograd.grad(value, coords, torch.ones_like(value), create_graph=True)[0]
    dxx = None
    if second:
        dxx = torch.autograd.grad(first[:, :1], coords, torch.ones_like(first[:, :1]), create_graph=True)[0][:, :1]
    return value, first, dxx


def pde_residual(model, coords, blocks):
    _, first, dxx = derivatives(model, coords, blocks)
    coefficient = torch.tensor(FO, dtype=coords.dtype)[blocks % 4, None]
    return first[:, 1:]-coefficient*dxx


def interface_residuals(model, x):
    left = torch.arange(7).repeat_interleave(len(x))
    xx = x.repeat(7)
    end = torch.stack((xx, torch.ones_like(xx)), dim=1)
    start = torch.stack((xx, torch.zeros_like(xx)), dim=1)
    diff = model(end, left)-model(start, left+1)
    cycle = left == 3
    return diff[~cycle], diff[cycle]


def energy_residual(model, times, quadrature_order=8):
    """Normalized dU/dtau - outward-gradient influx for each block/time.

    Actual U scale is Cvol*A*L*DeltaT J; actual rate scale divides by duration.
    Unlike a residual-defined storage label, both terms derive independently
    from the same differentiable state field and boundary gradients.
    """
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    x = torch.tensor((nodes+1)/2, dtype=torch.float64)
    w = torch.tensor(weights/2, dtype=torch.float64)
    blocks = torch.arange(8).repeat_interleave(len(times)*len(x))
    tt = times.repeat_interleave(len(x)).repeat(8)
    coords = torch.stack((x.repeat(8*len(times)), tt), dim=1).requires_grad_(True)
    _, first, _ = derivatives(model, coords, blocks, second=False)
    storage_rate = (first[:, 1].reshape(8, len(times), len(x))*w).sum(dim=2)
    edge_blocks = torch.arange(8).repeat_interleave(len(times)*2)
    edge = torch.stack((torch.tensor([0., 1.], dtype=torch.float64).repeat(8*len(times)),
                        times.repeat_interleave(2).repeat(8)), dim=1).requires_grad_(True)
    _, gradient, _ = derivatives(model, edge, edge_blocks, second=False)
    gx = gradient[:, 0].reshape(8, len(times), 2)
    fo = torch.tensor(np.tile(FO, 2), dtype=torch.float64)[:, None]
    boundary_influx = fo*(gx[:, :, 1]-gx[:, :, 0])
    return storage_rate-boundary_influx


def fit(coords, blocks, values, *, seed, variant, steps=600):
    if variant not in VARIANTS or type(steps) is not int or steps < 1:
        raise ValueError("invalid fit specification")
    coords, blocks, values = np.asarray(coords, float), np.asarray(blocks), np.asarray(values, float)
    if coords.shape != (len(blocks), 2) or values.shape != (len(blocks), 1) or not np.isfinite(values).all():
        raise ValueError("invalid sparse observations")
    if blocks.dtype.kind not in "iu":
        raise ValueError("integer block labels required")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = StagedHeat()
            obs = torch.tensor(coords, dtype=torch.float64)
            bid = torch.tensor(blocks, dtype=torch.long)
            target = torch.tensor(values, dtype=torch.float64)
            collocation = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed+11).draw(16).double().repeat(8, 1).requires_grad_(True)
            cb = torch.arange(8).repeat_interleave(16)
            ix = torch.linspace(0, 1, 13, dtype=torch.float64)
            et = torch.tensor([.2, .5, .8], dtype=torch.float64)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            history = []
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                collocation.grad = None
                data = ((model(obs, bid)-target)**2).mean()
                pde = (pde_residual(model, collocation, cb)**2).mean() if variant != "data" else data.new_zeros(())
                stage = cycle = energy = data.new_zeros(())
                if variant in ("pde_interfaces", "full"):
                    sr, cr = interface_residuals(model, ix)
                    stage, cycle = (sr**2).mean(), (cr**2).mean()
                if variant == "full":
                    energy = (energy_residual(model, et)**2).mean()
                total = 10*data+pde+stage+cycle+energy
                if not torch.isfinite(total):
                    raise RuntimeError("nonfinite staged loss, diagnose before a new run")
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                if step % 100 == 0 or step == steps-1:
                    history.append({"step": step+1, **{k: float(v.detach()) for k, v in
                        {"data": data, "pde": pde, "stage": stage, "cycle": cycle, "energy": energy, "total": total}.items()}})
            model.eval()
    finally:
        torch.set_num_threads(old_threads)
    return model, history


def evaluate(model):
    # Dense evaluation coordinates exclude all48sparseobservations by construction.
    x = (np.arange(31)+.5)/31
    t = (np.arange(41)+.5)/41
    xx, tt = np.meshgrid(x, t)
    coordinates = np.tile(np.column_stack((xx.ravel(), tt.ravel())), (8, 1))
    blocks = np.arange(8).repeat(len(xx.ravel()))
    truth = exact(coordinates, blocks)
    with torch.no_grad():
        prediction = model(torch.tensor(coordinates), torch.tensor(blocks)).numpy()
    # Independent diagnostic points and higher quadrature order than training.
    d = torch.quasirandom.SobolEngine(2, scramble=True, seed=9081).draw(37).double().repeat(8, 1).requires_grad_(True)
    b = torch.arange(8).repeat_interleave(37)
    r = pde_residual(model, d, b).detach().numpy()[:, 0]
    e = energy_residual(model, torch.tensor([.13, .37, .67, .91], dtype=torch.float64), 24).detach().numpy()
    sr, cr = interface_residuals(model, torch.linspace(0, 1, 39, dtype=torch.float64))
    sr, cr = sr.detach().numpy(), cr.detach().numpy()
    duration = np.tile(DURATION, 2)
    return {"unobserved_field_rmse_K": float(np.sqrt(np.mean((prediction-truth)**2))*TEMPERATURE_SCALE),
        "pde_rms_K_per_s": float(np.sqrt(np.mean((r*TEMPERATURE_SCALE/np.repeat(duration, 37))**2))),
        "stage_jump_rms_K": float(np.sqrt(np.mean(sr**2))*TEMPERATURE_SCALE),
        "cycle_jump_rms_K": float(np.sqrt(np.mean(cr**2))*TEMPERATURE_SCALE),
        "energy_rate_residual_rms_W": float(np.sqrt(np.mean((e*ENERGY_SCALE/duration[:, None])**2)))}, {
            "coordinates": coordinates, "blocks": blocks, "truth_normalized": truth, "prediction_normalized": prediction,
            "pde_residual_normalized": r, "energy_residual_normalized": e,
            "stage_jump_normalized": sr, "cycle_jump_normalized": cr}


def run(out, steps=600):
    if out.exists():
        raise FileExistsError("preserve staged physics evidence")
    # Fixed before training:6 interior observations in EACH of8 blocks.
    x, t = np.meshgrid([.2, .5, .8], [.25, .75])
    coords = np.tile(np.column_stack((x.ravel(), t.ravel())), (8, 1))
    blocks = np.arange(8).repeat(6)
    labels = exact(coords, blocks)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"learning_only": True, "fresh_solver_calls": 0, "dataset": "manufactured staged heat equation, NOT SMA observations",
        "equation": "T_t=alpha_stage*T_xx, zero excess-temperature end boundaries;4stages x2cycles",
        "length_m": LENGTH, "area_m2": AREA, "volumetric_capacity_J_per_m3K": VOLUMETRIC_CAPACITY,
        "temperature_scale_K": TEMPERATURE_SCALE, "energy_scale_J": ENERGY_SCALE,
        "duration_s": DURATION.tolist(), "Fourier_numbers": FO.tolist(), "diffusivity_m2_per_s": (FO*LENGTH**2/DURATION).tolist(),
        "seeds": [810, 811, 812], "steps": steps, "observations": 48, "variants": list(VARIANTS),
        "weights": {"data": 10, "pde": 1, "stage": 1, "cycle": 1, "energy": 1},
        "hard_conditions": "end Dirichlet every block; known analytic IC only very first block",
        "training": "8input-24-24-1Tanh, Adam.003, float64, gradientclip10; eachvariant same seed/start/observations/steps",
        "limitations": ["fixed PDE coordinate surrogate, not parameterized operator", "both cycles have sparse observed labels; not horizon extrapolation",
            "energy is global integral consequence of same PDE, not independent new physical law", "positive/negative comparisons retained; no test tuning"],
        "implementation_sha256": sha(Path(__file__))}
    write_json(out/"protocol.json", protocol)
    np.savez_compressed(out/"observations.npz", coordinates=coords, blocks=blocks, labels=labels)
    reports = []
    started = time.monotonic()
    for seed in protocol["seeds"]:
        for variant in VARIANTS:
            model, history = fit(coords, blocks, labels, seed=seed, variant=variant, steps=steps)
            root = out/f"s{seed}-{variant}"
            root.mkdir()
            with (root/"model.pt").open("xb") as stream:
                torch.save(model.state_dict(), stream)
            old = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                metrics, arrays = evaluate(model)
                with torch.random.fork_rng(devices=[]):
                    restored = StagedHeat()
                restored.load_state_dict(torch.load(root/"model.pt", weights_only=True))
                again, replay = evaluate(restored)
                if again != metrics:
                    raise ValueError("staged checkpoint metrics differ")
                for k, value in arrays.items():
                    np.testing.assert_array_equal(value, replay[k])
            finally:
                torch.set_num_threads(old)
            np.savez_compressed(root/"evaluation.npz", **arrays)
            report = {"seed": seed, "variant": variant, "metrics": metrics, "history": history}
            write_json(root/"report.json", report)
            reports.append(report)
    aggregate = {v: {k: {"mean": float(np.mean([r["metrics"][k] for r in reports if r["variant"] == v])),
                         "std": float(np.std([r["metrics"][k] for r in reports if r["variant"] == v], ddof=1))}
                     for k in reports[0]["metrics"]} for v in VARIANTS}
    result = {"status": "completed_manufactured_staged_physics_ablation", "learning_only": True, "fresh_solver_calls": 0,
        "model_fits": len(reports), "elapsed_seconds": time.monotonic()-started, "all_cold_replays_exact": True,
        "aggregate": aggregate, "performance_gate": "no claim every added loss improves every metric",
        "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.steps), indent=2))
