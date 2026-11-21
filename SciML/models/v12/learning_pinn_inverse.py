"""WP5 teaching benchmark: infer diffusivity AND an unobserved temperature field.

Manufactured dimensionless 1-D heat equation, NOT SMA solver observations.
No true diffusivity or dense target is passed to the neural fit function.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
import torch
from torch import nn

from .run_learning_existing import sha, write_json


def exact_temperature(x, t, diffusivity):
    """Two eigenmodes of u_t = a u_xx on [0,1], with zero Dirichlet BCs."""
    return (np.sin(np.pi*x)*np.exp(-diffusivity*np.pi**2*t)
            + .3*np.sin(2*np.pi*x)*np.exp(-4*diffusivity*np.pi**2*t))


def analytic_sensitivity(x, t, diffusivity):
    return (-np.pi**2*t*np.sin(np.pi*x)*np.exp(-diffusivity*np.pi**2*t)
            - 1.2*np.pi**2*t*np.sin(2*np.pi*x)*np.exp(-4*diffusivity*np.pi**2*t))


class HeatInversePINN(nn.Module):
    def __init__(self, initial_diffusivity=.25, hidden=24):
        super().__init__()
        if not np.isfinite(initial_diffusivity) or initial_diffusivity <= 0:
            raise ValueError("initial diffusivity must be positive/finite")
        self.hidden = hidden
        self.network = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)).double()
        self.log_diffusivity = nn.Parameter(torch.tensor(math.log(initial_diffusivity), dtype=torch.float64))

    @property
    def diffusivity(self):
        return torch.exp(self.log_diffusivity)

    def forward(self, coordinates):
        x, t = coordinates[:, :1], coordinates[:, 1:]
        initial = torch.sin(torch.pi*x)+.3*torch.sin(2*torch.pi*x)
        # Hard initial/boundary conditions, not a large penalty coefficient.
        return initial + t*x*(1-x)*self.network(2*coordinates-1)

    def residual(self, coordinates):
        if not coordinates.requires_grad:
            raise ValueError("collocation coordinates must require gradients")
        u = self(coordinates)
        first = torch.autograd.grad(u, coordinates, torch.ones_like(u), create_graph=True)[0]
        second = torch.autograd.grad(first[:, :1], coordinates,
                                    torch.ones_like(first[:, :1]), create_graph=True)[0][:, :1]
        return first[:, 1:] - self.diffusivity*second


def fit(observation_coordinates, observation_values, *, seed, epochs=1800,
        initial_diffusivity=.25, physics=True, lbfgs_steps=150):
    coordinates, values = np.asarray(observation_coordinates), np.asarray(observation_values)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or values.shape != (len(coordinates), 1):
        raise ValueError("expected observation coordinates [N,2] and values [N,1]")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ValueError("non-finite sparse observations")
    if np.any((coordinates < 0) | (coordinates > 1)) or epochs < 1 or lbfgs_steps < 0:
        raise ValueError("invalid domain or training budget")
    if len(coordinates) < 3:
        raise ValueError("at least three observations required for this exercise")
    history = []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = HeatInversePINN(initial_diffusivity)
            observed_x = torch.from_numpy(coordinates.astype(np.float64))
            observed_y = torch.from_numpy(values.astype(np.float64))
            collocation = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(128).double().requires_grad_(True)
            parameters = list(model.parameters()) if physics else list(model.network.parameters())
            optimizer = torch.optim.Adam(parameters, lr=.003)

            def losses():
                data = torch.mean((model(observed_x)-observed_y)**2)
                pde = torch.mean(model.residual(collocation)**2) if physics else data.new_tensor(0.)
                return 10*data+pde, data, pde

            for epoch in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                collocation.grad = None
                total, data, pde = losses()
                if not torch.isfinite(total):
                    raise RuntimeError("non-finite inverse PINN loss")
                total.backward()
                nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
                optimizer.step()
                if epoch % 100 == 0 or epoch == epochs-1:
                    history.append({"phase": "adam", "step": epoch+1,
                        "total": float(total.detach()), "data": float(data.detach()),
                        "pde": float(pde.detach()),
                        "diffusivity": float(model.diffusivity.detach()) if physics else None})
            evaluations = 0
            if lbfgs_steps:
                optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=lbfgs_steps,
                    tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

                def closure():
                    nonlocal evaluations
                    optimizer.zero_grad(set_to_none=True)
                    collocation.grad = None
                    total, _, _ = losses()
                    if not torch.isfinite(total):
                        raise RuntimeError("non-finite L-BFGS loss")
                    total.backward()
                    evaluations += 1
                    return total

                optimizer.step(closure)
            total, data, pde = losses()
            history.append({"phase": "final", "total": float(total.detach()),
                "data": float(data.detach()), "pde": float(pde.detach()),
                "diffusivity": float(model.diffusivity.detach()) if physics else None})
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("non-finite fitted parameter")
            model.eval()
    finally:
        torch.set_num_threads(old_threads)
    return model, {"seed": seed, "physics_loss_enabled": physics,
        "adam_epochs": epochs, "lbfgs_max_iterations": lbfgs_steps,
        "lbfgs_closure_evaluations": evaluations, "collocation_count": 128,
        "data_weight": 10., "pde_weight": 1. if physics else 0.,
        "architecture": "2-24-24-1 Tanh, hard IC/BC, positive exp(log_a)",
        "loss_history": history}


def predict(model, coordinates):
    with torch.no_grad():
        return model(torch.as_tensor(coordinates, dtype=torch.float64)).numpy()


def run(out, epochs=1800):
    if out.exists():
        raise FileExistsError("use a fresh PINN output directory")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    true_a = .12
    # Fix before observing results: three initializations/noise realizations.
    specifications = [(410, .05, 0.), (411, .25, .003), (412, .4, .003)]
    ox, ot = np.meshgrid([.2,.4,.6,.8], [.2,.5,.8])
    observed = np.column_stack((ox.ravel(), ot.ravel()))
    gx, gt = np.meshgrid(np.linspace(0,1,51), np.linspace(0,1,51))
    grid = np.column_stack((gx.ravel(), gt.ravel()))
    truth = exact_temperature(grid[:,0], grid[:,1], true_a)[:,None]
    protocol = {"learning_only": True, "dataset": "manufactured_heat_equation_NOT_SMA_data",
        "equation": "u_t=a*u_xx", "domain": "dimensionless x,t in [0,1]",
        "initial_condition": "sin(pi*x)+0.3*sin(2*pi*x)", "boundary": "u(0,t)=u(1,t)=0",
        "true_diffusivity_for_evaluation_only": true_a, "sparse_observation_count": len(observed),
        "initializations": specifications, "adam_epochs": epochs, "lbfgs_max_iterations": 150,
        "implementation_sha256": sha(Path(__file__)),
        "evaluation_policy": "Fixed parameters and observations; no dense labels in neural fit; no test tuning",
        "acceptance": {"clean_relative_parameter_error_max": .1,
                       "clean_dense_field_rmse_max": .03},
        "scope": "Algorithm/identifiability teaching test, not inferred SMA material diffusivity"}
    write_json(out/"protocol.json", protocol)
    results, saved = [], {"grid": grid, "truth": truth, "observation_coordinates": observed}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed, initial, noise in specifications:
            rng = np.random.default_rng(seed)
            observation = exact_temperature(observed[:,0], observed[:,1], true_a)[:,None]
            observation = observation + rng.normal(0, noise, observation.shape)
            # Classical reference knows the closed-form family; report that advantage.
            reference = least_squares(lambda p: exact_temperature(observed[:,0], observed[:,1], np.exp(p[0]))-observation[:,0],
                                      [math.log(initial)])
            reference_a = float(np.exp(reference.x[0]))
            for physics in (True, False):
                tag = f"s{seed}_{'pinn' if physics else 'data_only'}"
                model, training = fit(observed, observation, seed=seed, epochs=epochs,
                    initial_diffusivity=initial, physics=physics)
                prediction = predict(model, grid)
                residual = model.residual(torch.from_numpy(grid[::7]).requires_grad_(True)).detach().numpy()
                identified = float(model.diffusivity.detach()) if physics else None
                checkpoint = out/(tag+".pt")
                torch.save({"state_dict": model.state_dict(), "hidden": model.hidden,
                            "training": training, "physics_enabled": physics}, checkpoint)
                restored = HeatInversePINN(initial)
                restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"])
                if not np.array_equal(prediction, predict(restored, grid)):
                    raise RuntimeError("PINN checkpoint round trip mismatch")
                sensitivity = analytic_sensitivity(observed[:,0], observed[:,1], true_a)
                record = {"tag": tag, "seed": seed, "observation_noise_sd": noise,
                    "initial_diffusivity": initial, "identified_diffusivity": identified,
                    "relative_parameter_error": abs(identified-true_a)/true_a if physics else None,
                    "classical_known_solution_diffusivity": reference_a,
                    "classical_baseline_caveat": "uses closed-form solution family unavailable in a general inverse PDE",
                    "dense_field_rmse": float(np.sqrt(np.mean((prediction-truth)**2))),
                    "dense_field_max_abs": float(np.max(np.abs(prediction-truth))),
                    "sparse_observation_rmse": float(np.sqrt(np.mean((predict(model, observed)-observation)**2))),
                    "pde_residual_rms": float(np.sqrt(np.mean(residual**2))),
                    "pde_evaluation_diffusivity": float(model.diffusivity.detach()),
                    "parameter_identifiable_from_data_only_network": False,
                    "analytic_observation_sensitivity_norm": float(np.linalg.norm(sensitivity)),
                    "training": training, "checkpoint_sha256": sha(checkpoint), "roundtrip_exact": True}
                results.append(record)
                write_json(out/(tag+".json"), record)
                saved[tag+"_prediction"], saved[tag+"_observations"] = prediction, observation
    finally:
        torch.set_num_threads(previous_threads)
    np.savez_compressed(out/"predictions.npz", **saved)
    # Observing only boundary points gives zero sensitivity -> no parameter information.
    boundary_sensitivity = analytic_sensitivity(np.zeros(3), np.array([.2,.5,.8]), true_a)
    clean = next(r for r in results if r["tag"] == "s410_pinn")
    summary = {"status": "completed", "learning_only": True, "manufactured_not_SMA": True,
        "clean_acceptance_passed": clean["relative_parameter_error"] < .1 and clean["dense_field_rmse"] < .03,
        "boundary_only_identifiability": {"sensitivity_norm": float(np.linalg.norm(boundary_sensitivity)),
                                        "identifiable": False},
        "runs": [{k:r[k] for k in ("tag", "identified_diffusivity", "relative_parameter_error",
                                    "dense_field_rmse", "pde_residual_rms")} for r in results],
        "elapsed_seconds": time.monotonic()-start,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1800)
    args = parser.parse_args()
    report = run(args.output_root, args.epochs)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["clean_acceptance_passed"]:
        raise SystemExit(2)
