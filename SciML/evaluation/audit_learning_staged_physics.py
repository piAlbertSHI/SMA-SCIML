"""Fresh refits, independent physical-unit metrics and finite-difference audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit_learning_wp5 import verify_artifacts
from SciML.models.v12 import learning_staged_physics as core
from SciML.models.v12.run_learning_existing import sha, write_json


def finite_difference(model):
    # All points strictly interior. An independent numerical derivative can
    # detect a swapped coordinate or a missing chain-rule coefficient.
    coordinates = np.tile(np.array([[.23, .31], [.61, .73]]), (8, 1))
    blocks = np.arange(8).repeat(2)
    def value(c):
        with torch.no_grad():
            return model(torch.tensor(c), torch.tensor(blocks)).numpy()[:, 0]
    eps_x, eps_t = 1e-4, 1e-5
    dx = np.array([eps_x, 0.])
    dt = np.array([0., eps_t])
    second = (value(coordinates+dx)-2*value(coordinates)+value(coordinates-dx))/eps_x**2
    rate = (value(coordinates+dt)-value(coordinates-dt))/(2*eps_t)
    numerical = rate-core.FO[blocks % 4]*second
    analytic = core.pde_residual(model, torch.tensor(coordinates, requires_grad=True), torch.tensor(blocks)).detach().numpy()[:, 0]
    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=2e-8)
    # Storage derivative via centered finite differences of integrated states;
    # edge gradients via one-sided second-order differences, no autograd here.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    nodes, weights = (nodes+1)/2, weights/2
    times = np.array([.17, .53, .87])
    defects = []
    for block in range(8):
        by_time = []
        def v(x, t):
            c = np.column_stack((x, np.full(len(x), t)))
            with torch.no_grad():
                return model(torch.tensor(c), torch.full((len(x),), block, dtype=torch.long)).numpy()[:, 0]
        for time in times:
            storage = np.dot(weights, v(nodes, time+eps_t)-v(nodes, time-eps_t))/(2*eps_t)
            left = v(np.array([0., eps_x, 2*eps_x]), time)
            right = v(np.array([1., 1-eps_x, 1-2*eps_x]), time)
            dl = (-3*left[0]+4*left[1]-left[2])/(2*eps_x)
            dr = (3*right[0]-4*right[1]+right[2])/(2*eps_x)
            by_time.append(storage-core.FO[block % 4]*(dr-dl))
        defects.append(by_time)
    analytic_energy = core.energy_residual(model, torch.tensor(times), 24).detach().numpy()
    np.testing.assert_allclose(analytic_energy, defects, rtol=2e-5, atol=5e-8)
    return {"pde_max_difference_normalized": float(np.max(np.abs(analytic-numerical))),
            "energy_max_difference_normalized": float(np.max(np.abs(analytic_energy-defects))),
            "space_step": eps_x, "time_step": eps_t}


def run(root, out):
    if out.exists():
        raise FileExistsError("preserve staged physics audit")
    summary = verify_artifacts(root)
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["implementation_sha256"] != sha(Path(core.__file__)):
        raise ValueError("staged implementation changed")
    obs = np.load(root/"observations.npz", allow_pickle=False)
    checks, reports = [], []
    old = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in protocol["seeds"]:
            for variant in core.VARIANTS:
                dest = root/f"s{seed}-{variant}"
                arrays = np.load(dest/"evaluation.npz", allow_pickle=False)
                report = json.loads((dest/"report.json").read_text())
                # Evaluation has no coordinate/block pair coincident with a label.
                for c, b in zip(obs["coordinates"], obs["blocks"]):
                    if np.any((arrays["blocks"] == b) & np.all(arrays["coordinates"] == c, axis=1)):
                        raise ValueError("observed coordinate included in unobserved metric")
                fresh, history = core.fit(obs["coordinates"], obs["blocks"], obs["labels"], seed=seed, variant=variant, steps=protocol["steps"])
                if history != report["history"]:
                    raise ValueError("fresh loss trajectory differs")
                checkpoint = torch.load(dest/"model.pt", weights_only=True)
                for key, parameter in fresh.state_dict().items():
                    np.testing.assert_array_equal(parameter, checkpoint[key])
                _, replay = core.evaluate(fresh)
                for key in arrays.files:
                    np.testing.assert_array_equal(replay[key], arrays[key])
                duration = np.tile(protocol["duration_s"], 2)
                scale = protocol["temperature_scale_K"]
                metrics = {"unobserved_field_rmse_K": float(np.sqrt(np.mean((arrays["prediction_normalized"]-arrays["truth_normalized"])**2))*scale),
                    "pde_rms_K_per_s": float(np.sqrt(np.mean((arrays["pde_residual_normalized"]*scale/np.repeat(duration, 37))**2))),
                    "stage_jump_rms_K": float(np.sqrt(np.mean(arrays["stage_jump_normalized"]**2))*scale),
                    "cycle_jump_rms_K": float(np.sqrt(np.mean(arrays["cycle_jump_normalized"]**2))*scale),
                    "energy_rate_residual_rms_W": float(np.sqrt(np.mean((arrays["energy_residual_normalized"]*protocol["energy_scale_J"]/duration[:, None])**2)))}
                if metrics != report["metrics"]:
                    raise ValueError("independent dimensional metric mismatch")
                checks.append({"seed": seed, "variant": variant, **finite_difference(fresh)})
                reports.append(report)
    finally:
        torch.set_num_threads(old)
    for variant in core.VARIANTS:
        for metric in reports[0]["metrics"]:
            values = [r["metrics"][metric] for r in reports if r["variant"] == variant]
            observed = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
            if observed != summary["aggregate"][variant][metric]:
                raise ValueError("aggregate mean/SD mismatch")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"finite_difference_checks.json", checks)
    figure(summary["aggregate"], out)
    result = {"status": "passed_manufactured_staged_physics_audit", "learning_only": True,
        "fresh_solver_calls": 0, "fresh_fits_exact": len(reports), "dimensional_metrics_recomputed": 5*len(reports),
        "all_evaluation_coordinates_unobserved": True, "finite_difference_models_checked": len(checks),
        "max_pde_finite_difference_error": max(c["pde_max_difference_normalized"] for c in checks),
        "max_energy_finite_difference_error": max(c["energy_max_difference_normalized"] for c in checks),
        "scope": "manufactured fixed PDE loss comparison; neither SMA validation nor autonomous multicycle rollout",
        "source_summary_sha256": sha(root/"summary.json"), "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", result)
    return result


def figure(values, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title, unit in zip(axes,
        ("unobserved_field_rmse_K", "cycle_jump_rms_K", "energy_rate_residual_rms_W"),
        ("Unobserved temperature", "Cycle interface jump", "Energy-rate defect"), ("K", "K", "W")):
        mean = [values[v][metric]["mean"] for v in core.VARIANTS]
        sd = [values[v][metric]["std"] for v in core.VARIANTS]
        ax.bar(np.arange(4), mean, yerr=sd, capsize=4, color=["#777777", "#3b7fb6", "#dc8b35", "#3b9975"])
        ax.set(xticks=np.arange(4), xticklabels=["data", "+ PDE", "+ interfaces", "+ energy"], ylabel=unit, title=title, yscale="log")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Manufactured heat benchmark | 3-seed mean +/- SD | hard BC/first IC in all variants")
    fig.savefig(out/"staged_physics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment_root, args.output_root), indent=2))
