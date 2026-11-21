"""Render source-linked field comparisons for the paired audit and the loading benchmark."""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.loading_field_benchmark import PAIR_RESULT, ROOT
from examples.verify_loading_field_benchmark import BENCHMARK
from tools.audit_project_corpora import read_json, sha256
from tools.compare_solver_archives import first_cycles, write


def archive_fields(solver_repo, output):
    summary = read_json(PAIR_RESULT)
    case = summary["stages"]["stage4_cooldown_exchange"]["htf_K"]["worst_future_case_id"]
    pairs = read_json(PAIR_RESULT.parent / "pairs.json")
    row = next(r for r in pairs if r["case_id"] == case)
    sources = [solver_repo / "SciML/datasets" / name for name in summary["old_new"]]
    fields = []
    for source, expected in zip(sources, summary["dataset_manifest_sha256"]):
        if sha256(source / "artifact_manifest.json") != expected:
            raise ValueError("Changed paired data source")
        _, cases = first_cycles(source)
        cycle = cases[row["condition_id"]][1]["hdf5_cycle_group"]
        with h5py.File(source / "fields.h5", "r") as handle:
            group = handle[cycle + "/stage4_cooldown_exchange"]
            fields.append([group[key][:] for key in ("sma_strip_T_K_t", "htf_internal_T_K_t")])
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), layout="constrained")
    extent = (0, row["exchange_duration_s"], 0, 1)
    for i, axis_name in enumerate(("SMA strip position / normalized", "Fluid pipe index / normalized")):
        old, new = fields[0][i], fields[1][i]
        bounds = {"vmin": min(old.min(), new.min()), "vmax": max(old.max(), new.max())}
        for j, values in enumerate((old, new)):
            image = axes[i, j].imshow(values.T, origin="lower", aspect="auto", extent=extent,
                                      interpolation="nearest", cmap="viridis", **bounds)
        fig.colorbar(image, ax=axes[i, :2], label="Temperature / K", shrink=.9)
        delta = new - old
        amplitude = np.max(np.abs(delta))
        image = axes[i, 2].imshow(delta.T, origin="lower", aspect="auto", extent=extent,
                                  interpolation="nearest", cmap="RdBu_r", vmin=-amplitude, vmax=amplitude)
        fig.colorbar(image, ax=axes[i, 2], label="Difference / K", shrink=.9)
        axes[i, 0].set_ylabel(axis_name)
        for ax in axes[i]:
            ax.set_xlabel("Local time / s")
        axes[i, 2].set_title(f"Revised − historical\nFuture RMS {np.sqrt(np.mean(delta[1:]**2)):.3f} K")
    axes[0, 0].set_title("Historical archive")
    axes[0, 1].set_title("Revised archive")
    fig.suptitle(f"Cooling exchange | largest fluid-field change among 54 pairs\n"
                 f"{row['material']} · R = {1e3 * row['radius_m']:.0f} mm · P = {1e-6 * row['pressure_Pa']:.3f} MPa")
    fig.savefig(output / "cooling_version_difference.png", dpi=150)
    plt.close(fig)
    return case


def loading_fields(output):
    with np.load(BENCHMARK / "loading_data.npz", allow_pickle=False) as data:
        test = data["split"] == "test"
        use = np.flatnonzero(test)[0]
        case = str(data["case_ids"][use])
        controls = data["controls"][use]
        truth = data["fields"][use]
    methods = {"POD/Ridge": "pod_ridge", "Coordinate MLP": "mlp", "DeepONet": "deeponet", "Radial Fourier": "fourier"}
    errors = {}
    scores = {}
    with np.load(BENCHMARK / "test_predictions.npz", allow_pickle=False) as data:
        for method, key in methods.items():
            related = [name for name in data.files if name == key or name.startswith(key + "_")]
            pred = np.stack([data[name][0] for name in related])
            square = (pred - truth) ** 2
            errors[method] = np.sqrt(square.mean(axis=0))
            scores[method] = np.sqrt(np.mean(square[:, :, 1:], axis=(2, 3))).mean(axis=0)
    fig, axes = plt.subplots(4, 5, figsize=(14.4, 10), layout="constrained")
    labels = ("Radial displacement / mm", "Deflection / mm", "SMA temperature / K", "Martensite fraction")
    extent = (0, .1, 0, 1)
    for i, label in enumerate(labels):
        image = axes[i, 0].imshow(truth[i].T, origin="lower", aspect="auto", extent=extent,
                                  interpolation="nearest", cmap="viridis")
        fig.colorbar(image, ax=axes[i, 0], shrink=.88, label=label)
        axes[i, 0].set_ylabel("Normalized radius")
        limit = max(array[i].max() for array in errors.values())
        for j, (method, values) in enumerate(errors.items(), 1):
            image = axes[i, j].imshow(values[i].T, origin="lower", aspect="auto", extent=extent,
                                      interpolation="nearest", cmap="magma", vmin=0, vmax=limit)
            axes[i, j].set_title(f"{method}\nRMSE {scores[method][i]:.4f}", fontsize=10)
        fig.colorbar(image, ax=axes[i, 1:], shrink=.88, label="RMS error")
        for ax in axes[i]:
            ax.set_xlabel("Loading time / s")
    axes[0, 0].set_title("Numerical reference", fontsize=10)
    material = "recAs" if controls[3] else "recA"
    fig.suptitle(f"Loading-stage radial fields | first held-out condition\n"
                 f"{material} · R = {1e3 * controls[0]:.0f} mm · P = {1e-6 * controls[2]:.3f} MPa\n"
                 "Neural maps show pointwise RMS across the three fixed seeds", fontsize=13)
    fig.savefig(output / "loading_benchmark_fields.png", dpi=150)
    plt.close(fig)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    paired_case = archive_fields(args.solver_repo, args.output)
    loading_case = loading_fields(args.output)
    paths = (PAIR_RESULT, PAIR_RESULT.parent / "pairs.json", BENCHMARK / "loading_data.npz",
             BENCHMARK / "test_predictions.npz", Path(__file__).resolve())
    write(args.output / "manifest.json", {
        "showcase_selection": {"largest_paired_cooling_htf_rms": paired_case, "first_sorted_loading_test": loading_case},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths},
        "artifact_sha256": {path.name: sha256(path) for path in sorted(args.output.glob("*.png"))},
    })
    print(json.dumps({"status": "rendered_paired_and_loading_fields", "figures": 2}))


if __name__ == "__main__":
    main()
