"""Render recAs saturation-arrival errors before and after weighted readout."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.loading_field_benchmark import RADIUS, TIME
from examples.loading_phase_study import arrival_times


def draw_arrival(path, case_ids, controls, truth, baseline, weighted):
    recas = controls[:, 3] == 1.0
    case_ids, controls = case_ids[recas], controls[recas]
    truth, baseline, weighted = truth[recas], baseline[recas], weighted[recas]
    expected = arrival_times(truth, .9)
    base_arrival = arrival_times(baseline, .9)
    weighted_arrival = arrival_times(weighted, .9)
    base_error = np.abs(base_arrival - expected)
    weighted_error = np.abs(weighted_arrival - expected)
    base_error[~np.isfinite(base_error)] = np.nan
    weighted_error[~np.isfinite(weighted_error)] = np.nan
    reachable = np.isfinite(expected)
    common = reachable & np.isfinite(base_arrival) & np.isfinite(weighted_arrival)
    improvement = base_error - weighted_error
    improvement[~common] = np.nan
    vmax = max(float(np.nanmax(base_error)), float(np.nanmax(weighted_error)), 1e-9)
    gain = max(float(np.nanmax(np.abs(improvement))), 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), layout="constrained")
    fig.suptitle("recAs loading | arrival of the 0.9 phase threshold", fontsize=16)
    labels = [f"{i + 1}: R{controls[i, 0] * 1000:.0f} P{controls[i, 2] / 1e6:.3f}"
              for i in range(len(case_ids))]
    for ax, values, title in (
        (axes[0], base_error, "Material-conditioned readout"),
        (axes[1], weighted_error, "Near-saturation weighted readout"),
    ):
        image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
                          extent=(RADIUS[0], RADIUS[-1], -.5, len(case_ids) - .5),
                          cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Normalized radius r/R")
        ax.set_yticks(np.arange(len(case_ids)), labels if ax is axes[0] else [])
        ax.set_ylabel("Test case" if ax is axes[0] else "")
        missed = reachable & ~np.isfinite(values)
        case, radial = np.nonzero(missed)
        ax.scatter(RADIUS[radial], case, marker="x", color="#5be7ff", s=24, linewidths=1.4,
                   label="reference reached, prediction missed")
        if len(case):
            ax.legend(loc="upper left", fontsize=7.2, framealpha=.9)
        fig.colorbar(image, ax=ax, label="Absolute arrival error (ms)")
    image = axes[2].imshow(improvement, origin="lower", aspect="auto", interpolation="nearest",
                           extent=(RADIUS[0], RADIUS[-1], -.5, len(case_ids) - .5),
                           cmap="RdBu", vmin=-gain, vmax=gain)
    axes[2].set_title("Error reduction: baseline − weighted")
    axes[2].set_xlabel("Normalized radius r/R")
    axes[2].set_yticks(np.arange(len(case_ids)), [])
    axes[2].set_ylabel("")
    missed = reachable & ~np.isfinite(weighted_arrival)
    case, radial = np.nonzero(missed)
    axes[2].scatter(RADIUS[radial], case, marker="x", color="#151515", s=24, linewidths=1.4,
                    label="weighted prediction missed")
    if len(case):
        axes[2].legend(loc="upper left", fontsize=7.2, framealpha=.9)
    fig.colorbar(image, ax=axes[2], label="Reduction in error (ms)")
    for ax in axes:
        ax.set_ylim(-.5, len(case_ids) - .5)
    fig.savefig(path, dpi=155)
    plt.close(fig)
