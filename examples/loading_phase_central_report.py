"""Render the recAs central-radial saturation-arrival correction."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.loading_field_benchmark import RADIUS
from examples.loading_phase_study import arrival_times


def draw_arrival(path, case_ids, controls, truth, baseline, central):
    select = controls[:, 3] == 1.0
    controls, truth = controls[select], truth[select]
    baseline, central = baseline[select], central[select]
    expected = arrival_times(truth, .9)
    base_arrival, central_arrival = arrival_times(baseline, .9), arrival_times(central, .9)
    base_error = np.abs(base_arrival - expected)
    central_error = np.abs(central_arrival - expected)
    reachable = np.isfinite(expected)
    common = reachable & np.isfinite(base_arrival) & np.isfinite(central_arrival)
    base_error[~np.isfinite(base_error)] = np.nan
    central_error[~np.isfinite(central_error)] = np.nan
    improvement = base_error - central_error
    improvement[~common] = np.nan
    vmax = max(float(np.nanmax(base_error)), float(np.nanmax(central_error)), 1e-9)
    gain = max(float(np.nanmax(np.abs(improvement))), 1e-9)
    labels = [f"{i + 1}: R{controls[i, 0] * 1000:.0f} P{controls[i, 2] / 1e6:.3f}"
              for i in range(len(controls))]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), layout="constrained")
    fig.suptitle("recAs loading | central-radial correction at the 0.9 phase threshold", fontsize=15)
    for ax, values, title in ((axes[0], base_error, "Selected saturation readout"),
                              (axes[1], central_error, "Central-radial correction")):
        image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
                          extent=(RADIUS[0], RADIUS[-1], -.5, len(controls) - .5),
                          cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Normalized radius r/R")
        ax.set_yticks(np.arange(len(controls)), labels if ax is axes[0] else [])
        ax.set_ylabel("Test case" if ax is axes[0] else "")
        missed = reachable & ~np.isfinite(values)
        case, radial = np.nonzero(missed)
        ax.scatter(RADIUS[radial], case, marker="x", color="#5be7ff", s=24, linewidths=1.4,
                   label="reference reached, prediction missed")
        if len(case):
            ax.legend(loc="upper left", fontsize=7.2, framealpha=.9)
        fig.colorbar(image, ax=ax, label="Absolute arrival error (ms)")
    image = axes[2].imshow(improvement, origin="lower", aspect="auto", interpolation="nearest",
                           extent=(RADIUS[0], RADIUS[-1], -.5, len(controls) - .5),
                           cmap="RdBu", vmin=-gain, vmax=gain)
    axes[2].set_title("Error reduction: selected − central")
    axes[2].set_xlabel("Normalized radius r/R")
    axes[2].set_yticks(np.arange(len(controls)), [])
    missed = reachable & ~np.isfinite(central_arrival)
    case, radial = np.nonzero(missed)
    axes[2].scatter(RADIUS[radial], case, marker="x", color="#151515", s=24, linewidths=1.4,
                    label="central prediction missed")
    if len(case):
        axes[2].legend(loc="upper left", fontsize=7.2, framealpha=.9)
    fig.colorbar(image, ax=axes[2], label="Reduction in error (ms)")
    for ax in axes:
        ax.set_ylim(-.5, len(controls) - .5)
    fig.savefig(path, dpi=155)
    plt.close(fig)
