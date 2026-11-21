"""Render the location of phase-bound violations and their relation to the transformation front."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from examples.loading_field_benchmark import RADIUS, TIME


METHOD_LABELS = {
    "clipped": "Direct clipping", "monotone_linear": "Monotone linear",
    "monotone_cubic": "Monotone cubic", "material_pressure_monotone": "Material-pressure Ridge + monotone",
}


def draw(path, controls, reference, phases, band, selected, tolerance):
    raw = phases["raw"]
    below, above = raw < -tolerance, raw > 1.0 + tolerance
    worst_index = int(np.argmax(below.sum(axis=(1, 2))))
    r_mm = 1000.0 * controls[worst_index, 0]
    pressure = controls[worst_index, 2] / 1e6
    material = "recAs" if controls[worst_index, 3] else "recA"
    time_ms = 100 * TIME
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.7), layout="constrained")
    fig.suptitle("First loading | phase bounds and transformation front", fontsize=16)

    for ax, mask, label, color in (
        (axes[0, 0], below, "Raw phase < 0", "Purples"),
        (axes[0, 1], above, "Raw phase > 1", "Oranges"),
    ):
        count = mask.sum(axis=0)
        chart = ax.pcolormesh(time_ms, RADIUS, count.T, cmap=color, shading="nearest", vmin=0)
        fig.colorbar(chart, ax=ax, label="Cases per node")
        ax.set_title(f"{label} · {mask.sum()} points across 30 test cases")

    ax = axes[1, 0]
    chart = ax.pcolormesh(time_ms, RADIUS, reference[worst_index].T, cmap="viridis", shading="nearest", vmin=0, vmax=1)
    fig.colorbar(chart, ax=ax, label="Reference martensite fraction")
    ax.contour(time_ms, RADIUS, raw[worst_index].T, levels=[.5], colors=["#ef5b2d"], linewidths=2)
    ax.contour(time_ms, RADIUS, phases[selected][worst_index].T, levels=[.5], colors=["white"], linestyles="--", linewidths=2)
    ax.legend(handles=[Line2D([0], [0], color="#ef5b2d", lw=2, label="Original 0.5 contour"),
                       Line2D([0], [0], color="#ffffff", lw=2, ls="--", label="Selected 0.5 contour")],
              loc="lower left", framealpha=.85, facecolor="#303443", labelcolor="white", fontsize=8.5)
    ax.set_title(f"Most negative points · {material}, R = {r_mm:g} mm, P = {pressure:.3f} MPa")

    ax = axes[1, 1]
    difference = np.abs(phases["clipped"] - reference) - np.abs(phases[selected] - reference)
    count = band.sum(axis=0)
    gain = np.divide(np.sum(difference * band, axis=0), count, out=np.full(count.shape, np.nan), where=count > 0)
    limit = max(float(np.nanmax(np.abs(gain))), 1e-12)
    chart = ax.pcolormesh(time_ms, RADIUS, gain.T, cmap="RdBu", shading="nearest", vmin=-limit, vmax=limit)
    fig.colorbar(chart, ax=ax, label="Reduction in absolute phase error")
    ax.set_title(f"Reference transition band · {METHOD_LABELS[selected]} vs clipping", fontsize=9.5)

    for ax in axes.flat:
        ax.set(xlabel="Loading time (ms)", ylabel="Normalized radius r/R", xlim=(73, 100), ylim=(0, 1))
    fig.savefig(path, dpi=155)
    plt.close(fig)
