"""Bounded, time-monotone phase reconstruction for first loading."""

from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression
from sklearn.linear_model import Ridge

from SciML.models.v12.learning_coordinate_operator import interpolate


def project_phase_anchors(phase):
    """Least-squares projection onto [0, 1] and nondecreasing time, at each radius."""
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 3 or phase.shape[1] < 2 or not np.isfinite(phase).all():
        raise ValueError("Expected finite phase values with case, time and radius axes")
    # The known first frame is fixed at zero. Bounded PAVA on later times then
    # enforces the initial lower bound and the remaining temporal inequalities.
    trajectories = np.moveaxis(phase[:, 1:], 1, -1)
    rows = trajectories.reshape(-1, trajectories.shape[-1])
    projected = np.empty_like(rows)
    for i, row in enumerate(rows):
        projected[i] = np.clip(isotonic_regression(row).x, 0.0, 1.0)
    result = np.zeros_like(phase)
    result[:, 1:] = np.moveaxis(projected.reshape(trajectories.shape), -1, 1)
    return result


def resample_phase(phase, anchor_times, anchor_radii, times, radii, *, cubic_time=False):
    """Resample in time, then use linear interpolation between radial anchors."""
    if cubic_time:
        by_time = PchipInterpolator(anchor_times, phase, axis=1, extrapolate=False)(times)
        if not np.isfinite(by_time).all():
            raise ValueError("Phase query outside loading time support")
        by_time = torch.from_numpy(by_time)
    else:
        by_time = interpolate(torch.from_numpy(phase).transpose(-1, -2),
                              torch.from_numpy(anchor_times), torch.from_numpy(times)).transpose(-1, -2)
    return interpolate(by_time, torch.from_numpy(anchor_radii), torch.from_numpy(radii)).numpy()


def constrained_phase_fields(phase_anchors, anchor_times, anchor_radii, times, radii):
    """Apply common bounded anchors with linear or shape-preserving cubic time reconstruction."""
    bounded = project_phase_anchors(phase_anchors)
    return {
        "monotone_linear": np.clip(resample_phase(bounded, anchor_times, anchor_radii, times, radii), 0.0, 1.0),
        "monotone_cubic": np.clip(resample_phase(bounded, anchor_times, anchor_radii, times, radii, cubic_time=True), 0.0, 1.0),
    }


def material_pressure_design(controls, pod_parameters, pressure_centers):
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 4 or not np.isfinite(controls).all():
        raise ValueError("Expected finite radius, thickness, peak pressure and material controls")
    radius, thickness, peak_pressure, material = controls.T
    mean, scale = pod_parameters["control_mean"], pod_parameters["control_scale"]
    if (not np.isin(material, (0.0, 1.0)).all() or np.any(thickness <= 0)
            or not np.allclose(radius, (mean[0] / mean[1]) * thickness, rtol=1e-12, atol=0)):
        raise ValueError("The phase fit requires the archived fixed aspect ratio and its two material codes")
    radius_scaled = np.sqrt(2) * (radius - mean[0]) / scale[0]
    pressure_scaled = (peak_pressure - pressure_centers[material.astype(int)]) / scale[2]
    material_scaled = (material - mean[3]) / scale[3]
    return np.column_stack((radius_scaled, pressure_scaled, material_scaled, pressure_scaled * (2 * material - 1)))


class MaterialPressurePhaseRidge:
    """Loading POD with material-dependent pressure slopes in the phase coefficients."""

    def __init__(self, pod, pressure_centers, coefficients, intercept):
        self.pod = pod
        self.pressure_centers = pressure_centers
        self.coefficients = coefficients
        self.intercept = intercept

    @classmethod
    def fit(cls, pod, train_controls, normalized_phase):
        p = pod.parameters
        material = train_controls[:, 3]
        if set(material) != {0.0, 1.0}:
            raise ValueError("Both material codes must be present in training")
        centers = np.array([train_controls[material == value, 2].mean() for value in (0.0, 1.0)])
        design = material_pressure_design(train_controls, p, centers)
        flat = normalized_phase.reshape(len(train_controls), -1)
        modal_targets = (flat - p["mean_change"][3]) @ p["basis"][3].T
        ridge = Ridge(alpha=1.0).fit(design, modal_targets)
        return cls(pod, centers, ridge.coef_, ridge.intercept_)

    def phase_anchors(self, controls):
        p = self.pod.parameters
        design = material_pressure_design(controls, p, self.pressure_centers)
        modes = design @ self.coefficients.T + self.intercept
        change = modes @ p["basis"][3] + p["mean_change"][3]
        phase = change.reshape(len(controls), len(p["coordinates_time"]), len(p["coordinates_radius"])) * p["field_scale"][3]
        phase[:, 0] = 0.0
        return phase

    def predict(self, controls, times, radii):
        p = self.pod.parameters
        result = self.pod.predict(controls, times, radii)
        anchors = project_phase_anchors(self.phase_anchors(controls))
        phase = resample_phase(anchors, p["coordinates_time"], p["coordinates_radius"], times, radii)
        result[:, 3] = np.clip(phase, 0.0, 1.0)
        return result

    def save(self, path):
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, pressure_centers=self.pressure_centers, coefficients=self.coefficients,
                                intercept=self.intercept)

    @classmethod
    def load(cls, path, pod):
        with np.load(path, allow_pickle=False) as state:
            return cls(pod, state["pressure_centers"], state["coefficients"], state["intercept"])
