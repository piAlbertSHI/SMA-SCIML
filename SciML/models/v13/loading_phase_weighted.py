"""Near-saturation weighted phase readout built on the frozen loading POD basis."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .loading_phase_reconstruction import (
    MaterialPressurePhaseRidge,
    material_pressure_design,
    project_phase_anchors,
    resample_phase,
)


class SaturationWeightedPhaseRidge:
    """A recAs-gated phase head with a near-saturation weighted field fit.

    The eight POD modes and the material-conditioned baseline are frozen. The
    weighted head fits its modal readout through the decoded phase field, so
    points above the training-only saturation threshold receive additional
    influence. A gate applies that readout only where the baseline is already
    in its upper transition band. Mechanical and thermal channels remain the
    baseline fields.
    """

    SCHEMA = "first_loading_phase_saturation_weighted_v1"

    def __init__(self, baseline, coefficient, saturation_threshold, weight_strength,
                 blend_start, blend_end):
        self.baseline = baseline
        self.pod = baseline.pod
        self.coefficient = np.asarray(coefficient, dtype=np.float64)
        self.saturation_threshold = float(saturation_threshold)
        self.weight_strength = float(weight_strength)
        self.blend_start = float(blend_start)
        self.blend_end = float(blend_end)
        p = self.pod.parameters
        expected = (len(p["control_mean"]) + 1, len(p["basis"][3]))
        if self.coefficient.shape != expected:
            raise ValueError(f"Expected weighted coefficient shape {expected}, got {self.coefficient.shape}")
        if not 0 < self.saturation_threshold < 1 or not 0 <= self.blend_start < self.blend_end <= 1:
            raise ValueError("Invalid saturation or blend bounds")

    @classmethod
    def fit(cls, baseline, train_controls, normalized_phase, saturation_threshold=.8,
            weight_strength=100.0, blend_start=.8, blend_end=.88):
        p = baseline.pod.parameters
        controls = np.asarray(train_controls, dtype=np.float64)
        phase = np.asarray(normalized_phase, dtype=np.float64) * p["field_scale"][3]
        if controls.ndim != 2 or controls.shape[1] != 4 or phase.ndim != 3 or len(controls) != len(phase):
            raise ValueError("Controls and phase anchors have incompatible shapes")
        if phase.shape[1:] != (len(p["coordinates_time"]), len(p["coordinates_radius"])):
            raise ValueError("Phase anchors do not match the frozen POD grid")
        design = material_pressure_design(controls, p, baseline.pressure_centers)
        design = np.column_stack((design, np.ones(len(design))))
        basis = p["basis"][3]
        centered = phase.reshape(len(phase), -1) / p["field_scale"][3] - p["mean_change"][3]
        rows = np.einsum("ip,kj->ijpk", design, basis).reshape(len(phase) * centered.shape[1], -1)
        weights = np.ones_like(phase)
        weights[:, 1:] += float(weight_strength) * (phase[:, 1:] >= float(saturation_threshold))
        weights = weights.reshape(-1)
        # The first frame remains in the regression with unit weight. The
        # decoder still hard-fixes it to the declared initial phase.
        target = centered.reshape(-1)
        if not np.isfinite(rows).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite weighted phase regression inputs")
        ridge = Ridge(alpha=1.0, fit_intercept=False).fit(rows, target, sample_weight=weights)
        coefficient = ridge.coef_.reshape(design.shape[1], basis.shape[0])
        return cls(baseline, coefficient, saturation_threshold, weight_strength, blend_start, blend_end)

    def weighted_phase_anchors(self, controls):
        p = self.pod.parameters
        design = material_pressure_design(controls, p, self.baseline.pressure_centers)
        augmented = np.column_stack((design, np.ones(len(design))))
        modes = augmented @ self.coefficient
        centered = modes @ p["basis"][3] + p["mean_change"][3]
        phase = centered.reshape(len(controls), len(p["coordinates_time"]),
                                len(p["coordinates_radius"])) * p["field_scale"][3]
        phase[:, 0] = 0.0
        return phase

    def phase(self, controls, times, radii):
        controls = np.asarray(controls, dtype=np.float64)
        baseline = self.baseline.predict(controls, times, radii)
        weighted_anchors = project_phase_anchors(self.weighted_phase_anchors(controls))
        p = self.pod.parameters
        weighted = resample_phase(weighted_anchors, p["coordinates_time"],
                                  p["coordinates_radius"], times, radii)
        base_phase = baseline[:, 3]
        gate = np.clip((base_phase - self.blend_start) / (self.blend_end - self.blend_start), 0.0, 1.0)
        use_weighted = controls[:, 3] == 1.0
        blended = base_phase.copy()
        blended[use_weighted] = ((1.0 - gate[use_weighted]) * base_phase[use_weighted]
                                 + gate[use_weighted] * weighted[use_weighted])
        return project_phase_anchors(np.clip(blended, 0.0, 1.0))

    def predict(self, controls, times, radii):
        result = self.baseline.predict(controls, times, radii)
        result[:, 3] = self.phase(controls, times, radii)
        return result

    def save(self, path):
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(self.SCHEMA),
                coefficient=self.coefficient,
                saturation_threshold=np.asarray(self.saturation_threshold),
                weight_strength=np.asarray(self.weight_strength),
                blend_start=np.asarray(self.blend_start),
                blend_end=np.asarray(self.blend_end),
            )

    @classmethod
    def load(cls, path, baseline):
        with np.load(path, allow_pickle=False) as state:
            if str(state["schema"]) != cls.SCHEMA:
                raise ValueError("Invalid saturation-weighted phase checkpoint")
            return cls(baseline, state["coefficient"], float(state["saturation_threshold"]),
                       float(state["weight_strength"]), float(state["blend_start"]),
                       float(state["blend_end"]))
