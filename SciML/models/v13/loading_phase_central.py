"""Central-radial near-saturation readout for the recAs loading tail."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from .loading_phase_reconstruction import material_pressure_design, project_phase_anchors, resample_phase


class CentralSaturationPhaseRidge:
    """Add a smooth central-radial correction to the selected phase head."""

    SCHEMA = "first_loading_phase_central_saturation_v1"

    def __init__(self, baseline, coefficient, saturation_threshold, base_weight_strength,
                 central_weight_strength, central_radius, blend_start, blend_end, radial_smoothing):
        self.baseline = baseline
        self.pod = baseline.pod
        self.coefficient = np.asarray(coefficient, dtype=np.float64)
        self.saturation_threshold = float(saturation_threshold)
        self.base_weight_strength = float(base_weight_strength)
        self.central_weight_strength = float(central_weight_strength)
        self.central_radius = float(central_radius)
        self.blend_start = float(blend_start)
        self.blend_end = float(blend_end)
        self.radial_smoothing = float(radial_smoothing)
        expected = (len(self.pod.parameters["control_mean"]) + 1, len(self.pod.parameters["basis"][3]))
        if self.coefficient.shape != expected:
            raise ValueError(f"Expected central coefficient shape {expected}, got {self.coefficient.shape}")
        if not 0 < self.saturation_threshold < 1 or not 0 <= self.central_radius <= 1:
            raise ValueError("Invalid central saturation settings")
        if not 0 <= self.blend_start < self.blend_end <= 1 or self.radial_smoothing < 0:
            raise ValueError("Invalid central blend settings")

    @classmethod
    def fit(cls, baseline, train_controls, normalized_phase, saturation_threshold=.8,
            base_weight_strength=100.0, central_weight_strength=400.0, central_radius=.4,
            blend_start=.8, blend_end=.95, radial_smoothing=.04):
        p = baseline.pod.parameters
        controls = np.asarray(train_controls, dtype=np.float64)
        phase = np.asarray(normalized_phase, dtype=np.float64) * p["field_scale"][3]
        if controls.ndim != 2 or controls.shape[1] != 4 or phase.ndim != 3 or len(controls) != len(phase):
            raise ValueError("Controls and phase anchors have incompatible shapes")
        if phase.shape[1:] != (len(p["coordinates_time"]), len(p["coordinates_radius"])):
            raise ValueError("Phase anchors do not match the frozen POD grid")
        design = material_pressure_design(controls, p, baseline.baseline.pressure_centers)
        design = np.column_stack((design, np.ones(len(design))))
        basis = p["basis"][3]
        centered = phase.reshape(len(phase), -1) / p["field_scale"][3] - p["mean_change"][3]
        rows = np.einsum("ip,kj->ijpk", design, basis).reshape(len(phase) * centered.shape[1], -1)
        saturation = phase[:, 1:] >= float(saturation_threshold)
        recas = controls[:, 3].astype(bool)[:, None, None]
        weights = np.ones_like(phase)
        weights[:, 1:] += float(base_weight_strength) * saturation * recas
        central = p["coordinates_radius"] <= float(central_radius)
        weights[:, 1:][:, :, central] += float(central_weight_strength) * saturation[:, :, central] * recas
        target = centered.reshape(-1)
        if not np.isfinite(rows).all() or not np.isfinite(target).all():
            raise ValueError("Nonfinite central weighted regression inputs")
        ridge = Ridge(alpha=1.0, fit_intercept=False).fit(rows, target, sample_weight=weights.reshape(-1))
        coefficient = ridge.coef_.reshape(design.shape[1], basis.shape[0])
        return cls(baseline, coefficient, saturation_threshold, base_weight_strength,
                   central_weight_strength, central_radius, blend_start, blend_end, radial_smoothing)

    def central_phase_anchors(self, controls):
        p = self.pod.parameters
        design = material_pressure_design(controls, p, self.baseline.baseline.pressure_centers)
        augmented = np.column_stack((design, np.ones(len(design))))
        modes = augmented @ self.coefficient
        centered = modes @ p["basis"][3] + p["mean_change"][3]
        phase = centered.reshape(len(controls), len(p["coordinates_time"]),
                                len(p["coordinates_radius"])) * p["field_scale"][3]
        phase[:, 0] = 0.0
        return phase

    def phase(self, controls, times, radii):
        controls = np.asarray(controls, dtype=np.float64)
        baseline_fields = self.baseline.predict(controls, times, radii)
        p = self.pod.parameters
        central = project_phase_anchors(self.central_phase_anchors(controls))
        central = resample_phase(central, p["coordinates_time"], p["coordinates_radius"], times, radii)
        base_phase = baseline_fields[:, 3]
        time_gate = np.clip((base_phase - self.blend_start) / (self.blend_end - self.blend_start), 0.0, 1.0)
        if self.radial_smoothing:
            radial_gate = np.clip((self.central_radius + self.radial_smoothing - np.asarray(radii))
                                  / self.radial_smoothing, 0.0, 1.0)
        else:
            radial_gate = (np.asarray(radii) <= self.central_radius).astype(np.float64)
        gate = time_gate * radial_gate[None, None, :]
        blended = base_phase.copy()
        use_weighted = controls[:, 3] == 1.0
        blended[use_weighted] = ((1.0 - gate[use_weighted]) * base_phase[use_weighted]
                                 + gate[use_weighted] * central[use_weighted])
        return project_phase_anchors(np.clip(blended, 0.0, 1.0))

    def predict(self, controls, times, radii):
        result = self.baseline.predict(controls, times, radii)
        result[:, 3] = self.phase(controls, times, radii)
        return result

    def save(self, path):
        with Path(path).open("xb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(self.SCHEMA), coefficient=self.coefficient,
                saturation_threshold=np.asarray(self.saturation_threshold),
                base_weight_strength=np.asarray(self.base_weight_strength),
                central_weight_strength=np.asarray(self.central_weight_strength),
                central_radius=np.asarray(self.central_radius), blend_start=np.asarray(self.blend_start),
                blend_end=np.asarray(self.blend_end), radial_smoothing=np.asarray(self.radial_smoothing),
            )

    @classmethod
    def load(cls, path, baseline):
        with np.load(path, allow_pickle=False) as state:
            if str(state["schema"]) != cls.SCHEMA:
                raise ValueError("Invalid central saturation checkpoint")
            return cls(baseline, state["coefficient"], float(state["saturation_threshold"]),
                       float(state["base_weight_strength"]), float(state["central_weight_strength"]),
                       float(state["central_radius"]), float(state["blend_start"]),
                       float(state["blend_end"]), float(state["radial_smoothing"]))
