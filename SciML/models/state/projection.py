"""Numerical admissibility projection with a labelled capacity heuristic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import StateLayout


@dataclass(frozen=True)
class ProjectionConfig:
    """Validated numerical-admissibility and capacity-proxy settings."""

    temperature_bounds_k: tuple[float, float] = (240.0, 360.0)
    sma_capacity_proxy_per_node_j_per_k: float = 0.02
    htf_capacity_proxy_per_node_j_per_k: float = 0.005
    side_capacity_proxy_per_node_j_per_k: float = 0.01
    capacity_proxy_atol_j: float = 1e-9

    def __post_init__(self) -> None:
        bounds = np.asarray(self.temperature_bounds_k, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("temperature_bounds_k must be finite and strictly increasing")
        capacities = np.asarray(
            [
                self.sma_capacity_proxy_per_node_j_per_k,
                self.htf_capacity_proxy_per_node_j_per_k,
                self.side_capacity_proxy_per_node_j_per_k,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(capacities).all() or np.any(capacities <= 0.0):
            raise ValueError("all capacity-proxy coefficients must be finite and positive")
        if not np.isfinite(self.capacity_proxy_atol_j) or self.capacity_proxy_atol_j < 0.0:
            raise ValueError("capacity_proxy_atol_j must be finite and non-negative")


@dataclass(frozen=True)
class ProjectionReport:
    nonfinite_replaced: int
    temperature_clipped: int
    xi_clipped: int
    capacity_proxy_correction_applied: bool
    capacity_proxy_target_available: bool
    capacity_proxy_delta_j: np.ndarray
    capacity_proxy_target_j: np.ndarray | None
    capacity_proxy_residual_j: np.ndarray | None
    capacity_proxy_constraint_satisfied: bool | None
    intervention_count: int
    postcheck_passed: bool

    @property
    def intervened(self) -> bool:
        return self.intervention_count > 0


class AdmissibilityProjector:
    """Enforce finite values and configured bounds.

    The optional capacity correction is only a reduced, explicitly configured
    heat-capacity heuristic. It is not the Stage-4 system energy balance and
    must never be reported as formal energy conservation.
    """

    def __init__(
        self,
        layout: StateLayout | None = None,
        temperature_bounds_k: tuple[float, float] = (240.0, 360.0),
        sma_capacity_proxy_per_node_j_per_k: float = 0.02,
        htf_capacity_proxy_per_node_j_per_k: float = 0.005,
        side_capacity_proxy_per_node_j_per_k: float = 0.01,
        capacity_proxy_atol_j: float = 1e-9,
        config: ProjectionConfig | None = None,
    ):
        self.layout = layout or StateLayout()
        if config is not None and (
            temperature_bounds_k != (240.0, 360.0)
            or sma_capacity_proxy_per_node_j_per_k != 0.02
            or htf_capacity_proxy_per_node_j_per_k != 0.005
            or side_capacity_proxy_per_node_j_per_k != 0.01
            or capacity_proxy_atol_j != 1e-9
        ):
            raise ValueError("pass either config or individual projection settings, not both")
        self.config = config or ProjectionConfig(
            temperature_bounds_k=temperature_bounds_k,
            sma_capacity_proxy_per_node_j_per_k=sma_capacity_proxy_per_node_j_per_k,
            htf_capacity_proxy_per_node_j_per_k=htf_capacity_proxy_per_node_j_per_k,
            side_capacity_proxy_per_node_j_per_k=side_capacity_proxy_per_node_j_per_k,
            capacity_proxy_atol_j=capacity_proxy_atol_j,
        )
        bounds = np.asarray(self.config.temperature_bounds_k, dtype=np.float64)
        capacities = np.asarray(
            [
                self.config.sma_capacity_proxy_per_node_j_per_k,
                self.config.htf_capacity_proxy_per_node_j_per_k,
                self.config.side_capacity_proxy_per_node_j_per_k,
            ],
            dtype=np.float64,
        )
        self.temperature_bounds_k = (float(bounds[0]), float(bounds[1]))
        self.capacity_proxy_atol_j = float(self.config.capacity_proxy_atol_j)
        self.capacity_proxy = np.zeros(self.layout.size, dtype=np.float64)
        self.capacity_proxy[self.layout.sma_t] = capacities[0]
        self.capacity_proxy[self.layout.htf_t] = capacities[1]
        self.capacity_proxy[self.layout.side] = capacities[2]

    @staticmethod
    def _validate_target(
        target: float | np.ndarray, output_shape: tuple[int, ...]
    ) -> np.ndarray:
        value = np.asarray(target, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError("capacity_proxy_target_j must be finite")
        try:
            return np.broadcast_to(value, output_shape).copy()
        except ValueError as exc:
            raise ValueError(
                f"capacity_proxy_target_j cannot broadcast to batch shape {output_shape}"
            ) from exc

    def project(
        self,
        candidate: np.ndarray,
        previous: np.ndarray,
        capacity_proxy_target_j: float | np.ndarray | None = None,
        *,
        enforce_capacity_proxy_target: bool = False,
    ) -> tuple[np.ndarray, ProjectionReport]:
        candidate = self.layout.validate(candidate, name="candidate").copy()
        previous = self.layout.validate(previous, name="previous")
        if candidate.shape != previous.shape:
            raise ValueError("candidate and previous must have identical shapes")
        if enforce_capacity_proxy_target and capacity_proxy_target_j is None:
            raise ValueError(
                "enforce_capacity_proxy_target=True requires a finite explicit target"
            )
        target = (
            None
            if capacity_proxy_target_j is None
            else self._validate_target(capacity_proxy_target_j, candidate.shape[:-1])
        )
        previous_safe = previous.copy()
        temperature = self.layout.temperature_indices
        previous_safe[..., temperature] = np.where(
            np.isfinite(previous_safe[..., temperature]),
            previous_safe[..., temperature], 293.0,
        )
        previous_safe[..., self.layout.xi] = np.where(
            np.isfinite(previous_safe[..., self.layout.xi]),
            previous_safe[..., self.layout.xi], 0.0,
        )
        if not np.isfinite(previous_safe).all():
            raise ValueError("previous contains non-finite values outside recoverable channels")

        bad = ~np.isfinite(candidate)
        candidate[bad] = previous_safe[bad]
        low, high = self.temperature_bounds_k
        before_t = candidate[..., temperature].copy()
        candidate[..., temperature] = np.clip(before_t, low, high)
        temperature_clipped_mask = candidate[..., temperature] != before_t
        before_xi = candidate[..., self.layout.xi].copy()
        candidate[..., self.layout.xi] = np.clip(before_xi, 0.0, 1.0)
        xi_clipped = int(np.count_nonzero(candidate[..., self.layout.xi] != before_xi))

        delta = np.sum((candidate - previous_safe) * self.capacity_proxy, axis=-1)
        if not np.isfinite(delta).all():
            raise RuntimeError("capacity-proxy delta is non-finite after admissibility projection")
        correction_applied = bool(enforce_capacity_proxy_target)
        if correction_applied:
            assert target is not None  # guarded above
            # Bounded water-filling: redistribute the residual after nodes hit
            # a temperature bound. A single uniform shift followed by clipping
            # can miss a feasible target because saturated nodes consume none
            # of the remaining correction.
            weights = self.capacity_proxy[temperature]
            if not np.isfinite(weights).all() or np.any(weights <= 0.0):
                raise ValueError("temperature capacity proxies must be finite and positive")
            flat_candidate = candidate.reshape(-1, candidate.shape[-1])
            flat_previous = previous_safe.reshape(-1, previous_safe.shape[-1])
            flat_target = target.reshape(-1)
            flat_clip_mask = temperature_clipped_mask.reshape(-1, temperature.size)
            for row_index, target_value in enumerate(flat_target):
                row = flat_candidate[row_index, temperature]
                previous_row = flat_previous[row_index, temperature]
                for _ in range(temperature.size + 1):
                    current = float(np.sum((row - previous_row) * weights))
                    residual_value = float(target_value - current)
                    if abs(residual_value) <= self.capacity_proxy_atol_j:
                        break
                    if residual_value > 0.0:
                        movable = row < high - np.finfo(float).eps
                    else:
                        movable = row > low + np.finfo(float).eps
                    if not np.any(movable):
                        break
                    shift = residual_value / float(np.sum(weights[movable]))
                    proposed = row[movable] + shift
                    bounded = np.clip(proposed, low, high)
                    movable_indices = np.flatnonzero(movable)
                    hit_bound = bounded != proposed
                    if np.any(hit_bound):
                        flat_clip_mask[row_index, movable_indices[hit_bound]] = True
                    before = row.copy()
                    row[movable] = bounded
                    if np.array_equal(before, row):
                        break
                flat_candidate[row_index, temperature] = row
            delta = np.sum((candidate - previous_safe) * self.capacity_proxy, axis=-1)
            if not np.isfinite(delta).all():
                raise RuntimeError("capacity-proxy delta is non-finite after target correction")

        temperatures = candidate[..., temperature]
        xis = candidate[..., self.layout.xi]
        postcheck = bool(
            np.isfinite(candidate).all()
            and np.all((temperatures >= low) & (temperatures <= high))
            and np.all((xis >= 0.0) & (xis <= 1.0))
        )
        if not postcheck:
            raise RuntimeError("projection postcheck failed")
        residual = None if target is None else np.asarray(delta - target)
        capacity_constraint_satisfied = (
            None
            if residual is None
            else bool(np.all(np.abs(residual) <= self.capacity_proxy_atol_j))
        )
        t_clipped = int(np.count_nonzero(temperature_clipped_mask))
        intervention_count = (
            int(np.count_nonzero(bad)) + t_clipped + xi_clipped + int(correction_applied)
        )
        report = ProjectionReport(
            nonfinite_replaced=int(np.count_nonzero(bad)),
            temperature_clipped=t_clipped,
            xi_clipped=xi_clipped,
            capacity_proxy_correction_applied=correction_applied,
            capacity_proxy_target_available=target is not None,
            capacity_proxy_delta_j=np.asarray(delta),
            capacity_proxy_target_j=target,
            capacity_proxy_residual_j=residual,
            capacity_proxy_constraint_satisfied=capacity_constraint_satisfied,
            intervention_count=intervention_count,
            postcheck_passed=postcheck,
        )
        return candidate, report


PhysicalProjector = AdmissibilityProjector
