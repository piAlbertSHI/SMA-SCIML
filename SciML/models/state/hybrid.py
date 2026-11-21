"""Stage-conditioned heuristic baseline plus a learnable residual."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np

from .layout import StateLayout
from .projection import AdmissibilityProjector, ProjectionReport


OBSERVED_STAGES = (2, 4)


class Ablation(str, Enum):
    HYBRID = "hybrid"
    NO_SIDE_STATE = "no_side_state"
    PURE_DATA = "pure_data"
    HEURISTIC_ONLY = "heuristic_only"


@dataclass(frozen=True)
class StageControl:
    stage_index: int
    duration_s: float = 1.0
    valve_mask: np.ndarray | None = None
    side_observed: np.ndarray | None = None
    capacity_proxy_target_j: float | np.ndarray | None = None
    allow_unvalidated_stage_heuristic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.stage_index, (bool, np.bool_)) or not isinstance(
            self.stage_index, (int, np.integer)
        ) or self.stage_index not in (1, 2, 3, 4):
            raise ValueError("stage_index must be one of 1, 2, 3, 4")
        if not isinstance(self.allow_unvalidated_stage_heuristic, (bool, np.bool_)):
            raise ValueError("allow_unvalidated_stage_heuristic must be boolean")
        if (
            self.stage_index not in OBSERVED_STAGES
            and not self.allow_unvalidated_stage_heuristic
        ):
            raise ValueError(
                "dataset v1.0 validates only stages 2 and 4; set "
                "allow_unvalidated_stage_heuristic=True for explicit stage-1/3 extrapolation"
            )
        duration = np.asarray(self.duration_s, dtype=np.float64)
        if duration.ndim != 0 or not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        object.__setattr__(self, "duration_s", float(duration))
        if self.valve_mask is not None:
            valves = np.asarray(self.valve_mask, dtype=np.float64)
            if valves.shape != (6,) or not np.isfinite(valves).all():
                raise ValueError("valve_mask must be a finite vector with shape (6,)")
            if not np.all((valves == 0.0) | (valves == 1.0)):
                raise ValueError("valve_mask must contain only binary 0/1 values")
            valves = valves.copy()
            valves.setflags(write=False)
            object.__setattr__(self, "valve_mask", valves)
        if self.side_observed is not None:
            side = np.asarray(self.side_observed, dtype=np.float64)
            if side.shape != (8,) or not np.isfinite(side).all():
                raise ValueError("side_observed must have shape (8,)")
            if not np.all((side == 0) | (side == 1)):
                raise ValueError("side_observed must contain only boolean/0/1 values")
            side = np.asarray(side, dtype=bool).copy()
            side.setflags(write=False)
            object.__setattr__(self, "side_observed", side)
        if self.capacity_proxy_target_j is not None:
            target = np.asarray(self.capacity_proxy_target_j, dtype=np.float64)
            if target.size == 0 or not np.isfinite(target).all():
                raise ValueError("capacity_proxy_target_j must be non-empty and finite")
            target = target.copy()
            target.setflags(write=False)
            object.__setattr__(self, "capacity_proxy_target_j", target)


class ResidualModel(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


class ZeroResidual:
    def __init__(self, output_size: int):
        self.output_size = output_size

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        return np.zeros((*features.shape[:-1], self.output_size), dtype=np.float64)


class RidgeResidual:
    """Small-data linear residual learner with per-output missing-target masks.

    This is an auditable reliability baseline, not a substitute for the
    optional PyTorch MLP.  The dual ridge solution remains stable when only a
    handful of physical groups are available.
    """

    def __init__(self, output_size: int, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.output_size = output_size
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed: np.ndarray | None = None,
    ) -> "RidgeResidual":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0], self.output_size):
            raise ValueError("incompatible feature/target shapes")
        mask = np.isfinite(y) if observed is None else np.asarray(observed, dtype=bool)
        if mask.shape != y.shape:
            raise ValueError("observed mask must match targets")
        x1 = np.c_[np.ones(x.shape[0]), x]
        self.coef_ = np.zeros((x1.shape[1], self.output_size), dtype=np.float64)
        for column in range(self.output_size):
            keep = mask[:, column] & np.isfinite(y[:, column]) & np.all(np.isfinite(x1), axis=1)
            if not np.any(keep):
                continue
            design = x1[keep]
            # dual form: X.T (X X.T + alpha I)^-1 y
            gram = design @ design.T + self.alpha * np.eye(np.count_nonzero(keep))
            self.coef_[:, column] = design.T @ np.linalg.solve(gram, y[keep, column])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeResidual must be fitted before prediction")
        features = np.asarray(features, dtype=np.float64)
        flat = features.reshape(-1, features.shape[-1])
        pred = np.c_[np.ones(flat.shape[0]), flat] @ self.coef_
        return pred.reshape(*features.shape[:-1], self.output_size)


class HeuristicExchangeBaseline:
    """Explicit persistence/exchange heuristic, not a governing-equation solver."""

    def __init__(
        self,
        layout: StateLayout | None = None,
        exchange_by_observed_stage: dict[int, tuple[float, float]] | None = None,
        unvalidated_stage_exchange: tuple[float, float] = (0.02, 0.02),
        diffusion: float = 0.03,
        endpoint_weight: float = 0.35,
        reference_duration_s: float = 2.4,
        max_duration_scale: float = 1.0,
    ):
        self.layout = layout or StateLayout()
        self.exchange_by_observed_stage = exchange_by_observed_stage or {
            2: (0.22, 0.12),
            4: (0.28, 0.16),
        }
        if set(self.exchange_by_observed_stage) != set(OBSERVED_STAGES):
            raise ValueError("exchange coefficients must be specified exactly for stages 2 and 4")
        coefficients = np.asarray(
            [*self.exchange_by_observed_stage.values(), unvalidated_stage_exchange],
            dtype=np.float64,
        )
        if not np.isfinite(coefficients).all() or np.any((coefficients < 0) | (coefficients > 1)):
            raise ValueError("exchange coefficients must be finite and within [0, 1]")
        if not np.isfinite(diffusion) or not 0.0 <= diffusion <= 0.5:
            raise ValueError("diffusion must be finite and within [0, 0.5]")
        if not np.isfinite(endpoint_weight) or not 0.0 <= endpoint_weight <= 1.0:
            raise ValueError("endpoint_weight must be finite and within [0, 1]")
        if not np.isfinite(reference_duration_s) or reference_duration_s <= 0:
            raise ValueError("reference_duration_s must be finite and positive")
        if not np.isfinite(max_duration_scale) or max_duration_scale <= 0:
            raise ValueError("max_duration_scale must be finite and positive")
        self.unvalidated_stage_exchange = tuple(float(x) for x in unvalidated_stage_exchange)
        self.diffusion = float(diffusion)
        self.endpoint_weight = float(endpoint_weight)
        self.reference_duration_s = float(reference_duration_s)
        self.max_duration_scale = float(max_duration_scale)

    @staticmethod
    def _interp(values: np.ndarray, size: int) -> np.ndarray:
        old = np.linspace(0.0, 1.0, values.shape[-1])
        new = np.linspace(0.0, 1.0, size)
        flat = values.reshape(-1, values.shape[-1])
        result = np.stack([np.interp(new, old, row) for row in flat])
        return result.reshape(*values.shape[:-1], size)

    def step(self, state: np.ndarray, control: StageControl) -> np.ndarray:
        state = self.layout.validate(state)
        out = state.copy()
        sma = state[..., self.layout.sma_t]
        htf = state[..., self.layout.htf_t]
        if control.stage_index in OBSERVED_STAGES:
            alpha_sma, alpha_htf = self.exchange_by_observed_stage[control.stage_index]
        elif control.allow_unvalidated_stage_heuristic:
            alpha_sma, alpha_htf = self.unvalidated_stage_exchange
        else:  # defensive: StageControl normally rejects this earlier
            raise ValueError(f"stage {control.stage_index} has no observed exchange calibration")
        duration_scale = min(
            control.duration_s / self.reference_duration_s,
            self.max_duration_scale,
        )
        alpha_sma *= duration_scale
        alpha_htf *= duration_scale
        sma_on_htf = self._interp(sma, self.layout.htf_nodes)
        htf_on_sma = self._interp(htf, self.layout.sma_nodes)
        next_sma = sma + alpha_sma * (htf_on_sma - sma)
        lap = np.zeros_like(htf)
        lap[..., 1:-1] = htf[..., :-2] - 2.0 * htf[..., 1:-1] + htf[..., 2:]
        next_htf = htf + alpha_htf * (sma_on_htf - htf) + self.diffusion * lap
        # Valve orientation affects which interface gets the stronger endpoint
        # correction, while remaining deterministic when the mask is absent.
        orientation = 1.0
        if control.valve_mask is not None:
            valves = np.asarray(control.valve_mask, dtype=np.float64)
            if valves.shape != (6,):
                raise ValueError("valve_mask must have shape (6,)")
            orientation = 1.0 if np.sum(valves[:3]) >= np.sum(valves[3:]) else -1.0
        endpoint = -1 if orientation > 0 else 0
        next_htf[..., endpoint] = (
            (1.0 - self.endpoint_weight) * next_htf[..., endpoint]
            + self.endpoint_weight * next_sma[..., endpoint]
        )
        out[..., self.layout.sma_t] = next_sma
        out[..., self.layout.htf_t] = next_htf
        # xi and side-state are explicit persistence states. No xi_out exists
        # in dataset v1.0, so no transformation law is fabricated here.
        return out


@dataclass(frozen=True)
class RolloutResult:
    states: np.ndarray
    projection_reports: tuple[ProjectionReport, ...]
    handoff_copy_invariant: np.ndarray
    independent_next_stage_inputs_available: bool = False

    @property
    def all_handoffs_are_exact_copies(self) -> bool:
        """Software invariant only; this is not a physical interface metric."""
        return bool(np.all(self.handoff_copy_invariant))


class HybridStateSpaceModel:
    """Composable WP3 model with hybrid and controlled ablation modes."""

    def __init__(
        self,
        layout: StateLayout | None = None,
        baseline: HeuristicExchangeBaseline | None = None,
        residual: ResidualModel | None = None,
        projector: AdmissibilityProjector | None = None,
        ablation: Ablation | str = Ablation.HYBRID,
        enforce_capacity_proxy_target: bool = False,
    ):
        self.layout = layout or StateLayout()
        self.baseline = baseline or HeuristicExchangeBaseline(self.layout)
        self.residual = residual or ZeroResidual(self.layout.size)
        self.projector = projector or AdmissibilityProjector(self.layout)
        self.ablation = Ablation(ablation)
        self.enforce_capacity_proxy_target = enforce_capacity_proxy_target

    @property
    def uses_side_state(self) -> bool:
        return self.ablation != Ablation.NO_SIDE_STATE

    def features(self, state: np.ndarray, control: StageControl) -> np.ndarray:
        state = self.layout.validate(state)
        # Fixed affine scales keep the reference implementation deterministic;
        # production scalers must be fitted on training groups only.
        encoded = state.copy()
        encoded[..., self.layout.sma_t] = (encoded[..., self.layout.sma_t] - 293.0) / 25.0
        encoded[..., self.layout.htf_t] = (encoded[..., self.layout.htf_t] - 293.0) / 25.0
        encoded[..., self.layout.side] = (encoded[..., self.layout.side] - 293.0) / 10.0
        if not self.uses_side_state:
            encoded[..., self.layout.side] = 0.0
        stage = np.zeros((*state.shape[:-1], 4), dtype=np.float64)
        stage[..., control.stage_index - 1] = 1.0
        valves = np.zeros((*state.shape[:-1], 6), dtype=np.float64)
        if control.valve_mask is not None:
            valves[...] = np.asarray(control.valve_mask, dtype=np.float64)
        side_seen = np.zeros((*state.shape[:-1], self.layout.side_size), dtype=np.float64)
        if self.uses_side_state and control.side_observed is not None:
            side_seen[...] = np.asarray(control.side_observed, dtype=np.float64)
        duration = np.full((*state.shape[:-1], 1), control.duration_s / 2.4)
        return np.concatenate([encoded, stage, valves, side_seen, duration], axis=-1)

    def unprojected_step(self, state: np.ndarray, control: StageControl) -> np.ndarray:
        state = self.layout.validate(state)
        residual = self.residual.predict(self.features(state, control))
        if residual.shape != state.shape:
            raise ValueError(f"residual shape {residual.shape} does not match state {state.shape}")
        if self.ablation == Ablation.HEURISTIC_ONLY:
            return self.baseline.step(state, control)
        if self.ablation == Ablation.PURE_DATA:
            return state + residual
        return self.baseline.step(state, control) + residual

    def step(
        self, state: np.ndarray, control: StageControl
    ) -> tuple[np.ndarray, ProjectionReport]:
        state = self.layout.validate(state)
        candidate = self.unprojected_step(state, control)
        return self.projector.project(
            candidate,
            state,
            control.capacity_proxy_target_j,
            enforce_capacity_proxy_target=self.enforce_capacity_proxy_target,
        )

    def training_target(
        self, state_in: np.ndarray, state_out: np.ndarray, control: StageControl
    ) -> np.ndarray:
        state_in = self.layout.validate(state_in, name="state_in")
        state_out = self.layout.validate(state_out, name="state_out")
        if self.ablation == Ablation.PURE_DATA:
            return state_out - state_in
        return state_out - self.baseline.step(state_in, control)

    def rollout(
        self, initial_state: np.ndarray, controls: Sequence[StageControl]
    ) -> RolloutResult:
        current = self.layout.validate(initial_state, name="initial_state").copy()
        states = [current.copy()]
        reports: list[ProjectionReport] = []
        copy_checks: list[bool] = []
        for control in controls:
            stage_input = current.copy()
            next_state, report = self.step(stage_input, control)
            # This checks only our array-copy/handoff implementation. Without an
            # independently observed next-stage input, no physical interface
            # error can be computed from dataset v1.0.
            copy_checks.append(bool(np.array_equal(stage_input, states[-1])))
            states.append(next_state.copy())
            reports.append(report)
            current = next_state
        return RolloutResult(
            states=np.stack(states, axis=0),
            projection_reports=tuple(reports),
            handoff_copy_invariant=np.asarray(copy_checks, dtype=bool),
        )
