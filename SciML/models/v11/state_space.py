"""Four-stage physics--neural state-space baseline for v1.1-pilot-r3.

The model uses a finite canonical state shared by all four stages.  Hard
stage masks preserve mechanically inactive variables, and a projection enforces
clamped-edge and martensite-fraction constraints.  It is intended for audited
small-corpus overfit/rollout tests, not for a generalization claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


STAGE_NAMES = (
    "stage1_loading",
    "stage2_hold_exchange",
    "stage3_unloading",
    "stage4_cooldown_exchange",
)
PREREGISTERED_RESIDUAL_GAIN = 0.2
SIDE_KEYS = (
    ("hot_sink_temp_in", "hot_sink_temp_out"),
    ("cold_sink_temp_in", "cold_sink_temp_out"),
    ("chip_temp_in", "chip_temp_out"),
    ("hot_buffer_temp_in", "hot_buffer_temp_out"),
    ("cold_buffer_temp_in", "cold_buffer_temp_out"),
    ("stage2_path_pipe_T_state_in_K", "stage2_path_pipe_T_state_out_K"),
    ("stage4_path_pipe_T_state_in_K", "stage4_path_pipe_T_state_out_K"),
    ("simscape_cold_sink_pipe_T_state_in_K", "simscape_cold_sink_pipe_T_state_out_K"),
)


@dataclass(frozen=True)
class CanonicalLayout:
    u: slice = slice(0, 51)
    w: slice = slice(51, 102)
    sma_T: slice = slice(102, 153)
    xi: slice = slice(153, 204)
    htf_stage2_T: slice = slice(204, 305)
    htf_stage4_T: slice = slice(305, 406)
    side: slice = slice(406, 414)
    size: int = 414


LAYOUT = CanonicalLayout()


@dataclass(frozen=True)
class TransitionRecord:
    sample_id: str
    case_id: str
    condition_id: str
    material_id: str
    geometry_id: str
    cycle_index: int
    stage_index: int
    state_in: np.ndarray
    state_out: np.ndarray
    controls: np.ndarray


def _read_table(root: Path, stem: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(root / f"{stem}.parquet")
    except (ImportError, OSError):
        return pd.read_csv(root / f"{stem}.csv")


def _array(group: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(group[name], dtype=np.float64)


def _finite_update(previous: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = previous.copy()
    finite = np.isfinite(values)
    result[finite] = values[finite]
    return result


def _side(group: h5py.Group, suffix: str, previous: np.ndarray) -> np.ndarray:
    values = []
    nested = group["side_state"]
    key_index = 0 if suffix == "in" else 1
    for keys in SIDE_KEYS:
        key = keys[key_index]
        values.append(float(np.asarray(nested[key])))
    return _finite_update(previous, np.asarray(values, dtype=np.float64))


def _controls(row: pd.Series, stage: h5py.Group, stage_index: int, cycle_index: int) -> np.ndarray:
    time = _array(stage, "time_local_s").reshape(-1)
    onehot = np.eye(4, dtype=np.float64)[stage_index - 1]
    material = np.array(
        [float(str(row.material_id) == "recA"), float(str(row.material_id) == "recAs")]
    )
    return np.concatenate(
        (
            onehot,
            material,
            np.array(
                [
                    float(row.radius_m),
                    float(row.thickness_m),
                    float(row.pressure_Pa),
                    float(row.flush_volume_target_ratio),
                    float(row.exchange_duration_s),
                    float(time[-1] - time[0]),
                    float(cycle_index),
                ]
            ),
        )
    )


def build_transition_corpus(dataset_root: str | Path) -> list[TransitionRecord]:
    """Build 4 x cycle_count observed transitions without synthetic cycles."""

    root = Path(dataset_root)
    cases = _read_table(root, "case_index")
    cases = cases[cases["physics_ok"].astype(bool)].sort_values("case_id")
    records: list[TransitionRecord] = []
    with h5py.File(root / "fields.h5", "r") as handle:
        for row in cases.itertuples(index=False):
            case_group = handle[str(row.hdf5_group)]
            previous_cycle_state: np.ndarray | None = None
            for cycle_index in range(1, int(row.cycle_count) + 1):
                cycle = case_group[f"cycles/cycle_{cycle_index:03d}"]
                stage2 = cycle[STAGE_NAMES[1]]
                if previous_cycle_state is None:
                    state = np.empty(LAYOUT.size, dtype=np.float64)
                    state[LAYOUT.u] = _array(cycle[STAGE_NAMES[0]], "u_m_t")[0]
                    state[LAYOUT.w] = _array(cycle[STAGE_NAMES[0]], "w_m_t")[0]
                    state[LAYOUT.sma_T] = _array(cycle[STAGE_NAMES[0]], "T_K_t")[0]
                    state[LAYOUT.xi] = _array(cycle[STAGE_NAMES[0]], "xi_t")[0]
                    state[LAYOUT.htf_stage2_T] = _array(stage2, "htf_internal_T_K_t")[0]
                    state[LAYOUT.htf_stage4_T] = _array(
                        cycle[STAGE_NAMES[3]], "htf_internal_T_K_t"
                    )[0]
                    state[LAYOUT.side] = _side(stage2, "in", np.full(8, 293.0))
                else:
                    state = previous_cycle_state.copy()

                for stage_index, stage_name in enumerate(STAGE_NAMES, start=1):
                    stage = cycle[stage_name]
                    observed_in = state.copy()
                    observed_out = state.copy()
                    if stage_index in (1, 3):
                        for field, slc in (
                            ("u_m_t", LAYOUT.u),
                            ("w_m_t", LAYOUT.w),
                            ("T_K_t", LAYOUT.sma_T),
                            ("xi_t", LAYOUT.xi),
                        ):
                            history = _array(stage, field)
                            observed_out[slc] = history[-1]
                    else:
                        htf_slice = (
                            LAYOUT.htf_stage2_T if stage_index == 2 else LAYOUT.htf_stage4_T
                        )
                        observed_out = observed_in.copy()
                        observed_out[LAYOUT.sma_T] = _array(stage, "sma_T_out_51_K")
                        observed_out[htf_slice] = _array(stage, "htf_internal_T_K_t")[-1]
                        observed_out[LAYOUT.side] = _side(stage, "out", observed_in[LAYOUT.side])
                    if not np.isfinite(observed_in).all() or not np.isfinite(observed_out).all():
                        raise ValueError(f"non-finite canonical state in {row.case_id}/{cycle_index}/{stage_index}")
                    records.append(
                        TransitionRecord(
                            sample_id=f"{row.case_id}:cycle_{cycle_index:03d}:stage_{stage_index}",
                            case_id=str(row.case_id),
                            condition_id=str(row.condition_id),
                            material_id=str(row.material_id),
                            geometry_id=(
                                f"R{float(row.radius_m):.12g}_t{float(row.thickness_m):.12g}"
                            ),
                            cycle_index=cycle_index,
                            stage_index=stage_index,
                            state_in=observed_in,
                            state_out=observed_out,
                            controls=_controls(pd.Series(row._asdict()), stage, stage_index, cycle_index),
                        )
                    )
                    state = observed_out
                previous_cycle_state = state
    return records


def stage_mask(stage_index: np.ndarray) -> np.ndarray:
    mask = np.zeros((len(stage_index), LAYOUT.size), dtype=np.float64)
    for row, stage in enumerate(stage_index):
        if stage in (1, 3):
            mask[row, : LAYOUT.htf_stage2_T.start] = 1.0
        else:
            mask[row, LAYOUT.sma_T] = 1.0
            mask[row, LAYOUT.htf_stage2_T if stage == 2 else LAYOUT.htf_stage4_T] = 1.0
            mask[row, LAYOUT.side] = 1.0
    return mask


def project_state(state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=np.float64).copy()
    result[..., LAYOUT.u.stop - 1] = 0.0
    result[..., LAYOUT.w.stop - 1] = 0.0
    result[..., LAYOUT.xi] = np.clip(result[..., LAYOUT.xi], 0.0, 1.0)
    result[..., LAYOUT.sma_T] = np.clip(result[..., LAYOUT.sma_T], 200.0, 400.0)
    result[..., LAYOUT.htf_stage2_T] = np.clip(
        result[..., LAYOUT.htf_stage2_T], 200.0, 400.0
    )
    result[..., LAYOUT.htf_stage4_T] = np.clip(
        result[..., LAYOUT.htf_stage4_T], 200.0, 400.0
    )
    result[..., LAYOUT.side] = np.clip(result[..., LAYOUT.side], 200.0, 400.0)
    result[..., LAYOUT.u] = np.clip(result[..., LAYOUT.u], -0.2, 0.2)
    result[..., LAYOUT.w] = np.clip(result[..., LAYOUT.w], -0.2, 0.2)
    return result


class ResidualStateMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class HybridStageModel:
    """Physics-constrained stage-mean baseline plus a neural residual.

    The baseline itself is empirical.  Physics enters through the canonical
    state ownership, stage activity masks, exact Markov continuity and hard
    admissibility projection; it is not a reduced-order PDE solver.
    """

    def __init__(
        self,
        seed: int = 20260820,
        hidden: int = 128,
        residual_gain: float = PREREGISTERED_RESIDUAL_GAIN,
    ) -> None:
        if not 0.0 <= residual_gain <= 1.0:
            raise ValueError("residual_gain must be in [0, 1]")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.hidden = hidden
        self.residual_gain = float(residual_gain)
        self.model: ResidualStateMLP | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None
        self.stage_baseline = np.zeros((4, LAYOUT.size), dtype=np.float64)
        self.stage_training_features: dict[int, np.ndarray] = {}
        self.stage_trust_radius: dict[int, float] = {}
        self.training_ids: tuple[str, ...] = ()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "HybridStageModel":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("checkpoint_schema_version") != "sciml_wp3_checkpoint_v2":
            raise ValueError("unsupported WP3 checkpoint schema")
        config = payload.get("training_config", {})
        state_dict = payload["state_dict"]
        first_weight = state_dict["network.0.weight"]
        hidden = int(config.get("hidden", first_weight.shape[0]))
        model = cls(
            seed=int(config.get("seed", 20260820)),
            hidden=hidden,
            residual_gain=float(payload["residual_gain"]),
        )
        model.model = ResidualStateMLP(
            int(first_weight.shape[1]), LAYOUT.size, hidden
        )
        model.model.load_state_dict(state_dict)
        model.model.eval()
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        model.residual_scale = np.asarray(payload["residual_scale"], dtype=np.float64)
        model.stage_baseline = np.asarray(payload["stage_baseline"], dtype=np.float64)
        model.stage_training_features = {
            int(stage): np.asarray(values, dtype=np.float64)
            for stage, values in payload["stage_training_features"].items()
        }
        model.stage_trust_radius = {
            int(stage): float(value)
            for stage, value in payload["stage_trust_radius"].items()
        }
        model.training_ids = tuple(str(value) for value in payload["training_ids"])
        if model.parameter_sha256() != str(payload.get("parameter_sha256", "")):
            raise ValueError("WP3 checkpoint parameter hash mismatch")
        return model

    @staticmethod
    def _matrix(records: list[TransitionRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.stack([record.state_in for record in records])
        controls = np.stack([record.controls for record in records])
        targets = np.stack([record.state_out for record in records])
        stages = np.asarray([record.stage_index for record in records], dtype=np.int64)
        return states, controls, targets, stages

    def fit(self, records: list[TransitionRecord], epochs: int = 1200, learning_rate: float = 2e-3) -> dict[str, float]:
        if not records:
            raise ValueError("cannot fit an empty transition corpus")
        states, controls, targets, stages = self._matrix(records)
        masks = stage_mask(stages)
        delta = (targets - states) * masks
        for stage in range(1, 5):
            selected = stages == stage
            if selected.any():
                self.stage_baseline[stage - 1] = np.mean(delta[selected], axis=0)
        residual = (delta - self.stage_baseline[stages - 1]) * masks
        features = np.concatenate((states, controls), axis=1)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.where(features.std(axis=0) > 1e-10, features.std(axis=0), 1.0)
        scale = residual.std(axis=0)
        self.residual_scale = np.where(scale > 1e-10, scale, 1.0)
        x = torch.tensor((features - self.feature_mean) / self.feature_scale, dtype=torch.float32)
        y = torch.tensor(residual / self.residual_scale, dtype=torch.float32)
        mask = torch.tensor(masks, dtype=torch.float32)
        self.model = ResidualStateMLP(x.shape[1], LAYOUT.size, self.hidden)
        scaled_features = x.numpy().astype(np.float64)
        for stage in range(1, 5):
            selected = scaled_features[stages == stage]
            self.stage_training_features[stage] = selected
            if len(selected) >= 2:
                distances = np.sqrt(
                    np.sum((selected[:, None, :] - selected[None, :, :]) ** 2, axis=2)
                )
                distances[distances == 0.0] = np.nan
                nearest = np.nanmin(distances, axis=1)
                self.stage_trust_radius[stage] = max(float(np.nanmedian(nearest)), 1.0)
            else:
                self.stage_trust_radius[stage] = 1.0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
        for _ in range(epochs):
            optimizer.zero_grad()
            prediction = self.model(x) * mask
            loss = torch.sum((prediction - y) ** 2 * mask) / torch.clamp(mask.sum(), min=1.0)
            loss.backward()
            optimizer.step()
        self.training_ids = tuple(record.sample_id for record in records)
        predicted = self.predict_records(records)
        error = predicted - targets
        return {
            "rmse_all_state": float(np.sqrt(np.mean(error**2))),
            "max_abs_all_state": float(np.max(np.abs(error))),
        }

    def predict(self, state: np.ndarray, controls: np.ndarray, stage_index: int) -> np.ndarray:
        if self.model is None or self.feature_mean is None or self.feature_scale is None or self.residual_scale is None:
            raise RuntimeError("model is not fitted")
        feature = np.concatenate((np.asarray(state), np.asarray(controls)))[None, :]
        x = torch.tensor((feature - self.feature_mean) / self.feature_scale, dtype=torch.float32)
        with torch.no_grad():
            neural = self.model(x).numpy()[0] * self.residual_scale
        trust = self.trust_score(state, controls, stage_index)
        neural *= trust * self.residual_gain
        mask = stage_mask(np.asarray([stage_index]))[0]
        candidate = state + (self.stage_baseline[stage_index - 1] + neural) * mask
        return project_state(candidate)

    def trust_score(self, state: np.ndarray, controls: np.ndarray, stage_index: int) -> float:
        if self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("model is not fitted")
        feature = np.concatenate((np.asarray(state), np.asarray(controls)))[None, :]
        scaled = (feature - self.feature_mean) / self.feature_scale
        reference = self.stage_training_features[stage_index]
        distance = float(np.min(np.linalg.norm(reference - scaled, axis=1)))
        radius = self.stage_trust_radius[stage_index]
        return float(np.exp(-((distance / radius) ** 2)))

    def predict_records(self, records: list[TransitionRecord]) -> np.ndarray:
        return np.stack(
            [self.predict(record.state_in, record.controls, record.stage_index) for record in records]
        )

    def predict_stage_mean(self, state: np.ndarray, stage_index: int) -> np.ndarray:
        """Apply the train-fitted stage-mean delta without a neural residual."""

        mask = stage_mask(np.asarray([stage_index]))[0]
        candidate = np.asarray(state) + self.stage_baseline[stage_index - 1] * mask
        return project_state(candidate)

    def predict_stage_mean_records(self, records: list[TransitionRecord]) -> np.ndarray:
        return np.stack(
            [self.predict_stage_mean(record.state_in, record.stage_index) for record in records]
        )

    def rollout(self, records: list[TransitionRecord]) -> np.ndarray:
        if not records:
            return np.empty((0, LAYOUT.size))
        ordered = sorted(records, key=lambda record: (record.cycle_index, record.stage_index))
        state = ordered[0].state_in.copy()
        predictions = []
        for record in ordered:
            state = self.predict(state, record.controls, record.stage_index)
            predictions.append(state)
        return np.stack(predictions)

    def rollout_stage_mean(self, records: list[TransitionRecord]) -> np.ndarray:
        if not records:
            return np.empty((0, LAYOUT.size))
        ordered = sorted(records, key=lambda record: (record.cycle_index, record.stage_index))
        state = ordered[0].state_in.copy()
        predictions = []
        for record in ordered:
            state = self.predict_stage_mean(state, record.stage_index)
            predictions.append(state)
        return np.stack(predictions)

    def parameter_sha256(self) -> str:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        digest = hashlib.sha256()

        def update_array(name: str, value: np.ndarray) -> None:
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())

        for name, tensor in sorted(self.model.state_dict().items()):
            update_array(f"network:{name}", tensor.detach().cpu().numpy())
        for name, value in (
            ("feature_mean", self.feature_mean),
            ("feature_scale", self.feature_scale),
            ("residual_scale", self.residual_scale),
            ("stage_baseline", self.stage_baseline),
        ):
            if value is None:
                raise RuntimeError(f"model parameter {name} is unavailable")
            update_array(name, value)
        update_array("residual_gain", np.asarray([self.residual_gain], dtype=np.float64))
        for stage in sorted(self.stage_training_features):
            update_array(
                f"stage_training_features:{stage}",
                self.stage_training_features[stage],
            )
            update_array(
                f"stage_trust_radius:{stage}",
                np.asarray([self.stage_trust_radius[stage]], dtype=np.float64),
            )
        for sample_id in self.training_ids:
            digest.update(b"training_id\0")
            digest.update(sample_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


def continuity_audit(records: list[TransitionRecord]) -> dict[str, float]:
    by_case: dict[str, list[TransitionRecord]] = {}
    for record in records:
        by_case.setdefault(record.case_id, []).append(record)
    differences = []
    for case_records in by_case.values():
        ordered = sorted(case_records, key=lambda record: (record.cycle_index, record.stage_index))
        for left, right in zip(ordered[:-1], ordered[1:]):
            differences.append(float(np.max(np.abs(left.state_out - right.state_in))))
    return {
        "max_raw_stage_interface_abs": max(differences, default=0.0),
        "mean_raw_stage_interface_abs": float(np.mean(differences)) if differences else 0.0,
    }
