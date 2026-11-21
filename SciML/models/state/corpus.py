"""Audited adapter from dataset v1.0 operator checkpoints to WP3 transitions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .layout import StateLayout


@dataclass(frozen=True)
class CorpusAudit:
    sample_count: int
    physical_case_count: int
    physical_run_count: int
    stage_indices: tuple[int, ...]
    stage_indices_by_role: dict[str, tuple[int, ...]]
    train_count: int
    test_count: int
    golden_count: int
    train_group_count: int
    test_group_count: int
    golden_group_count: int
    aer_full_state_count: int
    role_counts_recomputed: dict[str, int]
    physical_group_counts_recomputed: dict[str, int]
    run_group_counts_recomputed: dict[str, int]
    protocols_by_role: dict[str, tuple[str, ...]]
    train_test_protocol_disjoint: bool
    train_test_physical_groups_disjoint: bool
    train_test_run_ids_disjoint: bool
    all_role_groups_disjoint: bool
    published_protocols_by_role: dict[str, tuple[str, ...]]
    protocol_declarations_match_rows: bool
    assignment_covers_table_exactly: bool
    unique_sample_ids: bool
    unique_hdf5_groups: bool
    all_hdf5_groups_exist: bool


@dataclass(frozen=True)
class StageStateTarget:
    """Stage output with observed and persistence-only channels separated."""

    values: np.ndarray
    observed_mask: np.ndarray
    persistence_only_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != self.observed_mask.shape:
            raise ValueError("observed_mask must match target values")
        if self.values.shape != self.persistence_only_mask.shape:
            raise ValueError("persistence_only_mask must match target values")
        if np.any(self.observed_mask & self.persistence_only_mask):
            raise ValueError("a target channel cannot be both observed and persistence-only")


@dataclass(frozen=True)
class Transition:
    sample_id: str
    physical_case_id: str
    stage_index: int
    state_in: np.ndarray
    target: StageStateTarget
    side_input_observed: np.ndarray
    valve_mask: np.ndarray
    duration_s: float
    energy_labels: tuple[str, ...]
    energy_values_j: np.ndarray

    @property
    def state_out(self) -> np.ndarray:
        """Compatibility view; inspect ``target`` before using as supervision."""
        return self.target.values

    @property
    def target_observed(self) -> np.ndarray:
        return self.target.observed_mask

    @property
    def energy_observed(self) -> np.ndarray:
        """Finite audit entries only; absent entries remain absent, never imputed."""
        return np.isfinite(self.energy_values_j)


class OperatorCorpus:
    """Read the 20 checkpoint corpus without silently manufacturing fields."""

    def __init__(self, dataset_dir: str | Path, layout: StateLayout | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.layout = layout or StateLayout()
        with (self.dataset_dir / "operator_samples.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            self.rows = list(csv.DictReader(handle))
        with (self.dataset_dir / "split_manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("dataset_version") != "1.0":
            raise ValueError("WP3 adapter is audited only for dataset_version='1.0'")
        try:
            scheme = self.manifest["schemes"]["protocol_frequency_holdout"]
            assignments = scheme["assignments"]
        except KeyError as exc:
            raise ValueError("protocol_frequency_holdout manifest is incomplete") from exc
        if scheme.get("task_family") != "spatiotemporal_operator":
            raise ValueError("protocol holdout must identify the spatiotemporal_operator task")
        if scheme.get("group_key") != "physical_case_id/run_id":
            raise ValueError("protocol holdout must group jointly by physical_case_id/run_id")
        sample_ids = [row.get("sample_id", "") for row in self.rows]
        if not self.rows or any(not sample_id for sample_id in sample_ids):
            raise ValueError("operator table must contain non-empty sample_id values")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("operator table contains duplicate sample_id values")
        hdf5_groups = [row.get("hdf5_group", "") for row in self.rows]
        if any(not group for group in hdf5_groups):
            raise ValueError("operator table must contain non-empty hdf5_group values")
        if len(hdf5_groups) != len(set(hdf5_groups)):
            raise ValueError("operator table contains duplicate hdf5_group values")
        row_ids = {row["sample_id"] for row in self.rows}
        assignment_ids = set(assignments)
        if row_ids != assignment_ids:
            raise ValueError(
                "operator table/split assignment mismatch: "
                f"table_only={sorted(row_ids - assignment_ids)}, "
                f"manifest_only={sorted(assignment_ids - row_ids)}"
            )
        unknown_roles = set(assignments.values()) - {"train", "test", "golden_validation"}
        if unknown_roles:
            raise ValueError(f"unsupported protocol split roles: {sorted(unknown_roles)}")
        self._sample_ids_unique = len(sample_ids) == len(set(sample_ids))
        self._hdf5_groups_unique = len(hdf5_groups) == len(set(hdf5_groups))
        fields_path = self.dataset_dir / "fields.h5"
        with h5py.File(fields_path, "r") as handle:
            missing_groups = [group for group in hdf5_groups if group not in handle]
        if missing_groups:
            raise ValueError(f"operator table references missing HDF5 groups: {missing_groups}")
        self._all_hdf5_groups_exist = True
        required_columns = ("physical_case_id", "run_id", "stage_index", "protocol_id")
        for row in self.rows:
            if any(not row.get(column, "") for column in required_columns):
                raise ValueError(f"operator row {row['sample_id']!r} lacks required metadata")
            try:
                stage_index = int(row["stage_index"])
            except ValueError as exc:
                raise ValueError(f"invalid stage_index in {row['sample_id']!r}") from exc
            if stage_index not in (1, 2, 3, 4):
                raise ValueError(f"invalid stage_index in {row['sample_id']!r}")

    @staticmethod
    def _decode(values: Iterable[object]) -> list[str]:
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]

    def _row(self, sample_id: str) -> dict[str, str]:
        try:
            return next(row for row in self.rows if row["sample_id"] == sample_id)
        except StopIteration as exc:
            raise KeyError(sample_id) from exc

    def split_ids(
        self, role: str, scheme: str = "protocol_frequency_holdout"
    ) -> list[str]:
        assignments = self.manifest["schemes"][scheme]["assignments"]
        return [sample_id for sample_id, assigned in assignments.items() if assigned == role]

    def assert_group_disjoint(
        self, scheme: str = "protocol_frequency_holdout"
    ) -> None:
        assignments = self.manifest["schemes"][scheme]["assignments"]
        for group_key in ("physical_case_id", "run_id"):
            roles_by_group: dict[str, set[str]] = {}
            for sample_id, role in assignments.items():
                group = self._row(sample_id)[group_key]
                roles_by_group.setdefault(group, set()).add(role)
            leaking = {
                group: roles for group, roles in roles_by_group.items() if len(roles) > 1
            }
            if leaking:
                raise ValueError(f"{group_key} leakage in {scheme}: {leaking}")

    def assert_protocol_declarations_match_rows(self) -> None:
        """Reject stale protocol summaries instead of trusting declarations."""
        audit = self.audit()
        if not audit.protocol_declarations_match_rows:
            raise ValueError(
                "published train/test protocol summaries do not match operator rows: "
                f"published={audit.published_protocols_by_role}, "
                f"rows={audit.protocols_by_role}"
            )

    def audit(self) -> CorpusAudit:
        scheme = self.manifest["schemes"]["protocol_frequency_holdout"]
        assignments = scheme["assignments"]
        groups_by_role: dict[str, set[str]] = {}
        runs_by_role: dict[str, set[str]] = {}
        protocols_by_role: dict[str, set[str]] = {}
        stages_by_role: dict[str, set[int]] = {}
        role_counts: dict[str, int] = {}
        for sample_id, role in assignments.items():
            row = self._row(sample_id)
            groups_by_role.setdefault(role, set()).add(row["physical_case_id"])
            runs_by_role.setdefault(role, set()).add(row["run_id"])
            protocols_by_role.setdefault(role, set()).add(row["protocol_id"])
            stages_by_role.setdefault(role, set()).add(int(row["stage_index"]))
            role_counts[role] = role_counts.get(role, 0) + 1
        published_counts = {str(k): int(v) for k, v in scheme.get("role_counts", {}).items()}
        if published_counts != role_counts:
            raise ValueError(
                f"published role_counts {published_counts} do not match assignments {role_counts}"
            )
        train_protocols = protocols_by_role.get("train", set())
        test_protocols = protocols_by_role.get("test", set())
        published_protocols = {
            "train": tuple(sorted(map(str, scheme.get("train_protocols", [])))),
            "test": tuple(sorted(map(str, scheme.get("test_protocols", [])))),
        }
        row_protocols = {
            role: tuple(sorted(protocols))
            for role, protocols in sorted(protocols_by_role.items())
        }
        declarations_match = all(
            published_protocols[role] == row_protocols.get(role, ())
            for role in ("train", "test")
        )
        role_names = tuple(sorted(role_counts))
        all_physical_groups = [groups_by_role.get(role, set()) for role in role_names]
        all_run_groups = [runs_by_role.get(role, set()) for role in role_names]
        physical_disjoint = all(
            left.isdisjoint(right)
            for index, left in enumerate(all_physical_groups)
            for right in all_physical_groups[index + 1 :]
        )
        runs_disjoint = all(
            left.isdisjoint(right)
            for index, left in enumerate(all_run_groups)
            for right in all_run_groups[index + 1 :]
        )
        return CorpusAudit(
            sample_count=len(self.rows),
            physical_case_count=len({row["physical_case_id"] for row in self.rows}),
            physical_run_count=len({row["run_id"] for row in self.rows}),
            stage_indices=tuple(sorted({int(row["stage_index"]) for row in self.rows})),
            stage_indices_by_role={
                role: tuple(sorted(stages)) for role, stages in sorted(stages_by_role.items())
            },
            train_count=role_counts.get("train", 0),
            test_count=role_counts.get("test", 0),
            golden_count=role_counts.get("golden_validation", 0),
            train_group_count=len(groups_by_role.get("train", set())),
            test_group_count=len(groups_by_role.get("test", set())),
            golden_group_count=len(groups_by_role.get("golden_validation", set())),
            aer_full_state_count=sum(
                row["aer_full_state_available"].lower() == "true" for row in self.rows
            ),
            role_counts_recomputed=dict(sorted(role_counts.items())),
            physical_group_counts_recomputed={
                role: len(groups) for role, groups in sorted(groups_by_role.items())
            },
            run_group_counts_recomputed={
                role: len(groups) for role, groups in sorted(runs_by_role.items())
            },
            protocols_by_role=row_protocols,
            train_test_protocol_disjoint=train_protocols.isdisjoint(test_protocols),
            train_test_physical_groups_disjoint=groups_by_role.get("train", set()).isdisjoint(
                groups_by_role.get("test", set())
            ),
            train_test_run_ids_disjoint=runs_by_role.get("train", set()).isdisjoint(
                runs_by_role.get("test", set())
            ),
            all_role_groups_disjoint=physical_disjoint and runs_disjoint,
            published_protocols_by_role=published_protocols,
            protocol_declarations_match_rows=declarations_match,
            assignment_covers_table_exactly=True,
            unique_sample_ids=self._sample_ids_unique,
            unique_hdf5_groups=self._hdf5_groups_unique,
            all_hdf5_groups_exist=self._all_hdf5_groups_exist,
        )

    def load_transition(self, sample_id: str) -> Transition:
        row = self._row(sample_id)
        with h5py.File(self.dataset_dir / "fields.h5", "r") as handle:
            group = handle[row["hdf5_group"]]
            labels = self._decode(group["side_state_labels"][...])
            side_values = np.asarray(group["side_state_values"][...], dtype=np.float64)
            side_map = dict(zip(labels, side_values))
            input_labels = [label for label in labels if label.endswith("_in") or label.endswith("_in_K")]
            # Preserve source order and derive each paired output name exactly.
            output_labels = [
                label[:-5] + "_out_K" if label.endswith("_in_K") else label[:-3] + "_out"
                for label in input_labels
            ]
            if len(input_labels) != self.layout.side_size or any(
                label not in side_map for label in output_labels
            ):
                raise ValueError("unexpected dataset-v1 side-state pairing")
            side_in_raw = np.asarray([side_map[x] for x in input_labels], dtype=np.float64)
            side_out_raw = np.asarray([side_map[x] for x in output_labels], dtype=np.float64)
            side_in_seen = np.isfinite(side_in_raw)
            # 293 K is an explicit neutral reference only for model input. The mask
            # preserves which values were truly observed.
            side_in = np.where(side_in_seen, side_in_raw, 293.0)
            side_out_seen = np.isfinite(side_out_raw)
            side_out = np.where(side_out_seen, side_out_raw, side_in)

            xi_in = np.asarray(group["sma_state_in_xi_51"][...], dtype=np.float64)
            state_in = self.layout.assemble(
                group["sma_T_in_51_K"][...],
                group["htf_directional_T_in_101_K"][...],
                xi_in,
                side_in,
            )
            state_out = self.layout.assemble(
                group["sma_T_out_51_K"][...],
                group["htf_directional_T_out_101_K"][...],
                xi_in,  # persistence carrier, deliberately excluded from supervision
                side_out,
            )
            observed = np.ones(self.layout.size, dtype=bool)
            observed[self.layout.xi] = False  # dataset v1.0 has no xi_out
            observed[self.layout.side] = side_out_seen
            energy_labels = tuple(self._decode(group["energy_labels"][...]))
            energy_values = np.asarray(group["energy_values_J"][...], dtype=np.float64)
            if energy_values.shape != (len(energy_labels),):
                raise ValueError("energy label/value shapes do not match")
            if len(set(energy_labels)) != len(energy_labels):
                raise ValueError("energy labels must be unique within a checkpoint")
            valve_mask = np.asarray(group["topology_valve_mask"][...], dtype=np.float64)
            if "time_local_s" not in group:
                raise ValueError(f"operator sample {sample_id!r} lacks time_local_s")
            time_local = np.asarray(group["time_local_s"][...], dtype=np.float64).reshape(-1)
            if (
                time_local.size < 2
                or not np.isfinite(time_local).all()
                or np.any(np.diff(time_local) < 0.0)
            ):
                raise ValueError(f"operator sample {sample_id!r} has an invalid time_local_s grid")
            duration_s = float(np.max(time_local) - np.min(time_local))
            if not np.isfinite(duration_s) or duration_s <= 0.0:
                raise ValueError(f"operator sample {sample_id!r} has a non-positive observed duration")
            published_duration = row.get("observed_stage_duration_s", "")
            if published_duration:
                try:
                    tabular_duration = float(published_duration)
                except ValueError as exc:
                    raise ValueError(
                        f"operator sample {sample_id!r} has an invalid tabular duration"
                    ) from exc
                if not np.isclose(tabular_duration, duration_s, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"operator sample {sample_id!r} duration disagrees with time_local_s"
                    )

        persistence_only = np.zeros(self.layout.size, dtype=bool)
        persistence_only[self.layout.xi] = True
        return Transition(
            sample_id=sample_id,
            physical_case_id=row["physical_case_id"],
            stage_index=int(row["stage_index"]),
            state_in=state_in,
            target=StageStateTarget(
                values=state_out,
                observed_mask=observed,
                persistence_only_mask=persistence_only,
            ),
            side_input_observed=side_in_seen,
            valve_mask=valve_mask,
            duration_s=duration_s,
            energy_labels=energy_labels,
            energy_values_j=energy_values,
        )
