"""Audited WP4 data access with physical-run split enforcement.

The v1.0 operator corpus is deliberately small.  This module does not create
additional pseudo-samples: each returned row maps one-to-one to an HDF5 group
and keeps the published protocol-frequency role.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


SMA_NODES = 51
HTF_NODES = 101
TIME_SAMPLES = 50
TEMPERATURE_PROBE_FIELDS = (
    "sma_T_mean_t_K",
    "sma_T_center_t_K",
    "sma_T_edge_t_K",
    "htf_T_in_t_K",
    "htf_T_out_t_K",
    "hot_sink_t_K",
    "cold_sink_t_K",
    "hot_hx_inlet_t_K",
    "hot_hx_outlet_t_K",
    "cold_hx_inlet_t_K",
    "cold_hx_outlet_t_K",
)


@dataclass(frozen=True)
class OperatorBatch:
    """Dense fields and conditioning variables for one published role."""

    sample_ids: tuple[str, ...]
    physical_case_ids: tuple[str, ...]
    protocol_ids: tuple[str, ...]
    branches: tuple[str, ...]
    duration_s: np.ndarray
    stage_index: np.ndarray
    sma_in: np.ndarray
    sma_out: np.ndarray
    htf_in: np.ndarray
    htf_out: np.ndarray
    topology: np.ndarray
    side_input: np.ndarray
    side_output_audit: np.ndarray
    side_input_observed: np.ndarray
    side_output_observed: np.ndarray
    side_input_labels: tuple[str, ...]
    side_output_labels: tuple[str, ...]
    time_local_s: np.ndarray
    mass_flow_t_kgps: np.ndarray
    temperature_probes_t_K: np.ndarray
    temperature_probe_labels: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def validate(self) -> None:
        n = len(self)
        expected = {
            "duration_s": (n,),
            "stage_index": (n,),
            "sma_in": (n, SMA_NODES),
            "sma_out": (n, SMA_NODES),
            "htf_in": (n, HTF_NODES),
            "htf_out": (n, HTF_NODES),
            "topology": (n, 6),
            "side_input": (n, 8),
            "side_output_audit": (n, 8),
            "side_input_observed": (n, 8),
            "side_output_observed": (n, 8),
            "time_local_s": (n, TIME_SAMPLES),
            "mass_flow_t_kgps": (n, TIME_SAMPLES),
            "temperature_probes_t_K": (n, TIME_SAMPLES, len(TEMPERATURE_PROBE_FIELDS)),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}; expected {shape}")
        for name in (
            "sma_in", "sma_out", "htf_in", "htf_out", "topology",
            "time_local_s", "mass_flow_t_kgps", "temperature_probes_t_K",
        ):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains non-finite values")
        if not np.isfinite(self.duration_s).all() or np.any(self.duration_s <= 0.0):
            raise ValueError("duration_s must contain finite positive observed stage durations")
        if len(self.branches) != n or not set(self.branches).issubset({"full", "half1", "half2"}):
            raise ValueError("branches must declare full, half1, or half2 for every sample")
        if len(self.side_input_labels) != 8 or len(self.side_output_labels) != 8:
            raise ValueError("side-state label layout must contain eight input/output pairs")
        if self.temperature_probe_labels != TEMPERATURE_PROBE_FIELDS:
            raise ValueError("temperature probe labels do not match the published contract")
        if np.any(np.diff(self.time_local_s, axis=1) < 0.0):
            raise ValueError("time_local_s must be nondecreasing")
        if not all(label.endswith(("_in", "_in_K")) for label in self.side_input_labels):
            raise ValueError("side_input contains a non-causal label")
        if not all(label.endswith(("_out", "_out_K")) for label in self.side_output_labels):
            raise ValueError("side_output_audit contains a non-output label")
        if not np.array_equal(self.side_input_observed, np.isfinite(self.side_input)):
            raise ValueError("side_input_observed disagrees with side_input missingness")
        if not np.array_equal(self.side_output_observed, np.isfinite(self.side_output_audit)):
            raise ValueError("side_output_observed disagrees with side_output_audit missingness")
        # Side-state NaNs are published missing values. Only the eight input
        # members may enter model features; outputs remain target/audit data.


def _read_table(root: Path) -> pd.DataFrame:
    parquet = root / "operator_samples.parquet"
    try:
        return pd.read_parquet(parquet)
    except (ImportError, OSError, ValueError):
        return pd.read_csv(root / "operator_samples.csv")


def _h5_path(dataset_root: Path, stored_path: str) -> Path:
    relative = Path(str(stored_path).replace("\\", "/"))
    # stored paths are rooted at SciML (for example datasets/v1.0/fields.h5)
    return dataset_root.parents[1] / relative


def _make_batch(rows: pd.DataFrame, dataset_root: Path) -> OperatorBatch:
    records: dict[str, list[np.ndarray]] = {
        "sma_in": [], "sma_out": [], "htf_in": [], "htf_out": [],
        "topology": [], "side_input": [], "side_output_audit": [],
        "side_input_observed": [], "side_output_observed": [],
        "duration_s": [],
        "time_local_s": [], "mass_flow_t_kgps": [], "temperature_probes_t_K": [],
    }
    arrays = {
        "sma_in": "sma_T_in_51_K",
        "sma_out": "sma_T_out_51_K",
        "htf_in": "htf_directional_T_in_101_K",
        "htf_out": "htf_directional_T_out_101_K",
        "topology": "topology_valve_mask",
    }
    handles: dict[Path, h5py.File] = {}
    input_labels_reference: tuple[str, ...] | None = None
    output_labels_reference: tuple[str, ...] | None = None
    try:
        for row in rows.itertuples(index=False):
            path = _h5_path(dataset_root, row.hdf5_file)
            if path not in handles:
                handles[path] = h5py.File(path, "r")
            handle = handles[path]
            group = handle[row.hdf5_group]
            for output_name, h5_name in arrays.items():
                records[output_name].append(np.asarray(group[h5_name], dtype=np.float64))
            if "time_local_s" not in group:
                raise ValueError(f"{row.sample_id} lacks time_local_s")
            time_local = np.asarray(group["time_local_s"], dtype=np.float64).reshape(-1)
            if time_local.size < 2 or not np.isfinite(time_local).all():
                raise ValueError(f"{row.sample_id} has an invalid local time grid")
            records["duration_s"].append(
                np.asarray(float(np.max(time_local) - np.min(time_local)), dtype=np.float64)
            )
            records["time_local_s"].append(time_local)
            if "mass_flow_t_kgps" not in group:
                raise ValueError(f"{row.sample_id} lacks mass_flow_t_kgps")
            records["mass_flow_t_kgps"].append(
                np.asarray(group["mass_flow_t_kgps"], dtype=np.float64).reshape(-1)
            )
            missing_probes = [name for name in TEMPERATURE_PROBE_FIELDS if name not in group]
            if missing_probes:
                raise ValueError(f"{row.sample_id} lacks temperature probes: {missing_probes}")
            records["temperature_probes_t_K"].append(
                np.column_stack(
                    [np.asarray(group[name], dtype=np.float64).reshape(-1) for name in TEMPERATURE_PROBE_FIELDS]
                )
            )
            labels = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in group["side_state_labels"][...]
            )
            values = np.asarray(group["side_state_values"][...], dtype=np.float64)
            side_map = dict(zip(labels, values))
            input_labels = tuple(
                label for label in labels if label.endswith("_in") or label.endswith("_in_K")
            )
            output_labels = tuple(
                label[:-5] + "_out_K" if label.endswith("_in_K") else label[:-3] + "_out"
                for label in input_labels
            )
            if len(input_labels) != 8 or any(label not in side_map for label in output_labels):
                raise ValueError("unexpected side-state input/output pairing")
            if input_labels_reference is None:
                input_labels_reference, output_labels_reference = input_labels, output_labels
            elif input_labels != input_labels_reference or output_labels != output_labels_reference:
                raise ValueError("side-state label order differs across operator samples")
            side_input = np.asarray([side_map[label] for label in input_labels], dtype=np.float64)
            side_output = np.asarray([side_map[label] for label in output_labels], dtype=np.float64)
            records["side_input"].append(side_input)
            records["side_output_audit"].append(side_output)
            records["side_input_observed"].append(np.isfinite(side_input))
            records["side_output_observed"].append(np.isfinite(side_output))
    finally:
        for handle in handles.values():
            handle.close()

    def stack(name: str, width: int) -> np.ndarray:
        if records[name]:
            return np.stack(records[name], axis=0)
        return np.empty((0, width), dtype=np.float64)

    batch = OperatorBatch(
        sample_ids=tuple(rows["sample_id"].astype(str)),
        physical_case_ids=tuple(rows["physical_case_id"].astype(str)),
        protocol_ids=tuple(rows["protocol_id"].astype(str)),
        branches=tuple(
            "full" if value == "full_cycle" else value
            for value in rows["branch"].astype(str)
        ),
        duration_s=stack("duration_s", 1).reshape(-1),
        stage_index=rows["stage_index"].to_numpy(dtype=np.int64),
        sma_in=stack("sma_in", SMA_NODES),
        sma_out=stack("sma_out", SMA_NODES),
        htf_in=stack("htf_in", HTF_NODES),
        htf_out=stack("htf_out", HTF_NODES),
        topology=stack("topology", 6),
        side_input=stack("side_input", 8),
        side_output_audit=stack("side_output_audit", 8),
        side_input_observed=stack("side_input_observed", 8).astype(bool),
        side_output_observed=stack("side_output_observed", 8).astype(bool),
        side_input_labels=input_labels_reference or (),
        side_output_labels=output_labels_reference or (),
        time_local_s=stack("time_local_s", TIME_SAMPLES),
        mass_flow_t_kgps=stack("mass_flow_t_kgps", TIME_SAMPLES),
        temperature_probes_t_K=stack(
            "temperature_probes_t_K", TIME_SAMPLES * len(TEMPERATURE_PROBE_FIELDS)
        ).reshape(-1, TIME_SAMPLES, len(TEMPERATURE_PROBE_FIELDS)),
        temperature_probe_labels=TEMPERATURE_PROBE_FIELDS,
    )
    batch.validate()
    return batch


def assert_group_isolation(batches: Iterable[OperatorBatch]) -> None:
    """Reject any physical run assigned to more than one role."""

    seen: set[str] = set()
    for batch in batches:
        current = set(batch.physical_case_ids)
        overlap = seen.intersection(current)
        if overlap:
            raise ValueError(f"physical_case_id leakage: {sorted(overlap)}")
        seen.update(current)


def load_protocol_holdout(
    dataset_root: str | Path | None = None,
) -> dict[str, OperatorBatch]:
    """Load train/test/golden roles exactly as published in dataset v1.0."""

    default = Path(__file__).resolve().parents[2] / "datasets" / "v1.0"
    root = Path(dataset_root).resolve() if dataset_root is not None else default
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    split = manifest["schemes"]["protocol_frequency_holdout"]
    if split.get("task_family") != "spatiotemporal_operator":
        raise ValueError("published protocol split is not an operator task")
    assignments = split["assignments"]
    table = _read_table(root)
    if table["sample_id"].duplicated().any():
        raise ValueError("operator table contains duplicate sample_id values")
    unknown = set(table["sample_id"]) - set(assignments)
    if unknown:
        raise ValueError(f"operator samples missing split assignments: {sorted(unknown)}")
    extra = set(assignments) - set(table["sample_id"])
    if extra:
        raise ValueError(f"split assignments reference unknown samples: {sorted(extra)}")

    result: dict[str, OperatorBatch] = {}
    for role in ("train", "test", "golden_validation"):
        ids = {sample_id for sample_id, value in assignments.items() if value == role}
        rows = table.loc[table["sample_id"].isin(ids)].copy()
        rows = rows.sort_values("sample_id", kind="stable").reset_index(drop=True)
        result[role] = _make_batch(rows, root)
    assert_group_isolation(result.values())
    observed_counts = {role: len(batch) for role, batch in result.items()}
    declared_counts = {role: int(split["role_counts"].get(role, 0)) for role in result}
    if observed_counts != declared_counts:
        raise ValueError(
            f"manifest role_counts disagree with assignments: {declared_counts} != {observed_counts}"
        )
    if set(result["train"].protocol_ids) & set(result["test"].protocol_ids):
        raise ValueError("train/test protocol identifiers overlap")
    return result


def manifest_coverage(dataset_root: str | Path | None = None) -> dict[str, object]:
    """Return recomputed evidence that every operator row is assigned once."""

    default = Path(__file__).resolve().parents[2] / "datasets" / "v1.0"
    root = Path(dataset_root).resolve() if dataset_root is not None else default
    table = _read_table(root)
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    split = manifest["schemes"]["protocol_frequency_holdout"]
    assignments = split["assignments"]
    table_ids = list(table["sample_id"].astype(str))
    table_set = set(table_ids)
    assigned_set = set(assignments)
    recomputed = {
        role: sum(value == role for value in assignments.values())
        for role in ("train", "test", "golden_validation")
    }
    declared = {role: int(split["role_counts"].get(role, 0)) for role in recomputed}
    return {
        "table_sample_count": len(table_ids),
        "assignment_count": len(assignments),
        "duplicate_table_sample_ids": len(table_ids) - len(table_set),
        "missing_assignments": sorted(table_set - assigned_set),
        "unknown_assignments": sorted(assigned_set - table_set),
        "recomputed_role_counts": recomputed,
        "declared_role_counts": declared,
        "role_counts_match": recomputed == declared,
        "coverage_complete": (
            len(table_ids) == len(table_set) == len(assignments)
            and table_set == assigned_set
            and recomputed == declared
        ),
    }
