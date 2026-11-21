"""Aggregate exported MATLAB v7 case artifacts into dataset v1.1-pilot-r3.

The release keeps the Simulink-internal SMA strip temperature and the
Simscape Pipe(TL) internal-fluid temperature as separate 50 x 101 fields.
Multiple input roots and every explicitly exported cycle are retained; no
cycle is copied or synthesized by this builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


SCHEMA_VERSION = "sciml_v11_pilot_dataset_v2"
CASE_SCHEMA_VERSION = "sciml_v11_case_artifact_v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    ROOT / "Output" / "sciml_v11_flowhtf_pump_r3",
    ROOT / "Output" / "sciml_v11_multicycle_r3",
)
DEFAULT_OUTPUT = ROOT / "SciML" / "datasets" / "v1.1-pilot-r3"

# Export and observation validation belong to the dataset pipeline.  The
# campaign driver is intentionally *not* excluded: it freezes ODE tolerances,
# Simscape local-solver settings and controller parameters, so changing it is
# a change in the numerical physics implementation even when the underlying
# .slx file is byte-identical.
PIPELINE_ONLY_FINGERPRINT_NAMES = {
    "Export_SCIML_V11_Case.m",
    "Validate_SCIML_Observation_Contract.m",
}

STAGES = {
    "stage1_loading": {
        "time_local_s": (50,),
        "pressure_Pa_t": (50,),
        "radial_coordinate_m": (51,),
        "u_m_t": (50, 51),
        "w_m_t": (50, 51),
        "T_K_t": (50, 51),
        "xi_t": (50, 51),
    },
    "stage2_hold_exchange": {
        "time_local_s": (50,),
        "sma_strip_coordinate_m": (101,),
        "sma_strip_T_K_t": (50, 101),
    },
    "stage3_unloading": {
        "time_local_s": (50,),
        "pressure_Pa_t": (50,),
        "radial_coordinate_m": (51,),
        "u_m_t": (50, 51),
        "w_m_t": (50, 51),
        "T_K_t": (50, 51),
        "xi_t": (50, 51),
    },
    "stage4_cooldown_exchange": {
        "time_local_s": (50,),
        "sma_strip_coordinate_m": (101,),
        "sma_strip_T_K_t": (50, 101),
    },
}

EXCHANGE_OPTIONAL_ARRAYS = (
    "time_global_s",
    "sma_T_in_51_K",
    "sma_T_out_51_K",
    "htf_directional_T_in_101_K",
    "htf_directional_T_out_101_K",
    "sma_T_mean_t_K",
    "sma_T_center_t_K",
    "sma_T_edge_t_K",
    "htf_T_in_t_K",
    "htf_T_out_t_K",
    "mass_flow_t_kgps",
    "pressure_pi_command_MPa_t",
    "hot_sink_t_K",
    "cold_sink_t_K",
    "hot_hx_inlet_t_K",
    "hot_hx_outlet_t_K",
    "cold_hx_inlet_t_K",
    "cold_hx_outlet_t_K",
    "htf_internal_pipe_index_101",
    "htf_internal_time_local_s",
    "htf_internal_T_K_t",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def require_shape(case_id: str, stage: str, name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = as_array(value)
    if array.shape != shape:
        raise ValueError(f"{case_id}: {stage}.{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{case_id}: {stage}.{name} contains non-finite values")
    return array


def scalar(mapping: dict[str, Any], name: str, default: float = np.nan) -> float:
    try:
        value = np.asarray(mapping.get(name, default), dtype=float)
        return float(value.reshape(-1)[0]) if value.size == 1 else default
    except (TypeError, ValueError):
        return default


def boolean(mapping: dict[str, Any], name: str, default: bool = False) -> bool:
    value = mapping.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def normalize_input_roots(input_roots: Path | Iterable[Path]) -> tuple[Path, ...]:
    if isinstance(input_roots, Path):
        roots = (input_roots,)
    else:
        roots = tuple(Path(root) for root in input_roots)
    if not roots:
        raise ValueError("At least one input root is required")
    return roots


def artifact_paths(input_roots: Path | Iterable[Path]) -> list[Path]:
    roots = normalize_input_roots(input_roots)
    paths = sorted(
        {path.resolve() for root in roots for path in root.rglob("sciml_case_v11.mat")}
    )
    if not paths:
        raise FileNotFoundError(f"No exported case artifacts found under {roots}")
    return paths


def load_case(path: Path) -> dict[str, Any]:
    artifact = loadmat(path, simplify_cells=True)["case_artifact"]
    if str(artifact.get("schema_version")) != CASE_SCHEMA_VERSION:
        raise ValueError(f"{path}: unexpected case schema {artifact.get('schema_version')!r}")
    return artifact


def semantic_case_sha256(value: Any) -> str:
    """Hash decoded MATLAB content, excluding container headers/timestamps."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                digest.update(b"\0")
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(f"sequence:{len(item)}\0".encode("ascii"))
            for child in item:
                update(child)
            return
        if isinstance(item, (str, bytes)):
            payload = item.encode("utf-8") if isinstance(item, str) else item
            digest.update(b"text\0")
            digest.update(payload)
            digest.update(b"\0")
            return
        array = np.asarray(item)
        if array.dtype.kind in {"O", "U", "S"}:
            digest.update(f"text_array:{array.shape}\0".encode("ascii"))
            for child in array.reshape(-1).tolist():
                update(child)
            return
        normalized = np.ascontiguousarray(array)
        digest.update(f"array:{normalized.dtype.str}:{normalized.shape}\0".encode("ascii"))
        digest.update(normalized.tobytes())

    update(value)
    return digest.hexdigest()


def cycle_payloads(case: dict[str, Any]) -> list[dict[str, Any]]:
    raw = case.get("cycles", [])
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        return [item for item in raw if isinstance(item, dict)]
    return []


def validate_case(case: dict[str, Any]) -> None:
    case_id = str(case["case_id"])
    if not boolean(case, "run_ok"):
        raise ValueError(f"{case_id}: run_ok is false")
    if not boolean(case, "observation_contract_ok"):
        raise ValueError(f"{case_id}: observation_contract_ok is false")
    availability = case["availability"]
    htf_available = boolean(availability, "htf_internal_full_field_history")
    for stage, fields in STAGES.items():
        payload = case[stage]
        for name, shape in fields.items():
            require_shape(case_id, stage, name, payload[name], shape)
        time = as_array(payload["time_local_s"]).reshape(-1)
        if np.any(np.diff(time) < 0.0):
            raise ValueError(f"{case_id}: {stage} time is not monotonic")
    for stage in ("stage1_loading", "stage3_unloading"):
        xi = as_array(case[stage]["xi_t"])
        if xi.min() < -1e-10 or xi.max() > 1.0 + 1e-10:
            raise ValueError(f"{case_id}: {stage} xi is outside [0, 1]")
    semantics = str(case["stage2_hold_exchange"]["sma_strip_history_semantics"]).lower()
    if "sma strip" not in semantics or "not the directional htf" not in semantics:
        raise ValueError(f"{case_id}: exchange field semantics are ambiguous")
    if htf_available:
        for stage in ("stage2_hold_exchange", "stage4_cooldown_exchange"):
            payload = case[stage]
            require_shape(case_id, stage, "htf_internal_T_K_t", payload["htf_internal_T_K_t"], (50, 101))
            require_shape(
                case_id,
                stage,
                "htf_internal_pipe_index_101",
                payload["htf_internal_pipe_index_101"],
                (101,),
            )
            htf_semantics = str(payload["htf_internal_field_semantics"]).lower()
            if "internal fluid" not in htf_semantics or "not the sma strip" not in htf_semantics:
                raise ValueError(f"{case_id}: {stage} HTF field semantics are ambiguous")
    cycles = cycle_payloads(case)
    expected_cycle_count = int(scalar(case, "cycle_count", 0.0))
    if len(cycles) != expected_cycle_count or expected_cycle_count < 1:
        raise ValueError(
            f"{case_id}: exported cycles={len(cycles)} does not match cycle_count={expected_cycle_count}"
        )
    for cycle_number, cycle in enumerate(cycles, start=1):
        if int(scalar(cycle, "cycle_index", 0.0)) != cycle_number:
            raise ValueError(f"{case_id}: non-contiguous cycle index at {cycle_number}")
        for stage, fields in STAGES.items():
            payload = cycle[stage]
            for name, shape in fields.items():
                require_shape(case_id, f"cycle_{cycle_number:03d}/{stage}", name, payload[name], shape)
        if htf_available:
            for stage in ("stage2_hold_exchange", "stage4_cooldown_exchange"):
                require_shape(
                    case_id,
                    f"cycle_{cycle_number:03d}/{stage}",
                    "htf_internal_T_K_t",
                    cycle[stage]["htf_internal_T_K_t"],
                    (50, 101),
                )


def write_numeric(group: h5py.Group, name: str, value: Any) -> None:
    array = as_array(value)
    if array.ndim == 0:
        group.create_dataset(name, data=array)
    else:
        group.create_dataset(name, data=array, compression="gzip", compression_opts=4)


def write_stage_fields(stage_group: h5py.Group, payload: dict[str, Any], required: dict[str, tuple[int, ...]]) -> None:
    for name in required:
        write_numeric(stage_group, name, payload[name])
    if "sma_strip_T_K_t" in required:
        for name in EXCHANGE_OPTIONAL_ARRAYS:
            if name in payload and np.asarray(payload[name]).size:
                write_numeric(stage_group, name, payload[name])
        for section in ("energy", "side_state"):
            nested = stage_group.create_group(section)
            for name, value in payload.get(section, {}).items():
                array = np.asarray(value)
                if np.issubdtype(array.dtype, np.number):
                    write_numeric(nested, name, value)


def write_case_fields(root: h5py.Group, case: dict[str, Any]) -> str:
    case_id = str(case["case_id"])
    group_name = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:20]
    case_group = root.create_group(group_name)
    case_group.attrs.update(
        case_id=case_id,
        material_id=str(case["material_id"]),
        physics_ok=boolean(case, "physics_ok"),
        sma_strip_field_semantics="time-resolved Simulink-internal SMA strip temperature; not HTF internal field",
        htf_internal_full_field_available=boolean(
            case["availability"], "htf_internal_full_field_history"
        ),
    )
    for stage_name, required in STAGES.items():
        payload = case[stage_name]
        stage_group = case_group.create_group(stage_name)
        write_stage_fields(stage_group, payload, required)
    cycles_group = case_group.create_group("cycles")
    for cycle in cycle_payloads(case):
        cycle_index = int(scalar(cycle, "cycle_index", 0.0))
        cycle_group = cycles_group.create_group(f"cycle_{cycle_index:03d}")
        for stage_name, required in STAGES.items():
            stage_group = cycle_group.create_group(stage_name)
            write_stage_fields(stage_group, cycle[stage_name], required)
        performance_group = cycle_group.create_group("performance")
        for name, value in cycle.get("performance", {}).items():
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number):
                write_numeric(performance_group, name, value)
    return f"/cases/{group_name}"


def validation_warnings(case_dir: Path) -> str:
    path = case_dir / "validation_summary.json"
    if not path.exists():
        return "missing_validation_summary"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unreadable_validation_summary"
    reasons: list[str] = [str(item) for item in payload.get("warning_reasons", [])]
    cycle_checks = payload.get("cycle_checks", [])
    if isinstance(cycle_checks, dict):
        cycle_checks = [cycle_checks]
    for cycle in cycle_checks:
        if isinstance(cycle, dict):
            reasons.extend(str(item) for item in cycle.get("warning_reasons", []))
    return ";".join(sorted(set(reasons)))


def fingerprint_logical_path(raw_path: str) -> str:
    """Return a workspace-independent identity for a registered source file."""
    normalized = str(raw_path).replace("\\", "/").strip()
    parts = [part for part in normalized.split("/") if part]
    if "SciML" in parts:
        # Use the last anchor defensively if a parent folder is also named SciML.
        anchor = len(parts) - 1 - parts[::-1].index("SciML")
        return "/".join(parts[anchor:])
    return parts[-1] if parts else ""


def _current_fingerprint_source(raw_path: str) -> Path:
    original = Path(str(raw_path))
    if original.is_file():
        return original
    logical = fingerprint_logical_path(raw_path)
    candidate = ROOT.joinpath(*logical.split("/"))
    return candidate


def request_config_identity(request_path: Path) -> tuple[str, str, bool, bool]:
    if not request_path.exists():
        return "", "", False, False
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", "", False, False
    source_config = str(request.get("source_config_path", ""))
    config_logical_path = fingerprint_logical_path(source_config)
    recorded = ""
    for entry in request.get("implementation_fingerprint", []):
        if fingerprint_logical_path(str(entry.get("path", ""))) == config_logical_path:
            recorded = str(entry.get("sha256", ""))
            break
    config_path = _current_fingerprint_source(source_config)
    current = sha256(config_path) if config_path.is_file() else ""
    # The request embeds case_spec and run_options, so it remains replayable
    # even when a campaign schedule later adds unrelated cases.
    self_contained = isinstance(request.get("case_spec"), dict) and isinstance(
        request.get("run_options"), dict
    )
    return recorded, current, bool(recorded and current and recorded == current), self_contained


def request_fingerprints(request_path: Path) -> tuple[str, str]:
    if not request_path.exists():
        return "", ""
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", ""
    entries = request.get("implementation_fingerprint", [])
    if not isinstance(entries, list):
        return "", ""

    def aggregate(selected: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for entry in sorted(
            selected,
            key=lambda item: fingerprint_logical_path(str(item.get("path", ""))),
        ):
            digest.update(
                fingerprint_logical_path(str(entry.get("path", ""))).encode("utf-8")
            )
            digest.update(b"\0")
            digest.update(str(entry.get("sha256", "")).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest() if selected else ""

    pipeline = [entry for entry in entries if isinstance(entry, dict)]
    pipeline_only_names = PIPELINE_ONLY_FINGERPRINT_NAMES | {
        Path(str(request.get("source_config_path", ""))).name
    }
    physics = [
        entry
        for entry in pipeline
        if Path(str(entry.get("path", ""))).name
        not in pipeline_only_names
    ]
    return aggregate(physics), aggregate(pipeline)


def export_provenance_identity(case_dir: Path, artifact_path: Path) -> dict[str, Any]:
    path = case_dir / "sciml_export_provenance.json"
    empty = {
        "export_provenance_available": False,
        "export_artifact_hash_matches": False,
        "exporter_sha256": "",
        "validator_sha256": "",
        "source_export_provenance_sha256": "",
    }
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    return {
        "export_provenance_available": True,
        "export_artifact_hash_matches": str(payload.get("artifact_sha256", ""))
        == sha256(artifact_path),
        "exporter_sha256": str(payload.get("exporter_sha256", "")),
        "validator_sha256": str(payload.get("validator_sha256", "")),
        "source_export_provenance_sha256": sha256(path),
    }


def condition_id(case: dict[str, Any]) -> str:
    geometry = case["geometry"]
    protocol = case["protocol"]
    payload = {
        "material_id": str(case["material_id"]),
        "radius_m": scalar(geometry, "radius_m"),
        "thickness_m": scalar(geometry, "thickness_m"),
        "pressure_Pa": scalar(geometry, "pressure_Pa"),
        "flush_volume_target_ratio": scalar(protocol, "flush_volume_target_ratio"),
        "exchange_duration_s": scalar(protocol, "exchange_duration_s"),
        "cold_load_mode": str(protocol["cold_load_mode"]),
        "cold_load_heat_W": scalar(protocol, "cold_load_heat_W"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "condition:" + hashlib.sha256(encoded).hexdigest()[:20]


def table_row(case: dict[str, Any], path: Path, hdf5_group: str) -> dict[str, Any]:
    geometry = case["geometry"]
    protocol = case["protocol"]
    performance = case["performance"]
    availability = case["availability"]
    request_path = path.parent / "sciml_case_request.json"
    result_path = path.parent / "run_result.mat"
    request_config_sha, current_config_sha, config_matches, request_self_contained = (
        request_config_identity(request_path)
    )
    physics_fingerprint, pipeline_fingerprint = request_fingerprints(request_path)
    export_identity = export_provenance_identity(path.parent, path)
    row = {
        "case_id": str(case["case_id"]),
        "physical_case_id": str(case["case_id"]),
        "condition_id": condition_id(case),
        "material_id": str(case["material_id"]),
        "solver_compatibility_commit": str(case["solver_compatibility_commit"]),
        "run_ok": boolean(case, "run_ok"),
        "physics_ok": boolean(case, "physics_ok"),
        "observation_contract_ok": boolean(case, "observation_contract_ok"),
        "paper_grade_valid": boolean(performance, "paper_grade_valid"),
        "validation_warnings": validation_warnings(path.parent),
        "radius_m": scalar(geometry, "radius_m"),
        "thickness_m": scalar(geometry, "thickness_m"),
        "pressure_Pa": scalar(geometry, "pressure_Pa"),
        "sma_mass_kg": scalar(geometry, "sma_mass_kg"),
        "flush_volume_target_ratio": scalar(protocol, "flush_volume_target_ratio"),
        "exchange_duration_s": scalar(protocol, "exchange_duration_s"),
        "cycle_count": int(scalar(protocol, "cycle_count", 0.0)),
        "cold_load_mode": str(protocol["cold_load_mode"]),
        "cold_load_heat_W": scalar(protocol, "cold_load_heat_W"),
        "Qcool_J": scalar(performance, "Qcool_J"),
        "W1_J": scalar(performance, "W1_J"),
        "Wunload_recovered_J": scalar(performance, "Wunload_recovered_J"),
        "Wpump_J": scalar(performance, "Wpump_J"),
        "Wtotal_for_COP_J": scalar(performance, "Wtotal_for_COP_J"),
        "COP": scalar(performance, "COP"),
        "SCP_W_per_kg": scalar(performance, "SCP_W_per_kg"),
        "deltaT_hot_cold_max_K": scalar(performance, "deltaT_hot_cold_max_K"),
        "cycle_duration_s": scalar(performance, "cycle_duration_s"),
        "pump_work_available": boolean(performance, "pump_work_available"),
        "htf_internal_full_field_available": boolean(
            availability, "htf_internal_full_field_history"
        ),
        "hdf5_group": hdf5_group,
        "source_artifact": str(path.resolve()),
        "source_artifact_sha256": sha256(path),
        "source_artifact_semantic_sha256": semantic_case_sha256(case),
        "source_request_sha256": sha256(request_path) if request_path.exists() else "",
        "request_config_sha256": request_config_sha,
        "current_config_sha256": current_config_sha,
        "request_config_matches_current": config_matches,
        "request_self_contained": request_self_contained,
        "physics_implementation_fingerprint_sha256": physics_fingerprint,
        "pipeline_implementation_fingerprint_sha256": pipeline_fingerprint,
        "source_run_result_sha256": sha256(result_path) if result_path.exists() else "",
    }
    row.update(export_identity)
    return row


def cycle_rows(case: dict[str, Any], case_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_columns = (
        "case_id",
        "physical_case_id",
        "condition_id",
        "material_id",
        "solver_compatibility_commit",
        "run_ok",
        "physics_ok",
        "observation_contract_ok",
        "radius_m",
        "thickness_m",
        "pressure_Pa",
        "sma_mass_kg",
        "flush_volume_target_ratio",
        "exchange_duration_s",
        "cycle_count",
        "cold_load_mode",
        "cold_load_heat_W",
        "hdf5_group",
        "source_artifact",
        "source_artifact_sha256",
    )
    for cycle in cycle_payloads(case):
        cycle_index = int(scalar(cycle, "cycle_index", 0.0))
        performance = cycle["performance"]
        row = {name: case_row[name] for name in base_columns}
        row.update(
            cycle_sample_id=f"{case_row['case_id']}:cycle_{cycle_index:03d}",
            cycle_index=cycle_index,
            hdf5_cycle_group=f"{case_row['hdf5_group']}/cycles/cycle_{cycle_index:03d}",
            Qcool_J=scalar(performance, "Qcool_J"),
            W1_J=scalar(performance, "W1_J"),
            Wunload_recovered_J=scalar(performance, "Wunload_recovered_J"),
            Wpump_J=scalar(performance, "Wpump_J"),
            Wtotal_for_COP_J=scalar(performance, "Wtotal_for_COP_J"),
            COP=scalar(performance, "COP"),
            SCP_W_per_kg=scalar(performance, "SCP_W_per_kg"),
            deltaT_hot_cold_max_K=scalar(performance, "deltaT_hot_cold_max_K"),
            cycle_duration_s=scalar(performance, "cycle_duration_s"),
            pump_work_available=boolean(performance, "pump_work_available"),
            cycle_fields_finite=boolean(performance, "complete_cycle_fields_finite"),
            artifact_complete=boolean(performance, "paper_grade_valid"),
        )
        rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    records = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(paths)
    ]
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(record["sha256"].encode("ascii"))
    return {"artifacts": records, "aggregate_sha256": aggregate.hexdigest()}


def source_content_hash(frame: pd.DataFrame) -> str:
    """Stable identity derived only from immutable case/source content."""
    digest = hashlib.sha256()
    columns = (
        "case_id",
        "source_artifact_semantic_sha256",
        "source_request_sha256",
        "source_run_result_sha256",
    )
    for row in frame.sort_values("case_id")[list(columns)].itertuples(index=False, name=None):
        for value in row:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def expected_case_ids(config_path: Path | None) -> list[str]:
    if config_path is None:
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    identifiers = [str(case["case_id"]) for case in payload.get("cases", [])]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Expected-case config is empty or has duplicate case_id values: {config_path}")
    return identifiers


def verified_attrition_ledger(
    ledger_path: Path | None,
    expected_config: Path | None,
    expected_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Validate preregistered missing-primary evidence without relabeling it.

    Supplemental artifacts may restore statistical power, but they must never
    make an absent primary look observed.  Every absent primary therefore
    needs immutable evidence in the campaign attrition ledger.  Two evidence
    classes are admitted: a deterministic execution-domain failure reproduced
    by at least three clean attempts, or a completed run that cannot be
    exported because its physics/observation contract is negative.
    """

    if ledger_path is None:
        return [], True, []
    errors: list[str] = []
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        return [], False, ["attrition ledger entries must be a list of mappings"]
    protocol = payload.get("protocol", {})
    maximum = int(protocol.get("max_primary_attrition", 0))
    if maximum <= 0 or len(entries) > maximum:
        errors.append(
            f"attrition entry count {len(entries)} exceeds registered maximum {maximum}"
        )
    identifiers = [str(item.get("case_id", "")) for item in entries]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        errors.append("attrition case_id values must be nonempty and unique")
    unexpected = sorted(set(identifiers) - set(expected_ids))
    if unexpected:
        errors.append(f"attrition ledger contains non-primary case_ids: {unexpected}")
    if expected_config is None:
        errors.append("attrition evidence requires an expected campaign config")
    else:
        recorded = str(payload.get("campaign_config_sha256", "")).lower()
        current = sha256(expected_config)
        if recorded != current:
            errors.append(
                f"attrition campaign config hash mismatch: recorded={recorded}, current={current}"
            )

    def verify_file(record: dict[str, Any], label: str) -> Path | None:
        relative = str(record.get("evidence", record.get("path", ""))).strip()
        recorded_hash = str(record.get("sha256", "")).lower()
        target = (ROOT / relative).resolve() if relative else None
        if target is None or not target.is_file():
            errors.append(f"{label}: evidence file is absent: {relative}")
            return None
        current_hash = sha256(target)
        if recorded_hash != current_hash:
            errors.append(
                f"{label}: evidence hash mismatch for {relative}: "
                f"recorded={recorded_hash}, current={current_hash}"
            )
            return None
        return target

    for entry in entries:
        case_id = str(entry.get("case_id", ""))
        disposition = str(entry.get("disposition", ""))
        if disposition == "terminal_execution_domain_failure":
            attempts = entry.get("attempts", [])
            attempt_count = int(entry.get("attempt_count", 0))
            if (
                not isinstance(attempts, list)
                or attempt_count < 3
                or len(attempts) < 3
                or attempt_count != len(attempts)
            ):
                errors.append(
                    f"{case_id}: terminal failure requires at least three enumerated attempts"
                )
                continue
            for index, attempt in enumerate(attempts, start=1):
                if not isinstance(attempt, dict):
                    errors.append(f"{case_id}: attempt {index} is not a mapping")
                    continue
                verify_file(attempt, f"{case_id} attempt {index}")
        elif disposition == "completed_nonexportable_negative":
            evidence_files = entry.get("evidence_files", [])
            if not isinstance(evidence_files, list):
                errors.append(f"{case_id}: evidence_files must be a list")
                continue
            by_role: dict[str, Path] = {}
            for record in evidence_files:
                if not isinstance(record, dict):
                    errors.append(f"{case_id}: evidence record is not a mapping")
                    continue
                role = str(record.get("role", ""))
                verified = verify_file(record, f"{case_id} {role}")
                if role and verified is not None:
                    by_role[role] = verified
            required = {"validation_summary", "observation_contract_summary", "run_result"}
            if not required <= set(by_role):
                errors.append(
                    f"{case_id}: completed negative evidence lacks roles "
                    f"{sorted(required - set(by_role))}"
                )
                continue
            validation = json.loads(by_role["validation_summary"].read_text(encoding="utf-8"))
            observation = json.loads(
                by_role["observation_contract_summary"].read_text(encoding="utf-8")
            )
            if not bool(validation.get("run_ok", False)):
                errors.append(f"{case_id}: completed-negative validation does not have run_ok=true")
            if bool(validation.get("overall_ok", False)) and bool(observation.get("ok", False)):
                errors.append(
                    f"{case_id}: completed-negative evidence actually passes both strict summaries"
                )
        else:
            errors.append(f"{case_id}: unsupported attrition disposition {disposition!r}")
    return entries, not errors, errors


def matrix_coverage(
    observed_ids: Iterable[str],
    expected_ids: Iterable[str],
    allow_supplemental_cases: bool,
) -> tuple[list[str], list[str], bool]:
    observed = set(observed_ids)
    expected = set(expected_ids)
    missing = sorted(expected - observed)
    supplemental = sorted(observed - expected) if expected else []
    complete = bool(
        not expected or (not missing and (allow_supplemental_cases or not supplemental))
    )
    return missing, supplemental, complete


def confirmation_quality_gate(
    frame: pd.DataFrame,
    *,
    expected_ids: Iterable[str],
    missing_expected: Iterable[str],
    unexpected: Iterable[str],
    allow_supplemental_cases: bool,
    target_version: str,
) -> dict[str, Any]:
    """Audit an exact confirmation matrix independently of training-set size.

    ``paper_grade_gate`` is the development-corpus gate and intentionally
    requires at least 200 independent physics-valid conditions.  A registered
    50- or 12-case confirmation matrix cannot and must not satisfy that size
    requirement.  Confirmation instead requires the exact preregistered
    matrix and strict validity/provenance for every observed case.
    """

    expected = tuple(str(value) for value in expected_ids)
    missing = tuple(str(value) for value in missing_expected)
    extra = tuple(str(value) for value in unexpected)
    applicable = target_version.startswith("1.2-confirm")
    blockers: list[str] = []
    if not applicable:
        blockers.append("dataset target is not a v1.2 confirmation matrix")
    if not expected:
        blockers.append("registered confirmation matrix is absent")
    if missing:
        blockers.append("registered confirmation cases are missing")
    if extra:
        blockers.append("unregistered supplemental cases are present")
    if allow_supplemental_cases:
        blockers.append("supplemental cases are permitted by the build")
    if len(frame) != len(expected):
        blockers.append("observed case count differs from the registered matrix")

    required_case_gates = (
        "run_ok",
        "physics_ok",
        "paper_grade_valid",
        "observation_contract_ok",
        "pump_work_available",
        "htf_internal_full_field_available",
        "request_config_matches_current",
        "request_self_contained",
        "export_provenance_available",
        "export_artifact_hash_matches",
    )
    for name in required_case_gates:
        if name not in frame or len(frame) == 0 or not frame[name].astype(bool).all():
            blockers.append(f"not every confirmation case passes {name}")
    fingerprint_count = (
        int(
            frame["physics_implementation_fingerprint_sha256"]
            .replace("", np.nan)
            .nunique()
        )
        if "physics_implementation_fingerprint_sha256" in frame
        else 0
    )
    if fingerprint_count != 1:
        blockers.append("physics implementation fingerprint is not unique")
    return {
        "applicable": applicable,
        "contract": "exact_registered_matrix_and_every_case_strict_validity",
        "passed": not blockers,
        "blocking_reasons": blockers,
    }


def build(
    input_roots: Path | Iterable[Path],
    output_root: Path,
    expected_config: Path | None = None,
    allow_supplemental_cases: bool = False,
    attrition_ledger: Path | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    roots = normalize_input_roots(input_roots)
    paths = artifact_paths(roots)
    h5_path = output_root / "fields.h5"
    rows: list[dict[str, Any]] = []
    all_cycle_rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "w") as h5:
        h5.attrs.update(
            schema_version=SCHEMA_VERSION,
            field_semantics=(
                "SMA mechanical histories, SMA strip histories, and distinct "
                "Simscape Pipe(TL) internal-fluid histories"
            ),
        )
        cases_group = h5.create_group("cases")
        for path in paths:
            case = load_case(path)
            validate_case(case)
            hdf5_group = write_case_fields(cases_group, case)
            row = table_row(case, path, hdf5_group)
            rows.append(row)
            all_cycle_rows.extend(cycle_rows(case, row))

    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    if not frame["case_id"].is_unique:
        raise ValueError("Duplicate case_id values in v1.1-pilot input")
    csv_path = output_root / "case_index.csv"
    parquet_path = output_root / "case_index.parquet"
    frame.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    frame.to_parquet(parquet_path, index=False)
    cycle_frame = pd.DataFrame(all_cycle_rows).sort_values(["case_id", "cycle_index"]).reset_index(drop=True)
    if not cycle_frame["cycle_sample_id"].is_unique:
        raise ValueError("Duplicate cycle_sample_id values in v1.1-pilot-r3 input")
    cycle_csv_path = output_root / "cycle_index.csv"
    cycle_parquet_path = output_root / "cycle_index.parquet"
    cycle_frame.to_csv(cycle_csv_path, index=False, encoding="utf-8", lineterminator="\n")
    cycle_frame.to_parquet(cycle_parquet_path, index=False)
    stable_content_hash = source_content_hash(frame)
    expected_ids = expected_case_ids(expected_config)
    expected_payload = (
        json.loads(expected_config.read_text(encoding="utf-8-sig"))
        if expected_config is not None
        else {}
    )
    target_version = str(expected_payload.get("dataset_target_version", ""))
    missing_expected, unexpected, expected_complete = matrix_coverage(
        frame["case_id"], expected_ids, allow_supplemental_cases
    )
    ledger_entries, attrition_evidence_complete, attrition_evidence_errors = (
        verified_attrition_ledger(attrition_ledger, expected_config, expected_ids)
    )
    ledger_ids = {str(entry.get("case_id", "")) for entry in ledger_entries}
    terminal_exclusion_complete = bool(
        expected_ids
        and attrition_evidence_complete
        and set(missing_expected).issubset(ledger_ids)
        and not (ledger_ids - set(missing_expected))
    )
    observed_run_ok = bool(frame["run_ok"].astype(bool).all()) if len(frame) else False
    observed_provenance_complete = bool(
        (
            frame["export_provenance_available"].astype(bool)
            & frame["export_artifact_hash_matches"].astype(bool)
        ).all()
    ) if len(frame) else False
    distinct_valid_conditions = int(
        frame.loc[frame["physics_ok"].astype(bool), "condition_id"].nunique()
    )
    paper_grade_evidence_complete = bool(
        (expected_complete or terminal_exclusion_complete)
        and len(frame) >= 200
        and distinct_valid_conditions >= 200
        and observed_run_ok
        and observed_provenance_complete
        and int(frame["physics_implementation_fingerprint_sha256"].replace("", np.nan).nunique()) == 1
    )
    confirmation_gate = confirmation_quality_gate(
        frame,
        expected_ids=expected_ids,
        missing_expected=missing_expected,
        unexpected=unexpected,
        allow_supplemental_cases=allow_supplemental_cases,
        target_version=target_version,
    )

    quality = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "case_count": int(len(frame)),
        "expected_case_count": len(expected_ids) if expected_ids else None,
        "missing_expected_case_ids": missing_expected,
        "unexpected_case_ids": unexpected,
        "allow_supplemental_cases": bool(allow_supplemental_cases),
        "expected_matrix_complete": expected_complete,
        "attrition_ledger": str(attrition_ledger.resolve()) if attrition_ledger else "",
        "attrition_evidence_complete": attrition_evidence_complete,
        "attrition_evidence_errors": attrition_evidence_errors,
        "terminal_exclusion_case_ids": sorted(ledger_ids & set(missing_expected)),
        "terminal_exclusion_complete": terminal_exclusion_complete,
        "request_config_match_count": int(frame["request_config_matches_current"].sum()),
        "request_config_mismatch_count": int((~frame["request_config_matches_current"].astype(bool)).sum()),
        "request_self_contained_count": int(frame["request_self_contained"].sum()),
        "physics_implementation_fingerprint_count": int(
            frame["physics_implementation_fingerprint_sha256"].replace("", np.nan).nunique()
        ),
        "pipeline_implementation_fingerprint_count": int(
            frame["pipeline_implementation_fingerprint_sha256"].replace("", np.nan).nunique()
        ),
        "export_provenance_complete_count": int(
            (
                frame["export_provenance_available"].astype(bool)
                & frame["export_artifact_hash_matches"].astype(bool)
            ).sum()
        ),
        "cycle_sample_count": int(len(cycle_frame)),
        "observed_stage_transition_count": int(4 * len(cycle_frame.loc[cycle_frame["physics_ok"].astype(bool)])),
        "run_ok_count": int(frame["run_ok"].sum()),
        "physics_ok_count": int(frame["physics_ok"].sum()),
        "paper_grade_valid_count": int(frame["paper_grade_valid"].sum()),
        "observation_contract_ok_count": int(frame["observation_contract_ok"].sum()),
        "materials": sorted(frame["material_id"].unique().tolist()),
        "exchange_durations_s": sorted(frame["exchange_duration_s"].unique().tolist()),
        "geometries": int(frame[["radius_m", "thickness_m"]].drop_duplicates().shape[0]),
        "all_required_fields_finite": True,
        "htf_internal_full_field_available": bool(
            frame["htf_internal_full_field_available"].astype(bool).all()
        ),
        "artifact_complete_case_count": int(frame["paper_grade_valid"].sum()),
        "paper_grade_gate": paper_grade_evidence_complete,
        "confirmation_quality_gate": confirmation_gate,
        "source_content_sha256": stable_content_hash,
        "limitations": [
            "Pilot corpus is too small for paper-grade model training.",
            f"{len(frame)} physical cases and {len(cycle_frame)} observed cycles are insufficient for generalization claims unless the registered readiness gates pass.",
            "SMA strip and HTF internal fields are distinct and must never be substituted for one another.",
            "No experimental data are present.",
            "Rows with physics_ok=false remain diagnostic and must not be used as valid targets.",
        ],
    }
    quality_path = output_root / "quality_report.json"
    write_json(quality_path, quality)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "repository_head_at_build": git_head(),
        "input_roots": [str(root.resolve()) for root in roots],
        "expected_config": str(expected_config.resolve()) if expected_config else "",
        "expected_config_sha256": sha256(expected_config) if expected_config else "",
        "allow_supplemental_cases": bool(allow_supplemental_cases),
        "attrition_ledger": str(attrition_ledger.resolve()) if attrition_ledger else "",
        "attrition_ledger_sha256": sha256(attrition_ledger) if attrition_ledger else "",
        "source_artifact_count": len(paths),
        "source_content_sha256": stable_content_hash,
    }
    provenance_path = output_root / "provenance.json"
    write_json(provenance_path, provenance)
    output_paths = [
        h5_path,
        csv_path,
        parquet_path,
        cycle_csv_path,
        cycle_parquet_path,
        quality_path,
        provenance_path,
    ]
    manifest_path = output_root / "artifact_manifest.json"
    manifest = artifact_manifest(output_paths)
    manifest.update(
        schema_version=SCHEMA_VERSION,
        generated_utc=utc_now(),
        source_content_sha256=stable_content_hash,
    )
    write_json(manifest_path, manifest)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        dest="input_roots",
        help="Repeat for each MATLAB artifact root; defaults to the two r3 roots.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-config", type=Path)
    parser.add_argument("--attrition-ledger", type=Path)
    parser.add_argument("--allow-supplemental-cases", action="store_true")
    args = parser.parse_args()
    report = build(
        tuple(args.input_roots) if args.input_roots else DEFAULT_INPUTS,
        args.output_root,
        args.expected_config,
        args.allow_supplemental_cases,
        args.attrition_ledger,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
