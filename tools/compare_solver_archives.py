"""Compare the archived first-cycle pairs under the two recorded solver configurations."""

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
from scipy.integrate import trapezoid

from SciML.evaluation.audit_learning_field_energy import port_heat, pump_work
from tools.audit_project_corpora import CORPORA, EXCHANGE, MECHANICAL, check_manifest, read_json, read_rows, recorded_true, sha256


ROOT = Path(__file__).resolve().parents[1]
PERFORMANCE = (
    "Qcool_J", "W1_J", "Wunload_recovered_J", "Wpump_J", "Wtotal_for_COP_J", "COP",
    "SCP_W_per_kg", "deltaT_hot_cold_max_K", "cycle_duration_s",
)
REGISTERED_NUMERIC = (
    "radius_m", "thickness_m", "pressure_Pa", "flush_volume_target_ratio",
    "exchange_duration_s", "cold_load_heat_W", "sma_mass_kg",
)
REGISTERED_TEXT = ("case_id", "material_id", "cold_load_mode")
MECHANICAL_FIELDS = {"u_mm": ("u_m_t", 1000.0), "w_mm": ("w_m_t", 1000.0),
                     "sma_K": ("T_K_t", 1.0), "phase_fraction": ("xi_t", 1.0),
                     "pressure_MPa": ("pressure_Pa_t", 1e-6)}
EXCHANGE_FIELDS = {
    "sma_K": ("sma_strip_T_K_t", 1.0), "htf_K": ("htf_internal_T_K_t", 1.0),
    "mass_flow_g_per_s": ("mass_flow_t_kgps", 1000.0), "pressure_command_MPa": ("pressure_pi_command_MPa_t", 1.0),
    "htf_inlet_K": ("htf_T_in_t_K", 1.0), "htf_outlet_K": ("htf_T_out_t_K", 1.0),
    "hot_sink_K": ("hot_sink_t_K", 1.0), "cold_sink_K": ("cold_sink_t_K", 1.0),
}
AUDIT_FLAGS = (
    "energy_audit_ok", "sma_htf_energy_audit_ok", "stage2_hot_end_energy_audit_ok",
    "measured_cold_end_energy_audit_ok", "system_energy_audit_ok",
    "system_energy_residual_included_in_pass_fail",
)


def first_cycles(dataset):
    """One accepted, originally single-cycle row per condition."""
    manifest, _ = check_manifest(dataset)
    cycles = {}
    for row in read_rows(dataset / "cycle_index.csv"):
        if row["cycle_index"] == "1":
            if row["case_id"] in cycles:
                raise ValueError("Duplicate first-cycle entry")
            cycles[row["case_id"]] = row
    cases = {}
    for row in read_rows(dataset / "case_index.csv"):
        if row["cycle_count"] != "1" or not all(recorded_true(row, flag) for flag in (
            "physics_ok", "paper_grade_valid", "observation_contract_ok", "pump_work_available",
        )):
            continue
        condition = row["condition_id"]
        cycle = cycles[row["case_id"]]
        if condition in cases or cycle["condition_id"] != condition or not recorded_true(cycle, "artifact_complete"):
            raise ValueError(f"Invalid accepted first-cycle record: {condition}")
        cases[condition] = (row, cycle)
    return manifest, cases


def field(group, key, shape=None):
    values = np.asarray(group[key][()], dtype=np.float64)
    if (shape is not None and values.shape != shape) or not np.isfinite(values).all():
        raise ValueError(f"Invalid archived values: {group.name}/{key}")
    return values


def scalar(group, key):
    value = float(group[key][()])
    return value if math.isfinite(value) else None


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def trajectory_delta(old, new):
    diff = new - old
    return {
        "initial_rms": rms(diff[0]),
        "future_rms": rms(diff[1:]),
        "future_signed_mean": float(diff[1:].mean()),
        "increment_rms": rms(diff[1:] - diff[0]),
        "endpoint_rms": rms(diff[-1]),
        "max_abs": float(np.abs(diff).max()),
        "elementwise_equal": bool(np.array_equal(old, new)),
    }


def read_case_evidence(row):
    directory = Path(row["source_artifact"]).parent
    request_file = directory / "sciml_case_request.json"
    export_file = directory / "sciml_export_provenance.json"
    validation_file = directory / "validation_summary.json"
    request_hash = sha256(request_file)
    export_hash = sha256(export_file)
    if request_hash != row["source_request_sha256"] or export_hash != row["source_export_provenance_sha256"]:
        raise ValueError(f"Sidecar hash mismatch: {row['case_id']}")
    if read_json(export_file)["artifact_sha256"] != row["source_artifact_sha256"]:
        raise ValueError(f"Export provenance mismatch: {row['case_id']}")
    validation = read_json(validation_file)
    log = {key: bool(validation[key]) for key in AUDIT_FLAGS}
    log["total_system_energy_residual_max_abs_J"] = float(validation["total_system_energy_residual_max_abs_J"])
    log["solver_Nt"] = int(validation["solver_Nt"])
    log["solver_Nt_unload"] = int(validation["solver_Nt_unload"])
    hashes = {"request": request_hash, "export_provenance": export_hash, "validation_at_audit": sha256(validation_file)}
    return read_json(request_file), log, hashes


def fingerprints(request):
    return {PurePosixPath(row["path"].replace("\\", "/")).name: row["sha256"]
            for row in request["implementation_fingerprint"]}


def inspect_defaults(solver_repo):
    name = "Build_Simulink_Exchange_Params.m"
    archived = "Transfer/gx101r1_sciml_paper220_remote_20260822_final"
    sources = (solver_repo / archived / name, solver_repo / name)
    defaults = {}
    for source in sources:
        body = source.read_text(encoding="utf-8-sig")
        numeric = {}
        for key in ("cp_water", "rho_water"):
            match = re.search(r"get_option_field\(exchange_options,\s*'" + key + r"',\s*([0-9.]+)\)", body)
            if match is None:
                raise ValueError(f"Missing archived default for {key}")
            numeric[key] = float(match[1])
        defaults[sha256(source)] = numeric
    return defaults


def water_properties(request, defaults):
    source = fingerprints(request)["Build_Simulink_Exchange_Params.m"]
    options = request["run_options"]["exchange_options"]
    return {name: float(options[name]) if name in options else defaults[source][name]
            for name in ("cp_water", "rho_water")}


def controls_delta(old, new, changes):
    for block in ("solver_options", "exchange_options"):
        a, b = (request["run_options"][block] for request in (old, new))
        for key in set(a) | set(b):
            if not any(term in key.lower() for term in ("path", "dir", "root", "file")):
                if a.get(key) != b.get(key):
                    summary = json.dumps([a.get(key), b.get(key)], sort_keys=True, allow_nan=False)
                    changes[f"{block}.{key}"][summary] += 1
    a, b = map(fingerprints, (old, new))
    for key in set(a) | set(b):
        if a.get(key) != b.get(key):
            changes[f"source.{key}"][json.dumps([a.get(key), b.get(key)])] += 1


def verify_identity(left, right):
    for key in REGISTERED_TEXT:
        if left[key] != right[key]:
            raise ValueError(f"Registered text changed: {key}")
    for key in REGISTERED_NUMERIC:
        if float(left[key]) != float(right[key]):
            raise ValueError(f"Registered scalar changed: {key}")


def stage_comparison(old, new, exchange, water):
    axes = ("time_local_s", "sma_strip_coordinate_m", "htf_internal_time_local_s",
            "htf_internal_pipe_index_101") if exchange else ("time_local_s", "radial_coordinate_m")
    for key in axes:
        np.testing.assert_array_equal(field(old, key), field(new, key), err_msg=f"Unaligned {key}")
    fields = EXCHANGE_FIELDS if exchange else MECHANICAL_FIELDS
    histories = {key: trajectory_delta(field(old, name) * factor, field(new, name) * factor)
                 for key, (name, factor) in fields.items()}
    record = {"histories": histories}
    if exchange:
        if set(old["energy"]) != set(new["energy"]):
            raise ValueError("Energy export schemas differ")
        energy = {name: [scalar(group["energy"], name) for group in (old, new)]
                  for name in sorted(old["energy"])}
        for side, group in enumerate((old, new)):
            t, flow, pressure, tin, tout = (field(group, key, (50,)) for key in (
                "time_local_s", "mass_flow_t_kgps", "pressure_pi_command_MPa_t", "htf_T_in_t_K", "htf_T_out_t_K",
            ))
            def record_sample(name, value):
                energy.setdefault(name, [None, None])[side] = value
            cp, rho = water[side]["cp_water"], water[side]["rho_water"]
            record_sample("sampled_signed_mass_kg", float(trapezoid(flow, t)))
            record_sample("sampled_throughput_mass_kg", float(trapezoid(np.abs(flow), t)))
            record_sample("sampled_port_heat_J", port_heat(t, flow, tin, tout, cp))
            record_sample("sampled_hydraulic_work_J", pump_work(t, flow, pressure, rho))
        record["energy_old_new"] = energy
    return record


def performance(group, row, cycle):
    result = {name: scalar(group["performance"], name) for name in PERFORMANCE}
    if any(value is None for value in result.values()):
        raise ValueError("Accepted case has a nonfinite performance label")
    for name, value in result.items():
        np.testing.assert_allclose([float(row[name]), float(cycle[name])], value, rtol=0, atol=1e-12)
    p = group["performance"]
    cold_heat = scalar(group["stage4_cooldown_exchange/energy"], "Q_sma_htf_J")
    q, work = result["Qcool_J"], result["Wtotal_for_COP_J"]
    identities = {
        "Qcool_minus_positive_stage4_heat_J": q - max(cold_heat, 0),
        "pump_minus_sum_of_stage_work_J": result["Wpump_J"] - scalar(p, "Wpump_stage2_J") - scalar(p, "Wpump_stage4_J"),
        "work_minus_load_plus_recovered_plus_pump_J": work - result["W1_J"] - result["Wunload_recovered_J"] - result["Wpump_J"],
        "COP_minus_ratio": result["COP"] - q / work,
        "SCP_minus_ratio_W_per_kg": result["SCP_W_per_kg"] - q / (float(row["sma_mass_kg"]) * result["cycle_duration_s"]),
    }
    return result, identities


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {key: float(value) for key, value in {
        "min": array.min(), "median": np.median(array), "max": array.max(),
        "mean": array.mean(), "p95_absolute": np.quantile(np.abs(array), .95),
        "median_absolute": np.median(np.abs(array)), "rmse": rms(array),
    }.items()}


def summarize_pairs(pairs, ids):
    valid = [(a, b, case) for (a, b), case in zip(pairs, ids) if a is not None and b is not None]
    if not valid:
        return {"comparable_pairs": 0}
    old, new, cases = zip(*valid)
    deltas = np.asarray(new) - np.asarray(old)
    return {
        "comparable_pairs": len(valid),
        "old": stats(old), "new": stats(new), "new_minus_old": stats(deltas),
        "changed_exactly": int(np.count_nonzero(deltas)),
        "worst_case_id": cases[int(np.argmax(np.abs(deltas)))],
    }


def summarize(rows):
    ids = [row["case_id"] for row in rows]
    performance = {name: summarize_pairs([row["performance_old_new"][name] for row in rows], ids)
                   for name in PERFORMANCE}
    energy, fields = {}, {}
    for name in rows[0]["stages"]:
        stages = [row["stages"][name] for row in rows]
        fields[name] = {}
        for key in stages[0]["histories"]:
            histories = [record["histories"][key] for record in stages]
            fields[name][key] = {
                "exact_pairs": sum(item["elementwise_equal"] for item in histories),
                **{metric: stats([item[metric] for item in histories])
                   for metric in histories[0] if metric != "elementwise_equal"},
                "worst_future_case_id": ids[int(np.argmax([item["future_rms"] for item in histories]))],
            }
        if name in EXCHANGE:
            energy[name] = {metric: summarize_pairs([s["energy_old_new"][metric] for s in stages], ids)
                            for metric in stages[0]["energy_old_new"]}
    return {"performance": performance, "stages": fields, "exchange_energy": energy}


def write(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(solver_repo, out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError(out)
    datasets = solver_repo / "SciML/datasets"
    old, new = (datasets / CORPORA[key] for key in ("historical_development", "single_cycle_learning"))
    old_manifest, old_cases = first_cycles(old)
    new_manifest, new_cases = first_cycles(new)
    shared = sorted(old_cases.keys() & new_cases.keys())
    if len(shared) != 54:
        raise ValueError(f"Expected 54 previously identified first-cycle pairs, found {len(shared)}")
    defaults = inspect_defaults(solver_repo)
    changes = defaultdict(Counter)
    rows = []
    with h5py.File(old / "fields.h5", "r") as old_h5, h5py.File(new / "fields.h5", "r") as new_h5:
        for condition in shared:
            (a, ca), (b, cb) = old_cases[condition], new_cases[condition]
            verify_identity(a, b)
            request_a, log_a, hash_a = read_case_evidence(a)
            request_b, log_b, hash_b = read_case_evidence(b)
            if request_a["case_spec"] != request_b["case_spec"]:
                raise ValueError(f"Paired request case specs differ: {condition}")
            controls_delta(request_a, request_b, changes)
            water = [water_properties(request, defaults) for request in (request_a, request_b)]
            ga, gb = (handle[cycle["hdf5_cycle_group"]] for handle, cycle in ((old_h5, ca), (new_h5, cb)))
            (ya, ida), (yb, idb) = performance(ga, a, ca), performance(gb, b, cb)
            stages = {name: stage_comparison(ga[name], gb[name], name in EXCHANGE, water)
                      for name in sorted((*MECHANICAL, *EXCHANGE))}
            qa, qb = ya["Qcool_J"], yb["Qcool_J"]
            wa, wb = ya["Wtotal_for_COP_J"], yb["Wtotal_for_COP_J"]
            numerator = .5 * (qb - qa) * (1 / wa + 1 / wb)
            denominator = .5 * (qa + qb) * (1 / wb - 1 / wa)
            np.testing.assert_allclose(numerator + denominator, yb["COP"] - ya["COP"], rtol=0, atol=1e-12)
            rows.append({
                "condition_id": condition, "case_id": a["case_id"], "material": a["material_id"],
                "radius_m": float(a["radius_m"]), "pressure_Pa": float(a["pressure_Pa"]),
                "flush_ratio": float(a["flush_volume_target_ratio"]), "exchange_duration_s": float(a["exchange_duration_s"]),
                "performance_old_new": {name: [ya[name], yb[name]] for name in PERFORMANCE},
                "stored_identity_residuals_old_new": [ida, idb],
                "cop_change_from_heat": numerator, "cop_change_from_work": denominator,
                "validation_old_new": [log_a, log_b], "record_hashes_old_new": [hash_a, hash_b],
                "water_old_new": water, "stages": stages,
            })

    out.mkdir(parents=True, exist_ok=False)
    write(out / "pairs.json", rows)
    by_material = {name: [row for row in rows if row["material"] == name] for name in sorted({r["material"] for r in rows})}
    summary = {
        "status": "completed_paired_solver_archive_comparison",
        "pair_count": len(rows), "pair_count_by_material": {name: len(group) for name, group in by_material.items()},
        "old_new": [old.name, new.name],
        "dataset_source_content_sha256": [old_manifest["source_content_sha256"], new_manifest["source_content_sha256"]],
        "dataset_manifest_sha256": [sha256(path / "artifact_manifest.json") for path in (old, new)],
        "axes_elementwise_equal": True,
        "case_request_spec_equal": True,
        "records_with_verified_request_export_hashes": 2 * len(rows),
        "validation_flags_true_old_new": {flag: [sum(row["validation_old_new"][side][flag] for row in rows) for side in (0, 1)]
                                          for flag in AUDIT_FLAGS},
        "implementation_and_request_changes": {
            key: [{"old_new": json.loads(value), "pairs": count} for value, count in sorted(observed.items())]
            for key, observed in sorted(changes.items())
        },
        "sampled_water_defaults_by_source_sha256": defaults,
        "cop_signed_difference_mean": {
            key: float(np.mean([row[key] for row in rows])) for key in ("cop_change_from_heat", "cop_change_from_work")
        },
        "max_identity_residual_by_side": {key: [max(abs(row["stored_identity_residuals_old_new"][side][key])
                                                   for row in rows) for side in (0, 1)]
                                          for key in rows[0]["stored_identity_residuals_old_new"][0]},
        **summarize(rows),
        "material_performance": {material: {key: summarize_pairs(
            [row["performance_old_new"][key] for row in group], [row["case_id"] for row in group])
            for key in PERFORMANCE} for material, group in by_material.items()},
        "sampled_integrals_scope": "Trapezoids on observed 50-point histories using source-verified water defaults or recorded overrides",
        "energy_scope": "Recorded energy terms and separate sampled port/pump calculations. No attribution of individual solver changes or new closure claim.",
        "pairs_sha256": sha256(out / "pairs.json"),
        "source_sha256": {p: sha256(ROOT / p) for p in (
            "tools/compare_solver_archives.py", "tools/audit_project_corpora.py",
            "SciML/evaluation/audit_learning_field_energy.py",
        )},
        "fresh_solver_calls": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    write(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.solver_repo, args.output)
    print(json.dumps({
        "status": result["status"], "pair_count": result["pair_count"], "runtime_seconds": result["runtime_seconds"],
        "median_absolute_COP_change": result["performance"]["COP"]["new_minus_old"]["median_absolute"],
        "fresh_solver_calls": result["fresh_solver_calls"],
    }, indent=2))


if __name__ == "__main__":
    main()
