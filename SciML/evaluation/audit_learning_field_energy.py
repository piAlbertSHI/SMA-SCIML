"""Audit field-to-scalar information requirements; never invent energy closure.

Uses existing63 only. Observed-port quadratures are diagnostics, not autonomous
operator predictions. A non-identifiability example concerns the output/readout
contract, not impossibility of learning a control-to-performance mapping.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import trapezoid

from SciML.models.v12.run_learning_existing import validate_dataset, sha, write_json


def series(time, *values):
    time = np.asarray(time, float)
    arrays = [np.asarray(v, float) for v in values]
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or not (np.diff(time) > 0).all():
        raise ValueError("finite strictly increasing time required")
    if any(a.shape != time.shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite aligned one-dimensional observations required")
    return time, arrays


def port_heat(time, mass_flow, inlet, outlet, cp):
    """J into SMA boundary: integral mdot*cp*(Tin-Tout), signed flow retained."""
    t, (m, tin, tout) = series(time, mass_flow, inlet, outlet)
    if not np.isfinite(cp) or cp <= 0:
        raise ValueError("positive finite specific heat required")
    return float(trapezoid(m*cp*(tin-tout), t))


def pump_work(time, mass_flow, pressure_command_MPa, rho):
    """Archive hydraulic-work convention, not motor/electrical pump consumption."""
    t, (m, p) = series(time, mass_flow, pressure_command_MPa)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("positive finite fluid density required")
    return float(trapezoid(np.abs(p*1e6*m/rho), t))


def counterexamples():
    t = np.array([0., .4, 1.])
    tin, tout = np.full(3, 300.), np.full(3, 299.)
    first = port_heat(t, np.full(3, .001), tin, tout, 4180.)
    second = port_heat(t, np.full(3, .002), tin, tout, 4180.)
    return {"type": "manufactured readout-contract examples, not physical solver trajectories",
        "same_port_temperatures_K": {"inlet": 300., "outlet": 299.},
        "mass_flow_kgps": [.001, .002], "heat_J": [first, second],
        "same_heat_different_work_COP": {"heat_J": first, "work_J": [1., 2.], "COP": [first, first/2]},
        "same_temperature_change_different_capacity_storage": {"delta_T_K": 1., "capacity_J_per_K": [1., 2.], "delta_U_J": [1., 2.]},
        "inference": "temperature values alone do not specify boundary heat, work, or storage; physical constitutive/control models can supply additional information"}


def run(dataset, repo, out):
    if out.exists():
        raise FileExistsError("preserve energy-readout diagnostic")
    table, quality, _ = validate_dataset(dataset)
    if len(table) != 63:
        raise ValueError("expected existing63 learning corpus")
    paths = ("Build_Simulink_Exchange_Params.m", "Run_Simulink_Exchange_Hook.m", "Run_GlobalXi_Simulink_Cycles.m", "SciML/builders/Export_SCIML_V11_Case.m")
    source_hashes = {p: sha(repo/p) for p in paths}
    stages, cases = [], []
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for row in table.itertuples(index=False):
            raw = Path(row.source_artifact)
            request_file = raw.parent/"sciml_case_request.json"
            if sha(raw) != row.source_artifact_sha256 or sha(request_file) != row.source_request_sha256:
                raise ValueError("raw source/request changed")
            request = json.loads(request_file.read_text())
            recorded = {Path(v["path"].replace("\\", "/")).name: v["sha256"] for v in request["implementation_fingerprint"]}
            matches = {p: recorded.get(Path(p).name) == h for p, h in source_hashes.items()}
            opts = request["run_options"]["exchange_options"]
            cp, rho = float(opts.get("cp_water", 4180.)), float(opts.get("rho_water", 998.))
            constants_provenance = "explicit request or source-verified defaults" if matches["Build_Simulink_Exchange_Params.m"] else "request values or assumed defaults; current source differs, diagnostic only"
            g = handle[row.hdf5_group+"/cycles/cycle_001"]
            perf = {k: float(v[()]) for k, v in g["performance"].items()}
            qcool_identity = perf["Qcool_J"]-max(float(g["stage4_cooldown_exchange/energy/Q_sma_htf_J"][()]), 0.)
            cop_identity = perf["COP"]-perf["Qcool_J"]/perf["Wtotal_for_COP_J"]
            pump_identity = perf["Wpump_J"]-perf["Wpump_stage2_J"]-perf["Wpump_stage4_J"]
            # Compare both work conventions; neither is silently substituted for
            # the stored target. Export lacks the signed unload-work scalar.
            plus = perf["W1_J"]+perf["Wunload_recovered_J"]+perf["Wpump_J"]
            minus = perf["W1_J"]-perf["Wunload_recovered_J"]+perf["Wpump_J"]
            cases.append({"case_id": row.case_id, "source_hashes_match_request": matches,
                "constant_provenance": constants_provenance, "cp_J_per_kgK": cp, "rho_kg_per_m3": rho,
                "export_identity_residuals": {"Qcool_J": qcool_identity, "COP": cop_identity, "Wpump_J": pump_identity},
                "work_convention": {"stored_Wtotal_J": perf["Wtotal_for_COP_J"], "load_plus_recovered_plus_pump_J": plus,
                    "load_minus_recovered_plus_pump_J": minus, "plus_minus_stored_J": plus-perf["Wtotal_for_COP_J"],
                    "minus_minus_stored_J": minus-perf["Wtotal_for_COP_J"]}})
            for s, name in ((2, "stage2_hold_exchange"), (4, "stage4_cooldown_exchange")):
                group = g[name]
                arrays = {k: np.asarray(group[k]) for k in ("time_local_s", "mass_flow_t_kgps", "pressure_pi_command_MPa_t", "htf_T_in_t_K", "htf_T_out_t_K", "htf_internal_T_K_t")}
                t, m, p, tin, tout = [arrays[k] for k in ("time_local_s", "mass_flow_t_kgps", "pressure_pi_command_MPa_t", "htf_T_in_t_K", "htf_T_out_t_K")]
                htf = arrays["htf_internal_T_K_t"]
                if htf.shape != (50, 101) or not np.isfinite(htf).all():
                    raise ValueError("invalid internal-fluid field")
                heat, work = port_heat(t, m, tin, tout, cp), pump_work(t, m, p, rho)
                heat_truth = float(group["energy/Q_sma_htf_J"][()])
                work_truth = perf[f"Wpump_stage{s}_J"]
                stages.append({"case_id": row.case_id, "stage": s, "samples": len(t),
                    "observed_port_heat_50point_J": heat, "stored_Q_sma_htf_J": heat_truth,
                    "heat_quadrature_minus_stored_J": heat-heat_truth,
                    "observed_pump_work_50point_J": work, "stored_Wpump_stage_J": work_truth,
                    "pump_quadrature_minus_stored_J": work-work_truth,
                    "internal_endpoint_minus_port_max_K": {"first_vs_inlet": float(np.max(np.abs(htf[:, 0]-tin))),
                        "last_vs_outlet": float(np.max(np.abs(htf[:, -1]-tout))),
                        "last_vs_inlet": float(np.max(np.abs(htf[:, -1]-tin))),
                        "first_vs_outlet": float(np.max(np.abs(htf[:, 0]-tout)))},
                    "energy_keys_exported": sorted(group["energy"].keys()),
                    "readout_information": "all measured ports/flow/pressure supplied; NOT forecast-only evaluation"})
    missing = {"temperature_only_operator_outputs": ["realized signed mass flow", "pressure-command trajectory", "port temperature ownership", "mechanical work", "capacities and latent-energy/restart convention"],
        "energy_export_not_present": sorted(set(("Q_environment_J", "Q_internal_pipe_fluid_temperature_estimate_J", "Q_internal_pipe_fluid_method"))-set(stages[0]["energy_keys_exported"])),
        "independent_energy_closure": "not established by summing exported residual-derived terms; source/storage provenance must be inspected"}
    examples = counterexamples()
    if np.isclose(*examples["heat_J"]):
        raise ValueError("readout counterexample is not distinct")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"case_diagnostics.json", cases)
    write_json(out/"stage_quadrature_diagnostics.json", stages)
    write_json(out/"readout_counterexamples.json", examples)
    summary = {"status": "completed_field_energy_information_audit", "learning_only": True, "fresh_solver_calls": 0,
        "case_count": len(cases), "exchange_stage_count": len(stages),
        "all_recorded_sources_match_current": all(all(c["source_hashes_match_request"].values()) for c in cases),
        "max_export_identity_abs_residuals": {k: max(abs(c["export_identity_residuals"][k]) for c in cases) for k in ("Qcool_J", "COP", "Wpump_J")},
        "max_abs_work_convention_residual_J": {k: max(abs(c["work_convention"][k]) for c in cases) for k in ("plus_minus_stored_J", "minus_minus_stored_J")},
        "observed_quadrature_diagnostic": {k: {"mae_J": float(np.mean([abs(s[k]) for s in stages])), "max_abs_J": max(abs(s[k]) for s in stages)} for k in ("heat_quadrature_minus_stored_J", "pump_quadrature_minus_stored_J")},
        "max_internal_endpoint_port_gaps_K": {k: max(s["internal_endpoint_minus_port_max_K"][k] for s in stages) for k in stages[0]["internal_endpoint_minus_port_max_K"]},
        "information_gaps": missing, "physics_gate_changed": False, "autonomous_field_to_COP_claim": False,
        "dataset_source_content_sha256": quality["source_content_sha256"], "sources_sha256": source_hashes,
        "implementation_sha256": sha(Path(__file__)),
        "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.repo, args.output_root), indent=2))
