"""Quantify record-initialization leakage and stage-start interface discrepancies.

Read-only physical/data inspection plus checkpoint sensitivity diagnostics.
No source solver change, no new labels, no claim of repaired autonomous model.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from SciML.models.v11.state_space import LAYOUT, STAGE_NAMES, SIDE_KEYS, build_transition_corpus
from SciML.models.v12.neural_state_space_candidate import load_frozen_checkpoint
from SciML.models.v12.run_learning_existing import sha, validate_dataset, write_json
from SciML.evaluation.audit_learning_wp5 import verify_artifacts


def request_initial_state(request):
    """Declared rest/uniform initial condition; NEVER reads stage2/4 histories.

    Thermal side defaults follow Build_Simulink_Exchange_Params. Uniform inactive
    HTF placeholders are declared here, not independently observed full states.
    This is an initialization sensitivity input, not an exact Markov closure.
    """
    options = request["run_options"]
    solver, exchange = options["solver_options"], options["exchange_options"]
    if solver.get("initial_checkpoint") or options.get("resume_from"):
        raise ValueError("this initial-condition diagnostic excludes resumed trajectories")
    tr = float(solver["Tr"])
    hot = float(exchange.get("hot_sink_T_init", 293.))
    cold = float(exchange.get("cold_sink_T_init", 293.))
    load = float(exchange.get("cold_load_T_init", exchange.get("chip_T_init", 293.)))
    result = np.zeros(LAYOUT.size)
    result[LAYOUT.sma_T] = tr
    result[LAYOUT.htf_stage2_T] = cold
    result[LAYOUT.htf_stage4_T] = hot
    result[LAYOUT.side] = [hot, cold, load, hot, cold, cold, hot, cold]
    if not np.isfinite(result).all() or not all(200 <= t <= 400 for t in (tr, hot, cold, load)):
        raise ValueError("invalid initial temperature configuration")
    return result


def requested_controls(request, stage):
    if stage not in (1, 2, 3, 4):
        raise ValueError("invalid stage")
    case = request["case_spec"]
    solver = request["run_options"]["solver_options"]
    duration = solver["T_load"] if stage == 1 else solver["T_unload"] if stage == 3 else case["exchange_duration_s"]
    material = case["material_id"]
    if material not in ("recA", "recAs"):
        raise ValueError("unknown material")
    return np.r_[np.eye(4)[stage-1], [material == "recA", material == "recAs"],
                 [case[k] for k in ("radius_m", "thickness_m", "pressure_Pa", "flush_volume_target_ratio", "exchange_duration_s")],
                 duration, 1.].astype(float)


def maximum_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid finite interface comparison")
    return float(np.max(np.abs(a-b)))


def run(dataset, original_result, out):
    if out.exists():
        raise FileExistsError("preserve state contract evidence")
    verify_artifacts(original_result)
    protocol = json.loads((original_result/"protocol.json").read_text())
    table, quality, missing = validate_dataset(dataset)
    if protocol["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("source mismatch")
    model = load_frozen_checkpoint(original_result/"wp3_checkpoint.pkl")
    records = build_transition_corpus(dataset)
    lookup = table.set_index("case_id", drop=False)
    by_case = {key: [r for r in records if r.case_id == key] for key in table.case_id}
    diagnostics, truth, record_predictions, request_predictions = [], [], [], []
    with h5py.File(dataset/"fields.h5", "r") as fields:
        for key in table.case_id:
            row = lookup.loc[key]
            request_path = Path(row.source_artifact).parent/"sciml_case_request.json"
            if sha(request_path) != row.source_request_sha256:
                raise ValueError(f"request hash changed: {key}")
            request = json.loads(request_path.read_text())
            initial = request_initial_state(request)
            group = fields[str(row.hdf5_group)+"/cycles/cycle_001"]
            transitions = by_case[key]
            state_record = transitions[0].state_in.copy()
            state_request = initial.copy()
            initial_differences = {name: maximum_difference(initial[slc], state_record[slc]) for name, slc in
                                   (("u_m", LAYOUT.u), ("w_m", LAYOUT.w), ("sma_K", LAYOUT.sma_T),
                                    ("xi", LAYOUT.xi), ("htf2_K", LAYOUT.htf_stage2_T),
                                    ("htf4_K", LAYOUT.htf_stage4_T), ("side_K", LAYOUT.side))}
            stage_rows = []
            for record in transitions:
                stage = record.stage_index
                raw = group[STAGE_NAMES[stage-1]]
                controls = requested_controls(request, stage)
                input_gaps = {}
                if stage in (1, 3):
                    for field, slc in (("u_m_t", LAYOUT.u), ("w_m_t", LAYOUT.w),
                                       ("T_K_t", LAYOUT.sma_T), ("xi_t", LAYOUT.xi)):
                        input_gaps[field] = maximum_difference(record.state_in[slc], np.asarray(raw[field])[0])
                else:
                    input_gaps["sma_T_in_51_K"] = maximum_difference(record.state_in[LAYOUT.sma_T], np.asarray(raw["sma_T_in_51_K"]))
                    active = LAYOUT.htf_stage2_T if stage == 2 else LAYOUT.htf_stage4_T
                    input_gaps["htf_internal_start_K"] = maximum_difference(record.state_in[active], np.asarray(raw["htf_internal_T_K_t"])[0])
                    for j, keys in enumerate(SIDE_KEYS):
                        value = float(np.asarray(raw["side_state/"+keys[0]]))
                        if np.isfinite(value):
                            input_gaps[keys[0]] = abs(float(record.state_in[LAYOUT.side][j])-value)
                control_gap = maximum_difference(controls, record.controls)
                time = np.asarray(raw["time_local_s"])
                if not np.isfinite(time).all() or time[0] != 0 or not np.all(np.diff(time) > 0):
                    raise ValueError("invalid local time coordinate")
                state_record = model.predict(state_record, record.controls, stage)
                state_request = model.predict(state_request, controls, stage)
                record_predictions.append(state_record.copy())
                request_predictions.append(state_request.copy())
                truth.append(record.state_out)
                stage_rows.append({"stage": stage, "canonical_vs_observed_start_max_abs": input_gaps,
                                   "requested_vs_observed_control_max_abs": control_gap,
                                   "observed_duration_s": float(time[-1]), "requested_duration_s": float(controls[-2]),
                                   "field_shapes": {k: list(v.shape) for k, v in raw.items() if isinstance(v, h5py.Dataset)}})
            diagnostics.append({"case_id": key, "condition_id": row.condition_id,
                                "request_sha256": sha(request_path), "declared_initial_differences": initial_differences,
                                "sma_fluid_T_init_policy": request["run_options"]["exchange_options"].get("sma_fluid_T_init_policy", "source_default"),
                                "stages": stage_rows})
    truth, record_predictions, request_predictions = map(np.stack, (truth, record_predictions, request_predictions))
    scores = {}
    ordered = [r for key in table.case_id for r in by_case[key]]
    for part, conditions in protocol["condition_splits"].items():
        scores[part] = {}
        for stage in (1, 2, 3, 4):
            mask = np.array([r.condition_id in conditions and r.stage_index == stage for r in ordered])
            scores[part][str(stage)] = {name: {"record_initial_rmse": float(np.sqrt(np.mean((record_predictions[mask, slc]-truth[mask, slc])**2))),
                                                 "request_initial_rmse": float(np.sqrt(np.mean((request_predictions[mask, slc]-truth[mask, slc])**2)))}
                                       for name, slc in (("sma_K", LAYOUT.sma_T), ("side_K", LAYOUT.side), ("xi", LAYOUT.xi))}
    maxima = {name: max(d["declared_initial_differences"][name] for d in diagnostics)
              for name in diagnostics[0]["declared_initial_differences"]}
    gap_maxima = {str(stage): {field: max(d["stages"][stage-1]["canonical_vs_observed_start_max_abs"].get(field, 0.) for d in diagnostics)
                             for field in set().union(*(d["stages"][stage-1]["canonical_vs_observed_start_max_abs"] for d in diagnostics))}
                  for stage in (1, 2, 3, 4)}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"case_contracts.json", diagnostics)
    np.savez_compressed(out/"initialization_sensitivity.npz", sample_ids=np.array([r.sample_id for r in ordered]),
                        truth=truth, record_initial_prediction=record_predictions, request_initial_prediction=request_predictions)
    report = {"status": "completed_state_input_contract_inspection", "learning_only": True, "fresh_solver_calls": 0,
              "case_count": len(diagnostics), "observed_cycle_count": 63,
              "initial_record_vs_request_max_abs": maxima, "stage_start_interface_gap_maxima": gap_maxima,
              "cases_with_stage4_future_initial_difference": sum(d["declared_initial_differences"]["htf4_K"] > 1e-12 for d in diagnostics),
              "requested_vs_observed_controls_max_abs": max(s["requested_vs_observed_control_max_abs"] for d in diagnostics for s in d["stages"]),
              "initialization_sensitivity": scores, "known_inactive_missing_fields": missing,
              "limitations": ["Existing model trained with record-derived state; request-only rerun is sensitivity, NOT a retrained causal model",
                              "Canonical continuity may hide observed stage-start jumps; explicit transition/restart maps still needed",
                              "Changing initialization alone does not repair stage ownership, hidden storage or restart-latent coupling",
                              "Uniform inactive fluids are declared initialization placeholders, not independent complete-state observations"],
              "dataset_source_content_sha256": quality["source_content_sha256"], "original_summary_sha256": sha(original_result/"summary.json"),
              "implementation_sha256": sha(Path(__file__)),
              "source_map": {p: sha(Path(p)) for p in ("SciML/models/v11/state_space.py", "SciML/models/v12/state_space_candidate.py",
                    "Build_Simulink_Exchange_Params.m", "Run_GlobalXi_Simulink_Cycles.m")},
              "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_result, args.output_root), indent=2))
