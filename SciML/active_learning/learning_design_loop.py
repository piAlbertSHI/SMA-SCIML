"""Fixed-pool Pareto proposals and archived verification; never engineering certification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np

from SciML.active_learning.learning_pool_loop import pareto_mask, hypervolume
from SciML.models.v12.learning_ensemble import LearningEnsemble, ENSEMBLE_TARGETS, design_matrix
from SciML.models.v12.run_learning_existing import sha, validate_dataset, write_json


def propose(ids, mean, std, cal_y, cal_mean, cal_std, scale, known_y, budget=3):
    """No candidate targets or candidate mechanical outcomes are accepted here.

    Marginal split-conformal calibration is NOT guaranteed after optimization.
    Lower endpoints are a conservative ranking heuristic, not a safety proof.
    """
    mean, std, scale = np.asarray(mean), np.asarray(std), np.asarray(scale)
    if (mean.shape != (len(ids), 6) or std.shape != mean.shape or len(set(ids)) != len(ids)
        or scale.shape != (6,) or np.any(scale <= 0) or np.any(std < 0)
        or not all(np.isfinite(x).all() for x in (mean, std, scale, cal_y, cal_mean, cal_std, known_y))
        or type(budget) is not int or budget < 1):
        raise ValueError("invalid proposal data")
    scores = np.max(np.abs(cal_y-cal_mean)/np.maximum(cal_std, .01*scale), axis=1)
    rank = int(np.ceil((len(scores)+1)*.9))
    if not 1 <= rank <= len(scores):
        raise ValueError("insufficient calibration for finite 90% interval")
    q = float(np.sort(scores)[rank-1])
    width = q*np.maximum(std, .01*scale)
    lower = mean-width
    # Input/prediction-only cutoff, not tuned to candidate outcomes.
    cutoff = float(np.quantile(np.mean(cal_std/scale, axis=1), .9))
    eligible = (lower[:, [0, 2, 3, 5]] > 0).all(axis=1) & (mean[:, 1] >= 0)
    eligible &= np.mean(std/scale, axis=1) <= cutoff
    front = np.zeros(len(ids), dtype=bool)
    if eligible.any():
        front[eligible] = pareto_mask(lower[eligible][:, [2, 3]])
    objectives = known_y[:, [2, 3]]/scale[[2, 3]]
    reference = objectives.min(axis=0)-1
    base = hypervolume(objectives, reference)
    gains = np.array([max(0., hypervolume(np.vstack((objectives, p)), reference)-base)
                     for p in lower[:, [2, 3]]/scale[[2, 3]]])
    # If pessimistic gain is tied at zero, retain the nondominated tradeoff with
    # largest equal-weight normalized lower objective sum; case ID breaks ties.
    score = (lower[:, [2, 3]]/scale[[2, 3]]).sum(axis=1)
    order = np.lexsort((np.asarray(ids), -score, -gains))
    selected = [ids[i] for i in order if front[i]][:budget]
    return {"candidate_ids": list(ids), "selected_ids": selected,
            "mean": mean.tolist(), "std": std.tolist(), "half_width": width.tolist(),
            "lower": lower.tolist(), "proposal_eligible": eligible.tolist(),
            "lower_pareto_mask": front.tolist(), "pessimistic_hypervolume_gain": gains.tolist(),
            "tie_break_score": score.tolist(), "calibration_quantile": q,
            "calibration_rank": rank, "uncertainty_cutoff": cutoff,
            "candidate_labels_used": False, "budget_max": budget,
            "post_selection_coverage_guaranteed": False}


def verified_scalar(handle, key):
    values = np.asarray(handle[key], dtype=float)
    if values.size != 1 or not np.isfinite(values).all():
        raise ValueError(f"invalid archived scalar: {key}")
    return float(values.item())


def archive_observation(row, fields):
    """Read only explicitly revealed case, verifying original MAT provenance first."""
    path = Path(row.source_artifact).parent/"run_result.mat"
    digest = sha(path)
    if digest != row.source_run_result_sha256:
        raise ValueError(f"archive source hash mismatch: {row.case_id}")
    with h5py.File(path, "r") as handle:
        stress = max(verified_scalar(handle, f"run_result/cycle_results/{stage}/metrics/global_xi_time_sigma_eq_max_MPa")
                     for stage in ("load_result", "unload_result"))
        wr = verified_scalar(handle, "run_result/cycle_diagnostics/center_deflection_peak_over_radius")
        energy_ok = bool(verified_scalar(handle, "run_result/cycle_diagnostics/hold_system_energy_audit_ok"))
        energy_ok &= bool(verified_scalar(handle, "run_result/cycle_diagnostics/cooldown_system_energy_audit_ok"))
    cycle = fields[str(row.hdf5_group)+"/cycles/cycle_001"]
    phase = np.concatenate([np.asarray(cycle[stage+"/xi_t"]).ravel()
                            for stage in ("stage1_loading", "stage3_unloading")])
    if not np.isfinite(phase).all() or stress < 0 or wr < 0:
        raise ValueError("malformed archived mechanical observation")
    return {"case_id": str(row.case_id), "run_result_path": str(path), "source_sha256": digest,
            "targets": [float(getattr(row, key)) for key in ENSEMBLE_TARGETS],
            "sigma_eq_peak_MPa": stress, "center_deflection_peak_over_radius": wr,
            "mechanical_phase_min": float(phase.min()), "mechanical_phase_max": float(phase.max()),
            "exchange_system_energy_audits_ok": energy_ok,
            "existing_physics_gate_ok": bool(row.physics_ok)}


def adjudicate(observation, reference, limits=None):
    """Three-valued engineering verdict; absent limits must never become a pass."""
    limits = limits or {}
    numerical = (observation["existing_physics_gate_ok"] and observation["exchange_system_energy_audits_ok"]
                 and 0 <= observation["mechanical_phase_min"] <= observation["mechanical_phase_max"] <= 1)
    metrics = ("sigma_eq_peak_MPa", "center_deflection_peak_over_radius")
    checks = {}
    for key in metrics:
        limit = limits.get(key)
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError("invalid externally specified engineering limit")
        checks[key] = None if limit is None else bool(observation[key] <= limit)
    rejected = not numerical or any(v is False for v in checks.values())
    return {"numerical_gates_pass": bool(numerical), "engineering_constraint_checks": checks,
            "engineering_safety": "rejected" if rejected else "unknown",
            "reason": "Missing approved limits and fatigue/equipment evidence prevent certification",
            "within_revealed_reference_envelope": {k: bool(observation[k] <= reference[k]) for k in metrics},
            "reference_envelope_is_safety_limit": False}


def run(root, original_protocol, active_root, out):
    if out.exists():
        raise FileExistsError("preserve existing design evidence")
    table, quality, _ = validate_dataset(root)
    original = json.loads(original_protocol.read_text())
    active_protocol = json.loads((active_root/"protocol.json").read_text())
    if not (original["dataset_source_content_sha256"] == quality["source_content_sha256"]
            == active_protocol["dataset_source_content_sha256"]):
        raise ValueError("design/active/split provenance mismatch")
    active_summary = json.loads((active_root/"summary.json").read_text())
    for name, digest in active_summary["artifacts"].items():
        if sha(active_root/name) != digest:
            raise ValueError(f"active-learning evidence changed: {name}")
    # Fixed first registered seed and composite method, NOT best test performance.
    checkpoint = active_root/"s701_composite/final_ensemble.pt"
    model = LearningEnsemble.load(checkpoint)
    known_ids = list(model.training_ids)
    pool_ids = active_protocol["pool_case_ids"]
    if not set(known_ids) <= set(pool_ids) or len(known_ids) != 22:
        raise ValueError("unexpected prior query budget")
    candidates = sorted(set(pool_ids)-set(known_ids))
    lookup = table.set_index("case_id", drop=False)
    cal = table[table.condition_id.isin(original["condition_splits"]["calibration"])]
    if set(cal.case_id) & set(pool_ids):
        raise ValueError("calibration leaks into candidate pool")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol = {"learning_only": True, "fresh_solver_calls": 0,
                "dataset_source_content_sha256": quality["source_content_sha256"],
                "original_protocol_sha256": sha(original_protocol),
                "active_summary_sha256": sha(active_root/"summary.json"),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                "implementation_sha256": sha(Path(__file__)), "training_ids": known_ids,
                "calibration_ids": cal.case_id.tolist(), "candidate_ids": candidates,
                "selection": "Fixed s701 composite final22; max3 lower-bound Pareto proposals; no test-based model choice",
                "domain": "Finite registered candidate pool only; preserve observed paired geometry, no rectangular R/thickness extrapolation",
                "engineering_limits": None,
                "limit_policy": "No approved stress/deflection limits located; record unknown rather than invent limits",
                "reference_policy": "Maxima of already revealed22 mechanical observations; descriptive NOT certified safe",
                "coverage_policy": "90% marginal calibration exercise; adaptive design selection voids any automatic coverage claim"}
    write_json(out/"protocol.json", protocol)
    mean, std = model.predict(design_matrix(lookup.loc[candidates]))
    cal_mean, cal_std = model.predict(design_matrix(cal))
    known_y = lookup.loc[known_ids, list(ENSEMBLE_TARGETS)].to_numpy(float)
    proposal = propose(candidates, mean, std, cal[list(ENSEMBLE_TARGETS)].to_numpy(float),
                       cal_mean, cal_std, model.y_scaler.scale_, known_y)
    # The immutable decision is persisted before touching any candidate outcome.
    write_json(out/"proposal_before_reveal.json", proposal)
    observations = []
    with h5py.File(root/"fields.h5", "r") as fields:
        known = [archive_observation(lookup.loc[key], fields) for key in known_ids]
        reference = {key: max(row[key] for row in known) for key in
                     ("sigma_eq_peak_MPa", "center_deflection_peak_over_radius")}
        write_json(out/"revealed22_mechanical_reference.json", {"observations": known, "reference": reference})
        for key in proposal["selected_ids"]:
            observation = archive_observation(lookup.loc[key], fields)
            observation["verdict"] = adjudicate(observation, reference)
            i = candidates.index(key)
            observation["prediction_error"] = (mean[i]-observation["targets"]).tolist()
            observations.append(observation)
    objectives = known_y[:, [2, 3]]/model.y_scaler.scale_[[2, 3]]
    reference_hv = objectives.min(axis=0)-1
    base = hypervolume(objectives, reference_hv)
    augmented = np.vstack((known_y, [o["targets"] for o in observations])) if observations else known_y
    actual_gain = hypervolume(augmented[:, [2, 3]]/model.y_scaler.scale_[[2, 3]], reference_hv)-base
    write_json(out/"candidate_verification.json", {"observations": observations,
               "reference": reference, "normalized_actual_hypervolume_gain": actual_gain,
               "all_verified_numerical_not_certified_safe": all(o["verdict"]["numerical_gates_pass"] for o in observations) if observations else None})
    summary = {"status": "completed_archived_design_experiment", "learning_only": True,
               "fresh_solver_calls": 0, "candidate_count": len(candidates),
               "selected_ids": proposal["selected_ids"], "newly_revealed_archive_count": len(observations),
               "engineering_safety_certified": False, "actual_normalized_hypervolume_gain": actual_gain,
               "elapsed_seconds": time.monotonic()-start,
               "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_protocol, args.active_root, args.output_root), indent=2))
