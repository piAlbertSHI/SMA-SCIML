"""Same-profile multicycle boundary-to-DeepONet field cascade, no future starts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.evaluation.audit_learning_operators import matched_pod
from SciML.models.v11.spatiotemporal_operator import build_operator_corpus
from .temporal_operator_candidate import causal_delta_features
from .learning_historical_rollout import HistoricalModel, corpus, rollout, MEMORY
from .learning_coordinate_operator import CoordinateOperator
from .run_learning_existing import sha, write_json


def field_inputs(state, control, stage):
    if state.shape != (515,) or control.shape != (9,) or stage not in (2, 4):
        raise ValueError("invalid historical boundary-to-field input")
    radial = state[102:153]
    solid = np.r_[radial[:0:-1], radial]+state[MEMORY]
    fluid = state[204:305] if stage == 2 else state[305:406]
    side = state[406:414].copy()
    side[6 if stage == 2 else 5] = 293.
    branch = np.r_[solid-solid.mean(), fluid-fluid.mean(), solid.mean(), fluid.mean(),
                   stage == 2, stage == 4, control[5:7], control[:5], side]
    initial = np.stack((solid, fluid))
    if not np.isfinite(branch).all():
        raise ValueError("nonfinite forecast field input")
    return branch, initial


def cascade_inputs(boundary, starts, cases):
    """Known starts/controls/horizons only. Returned keys are case/cycle/stage."""
    rows = {}
    for case in cases:
        values = rollout(boundary, **starts[case])
        index = 0
        for cycle in range(1, starts[case]["cycles"]+1):
            if cycle > 1:
                index += 1  # explicitly learned restart output precedes loading
            for stage, offset in ((2, 0), (4, 2)):
                rows[(case, cycle, stage)] = field_inputs(values[index+offset], starts[case]["control"], stage)
            index += 4
        if index != len(values):
            raise ValueError("rollout boundary indexing mismatch")
    return rows


def metrics(prediction, target, samples, selected):
    result = {}
    for cycle in sorted({s.cycle_index for s in samples}):
        for stage in (2, 4):
            use = selected & np.array([s.cycle_index == cycle and s.stage_index == stage for s in samples])
            if not use.any():
                continue
            result[f"cycle{cycle}_stage{stage}"] = {field: {"case_count": int(use.sum()),
                "future_rmse_K": float(np.sqrt(np.mean((prediction[use, j, 1:]-target[use, j, 1:])**2))),
                "future_max_abs_K": float(np.abs(prediction[use, j, 1:]-target[use, j, 1:]).max()),
                "initial_max_abs_K": float(np.abs(prediction[use, j, 0]-target[use, j, 0]).max())}
                for j, field in enumerate(("sma", "htf"))}
    return result


def run(dataset, campaign, out):
    if out.exists():
        raise FileExistsError("preserve historical operator cascade")
    root = campaign/"historical-rollout-r1"
    verify_artifacts(root)
    previous = json.loads((root/"protocol.json").read_text())
    if previous["implementation_sha256"] != sha(Path(__file__).with_name("learning_historical_rollout.py")):
        raise ValueError("boundary source changed")
    table, records, starts, fps, checks = corpus(dataset)
    if fps != previous["fingerprints"]:
        raise ValueError("historical field/boundary profile mismatch")
    samples = build_operator_corpus(dataset)
    branch = np.stack([causal_delta_features(s) for s in samples])
    truth = np.stack([np.stack((s.sma_history, s.htf_history)) for s in samples])
    initial = truth[:, :, 0].copy()
    train = np.array([s.case_id in previous["groups"]["train"] for s in samples])
    test = ~train
    fixed = np.array([s.case_id in previous["groups"]["test"] and starts[s.case_id]["cycles"] == 3 for s in samples])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"learning_only": True, "fresh_solver_calls": 0,
        "groups": previous["groups"], "fingerprints": fps, "training_field_samples": int(train.sum()),
        "test_field_samples": int(test.sum()), "training_cycles": [1, 2], "three_cycle_cases_held_out": True,
        "operator": "Coordinate DeepONet,400updates,26x51supervised coordinates,observed-stage-start training",
        "cascade": "515state autonomous boundary model supplies every stage2/4 start; field model does not feed back to boundary model",
        "field_input_signature": "state515,control9,stage2or4; no observed future fields",
        "comparisons": "same-weight observed-start diagnostic versus autonomous-boundary inputs; matched POD observed-start baseline",
        "scope": "same-historical-profile multicycle temperature-field decoder, not tightly coupled physical closure or current63 evidence",
        "implementation_sha256": sha(Path(__file__)), "operator_source_sha256": sha(Path(__file__).with_name("learning_coordinate_operator.py")),
        "boundary_summary_sha256": sha(root/"summary.json")})
    pod, parameters = matched_pod(branch, initial, truth, np.array([s.stage_index for s in samples]), train)
    np.savez_compressed(out/"pod_parameters.npz", **parameters)
    np.savez_compressed(out/"pod_predictions.npz", prediction=pod, sample_ids=np.array([s.sample_id for s in samples]))
    reports = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in (20260910, 20260911, 20260912):
            dest = out/f"s{seed}"
            dest.mkdir()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = CoordinateOperator().double().fit(branch[train], initial[train], truth[train],
                    [s.sample_id for s, use in zip(samples, train) if use], seed=seed)
            observed = model.predict(branch, initial)
            boundary = HistoricalModel.load(root/f"s{seed}/model.pt")
            forecast = cascade_inputs(boundary, starts, table.case_id.tolist())
            pairs = [forecast[(s.case_id, s.cycle_index, s.stage_index)] for s in samples]
            cb, ci = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
            autonomous = model.predict(cb, ci)
            model.save(dest/"operator.pt")
            np.testing.assert_array_equal(autonomous, CoordinateOperator.load(dest/"operator.pt").predict(cb, ci))
            np.testing.assert_array_equal(autonomous[:, :, 0], ci)
            np.savez_compressed(dest/"predictions.npz", sample_ids=np.array([s.sample_id for s in samples]),
                observed_start=observed, autonomous_boundary=autonomous, forecast_initial=ci, forecast_branch=cb)
            report = {"seed": seed, "history": model.history,
                "test": {"observed_start": metrics(observed, truth, samples, test), "autonomous_boundary": metrics(autonomous, truth, samples, test)},
                "fixed_three_cycle_cases": {"observed_start": metrics(observed, truth, samples, fixed), "autonomous_boundary": metrics(autonomous, truth, samples, fixed)}}
            write_json(dest/"report.json", report)
            reports.append(report)
    finally:
        torch.set_num_threads(threads)
    mean = {mode: {key: {field: float(np.mean([r["fixed_three_cycle_cases"][mode][key][field]["future_rmse_K"] for r in reports]))
        for field in ("sma", "htf")} for key in reports[0]["fixed_three_cycle_cases"][mode]} for mode in ("observed_start", "autonomous_boundary")}
    summary = {"status": "completed_historical_multicycle_field_cascade", "learning_only": True, "fresh_solver_calls": 0,
        "fixed_three_cycle_seed_mean_rmse_K": mean, "fixed_three_cycle_pod_scores": metrics(pod, truth, samples, fixed),
        "all_checkpoint_replays_exact": True, "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    write_json(out/"summary.json", summary)
    return {"status": summary["status"], "fixed_three_cycle_seed_mean_rmse_K": mean, "fresh_solver_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.output_root), indent=2))
