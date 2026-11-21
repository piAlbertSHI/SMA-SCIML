"""Restore the observed antisymmetric SMA memory in the request-only field cascade."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.evaluation.audit_learning_operators import request_field_inputs
from .learning_coordinate_operator import CoordinateOperator, corpus, evaluate
from .learning_boundary_state import BoundaryStateModel, autonomous
from .run_learning_existing import sha, write_json, LAYOUT


def antisymmetric(field):
    field = np.asarray(field, dtype=float)
    if field.shape[-1] != 101 or not np.isfinite(field).all():
        raise ValueError("finite101point strip required")
    return .5*(field-field[..., ::-1])


def field_inputs_with_memory(state, request, memory):
    branch, initial = request_field_inputs(state, 4, request)
    memory = np.asarray(memory, dtype=float)
    if memory.shape != (101,) or not np.isfinite(memory).all() or not np.allclose(memory, -memory[::-1], rtol=0, atol=1e-12):
        raise ValueError("invalid antisymmetric spatial memory")
    initial[0] += memory
    branch[:101] = initial[0]-initial[0].mean()
    branch[202] = initial[0].mean()
    return branch, initial


def predict_cascade(boundary, operator, requests, case_ids):
    """No observed field/target arguments; stage2 PREDICTED endpoint supplies memory."""
    states = autonomous(boundary, requests, case_ids).reshape(len(case_ids), 4, 414)
    b2, i2 = zip(*(request_field_inputs(state[0], 2, requests[case]) for case, state in zip(case_ids, states)))
    stage2 = operator.predict(np.stack(b2), np.stack(i2))
    memory = antisymmetric(stage2[:, 0, -1])
    b4, i4 = zip(*(field_inputs_with_memory(state[2], requests[case], m) for case, state, m in zip(case_ids, states, memory)))
    stage4 = operator.predict(np.stack(b4), np.stack(i4))
    # Rows arecase1stage2,case1stage4,case2stage2,..., same as original corpus.
    return np.stack((stage2, stage4), axis=1).reshape(2*len(case_ids), 2, 50, 101), memory


def assert_roundoff_only(actual, expected):
    """Cross-batch float64 check, NOT a temperature/physics acceptance gate."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    tolerance = 8*np.finfo(np.float64).eps*max(1., float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    return {"max_abs_K": float(np.max(np.abs(actual-expected))), "roundoff_bound_K": tolerance,
            "exact_equal": bool(np.array_equal(actual, expected))}


def run(dataset, campaign, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve spatial-memory repair evidence")
    verify_artifacts(campaign/"wp3-boundary-r2")
    verify_artifacts(campaign/"wp4-deeponet-r1")
    verify_artifacts(campaign/"wp4-operator-audit-r1")
    samples, branch, initial, truth, quality = corpus(dataset)
    original = json.loads(original_protocol.read_text())
    if quality["source_content_sha256"] != original["dataset_source_content_sha256"]:
        raise ValueError("split/source mismatch")
    table = pd.read_parquet(dataset/"case_index.parquet").set_index("case_id")
    ids = sorted(table.index.tolist())
    if [s.case_id for s in samples[::2]] != ids:
        raise ValueError("case order mismatch")
    requests, identities = {}, []
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for case in ids:
            row = table.loc[case]
            path = Path(row.source_artifact).parent/"sciml_case_request.json"
            if sha(path) != row.source_request_sha256:
                raise ValueError("request hash changed")
            requests[case] = json.loads(path.read_text())
            cycle = handle[str(row.hdf5_group)+"/cycles/cycle_001"]
            hold = np.asarray(cycle["stage2_hold_exchange/sma_strip_T_K_t"])[-1]
            cold = np.asarray(cycle["stage4_cooldown_exchange/sma_strip_T_K_t"])[0]
            radial = np.asarray(cycle["stage4_cooldown_exchange/sma_T_in_51_K"])
            mirror = np.r_[radial[:0:-1], radial]
            restored = mirror+antisymmetric(hold)
            error = float(np.max(np.abs(restored-cold)))
            if error > 1e-10:
                raise ValueError("recorded phase-interface does not obey proposed symmetry transport")
            identities.append({"case_id": case, "mirror_only_max_error_K": float(np.max(np.abs(mirror-cold))),
                               "memory_transport_max_error_K": error})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"learning_only": True, "fresh_solver_calls": 0,
        "dataset_source_content_sha256": quality["source_content_sha256"],
        "rule": "Tstrip_stage4_in=mirror(T51_stage4_in)+antisymmetric(Tstrip_stage2_end)",
        "source_rule_verified_cases": len(identities),
        "inference": "Predicted stage2 endpoint memory only; stage4 observed initial never read by predict_cascade",
        "boundary_seed": 20260910, "operator_seed": 20260910, "new_model_training": False,
        "scope": "Directional-temperature carry added to boundary/field cascade; not full AER energy or multicycle closure",
        "implementation_sha256": sha(Path(__file__)),
        "physical_source_sha256": sha(Path("AER_OperatorSplit_Bed.m"))})
    boundary = BoundaryStateModel.load(campaign/"wp3-boundary-r2/s20260910/model.pt")
    operator = CoordinateOperator.load(campaign/"wp4-deeponet-r1/model.pt")
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        prediction, memory = predict_cascade(boundary, operator, requests, ids)
        # Fresh input dictionaries with poisoned, unused future fields cannot affect inference.
        poisoned = {case: {**request, "future_field": np.full((50, 101), np.nan)} for case, request in requests.items()}
        replay, replay_memory = predict_cascade(boundary, operator, poisoned, ids)
        np.testing.assert_array_equal(prediction, replay)
        np.testing.assert_array_equal(memory, replay_memory)
        # Reproduce the OLD alternating-stage batching before attributing any
        # cross-batch differences to floating point. Same checkpoint/inputs must
        # still reproduce the original saved baseline bit for bit.
        states = autonomous(boundary, requests, ids).reshape(len(ids), 4, 414)
        pairs = [request_field_inputs(state[0 if stage == 2 else 2], stage, requests[case])
                 for case, state in zip(ids, states) for stage in (2, 4)]
        baseline_replay = operator.predict(np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs]))
    finally:
        torch.set_num_threads(threads)
    prior = np.load(campaign/"wp4-operator-audit-r1/comparison_predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(prior["sample_ids"], [s.sample_id for s in samples])
    # Repair affects onlystage4, not the already-completed stage2 predictions.
    np.testing.assert_array_equal(baseline_replay, prior["request_boundary_cascade"])
    roundoff = assert_roundoff_only(prediction[::2], baseline_replay[::2])
    scores = {part: evaluate(prediction, truth, samples, np.array([s.condition_id in conditions for s in samples]))
              for part, conditions in original["condition_splits"].items()}
    np.savez_compressed(out/"predictions.npz", sample_ids=np.array([s.sample_id for s in samples]),
                        prediction=prediction, predicted_memory=memory)
    write_json(out/"source_identity_checks.json", identities)
    report = {"status": "completed_spatial_memory_cascade_repair", "learning_only": True, "fresh_solver_calls": 0,
              "source_transport_max_error_K": max(i["memory_transport_max_error_K"] for i in identities),
              "mirror_only_source_max_error_K": max(i["mirror_only_max_error_K"] for i in identities),
              "source_case_count": len(identities), "future_payload_invariance": True,
              "stage2_predictions_unchanged_within_float64_roundoff": True,
              "old_alternating_batch_baseline_replay_exact": True, "stage2_cross_batch_comparison": roundoff,
              "scores": scores,
              "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.original_protocol, args.output_root), indent=2))
