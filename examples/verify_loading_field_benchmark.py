"""Independently check the paired results, replay all loading predictors and audit their metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from examples.loading_field_benchmark import (
    BATCH_SIZE, EPOCHS, PAIR_RESULT, POD_RANK, RADIUS, ROOT, SEEDS, TIME, TI, RI,
    aggregate, predict, score, snapshot, stratified_split, training_arrays,
)
from SciML.models.v13.loading_field_operator import FIELDS, NEURAL_METHODS, NeuralLoadingOperator, PODRidgeLoading
from tools.audit_project_corpora import read_json, sha256
from tools.compare_solver_archives import write


BENCHMARK = ROOT / "evidence/local_studies/loading_field_benchmark_20261121"


def compare(actual, recorded):
    if isinstance(actual, dict):
        assert set(actual) == set(recorded)
        for key, value in actual.items():
            compare(value, recorded[key])
    elif isinstance(actual, list):
        assert len(actual) == len(recorded)
        for a, b in zip(actual, recorded):
            compare(a, b)
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, recorded, rtol=0, atol=1e-12)
    else:
        assert actual == recorded, (actual, recorded)


def verify_paired_results():
    pair_report = read_json(PAIR_RESULT)
    records_path = PAIR_RESULT.parent / "pairs.json"
    for name, digest in pair_report["source_sha256"].items():
        assert sha256(ROOT / name) == digest
    assert sha256(records_path) == pair_report["pairs_sha256"]
    rows = read_json(records_path)
    assert len({row["condition_id"] for row in rows}) == len(rows) == 54
    for field in ("Qcool_J", "W1_J", "Wunload_recovered_J", "Wpump_J", "COP", "SCP_W_per_kg"):
        x = np.array([r["performance_old_new"][field] for r in rows])
        d = x[:, 1] - x[:, 0]
        recorded = pair_report["performance"][field]["new_minus_old"]
        np.testing.assert_allclose(np.median(np.abs(d)), recorded["median_absolute"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean(d**2)), recorded["rmse"], rtol=0, atol=1e-12)
    for row in rows:
        for key in ("u_mm", "w_mm", "sma_K", "phase_fraction", "pressure_MPa"):
            assert row["stages"]["stage1_loading"]["histories"][key]["max_abs"] == 0
        assert row["performance_old_new"]["W1_J"][0] == row["performance_old_new"]["W1_J"][1]
        for side in (0, 1):
            assert row["validation_old_new"][side]["system_energy_audit_ok"]
            qcool = row["performance_old_new"]["Qcool_J"][side]
            stage_heat = row["stages"]["stage4_cooldown_exchange"]["energy_old_new"]["Q_sma_htf_J"][side]
            assert qcool == max(0, stage_heat)
            values = row["performance_old_new"]
            np.testing.assert_allclose(values["COP"][side], qcool / values["Wtotal_for_COP_J"][side], rtol=0, atol=1e-12)
    return {"pairs": len(rows), "loading_fields_and_work_exact": True, "archived_COP_and_heat_identity_checked": True}


def phase_and_edge_diagnostic(prediction, truth, scales):
    """Use the known mechanics/phase bounds as a separate projection diagnostic."""
    np.testing.assert_array_equal(truth[:, :2, :, -1], 0)
    assert (truth[:, 3] >= 0).all() and (truth[:, 3] <= 1).all()
    bounded = prediction.copy()
    bounded[:, :2, :, -1] = 0
    bounded[:, 3] = np.clip(bounded[:, 3], 0, 1)
    def error(x):
        return np.sqrt(np.mean(((x[:, :, 1:] - truth[:, :, 1:]) / scales[None, :, None, None]) ** 2))
    before, after = error(prediction), error(bounded)
    assert after <= before + 1e-12
    return {
        "raw_normalized_test_rmse": float(before),
        "projected_normalized_test_rmse": float(after),
        "minimum_raw_phase": float(prediction[:, 3].min()),
        "maximum_raw_phase": float(prediction[:, 3].max()),
        "raw_edge_radial_displacement_max_abs_mm": float(np.abs(prediction[:, 0, :, -1]).max()),
    }


def verify(archive, dataset):
    pairs = verify_paired_results()
    for name, digest in read_json(archive / "manifest.json")["sha256"].items():
        assert sha256(archive / name) == digest, name
    protocol = read_json(archive / "protocol.json")
    for name, digest in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    assert sha256(PAIR_RESULT) == protocol["paired_comparison_sha256"]

    with np.load(archive / "loading_data.npz", allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    matches_source = None
    if dataset is not None:
        original, provenance = snapshot(dataset)
        for name, values in original.items():
            np.testing.assert_array_equal(data[name], values)
        assert all(protocol[k] == v for k, v in provenance.items())
        matches_source = True
    assert len(set(data["case_ids"])) == len(set(data["condition_ids"])) == 200
    np.testing.assert_array_equal(stratified_split(data["condition_ids"], data["controls"]), data["split"])
    for name, ids in protocol["condition_splits"].items():
        assert data["condition_ids"][data["split"] == name].tolist() == ids
    assert [int((data["split"] == name).sum()) for name in ("train", "validation", "test")] == [140, 30, 30]

    x, change, mean, scale, field_scale = training_arrays(data)
    for key, value in (("control_mean", mean), ("control_scale", scale), ("field_scale", field_scale)):
        np.testing.assert_array_equal(value, protocol[key])
    refit = PODRidgeLoading.fit(x, change, TIME[TI], RADIUS[RI], mean, scale, field_scale, POD_RANK)
    records = read_json(archive / "runs.json")
    accepted = read_json(archive / "summary.json")
    expected = {(kind, seed) for kind in NEURAL_METHODS for seed in SEEDS} | {("pod_ridge", None), ("persistence", None)}
    assert {(r["method"], r["seed"]) for r in records} == expected and len(records) == 11
    assert sum(row["updates"] for row in records) == accepted["neural_updates_total"] == 2160
    replay = {}
    with np.load(archive / "test_predictions.npz", allow_pickle=False) as forecasts:
        assert set(forecasts.files) == {row["run_id"] for row in records}
        for row in records:
            kind = row["method"]
            if kind in NEURAL_METHODS:
                model = NeuralLoadingOperator.load(archive / row["checkpoint"])
                assert model.kind == kind and row["updates"] == EPOCHS * (len(x) // BATCH_SIZE)
                for a, b in ((model.control_mean, mean), (model.control_scale, scale), (model.field_scale, field_scale)):
                    np.testing.assert_array_equal(a.numpy(), b.astype(np.float32))
                assert sum(p.numel() for p in model.parameters()) == row["parameters"]
            elif kind == "pod_ridge":
                model = PODRidgeLoading.load(archive / row["checkpoint"])
                use = data["split"] != "train"
                np.testing.assert_allclose(predict(model, data["controls"][use]),
                                           predict(refit, data["controls"][use]), rtol=0, atol=1e-10)
            else:
                model = None
            for name in ("validation", "test"):
                use = data["split"] == name
                pred = predict(model, data["controls"][use])
                metrics = score(pred, data["fields"][use], field_scale, data["case_ids"][use], data["controls"][use])
                compare(metrics, {key: value for key, value in row[name].items() if key != "prediction_seconds"})
                if name == "test":
                    stored = forecasts[row["run_id"]]
                    np.testing.assert_array_equal(pred, stored)
                    replay[row["run_id"]] = {
                        "prediction_replay_max_difference": float(np.max(np.abs(pred - stored))),
                        "bounds_diagnostic": phase_and_edge_diagnostic(pred, data["fields"][use], field_scale),
                    }
    compare(aggregate(records), accepted["scores"])
    winner = min(accepted["scores"], key=lambda method: accepted["scores"][method]["validation"]["normalized_pooled_rmse"]["mean"])
    assert winner == "pod_ridge"
    return {
        "status": "verified_paired_data_and_loading_benchmark",
        "paired_diagnostics": pairs,
        "matches_source_HDF5": matches_source,
        "checkpoints_replayed": 10,
        "neural_updates_verified": accepted["neural_updates_total"],
        "all_initial_fields_exact": True,
        "pod_train_only_refit_verified": True,
        "validation_selected_baseline": winner,
        "replay_by_run": replay,
        "bound_projection_used_only_as_diagnostic": True,
        "fresh_solver_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=BENCHMARK)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad(), threadpool_limits(limits=1):
        result = verify(args.archive, args.dataset)
    if args.output is not None:
        write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "replay_by_run"}, indent=2))


if __name__ == "__main__":
    main()
