"""Compare fixed temperature operators across query grids on the original numerical mesh."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from SciML.evaluation.audit_learning_operators import matched_pod
from SciML.models.v12.learning_coordinate_operator import (
    CoordinateOperator, SpectralLayer, corpus, evaluate, interpolate,
)
from SciML.models.v12.run_learning_existing import sha, write_json


ROOT = Path(__file__).resolve().parents[1]
GRIDS = (26, 50, 51, 101, 201)
NATIVE_X = np.linspace(0.0, 1.0, 101)
CHECKPOINTS = {
    "deeponet": ("wp4-deeponet-r1", "wp4-repeat-s20260911/deeponet", "wp4-repeat-s20260912/deeponet"),
    "fourier": ("wp4-fourier-r1", "wp4-repeat-s20260911/fourier", "wp4-repeat-s20260912/fourier"),
}
METHODS = (*CHECKPOINTS, "pod_ridge")


def resample(values, source, target):
    return interpolate(torch.from_numpy(values), torch.from_numpy(source), torch.from_numpy(target)).numpy()


def spectral_identity_check():
    """Check retained modes across odd/even lengths and below the mode budget."""
    sizes = sorted({8, 9, *GRIDS})
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        layer = SpectralLayer(width=2, modes=8).double()
        layer.weight.zero_()
        diagonal = torch.arange(2)
        layer.weight[diagonal, diagonal, :, 0] = 1.0
        layer.local.weight.zero_()
        layer.local.bias.zero_()
        errors = {}
        for size in sizes:
            angle = 2 * torch.pi * torch.arange(size, dtype=torch.float64) / size
            signals = torch.stack((torch.ones_like(angle), torch.sin(2 * angle)))[None]
            actual = layer(signals)
            torch.testing.assert_close(actual, signals, rtol=0, atol=1e-12)
            errors[str(size)] = float((actual - signals).abs().max())
    return errors


def scores(prediction, truth, samples):
    raw = evaluate(prediction, truth, samples, np.ones(len(samples), dtype=bool))
    return {
        f"stage{stage}_{field}": {
            "rmse_K": metric["future_rmse_K"],
            "max_abs_K": metric["future_max_abs_K"],
        }
        for stage, fields in raw.items()
        for field, metric in fields.items()
    }


def check_native_replay(prediction, truth, samples, archived_scores):
    actual = evaluate(prediction, truth, samples, np.ones(len(samples), dtype=bool))
    for stage, fields in actual.items():
        for name, metrics in fields.items():
            for metric, value in metrics.items():
                np.testing.assert_allclose(value, archived_scores[stage][name][metric], rtol=0, atol=1e-9)


def grid_evaluation(prediction, positions, native_prediction, truth, samples, initial):
    np.testing.assert_array_equal(
        prediction[:, :, 0], resample(initial, NATIVE_X, positions)
    )
    aligned = resample(prediction, positions, NATIVE_X)
    # This control changes the grid through interpolation only. It isolates
    # the change caused by evaluating the operator itself on a different grid.
    interpolation_control = resample(
        resample(native_prediction, NATIVE_X, positions), positions, NATIVE_X
    )
    return {
        "native_truth_scores": scores(aligned, truth, samples),
        "native_forecast_resampling_control": scores(interpolation_control, truth, samples),
        "future_rmse_vs_resampling_control_K": float(np.sqrt(
            np.mean((aligned[:, :, 1:] - interpolation_control[:, :, 1:]) ** 2)
        )),
        "entry_field_exact_on_query_grid": True,
    }


def archived_json(experiment, filename):
    path = ROOT / "evidence/archive" / experiment / filename
    return path, json.loads(path.read_text(encoding="utf-8"))


def model_evidence(experiment, kind, quality, splits, training_ids):
    protocol_path, protocol = archived_json(experiment, "protocol.json")
    summary_path, summary = archived_json(experiment, "summary.json")
    checkpoint = ROOT / "checkpoints" / experiment / "model.pt"
    digest = sha(checkpoint)
    if (digest != summary["artifacts"]["model.pt"]
            or protocol["dataset_source_content_sha256"] != quality["source_content_sha256"]
            or protocol["condition_splits"] != splits
            or protocol["implementation_sha256"] != sha(ROOT / "SciML/models/v12/learning_coordinate_operator.py")):
        raise ValueError(f"Archived data or model binding changed for {experiment}")
    model = CoordinateOperator.load(checkpoint)
    if model.kind != kind or model.seed != summary["seed"] or model.training_ids != training_ids:
        raise ValueError(f"Model or training split mismatch for {experiment}")
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (protocol_path, summary_path, checkpoint)}
    return model, summary, sources


def run(dataset):
    start = time.perf_counter()
    samples, branch, initial, truth, quality = corpus(dataset)
    _, protocol = archived_json(CHECKPOINTS["fourier"][0], "protocol.json")
    splits = protocol["condition_splits"]
    train = np.array([row.condition_id in splits["train"] for row in samples])
    test = np.array([row.condition_id in splits["test"] for row in samples])
    training_ids = [row.sample_id for row, selected in zip(samples, train) if selected]
    test_samples = [row for row, selected in zip(samples, test) if selected]
    stages = np.array([row.stage_index for row in samples])
    source_sha256 = {}
    stages_report = {kind: [] for kind in METHODS}
    for kind, experiments in CHECKPOINTS.items():
        for experiment in experiments:
            model, summary, sources = model_evidence(experiment, kind, quality, splits, training_ids)
            source_sha256.update(sources)
            native = model.predict(branch[test], initial[test])
            check_native_replay(native, truth[test], test_samples, summary["scores"]["test"]["model"])
            grids = {}
            for size in GRIDS:
                positions = np.linspace(0.0, 1.0, size)
                prediction = native if size == 101 else model.predict(branch[test], initial[test], positions=positions)
                grids[str(size)] = grid_evaluation(
                    prediction, positions, native, truth[test], test_samples, initial[test]
                )
            stages_report[kind].append({"seed": model.seed, "grids": grids})

    native_pod, _ = matched_pod(branch, initial, truth, stages, train)
    pod_native = native_pod[test]
    pod_archive = ROOT / "data/metrics.json"
    pod_reference = json.loads(pod_archive.read_text(encoding="utf-8"))["operator_baselines"]
    check_native_replay(pod_native, truth[test], test_samples, pod_reference["test_comparison"]["coarse_pod_ridge"])
    source_sha256["data/metrics.json"] = sha(pod_archive)
    pod_grids = {}
    for size in GRIDS:
        positions = np.linspace(0.0, 1.0, size)
        pod_grids[str(size)] = grid_evaluation(
            resample(pod_native, NATIVE_X, positions), positions, pod_native,
            truth[test], test_samples, initial[test]
        )
    stages_report["pod_ridge"].append({"seed": None, "grids": pod_grids})

    return {
        "status": "completed_query_grid_audit",
        "fixed_query_nodes": list(GRIDS),
        "native_truth_shape": [50, 101],
        "train_conditions": len(splits["train"]),
        "test_conditions": len(splits["test"]),
        "test_exchange_segments": len(test_samples),
        "test_scope": "Previously examined fixed conditions, observed stage entry, original numerical labels",
        "target_scope": "All forecasts mapped back to the original 101 positions before measuring future error",
        "pod_scope": "Deterministic rank-4 train-only refit, then interpolation from native predictions",
        "spectral_constant_and_sine_max_error": spectral_identity_check(),
        "dataset_source_sha256": quality["source_content_sha256"],
        "dataset_manifest_sha256": sha(dataset / "artifact_manifest.json"),
        "inputs_sha256": source_sha256,
        "implementation_sha256": {
            name: sha(ROOT / name) for name in (
                "examples/operator_resolution_audit.py", "SciML/models/v12/learning_coordinate_operator.py",
                "SciML/evaluation/audit_learning_operators.py",
            )
        },
        "methods": stages_report,
        "fresh_solver_calls": 0,
        "neural_parameter_updates": 0,
        "runtime_seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh audit output path")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = run(args.dataset)
    write_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "query_nodes": result["fixed_query_nodes"],
        "test_exchange_segments": result["test_exchange_segments"],
        "runtime_seconds": result["runtime_seconds"],
        "fresh_solver_calls": 0,
        "neural_parameter_updates": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
