"""Operator evidence audit, matched coarse POD/Ridge and request-to-field cascade."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator, corpus, TIME_INDICES, SPACE_INDICES, evaluate
from SciML.models.v12.learning_boundary_state import BoundaryStateModel, autonomous
from SciML.models.v12.temporal_operator_candidate import _feature_scale_floor
from SciML.models.v12.run_learning_existing import sha, write_json, LAYOUT


def matched_pod(branch, initial, truth, stages, train):
    """Rank4 per field, stage-mean residual, Ridge coefficients; coarse labels only."""
    mean = branch[train].mean(axis=0)
    scale = np.maximum(branch[train].std(axis=0), _feature_scale_floor())
    x = (branch-mean)/scale
    coarse = truth[:, :, TIME_INDICES][:, :, :, SPACE_INDICES]-initial[:, :, None, SPACE_INDICES]
    centered = coarse.copy()
    stage_means = np.stack([coarse[train & (stages == s)].mean(axis=0) for s in (2, 4)])
    for i, s in enumerate((2, 4)):
        centered[stages == s] -= stage_means[i]
    predictions, parameters = [], {"feature_mean": mean, "feature_scale": scale, "stage_means": stage_means}
    native_t, native_x = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 101), indexing="ij")
    coordinates = np.stack((native_t, native_x), axis=-1)
    for field in (0, 1):
        flat = centered[:, field].reshape(len(branch), -1)
        with threadpool_limits(limits=1):
            basis = np.linalg.svd(flat[train], full_matrices=False)[2][:4]
            coef = flat[train] @ basis.T
            ridge = Ridge(alpha=1.).fit(x[train], coef)
            fitted = ridge.predict(x) @ basis
        coarse_prediction = fitted.reshape(len(branch), 26, 51)+stage_means[(stages == 4).astype(int), field]
        full = np.stack([RegularGridInterpolator((TIME_INDICES/49., SPACE_INDICES/100.), values)(coordinates) for values in coarse_prediction])
        full += initial[:, field, None]
        full[:, 0] = initial[:, field]
        predictions.append(full)
        parameters.update({f"basis_{field}": basis, f"coef_{field}": ridge.coef_, f"intercept_{field}": ridge.intercept_})
    return np.stack(predictions, axis=1), parameters


def request_field_inputs(state, stage, request):
    """Mirror radial SMA coordinates to the symmetric diameter; inactive side masked."""
    if stage not in (2, 4):
        raise ValueError("only exchange stages")
    radial = state[LAYOUT.sma_T]
    sma = np.r_[radial[:0:-1], radial]
    htf = state[LAYOUT.htf_stage2_T if stage == 2 else LAYOUT.htf_stage4_T]
    side = state[LAYOUT.side].copy()
    side[6 if stage == 2 else 5] = 293.  # same explicitly inactive encoding as training
    case = request["case_spec"]
    mat = [case["material_id"] == "recA", case["material_id"] == "recAs"]
    branch = np.r_[sma-sma.mean(), htf-htf.mean(), sma.mean(), htf.mean(), np.eye(2)[stage//2-1], mat,
                   [case[k] for k in ("radius_m", "thickness_m", "pressure_Pa", "flush_volume_target_ratio", "exchange_duration_s")], side]
    return branch, np.stack((sma, htf))


def run(dataset, campaign, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve operator audit")
    samples, branch, initial, truth, quality = corpus(dataset)
    original = json.loads(original_protocol.read_text())
    masks = {part: np.array([s.condition_id in ids for s in samples]) for part, ids in original["condition_splits"].items()}
    train, stages = masks["train"], np.array([s.stage_index for s in samples])
    roots = {"deeponet": "wp4-deeponet-r1", "fourier": "wp4-fourier-r1",
             "deeponet_no_stage": "wp4-no-stage-r1", "deeponet_no_ic": "wp4-no-ic-r1"}
    reports, saved_predictions = {}, {}
    current_sha = sha(Path("SciML/models/v12/learning_coordinate_operator.py"))
    threads = torch.get_num_threads()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"learning_only": True, "matched_pod_rank": 4, "matched_pod_regression": "Ridge(alpha=1)",
        "training_time_indices": TIME_INDICES.tolist(), "training_space_indices": SPACE_INDICES.tolist(),
        "cascade": "Fixed first seed boundary-r2 B2/B4 supplies DeepONet initial fields, no observed starts",
        "cascade_not_tightly_coupled_operator": True, "fine_mesh_ground_truth": "existing native discretization only"})
    try:
        torch.set_num_threads(1)
        for kind, name in roots.items():
            root = campaign/name
            summary = verify_artifacts(root)
            protocol = json.loads((root/"protocol.json").read_text())
            if protocol["implementation_sha256"] != current_sha or protocol["condition_splits"] != original["condition_splits"]:
                raise ValueError("operator implementation or split mismatch")
            model = CoordinateOperator.load(root/"model.pt")
            if model.training_ids != [s.sample_id for s, used in zip(samples, train) if used]:
                raise ValueError("operator training IDs changed")
            np.testing.assert_allclose(model.feature_mean.numpy(), branch[train].mean(axis=0), atol=0, rtol=0)
            np.testing.assert_allclose(model.feature_scale.numpy(), np.maximum(branch[train].std(axis=0), _feature_scale_floor()), atol=0, rtol=0)
            data = np.load(root/"predictions.npz", allow_pickle=False)
            np.testing.assert_array_equal(data["sample_ids"], [s.sample_id for s in samples])
            np.testing.assert_array_equal(data["truth"], truth)
            prediction = model.predict(branch, initial)
            np.testing.assert_array_equal(prediction, data["prediction"])
            for part, mask in masks.items():
                # Independent array expression rather than calling production metrics.
                for stage in (2, 4):
                    active = mask & (stages == stage)
                    for j, field in enumerate(("sma", "htf")):
                        error = prediction[active, j]-truth[active, j]
                        actual = {"future_rmse_K": float(np.sqrt(np.mean(error[:, 1:]**2))),
                                  "initial_max_abs_K": float(np.abs(error[:, 0]).max()),
                                  "future_max_abs_K": float(np.abs(error[:, 1:]).max())}
                        if actual != summary["scores"][part]["model"][str(stage)][field]:
                            raise ValueError("field metric mismatch")
            if kind != "deeponet_no_ic":
                np.testing.assert_array_equal(prediction[:, :, 0], initial)
            reports[kind], saved_predictions[kind] = summary, prediction
        # Deliberate same-seed refit with withheld target coordinates corrupted.
        # If they leak into target scales or loss, this cannot reproduce the fit.
        seen = np.zeros((50, 101), dtype=bool)
        seen[np.ix_(TIME_INDICES, SPACE_INDICES)] = True
        poisoned = truth[train].copy()
        poisoned[:, :, ~seen] += 1000.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260910)
            replay = CoordinateOperator().double().fit(branch[train], initial[train], poisoned,
                [s.sample_id for s, used in zip(samples, train) if used])
        np.testing.assert_array_equal(replay.predict(branch, initial), saved_predictions["deeponet"])
        pod, parameters = matched_pod(branch, initial, truth, stages, train)
        np.savez_compressed(out/"coarse_pod_parameters.npz", **parameters)
        # Read the existing request-derived boundary checkpoint after verifying artifacts.
        verify_artifacts(campaign/"wp3-boundary-r2")
        boundary = BoundaryStateModel.load(campaign/"wp3-boundary-r2/s20260910/model.pt")
        import pandas as pd
        table = pd.read_parquet(dataset/"case_index.parquet").set_index("case_id")
        requests = {}
        for case, row in table.iterrows():
            path = Path(row.source_artifact).parent/"sciml_case_request.json"
            if sha(path) != row.source_request_sha256:
                raise ValueError("changed request in boundary-to-field cascade")
            requests[case] = json.loads(path.read_text())
        ids = list(table.index)
        states = autonomous(boundary, requests, ids).reshape(len(ids), 4, 414)
        state_lookup = {case: values for case, values in zip(ids, states)}
        forecast_branch, forecast_initial = [], []
        for sample in samples:
            state = state_lookup[sample.case_id][0 if sample.stage_index == 2 else 2]
            b, f = request_field_inputs(state, sample.stage_index, requests[sample.case_id])
            forecast_branch.append(b); forecast_initial.append(f)
        cascade = replay.predict(np.stack(forecast_branch), np.stack(forecast_initial))
    finally:
        torch.set_num_threads(threads)
    comparison = {key: report["scores"]["test"]["model"] for key, report in reports.items()}
    comparison["coarse_pod_ridge"] = evaluate(pod, truth, samples, masks["test"])
    comparison["request_boundary_to_deeponet"] = evaluate(cascade, truth, samples, masks["test"])
    np.savez_compressed(out/"comparison_predictions.npz", sample_ids=np.array([s.sample_id for s in samples]),
                        coarse_pod=pod, request_boundary_cascade=cascade)
    figure(samples, truth, saved_predictions, pod, cascade, masks["test"], comparison, out)
    audit = {"status": "passed_operator_learning_audit", "learning_only": True, "fresh_solver_calls": 0,
             "all_four_checkpoint_predictions_exact": True, "withheld_label_poison_refit_exact": True,
             "test_comparison": comparison, "cascade_boundary_model_seed": 20260910,
             "native_fine_coordinates_are_not_new_solver_mesh": True, "multi_seed_architecture_comparison": False,
             "dataset_source_content_sha256": quality["source_content_sha256"], "implementation_sha256": sha(Path(__file__)),
             "source_summary_hashes": {key: sha(campaign/name/"summary.json") for key, name in roots.items()},
             "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", audit)
    return audit


def figure(samples, truth, predictions, pod, cascade, test, comparison, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["DeepONet", "Fourier", "No stage", "No hard IC", "Coarse POD/Ridge", "Request cascade"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for index, (key, metrics) in enumerate(comparison.items()):
        values = [metrics[str(s)][field]["future_rmse_K"] for s in (2, 4) for field in ("sma", "htf")]
        axes[0].plot(range(4), values, marker="o", label=labels[index])
    axes[0].set(xticks=range(4), xticklabels=["S2 SMA", "S2 HTF", "S4 SMA", "S4 HTF"],
                ylabel="Test future-field RMSE (K)", title="Matched coarse-label operator comparisons")
    axes[0].legend(fontsize=8)
    # Fixed first held-out stage4, not a hand-selected best image.
    row = next(i for i, s in enumerate(samples) if test[i] and s.stage_index == 4)
    for key, values in (("Truth", truth), ("DeepONet", predictions["deeponet"]), ("Fourier", predictions["fourier"]),
                        ("Coarse POD/Ridge", pod), ("Request cascade", cascade)):
        axes[1].plot(np.linspace(0, 1, 50), values[row, 0].mean(axis=1), label=key)
    axes[1].set(xlabel="Normalized stage time", ylabel="Spatial-mean SMA temperature (K)",
                title=f"First test stage4: {samples[row].case_id.split('_')[2]}")
    axes[1].legend(fontsize=8)
    fig.savefig(out/"operator_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for j, name in enumerate(("SMA", "HTF")):
        actual, forecast = truth[row, j], predictions["deeponet"][row, j]
        low, high = actual.min(), actual.max()
        for i, values in enumerate((actual, forecast, forecast-actual)):
            options = dict(cmap="coolwarm", vmin=-np.abs(values).max(), vmax=np.abs(values).max()) if i == 2 else dict(cmap="viridis", vmin=low, vmax=high)
            im = axes[j, i].imshow(values, origin="lower", aspect="auto", extent=(0, 1, 0, 1), **options)
            axes[j, i].set(title=f"{name}: {['observed','DeepONet','error'][i]}", xlabel="Normalized field coordinate", ylabel="Normalized time")
            fig.colorbar(im, ax=axes[j, i], label="K", shrink=.8)
    fig.savefig(out/"operator_fields.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.campaign_root, args.original_protocol, args.output_root), indent=2))
