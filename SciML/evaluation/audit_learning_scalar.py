"""Cold neural replay, deliberate deterministic refits and five-target scalar audit."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from SciML.evaluation.audit_learning_wp5 import verify_artifacts
from SciML.models.v12.learning_ensemble import LearningEnsemble, design_matrix
from SciML.models.v12.learning_scalar_benchmarks import fit_tracked, classical_models
from SciML.models.v12.run_learning_existing import TARGETS, sha, validate_dataset, write_json


def run(dataset, benchmark, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve previous scalar audit")
    summary = verify_artifacts(benchmark)
    protocol = json.loads((benchmark/"protocol.json").read_text())
    original = json.loads(original_protocol.read_text())
    table, quality, _ = validate_dataset(dataset)
    if quality["source_content_sha256"] != protocol["dataset_source_content_sha256"]:
        raise ValueError("dataset provenance mismatch")
    if sha(original_protocol) != protocol["original_protocol_sha256"] or original["condition_splits"] != protocol["condition_splits"]:
        raise ValueError("split mismatch")
    x, y = design_matrix(table), table[list(TARGETS)].to_numpy(float)
    masks = {key: table.condition_id.isin(ids).to_numpy() for key, ids in protocol["condition_splits"].items()}
    train, cal = masks["train"], masks["calibration"]
    ids = table.case_id[train].tolist()
    xs, ys = StandardScaler().fit(x[train]), StandardScaler().fit(y[train])
    sx, sy = xs.transform(x), ys.transform(y)
    reports, predictions = [], []
    for seed in protocol["seeds"]:
        dest = benchmark/f"s{seed}"
        report = json.loads((dest/"report.json").read_text())
        data = np.load(dest/"predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(data["case_ids"], table.case_id.to_numpy(str))
        np.testing.assert_array_equal(data["targets"], y)
        model = LearningEnsemble.load(dest/"ensemble5.pt")
        raw = LearningEnsemble.load(dest/"raw_targets.pt")
        if list(model.training_ids) != ids or list(raw.training_ids) != ids:
            raise ValueError("training IDs mismatch")
        np.testing.assert_array_equal(model.x_scaler.mean_, xs.mean_)
        np.testing.assert_array_equal(model.x_scaler.scale_, xs.scale_)
        np.testing.assert_array_equal(model.y_scaler.mean_, ys.mean_)
        np.testing.assert_array_equal(model.y_scaler.scale_, ys.scale_)
        actual = model.predict_members(x)
        np.testing.assert_array_equal(actual, data["member_predictions"])
        np.testing.assert_array_equal(actual[0], data["mlp"])
        np.testing.assert_array_equal(actual.mean(axis=0), data["ensemble5"])
        np.testing.assert_array_equal(raw.predict_members(x)[0], data["mlp_raw_targets"])
        # Genuine fresh fits for reproducibility, not a retry of failed production.
        for normalized, saved in ((True, model), (False, raw)):
            replay = fit_tracked(x[train], y[train], ids, x[cal], y[cal], seed=seed,
                                 epochs=300, members=5 if normalized else 2, normalize_targets=normalized)
            np.testing.assert_array_equal(replay.predict_members(x), saved.predict_members(x))
            if replay.history != saved.history or replay.history != report["normalized_history" if normalized else "raw_target_history"]:
                raise ValueError("training/diagnostic curves cannot be reproduced")
        # Only load own locally generated, hash-checked pickle, never an external model.
        with (dest/"classical.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("learning_only") is not True or payload["training_ids"] != ids:
            raise ValueError("classical checkpoint authority mismatch")
        for scaler, expected in ((payload["x_scaler"], xs), (payload["y_scaler"], ys)):
            np.testing.assert_array_equal(scaler.mean_, expected.mean_)
            np.testing.assert_array_equal(scaler.scale_, expected.scale_)
        with threadpool_limits(limits=1):
            for name, estimator in classical_models(seed).items():
                prediction = ys.inverse_transform(estimator.fit(sx[train], sy[train]).predict(sx))
                np.testing.assert_array_equal(prediction, data[name])
                np.testing.assert_array_equal(prediction, payload["y_scaler"].inverse_transform(
                    payload["estimators"][name].predict(payload["x_scaler"].transform(x))))
        # Metrics implemented here independently, including R2 denominator distinction.
        baseline = np.broadcast_to(y[train].mean(axis=0), y.shape)
        np.testing.assert_array_equal(baseline, data["train_mean"])
        for name, parts in report["model_scores"].items():
            for part, mask in masks.items():
                e = data[name][mask]-y[mask]
                mse = (e*e).mean(axis=0)
                variance = ((y[mask]-y[mask].mean(axis=0))**2).mean(axis=0)
                base_mse = ((baseline[mask]-y[mask])**2).mean(axis=0)
                expected = dict(rmse=np.sqrt(mse), mae=np.abs(e).mean(axis=0), max_abs=np.abs(e).max(axis=0),
                                r2=1-mse/variance, rmse_skill_vs_train_mean=1-np.sqrt(mse/base_mse))
                for metric, values in expected.items():
                    np.testing.assert_allclose(values, parts[part][metric], rtol=1e-12, atol=1e-12)
        reports.append(report)
        predictions.append({name: data[name] for name in report["model_scores"]})
    # Five-output first seed should also reproduce the original WP2 MLP exactly.
    original_predictions = np.load(original_protocol.parent/"wp2_predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(predictions[0]["mlp"], original_predictions["neural"])
    for name, info in summary["comparison"].items():
        rmse = np.array([r["model_scores"][name]["test"]["rmse"] for r in reports])
        np.testing.assert_array_equal(rmse.mean(axis=0), info["seed_mean_test_rmse"])
        np.testing.assert_array_equal(rmse.std(axis=0, ddof=1), info["seed_std_test_rmse"])
    # Demonstrate why default min_samples_leaf20 is inappropriate for37 samples:
    # two leaves would require40 samples, so default tree cannot split at all.
    with threadpool_limits(limits=1):
        default = HistGradientBoostingRegressor(early_stopping=False, random_state=20260910).fit(sx[train], sy[train, 0])
        default_prediction = default.predict(sx)
    np.testing.assert_allclose(default_prediction, np.mean(sy[train, 0]), atol=1e-12)
    out.mkdir(parents=True, exist_ok=False)
    make_figures(summary, reports, predictions, masks, ys.scale_, out)
    record = {"status": "passed_scalar_strong_baseline_audit", "learning_only": True,
              "fresh_solver_calls": 0, "seeds_refit": protocol["seeds"],
              "neural_network_fits_replayed": 21, "classical_model_families_refit_per_seed": 3,
              "curves_checkpoint_and_predictions_exact": True, "first_mlp_matches_original_wp2": True,
              "default_hist_leaf20_demonstration": "37 training rows cannot split into two20-row leaves; constant prediction verified",
              "comparison": summary["comparison"], "benchmark_summary_sha256": sha(benchmark/"summary.json"),
              "implementation_sha256": sha(Path(__file__)),
              "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_json(out/"summary.json", record)
    return record


def make_figures(summary, reports, predictions, masks, scale, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    names = ["ridge", "random_forest", "hist_gradient_boosting", "mlp", "ensemble5"]
    matrix = np.array([summary["comparison"][n]["seed_mean_test_rmse"] for n in names])/scale
    colors = axes[0].imshow(np.log10(matrix), cmap="viridis", aspect="auto", vmin=-2, vmax=0)
    axes[0].set(xticks=np.arange(5), xticklabels=["Qcool", "Wpump", "COP", "SCP", "deltaT"],
                yticks=np.arange(5), yticklabels=["Ridge", "RF", "HistGB", "MLP", "Ensemble5"],
                title="Mean test RMSE / train target scale")
    for i in range(5):
        for j in range(5):
            axes[0].text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] < .25 else "black", fontsize=9)
    fig.colorbar(colors, ax=axes[0], shrink=.7, label="log10 normalized RMSE")
    for report in reports:
        first = report["normalized_history"][0]["curve"]
        steps = [row["updates_completed"] for row in first]
        axes[1].plot(steps, [r["train_standardized_mse"] for r in first], c="steelblue", alpha=.6,
                     label="Training" if report is reports[0] else None)
        axes[1].plot(steps, [r["calibration_standardized_mse"] for r in first], c="darkorange", alpha=.6,
                     label="Calibration (no stopping)" if report is reports[0] else None)
    axes[1].set(yscale="log", xlabel="Completed optimizer updates", ylabel="Five-target standardized MSE",
                title="MLP learning curves: three seeds")
    axes[1].legend(fontsize=8)
    ablation = np.array([summary["comparison"][key]["seed_mean_test_rmse"] for key in ("mlp", "mlp_raw_targets")])/scale
    for i, label in enumerate(("Train-normalized targets", "Raw targets (ablation)")):
        axes[2].bar(np.arange(5)+(i-.5)*.35, ablation[i], width=.35, label=label)
    axes[2].set(xticks=np.arange(5), xticklabels=["Qcool", "Wpump", "COP", "SCP", "deltaT"],
                yscale="log", ylabel="Test RMSE / train target scale", title="Same architecture, seed and300 updates")
    axes[2].legend(fontsize=8)
    fig.savefig(out/"scalar_benchmarks.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.benchmark_root, args.original_protocol, args.output_root), indent=2))
