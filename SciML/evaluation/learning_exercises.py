"""Executable reference answers on tiny manufactured arrays; no project training.

Passing these answers verifies the examples, not the reader's independent skill.
See chapter14 for problem statements, staged hints and advanced acceptance tests.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from SciML.active_learning.learning_pool_loop import ArchivedLabelOracle, choose_batch, hypervolume, pareto_mask
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator
from SciML.models.v12.learning_ensemble import calibrated_intervals
from SciML.models.v12.learning_staged_physics import FO, pde_residual, energy_residual
from SciML.models.v12.run_learning_existing import sha, write_json


def strict_rmse(prediction, truth):
    """Keep identical [sample,target] layouts; never permit implicit broadcast."""
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if prediction.ndim != 2 or prediction.shape != truth.shape or not prediction.size:
        raise ValueError("identical nonempty [N,D] arrays required")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("finite inputs required")
    return np.sqrt(np.mean((prediction-truth)**2, axis=0))


def training_statistics(values, train):
    values, train = np.asarray(values, float), np.asarray(train)
    if values.ndim != 2 or train.dtype != np.bool_ or train.shape != (len(values),) or not train.any():
        raise ValueError("2D values and a nonempty Boolean training mask required")
    if not np.isfinite(values[train]).all():
        raise ValueError("nonfinite training input")
    mean, scale = values[train].mean(0), values[train].std(0)
    return mean, np.where(scale == 0, 1., scale)


def split_directional_memory(strip):
    strip = np.asarray(strip, float)
    if strip.ndim < 1 or strip.shape[-1] != 101 or not np.isfinite(strip).all():
        raise ValueError("finite [...,101] strip required")
    reverse = strip[..., ::-1]
    return .5*(strip+reverse), .5*(strip-reverse)


class AnalyticStagedHeat(torch.nn.Module):
    """Known benchmark only, not an SMA predictor or a trained network."""
    def forward(self, coords, blocks):
        fo = torch.tensor(FO, dtype=coords.dtype, device=coords.device)
        starts = torch.cat((fo.new_zeros(1), fo.repeat(2).cumsum(0)))[:8]
        x, tau = coords[:, :1], coords[:, 1:]
        return torch.sin(torch.pi*x)*torch.exp(-torch.pi**2*(starts[blocks, None]+fo[blocks % 4, None]*tau))


def rejected(call):
    try:
        call()
    except ValueError:
        return True
    raise AssertionError("deliberate invalid input was not rejected")


def reference_checks():
    answers = {}
    # E01: the wrong expression yields [3,3] pairwise differences, not [3,1].
    y = np.array([[1.], [2.], [3.]])
    np.testing.assert_array_equal(strict_rmse(y, y), [0.])
    answers["E01"] = {"correct_rmse": strict_rmse(y, y).tolist(),
        "wrong_broadcast_shape": list((y-y[:, 0]).shape),
        "invalid_shape_rejected": rejected(lambda: strict_rmse(y, y[:, 0]))}

    # E02: holdout mutation leaves train-fitted mean and scale unchanged.
    x = np.arange(12., dtype=float).reshape(6, 2)
    mask = np.array([True, True, True, False, False, False])
    copy = x.copy()
    copy[~mask] += 1e6
    a, b = training_statistics(x, mask), training_statistics(copy, mask)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    answers["E02"] = {"mean": a[0].tolist(), "scale": a[1].tolist(),
        "holdout_poison_train_statistics_unchanged": True,
        "wrong_global_mean_change": float(np.max(abs(copy.mean(0)-x.mean(0))))}

    # E03: manual one-parameter MSE update agrees with autograd.
    tx = torch.tensor([1., 2.], dtype=torch.float64)
    target = 2*tx
    weight = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    before = torch.mean((weight*tx-target)**2)
    before.backward()
    gradient = float(weight.grad)
    with torch.no_grad():
        weight -= .1*weight.grad
    after = torch.mean((weight*tx-target)**2)
    np.testing.assert_array_equal([float(before.detach()), gradient, float(weight.detach()), float(after.detach())], [10., -10., 1., 2.5])
    answers["E03"] = {"loss_before": 10., "gradient": gradient, "weight_after": 1., "loss_after": 2.5,
        "scope": "one scalar instructional gradient update, not project model training"}

    # E04: two different strips have identical even part but opposite memory.
    grid = np.linspace(-1., 1., 101)
    strip = 293.+grid**2+2*grid
    even, odd = split_directional_memory(strip)
    np.testing.assert_allclose(even+odd, strip, atol=1e-13, rtol=0)
    np.testing.assert_allclose(odd, -odd[::-1], atol=0, rtol=0)
    second_even, second_odd = split_directional_memory(even-odd)
    np.testing.assert_array_equal(second_even, even)
    np.testing.assert_array_equal(second_odd, -odd)
    answers["E04"] = {"mirror_only_max_error_K": float(abs(strip-even).max()),
        "odd_center_K": float(odd[50]), "odd_independent_degrees": 50,
        "same_even_different_full_strip": True}

    # E05: no fitting needed to test an architectural hard initial condition.
    with torch.random.fork_rng():
        torch.manual_seed(14)
        operator = CoordinateOperator().double()
    branch = torch.zeros((1, 221), dtype=torch.float64)
    initial = torch.tensor(np.stack((strip, strip+.5))[None], dtype=torch.float64)
    positions = torch.tensor([0., .5, 1.], dtype=torch.float64)
    with torch.no_grad():
        predicted = operator(branch, initial, torch.tensor([0., .5], dtype=torch.float64), positions)
    expected = initial[:, :, [0, 50, 100]]
    torch.testing.assert_close(predicted[:, :, 0, :], expected, rtol=0, atol=0)
    answers["E05"] = {"shape": list(predicted.shape), "initial_max_error_K": float((predicted[:, :, 0, :]-expected).abs().max()),
        "scope": "untrained architecture test; no predictive-accuracy claim"}

    # E06: exercise the actual residual functions on a known analytic family.
    coords = torch.tensor([[.2, .3], [.6, .7]]*8, dtype=torch.float64, requires_grad=True)
    blocks = torch.arange(8).repeat_interleave(2)
    pde = pde_residual(AnalyticStagedHeat(), coords, blocks)
    energy = energy_residual(AnalyticStagedHeat(), torch.tensor([.2, .7], dtype=torch.float64), quadrature_order=32)
    pde_max, energy_max = float(pde.detach().abs().max()), float(energy.detach().abs().max())
    if max(pde_max, energy_max) > 1e-12:
        raise AssertionError("analytic residual not numerically zero")
    answers["E06"] = {"normalized_pde_max": pde_max, "normalized_energy_max": energy_max,
        "tolerance": 1e-12, "scope": "manufactured dimensionless heat equation, not real SMA closure"}

    # E07: independent manual rank and quantile, not rank clamping.
    truth = np.arange(1., 14.)[:, None]
    zeros, ones = np.zeros_like(truth), np.ones_like(truth)
    info90, width90 = calibrated_intervals(truth, zeros, ones, truth, zeros, ones, np.ones(1), .90)
    info95, width95 = calibrated_intervals(truth, zeros, ones, truth, zeros, ones, np.ones(1), .95)
    if info90["order_statistic_rank"] != 13 or info90["quantile"] != 13. or info95["order_statistic_rank"] != 14 or width95 is not None:
        raise AssertionError("finite-sample order-statistic mismatch")
    np.testing.assert_array_equal(width90, np.full_like(truth, 13.))
    answers["E07"] = {"rank90": 13, "half_width90": 13., "rank95": 14, "half_width95": None}

    # E08: selector has no oracle input; unrevealed labels cannot set ranking.
    oracle = ArchivedLabelOracle(["a", "b", "c"], np.array([[1, 3], [2, 2], [3, 1]]))
    chosen = choose_batch(["b", "a", "c"], {"uncertainty": np.ones(3)}, "uncertainty", 1, 14)
    if chosen != ["a"]:
        raise AssertionError("tie-break is not deterministic by ID")
    oracle.reveal(chosen, "reference first query")
    rejected(lambda: oracle.reveal(["test"], "forbidden"))
    rejected(lambda: oracle.reveal(chosen, "duplicate"))
    points = np.array([[1., 3.], [2., 2.], [1., 1.]])
    np.testing.assert_array_equal(pareto_mask(points), [True, True, False])
    before_hv, after_hv = hypervolume(points, [0, 0]), hypervolume(np.vstack((points, [3, 1])), [0, 0])
    # Areas 1*3 + 2*2 - 1*2 = 5; adding [3,1] gives 6.
    if (before_hv, after_hv) != (5., 6.) or oracle.revealed != {"a"}:
        raise AssertionError("query isolation or rectangle union incorrect")
    answers["E08"] = {"chosen": chosen, "revealed_ids": sorted(oracle.revealed),
        "rejected_queries_do_not_mutate_history": len(oracle.events) == 1,
        "hypervolume_before": before_hv, "hypervolume_after": after_hv, "gain": after_hv-before_hv}
    return answers


def run(out):
    if out.exists():
        raise FileExistsError("preserve previous exercise evidence")
    answers = reference_checks()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"reference_answers.json", answers)
    sources = [Path(__file__), Path("SciML/active_learning/learning_pool_loop.py"),
        Path("SciML/models/v12/learning_coordinate_operator.py"), Path("SciML/models/v12/learning_ensemble.py"),
        Path("SciML/models/v12/learning_staged_physics.py")]
    report = {"status": "passed_learning_reference_exercises", "exercise_count": len(answers),
        "project_training_calls": 0, "fresh_solver_calls": 0,
        "reader_mastery_verified": False, "scope": "reference implementations and negative-input checks on manufactured arrays; independent learner submissions not assessed",
        "source_sha256": {str(p): sha(p) for p in sources},
        "artifacts": {"reference_answers.json": sha(out/"reference_answers.json")}}
    write_json(out/"summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(args.output_root), indent=2))
