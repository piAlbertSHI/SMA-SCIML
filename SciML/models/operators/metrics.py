"""Auditable endpoint-field metrics for the WP4 plumbing prototype.

SMA is stored on a radial grid while directional HTF is stored on a different
ordered grid. Dataset v1.0 provides no audited cross-grid coordinate mapping,
so this module deliberately does not manufacture a node-wise continuity score.
"""

from __future__ import annotations

import numpy as np

from .data import OperatorBatch


def training_temperature_range(
    train: OperatorBatch, margin_k: float = 0.0
) -> tuple[float, float]:
    """Derive a transparent admissibility envelope from train inputs/targets."""

    if not np.isfinite(margin_k) or margin_k < 0:
        raise ValueError("margin_k must be finite and non-negative")
    values = np.concatenate(
        (train.sma_in.ravel(), train.sma_out.ravel(), train.htf_in.ravel(), train.htf_out.ravel())
    )
    return float(np.min(values) - margin_k), float(np.max(values) + margin_k)


def _sample_relative_l2(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(np.sum(error**2, axis=1))
    denominator = np.sqrt(np.sum(reference**2, axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def operator_metrics(
    batch: OperatorBatch,
    sma_prediction: np.ndarray,
    htf_prediction: np.ndarray,
    *,
    temperature_range_k: tuple[float, float],
    endpoint_tolerance_k: float = 2.0,
    mean_delta_t_balance_tolerance_k: float = 1.0,
) -> dict[str, float]:
    """Compute primary Kelvin errors and explicitly heuristic diagnostics."""

    expected_sma = batch.sma_out.shape
    expected_htf = batch.htf_out.shape
    if sma_prediction.shape != expected_sma or htf_prediction.shape != expected_htf:
        raise ValueError(
            f"prediction shapes {(sma_prediction.shape, htf_prediction.shape)} do not match "
            f"{(expected_sma, expected_htf)}"
        )
    if not np.isfinite(sma_prediction).all() or not np.isfinite(htf_prediction).all():
        raise ValueError("predictions contain non-finite values")
    low, high = map(float, temperature_range_k)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("temperature_range_k must be finite and increasing")

    prediction = np.concatenate((sma_prediction, htf_prediction), axis=1)
    truth = np.concatenate((batch.sma_out, batch.htf_out), axis=1)
    inlet = np.concatenate((batch.sma_in, batch.htf_in), axis=1)
    error = prediction - truth
    true_delta = truth - inlet

    endpoint_error = np.column_stack(
        (
            sma_prediction[:, 0] - batch.sma_out[:, 0],
            sma_prediction[:, -1] - batch.sma_out[:, -1],
            htf_prediction[:, 0] - batch.htf_out[:, 0],
            htf_prediction[:, -1] - batch.htf_out[:, -1],
        )
    )
    true_balance = np.mean(batch.sma_out - batch.sma_in, axis=1) + np.mean(
        batch.htf_out - batch.htf_in, axis=1
    )
    predicted_balance = np.mean(sma_prediction - batch.sma_in, axis=1) + np.mean(
        htf_prediction - batch.htf_in, axis=1
    )
    balance_error = predicted_balance - true_balance
    out_of_range = (prediction < low) | (prediction > high)

    values = {
        "rmse_K": float(np.sqrt(np.mean(error**2))),
        "mae_K": float(np.mean(np.abs(error))),
        "sma_rmse_K": float(np.sqrt(np.mean((sma_prediction - batch.sma_out) ** 2))),
        "htf_rmse_K": float(np.sqrt(np.mean((htf_prediction - batch.htf_out) ** 2))),
        # Prediction residual equals predicted delta-T minus true delta-T
        # because both deltas use the same observed inlet field.
        "relative_delta_temperature_residual_l2": float(
            np.mean(_sample_relative_l2(error, true_delta))
        ),
        "endpoint_rmse_K": float(np.sqrt(np.mean(endpoint_error**2))),
        "endpoint_mae_K": float(np.mean(np.abs(endpoint_error))),
        "endpoint_violation_rate": float(
            np.mean(np.max(np.abs(endpoint_error), axis=1) > endpoint_tolerance_k)
        ),
        "unweighted_mean_deltaT_balance_heuristic_mae_K": float(
            np.mean(np.abs(balance_error))
        ),
        "unweighted_mean_deltaT_balance_heuristic_violation_rate": float(
            np.mean(np.abs(balance_error) > mean_delta_t_balance_tolerance_k)
        ),
        "predicted_temperature_min_K": float(np.min(prediction)),
        "predicted_temperature_max_K": float(np.max(prediction)),
        "train_temperature_range_lower_K": low,
        "train_temperature_range_upper_K": high,
        "temperature_out_of_train_range_rate": float(np.mean(out_of_range)),
        # Auxiliary only: absolute temperatures near 293 K can make this look
        # deceptively small even when temperature-change prediction is poor.
        "aux_absolute_temperature_relative_l2": float(
            np.mean(_sample_relative_l2(error, truth))
        ),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("metric computation produced a non-finite value")
    return values
