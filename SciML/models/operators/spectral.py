"""One-dimensional regular-grid spectral residual baseline for WP4."""

from __future__ import annotations

import numpy as np

from .data import OperatorBatch


class SpectralResidualBaseline:
    """Fit a diagonal Fourier transfer function from inlet to temperature rise.

    This baseline assumes a regular one-dimensional grid and ignores topology,
    side-state, and stage.  It is included only as an FNO-adjacent spectral
    comparator; it is not represented as a trained FNO.
    """

    def __init__(self, modes: int = 12, ridge: float = 1e-6) -> None:
        self.modes = int(modes)
        self.ridge = float(ridge)
        self._state: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self.fitted_physical_ids: tuple[str, ...] = ()

    def _fit_grid(self, inputs: np.ndarray, outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        n = inputs.shape[1]
        x_fft = np.fft.rfft(inputs, axis=1)
        delta_fft = np.fft.rfft(outputs - inputs, axis=1)
        mean_x = np.mean(x_fft, axis=0)
        mean_delta = np.mean(delta_fft, axis=0)
        centered = x_fft - mean_x
        transfer = np.zeros_like(mean_delta)
        keep = min(self.modes, transfer.size)
        numerator = np.sum(np.conj(centered[:, :keep]) * (delta_fft[:, :keep] - mean_delta[:keep]), axis=0)
        denominator = np.sum(np.abs(centered[:, :keep]) ** 2, axis=0) + self.ridge
        transfer[:keep] = numerator / denominator
        return mean_x, mean_delta, transfer, n

    def fit(self, train: OperatorBatch) -> "SpectralResidualBaseline":
        train.validate()
        self._state = [
            self._fit_grid(train.sma_in, train.sma_out),
            self._fit_grid(train.htf_in, train.htf_out),
        ]
        self.fitted_physical_ids = tuple(sorted(set(train.physical_case_ids)))
        return self

    @staticmethod
    def _predict_grid(inputs: np.ndarray, state: tuple[np.ndarray, np.ndarray, np.ndarray, int]) -> np.ndarray:
        mean_x, mean_delta, transfer, n = state
        x_fft = np.fft.rfft(inputs, axis=1)
        delta_fft = mean_delta + (x_fft - mean_x) * transfer
        return inputs + np.fft.irfft(delta_fft, n=n, axis=1)

    def predict(self, batch: OperatorBatch) -> tuple[np.ndarray, np.ndarray]:
        if not self._state:
            raise RuntimeError("baseline is not fitted")
        return (
            self._predict_grid(batch.sma_in, self._state[0]),
            self._predict_grid(batch.htf_in, self._state[1]),
        )
