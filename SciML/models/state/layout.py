"""Canonical vector layout for a stage-boundary state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateLayout:
    """Fixed dataset-v1 state layout.

    AER is deliberately absent because dataset v1.0 does not contain its full
    101-node state.  The eight side variables are the input or output member
    of the eight ``*_in``/``*_out`` pairs stored in ``fields.h5``.
    """

    sma_nodes: int = 51
    htf_nodes: int = 101
    xi_nodes: int = 51
    side_size: int = 8

    @property
    def sma_t(self) -> slice:
        return slice(0, self.sma_nodes)

    @property
    def htf_t(self) -> slice:
        start = self.sma_t.stop
        return slice(start, start + self.htf_nodes)

    @property
    def xi(self) -> slice:
        start = self.htf_t.stop
        return slice(start, start + self.xi_nodes)

    @property
    def side(self) -> slice:
        start = self.xi.stop
        return slice(start, start + self.side_size)

    @property
    def size(self) -> int:
        return self.side.stop

    @property
    def temperature_indices(self) -> np.ndarray:
        return np.r_[np.arange(self.sma_t.start, self.sma_t.stop),
                     np.arange(self.htf_t.start, self.htf_t.stop),
                     np.arange(self.side.start, self.side.stop)]

    def validate(self, state: np.ndarray, *, name: str = "state") -> np.ndarray:
        array = np.asarray(state, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != self.size:
            raise ValueError(
                f"{name} must end in dimension {self.size}; got {array.shape}"
            )
        return array

    def assemble(
        self,
        sma_t: np.ndarray,
        htf_t: np.ndarray,
        xi: np.ndarray,
        side: np.ndarray,
    ) -> np.ndarray:
        parts = [np.asarray(x, dtype=np.float64) for x in (sma_t, htf_t, xi, side)]
        expected = [self.sma_nodes, self.htf_nodes, self.xi_nodes, self.side_size]
        for part, width in zip(parts, expected):
            if part.shape[-1] != width:
                raise ValueError(f"state component must end in {width}; got {part.shape}")
        return np.concatenate(parts, axis=-1)
