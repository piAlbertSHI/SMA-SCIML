"""CPU-only stage-endpoint DeepONet-style random-feature regressor.

Because v1.0 has only three fit-eligible operator checkpoints, the encoders are
seeded random features and only a tensor-product branch/trunk head is fitted by ridge
regression. This is neither an end-to-end-trained DeepONet nor a temporal
operator: the 50-point time series in the source HDF5 is not used. All
preprocessing statistics are fitted on the training role.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .data import HTF_NODES, SMA_NODES, OperatorBatch


def _safe_scale(values: np.ndarray, axis: int | tuple[int, ...] = 0) -> np.ndarray:
    scale = np.nanstd(values, axis=axis)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


@dataclass
class _Preprocessor:
    use_side_state: bool
    sma_mean: np.ndarray | None = None
    sma_scale: np.ndarray | None = None
    htf_mean: np.ndarray | None = None
    htf_scale: np.ndarray | None = None
    side_median: np.ndarray | None = None
    side_scale: np.ndarray | None = None
    residual_scale: np.ndarray | None = None

    def fit(self, batch: OperatorBatch) -> None:
        if len(batch) == 0:
            raise ValueError("cannot fit preprocessing on an empty batch")
        self.sma_mean = np.mean(batch.sma_in, axis=0)
        self.sma_scale = _safe_scale(batch.sma_in, axis=0)
        self.htf_mean = np.mean(batch.htf_in, axis=0)
        self.htf_scale = _safe_scale(batch.htf_in, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(batch.side_input, axis=0)
        self.side_median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(batch.side_input), batch.side_input, self.side_median)
        self.side_scale = _safe_scale(filled, axis=0)
        residual = np.concatenate(
            (batch.sma_out - batch.sma_in, batch.htf_out - batch.htf_in), axis=1
        )
        per_target = np.array(
            [_safe_scale(residual[:, :SMA_NODES], axis=None),
             _safe_scale(residual[:, SMA_NODES:], axis=None)], dtype=np.float64
        )
        self.residual_scale = np.concatenate(
            (np.full(SMA_NODES, per_target[0]), np.full(HTF_NODES, per_target[1]))
        )

    def fields(self, batch: OperatorBatch) -> tuple[np.ndarray, np.ndarray]:
        if self.sma_mean is None or self.htf_mean is None:
            raise RuntimeError("preprocessor is not fitted")
        return (
            (batch.sma_in - self.sma_mean) / self.sma_scale,
            (batch.htf_in - self.htf_mean) / self.htf_scale,
        )

    def context(self, batch: OperatorBatch) -> np.ndarray:
        stage = np.zeros((len(batch), 4), dtype=np.float64)
        valid = (batch.stage_index >= 1) & (batch.stage_index <= 4)
        stage[np.arange(len(batch))[valid], batch.stage_index[valid] - 1] = 1.0
        # The stored local-time grid is authoritative. Protocol labels can be
        # rounded audit summaries, and half-stage labels may already encode the
        # shorter horizon, so parsing/halving them would corrupt the control.
        durations = batch.duration_s
        protocol = np.column_stack(
            (durations, np.log1p(durations),
             ["formal" in x for x in batch.protocol_ids],
             ["smoke" in x for x in batch.protocol_ids])
        ).astype(np.float64)
        branch = np.column_stack(
            [[value == label for value in batch.branches] for label in ("full", "half1", "half2")]
        ).astype(np.float64)
        pieces = [stage, protocol, branch, batch.topology]
        if self.use_side_state:
            assert self.side_median is not None and self.side_scale is not None
            missing = ~np.isfinite(batch.side_input)
            filled = np.where(~missing, batch.side_input, self.side_median)
            normalized = (filled - self.side_median) / self.side_scale
            pieces.extend((normalized, missing.astype(np.float64)))
        return np.concatenate(pieces, axis=1)


class DeepONetStyleRandomFeatureRegressor:
    """Multi-branch endpoint regressor with shared or per-stage ridge heads.

    The SMA field, HTF field, and context are encoded by separate branches.
    Context contains stage, protocol duration/type, valve topology, and—unless
    ablated—only the eight ``*_in`` side-state values plus missingness. The
    eight paired ``*_out`` values are never read by this class.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        ridge: float = 1e-4,
        seed: int = 20260819,
        use_side_state: bool = True,
        stage_specific: bool = False,
    ) -> None:
        if latent_dim < 4:
            raise ValueError("latent_dim must be at least 4")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.latent_dim = int(latent_dim)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.use_side_state = bool(use_side_state)
        self.stage_specific = bool(stage_specific)
        self.preprocessor = _Preprocessor(use_side_state=self.use_side_state)
        self._weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._heads: dict[str, np.ndarray] = {}
        self._fitted_sample_ids: tuple[str, ...] = ()
        self._fitted_physical_ids: tuple[str, ...] = ()
        self._trained_stage_ids: tuple[int, ...] = ()
        self._context_dimension = 0

    @property
    def fitted_sample_ids(self) -> tuple[str, ...]:
        return self._fitted_sample_ids

    @property
    def fitted_physical_ids(self) -> tuple[str, ...]:
        return self._fitted_physical_ids

    @property
    def trained_stage_ids(self) -> tuple[int, ...]:
        return self._trained_stage_ids

    @property
    def context_dimension(self) -> int:
        """Dimension after deterministic stage/protocol/topology preprocessing."""
        return self._context_dimension

    @property
    def input_signature(self) -> tuple[str, ...]:
        signature = [
            "sma_in_51", "htf_in_101", "stage_onehot_4",
            "effective_duration_and_protocol_features_4", "branch_onehot_3", "topology_6",
        ]
        if self.use_side_state:
            signature.extend(("side_input_8", "side_input_missing_mask_8"))
        return tuple(signature)

    def _initialize(self, context_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        for name, width in (("sma", SMA_NODES), ("htf", HTF_NODES), ("context", context_dim)):
            weight = rng.normal(0.0, 1.0 / np.sqrt(width), (width, self.latent_dim))
            bias = rng.uniform(-0.25, 0.25, self.latent_dim)
            self._weights[name] = (weight, bias)
        trunk_width = 10
        self._weights["trunk"] = (
            rng.normal(0.0, 1.0 / np.sqrt(trunk_width), (trunk_width, self.latent_dim)),
            rng.uniform(-0.25, 0.25, self.latent_dim),
        )

    def _encode(self, batch: OperatorBatch) -> np.ndarray:
        sma, htf = self.preprocessor.fields(batch)
        context = self.preprocessor.context(batch)
        encoded = []
        for name, values in (("sma", sma), ("htf", htf), ("context", context)):
            weight, bias = self._weights[name]
            encoded.append(np.tanh(values @ weight + bias))
        # Multiplicative fusion retains cross-branch interactions while the
        # mean term keeps a stable path when a branch is near zero.
        return np.tanh(encoded[0] + encoded[1] + encoded[2]) + 0.25 * (
            encoded[0] * encoded[1] * encoded[2]
        )

    def _trunk(self) -> np.ndarray:
        xs = np.concatenate((np.linspace(0.0, 1.0, SMA_NODES), np.linspace(0.0, 1.0, HTF_NODES)))
        kind = np.concatenate((np.zeros(SMA_NODES), np.ones(HTF_NODES)))
        features = np.column_stack(
            (np.ones_like(xs), xs, xs**2, xs**3,
             np.sin(np.pi * xs), np.cos(np.pi * xs),
             np.sin(2.0 * np.pi * xs), np.cos(2.0 * np.pi * xs),
             kind, kind * xs)
        )
        weight, bias = self._weights["trunk"]
        return np.tanh(features @ weight + bias)

    def _design(self, branch: np.ndarray) -> np.ndarray:
        trunk = self._trunk()
        # A full tensor product is used instead of only pairing equal latent
        # coordinates. It remains a fixed random-feature map, but gives the
        # ridge head enough spatial basis capacity for the 51/101-node fields.
        interaction = (
            branch[:, None, :, None] * trunk[None, :, None, :]
        ).reshape(len(branch), len(trunk), self.latent_dim**2)
        # Trunk-only features provide the constant branch of DeepONet.
        trunk_only = np.broadcast_to(
            trunk[None, :, :], (len(branch), len(trunk), self.latent_dim)
        )
        # Branch-only features allow each observed inlet/context state to set
        # a spatially uniform offset. They are still causal endpoint inputs
        # and materially improve the small-corpus ridge fit without adding
        # target-derived or sample-identity features.
        branch_only = np.broadcast_to(
            branch[:, None, :], (len(branch), len(trunk), self.latent_dim)
        )
        return np.concatenate((interaction, trunk_only, branch_only), axis=2)

    def _fit_head(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        matrix = design.reshape(-1, design.shape[-1])
        vector = target.reshape(-1)
        if matrix.shape[1] > matrix.shape[0]:
            gram = matrix @ matrix.T
            penalty = self.ridge * np.eye(gram.shape[0])
            return matrix.T @ np.linalg.solve(gram + penalty, vector)
        gram = matrix.T @ matrix
        penalty = self.ridge * np.eye(gram.shape[0])
        return np.linalg.solve(gram + penalty, matrix.T @ vector)

    def fit(self, train: OperatorBatch) -> "DeepONetStyleRandomFeatureRegressor":
        train.validate()
        self.preprocessor.fit(train)
        context = self.preprocessor.context(train)
        self._context_dimension = context.shape[1]
        self._initialize(context.shape[1])
        design = self._design(self._encode(train))
        baseline = np.concatenate((train.sma_in, train.htf_in), axis=1)
        scale = self.preprocessor.residual_scale
        assert scale is not None
        target = (np.concatenate((train.sma_out, train.htf_out), axis=1) - baseline) / scale
        self._heads = {
            "shared_sma": self._fit_head(
                design[:, :SMA_NODES], target[:, :SMA_NODES]
            ),
            "shared_htf": self._fit_head(
                design[:, SMA_NODES:], target[:, SMA_NODES:]
            ),
        }
        if self.stage_specific:
            for stage in np.unique(train.stage_index):
                selected = train.stage_index == stage
                self._heads[f"stage_{int(stage)}_sma"] = self._fit_head(
                    design[selected, :SMA_NODES], target[selected, :SMA_NODES]
                )
                self._heads[f"stage_{int(stage)}_htf"] = self._fit_head(
                    design[selected, SMA_NODES:], target[selected, SMA_NODES:]
                )
        self._fitted_sample_ids = train.sample_ids
        self._fitted_physical_ids = tuple(sorted(set(train.physical_case_ids)))
        self._trained_stage_ids = tuple(int(stage) for stage in sorted(np.unique(train.stage_index)))
        return self

    def save(self, path: str | "Path") -> None:
        """Save all fitted numerical state without serializing source batches."""

        from pathlib import Path
        import json

        if not self._heads:
            raise RuntimeError("model is not fitted")
        metadata = {
            "latent_dim": self.latent_dim,
            "ridge": self.ridge,
            "seed": self.seed,
            "use_side_state": self.use_side_state,
            "stage_specific": self.stage_specific,
            "fitted_sample_ids": list(self._fitted_sample_ids),
            "fitted_physical_ids": list(self._fitted_physical_ids),
            "trained_stage_ids": list(self._trained_stage_ids),
            "context_dimension": self._context_dimension,
            "weight_names": sorted(self._weights),
            "head_names": [str(name) for name in self._heads],
        }
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        for name in ("sma_mean", "sma_scale", "htf_mean", "htf_scale", "side_median", "side_scale", "residual_scale"):
            value = getattr(self.preprocessor, name)
            if value is None:
                raise RuntimeError(f"missing fitted preprocessor field {name}")
            arrays[f"pre_{name}"] = value
        for name, (weight, bias) in self._weights.items():
            arrays[f"weight_{name}"] = weight
            arrays[f"bias_{name}"] = bias
        for name, head in self._heads.items():
            arrays[f"head_{name}"] = head
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | "Path") -> "DeepONetStyleRandomFeatureRegressor":
        """Load a fitted model saved by :meth:`save`."""

        import json

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            model = cls(
                latent_dim=int(metadata["latent_dim"]),
                ridge=float(metadata["ridge"]),
                seed=int(metadata["seed"]),
                use_side_state=bool(metadata["use_side_state"]),
                stage_specific=bool(metadata["stage_specific"]),
            )
            for name in ("sma_mean", "sma_scale", "htf_mean", "htf_scale", "side_median", "side_scale", "residual_scale"):
                setattr(model.preprocessor, name, np.asarray(archive[f"pre_{name}"], dtype=np.float64))
            model._weights = {
                name: (
                    np.asarray(archive[f"weight_{name}"], dtype=np.float64),
                    np.asarray(archive[f"bias_{name}"], dtype=np.float64),
                )
                for name in metadata["weight_names"]
            }
            heads: dict[str, np.ndarray] = {}
            for name in metadata["head_names"]:
                heads[name] = np.asarray(archive[f"head_{name}"], dtype=np.float64)
            model._heads = heads
            model._fitted_sample_ids = tuple(metadata["fitted_sample_ids"])
            model._fitted_physical_ids = tuple(metadata["fitted_physical_ids"])
            model._trained_stage_ids = tuple(int(value) for value in metadata["trained_stage_ids"])
            model._context_dimension = int(metadata["context_dimension"])
        return model

    def predict(self, batch: OperatorBatch) -> tuple[np.ndarray, np.ndarray]:
        if not self._heads:
            raise RuntimeError("model is not fitted")
        batch.validate()
        design = self._design(self._encode(batch))
        scaled = np.empty((len(batch), SMA_NODES + HTF_NODES), dtype=np.float64)
        for index, stage in enumerate(batch.stage_index):
            prefix = f"stage_{int(stage)}" if self.stage_specific else "shared"
            sma_head = self._heads.get(f"{prefix}_sma", self._heads["shared_sma"])
            htf_head = self._heads.get(f"{prefix}_htf", self._heads["shared_htf"])
            scaled[index, :SMA_NODES] = design[index, :SMA_NODES] @ sma_head
            scaled[index, SMA_NODES:] = design[index, SMA_NODES:] @ htf_head
        scale = self.preprocessor.residual_scale
        assert scale is not None
        baseline = np.concatenate((batch.sma_in, batch.htf_in), axis=1)
        prediction = baseline + scaled * scale
        return prediction[:, :SMA_NODES], prediction[:, SMA_NODES:]

    def loss(self, batch: OperatorBatch) -> float:
        sma, htf = self.predict(batch)
        scale = self.preprocessor.residual_scale
        assert scale is not None
        error = np.concatenate((sma - batch.sma_out, htf - batch.htf_out), axis=1) / scale
        return float(np.mean(error**2))
