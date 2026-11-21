"""Request-initialized learning model of measured stage-plus-interface boundaries.

Unlike the old canonical corpus, restart re-equilibration is explicitly assigned
to the preceding exchange/adiabatic composite transition. This is an empirical
Ridge+neural map with admissibility guards, NOT a complete reduced physical solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from SciML.models.v11.state_space import LAYOUT, SIDE_KEYS, STAGE_NAMES, TransitionRecord, project_state
from SciML.evaluation.inspect_learning_state_contract import request_initial_state, requested_controls
from .run_learning_existing import sha, validate_dataset, write_json


def boundary_mask(stage):
    if stage not in (1, 2, 3, 4):
        raise ValueError("invalid stage boundary")
    mask = np.zeros(LAYOUT.size, dtype=bool)
    if stage in (1, 2, 3):
        mask[:204] = True
    else:
        mask[LAYOUT.sma_T] = True
    if stage == 2:
        mask[LAYOUT.htf_stage2_T] = True
    if stage in (3, 4):
        mask[LAYOUT.htf_stage4_T] = True
    if stage != 1:
        mask[LAYOUT.side] = True
    return mask


def features(state, controls):
    state, controls = np.asarray(state), np.asarray(controls)
    if state.shape != (414,) or controls.shape != (13,):
        raise ValueError("invalid state/control shape")
    result = np.r_[state, controls[:-1]]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite boundary feature")
    return result


def update_mechanical(state, group, endpoint):
    result = state.copy()
    for key, slc in (("u_m_t", LAYOUT.u), ("w_m_t", LAYOUT.w), ("T_K_t", LAYOUT.sma_T), ("xi_t", LAYOUT.xi)):
        result[slc] = np.asarray(group[key])[endpoint]
    return result


def update_side(state, group, endpoint, stage):
    result = state.copy()
    for j, keys in enumerate(SIDE_KEYS):
        value = float(np.asarray(group["side_state/"+keys[endpoint]]))
        if np.isfinite(value):
            result[LAYOUT.side.start+j] = value
        elif j != (6 if stage == 2 else 5):
            raise ValueError("unexpected missing active side observation")
    return result


def build_boundaries(dataset):
    table, quality, _ = validate_dataset(dataset)
    records, requests, checks = [], {}, []
    with h5py.File(dataset/"fields.h5", "r") as handle:
        for row in table.itertuples(index=False):
            path = Path(row.source_artifact).parent/"sciml_case_request.json"
            if sha(path) != row.source_request_sha256:
                raise ValueError("changed initial request")
            request = json.loads(path.read_text())
            requests[row.case_id] = request
            cycle = handle[str(row.hdf5_group)+"/cycles/cycle_001"]
            s1, s2, s3, s4 = [cycle[name] for name in STAGE_NAMES]
            initial = request_initial_state(request)
            # Boundary1: immediately before stage1; no later field reads.
            # Boundary2: loading completed, observed hold-exchange starts.
            b2 = update_mechanical(initial, s1, -1)
            b2[LAYOUT.sma_T] = np.asarray(s2["sma_T_in_51_K"])
            b2[LAYOUT.htf_stage2_T] = np.asarray(s2["htf_internal_T_K_t"])[0]
            b2 = update_side(b2, s2, 0, 2)
            # Boundary3: after exchange AND mechanical restart re-equilibration.
            b3 = update_mechanical(b2, s3, 0)
            b3[LAYOUT.htf_stage2_T] = np.asarray(s2["htf_internal_T_K_t"])[-1]
            b3 = update_side(b3, s2, 1, 2)
            # Boundary4: after unloading AND observed coldflush setup/side transfer.
            b4 = update_mechanical(b3, s3, -1)
            b4[LAYOUT.sma_T] = np.asarray(s4["sma_T_in_51_K"])
            b4[LAYOUT.htf_stage4_T] = np.asarray(s4["htf_internal_T_K_t"])[0]
            b4 = update_side(b4, s4, 0, 4)
            # Last boundary is observed stage4 output, BEFORE any unobserved next
            # cycle restart. No fabricated periodic or later-cycle target.
            b5 = b4.copy()
            b5[LAYOUT.sma_T] = np.asarray(s4["sma_T_out_51_K"])
            b5[LAYOUT.htf_stage4_T] = np.asarray(s4["htf_internal_T_K_t"])[-1]
            b5 = update_side(b5, s4, 1, 4)
            states = [initial, b2, b3, b4, b5]
            if not all(np.isfinite(a).all() for a in states):
                raise ValueError("nonfinite boundary target")
            for i in range(4):
                mask = boundary_mask(i+1)
                if not np.allclose(states[i+1][~mask], states[i][~mask], rtol=0, atol=1e-12):
                    raise ValueError("measured boundary conflicts with declared ownership")
                records.append(TransitionRecord(f"{row.case_id}:boundary_{i+1}_to_{i+2}", row.case_id,
                    row.condition_id, row.material_id, f"R{row.radius_m}_t{row.thickness_m}", 1, i+1,
                    states[i], states[i+1], requested_controls(request, i+1)))
            raw_initial = update_mechanical(initial, s1, 0)
            if not np.array_equal(raw_initial[:204], initial[:204]):
                raise ValueError("declared mechanical rest not equal to archived first frame")
            checks.append({"case_id": row.case_id, "request_sha256": sha(path),
                           "initial_mechanics_equal_declared_rest": True,
                           "boundary_masks_preserve_inactive_targets": True})
    return table, quality, records, requests, checks


class BoundaryStateModel:
    def __init__(self, seed=20260910, hidden=32):
        self.seed, self.hidden = seed, hidden
        self.linear, self.networks, self.residual_scale = {}, {}, {}
        self.training_ids = []

    def fit(self, records, epochs=300):
        if not records or len({r.sample_id for r in records}) != len(records) or type(epochs) is not int or epochs < 1:
            raise ValueError("invalid training records or epochs")
        raw = np.stack([features(r.state_in, r.controls) for r in records])
        self.scaler = StandardScaler().fit(raw)
        x = self.scaler.transform(raw)
        delta = np.stack([r.state_out-r.state_in for r in records])
        stages = np.array([r.stage_index for r in records])
        if set(stages) != {1, 2, 3, 4} or not np.isfinite(delta).all():
            raise ValueError("missing stage or invalid boundary delta")
        self.training_ids = []
        history = {}
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                for stage in (1, 2, 3, 4):
                    selected, mask = stages == stage, boundary_mask(stage)
                    target = delta[selected][:, mask]
                    linear = Ridge(alpha=1.).fit(x[selected], target)
                    residual = target-linear.predict(x[selected])
                    scale = np.sqrt(np.mean(residual**2, axis=0))
                    # Zero/negligible residual is not a one-unit physical scale.
                    # A 1m fallback would amplify tiny center-displacement noise.
                    scale = np.where(scale > 1e-10, scale, 0.)
                    torch.manual_seed(self.seed+stage*1009)
                    net = nn.Sequential(nn.Linear(426, self.hidden), nn.SiLU(), nn.Linear(self.hidden, self.hidden),
                                        nn.SiLU(), nn.Linear(self.hidden, int(mask.sum()))).double()
                    normalized_residual = np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 0)
                    tx, ty = torch.from_numpy(x[selected]), torch.from_numpy(normalized_residual)
                    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=1e-4)
                    losses = []
                    for _ in range(epochs):
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.mean((net(tx)-ty)**2)
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite boundary loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 10., error_if_nonfinite=True)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    net.eval()
                    self.linear[stage], self.networks[stage], self.residual_scale[stage] = linear, net, scale
                    history[str(stage)] = {"first_loss": losses[0], "last_preupdate_loss": losses[-1], "active_count": int(mask.sum())}
        finally:
            torch.set_num_threads(old_threads)
        self.training_ids = [r.sample_id for r in records]
        self.epochs = epochs
        self.history = history
        return self

    def predict(self, state, controls, stage, neural=True):
        if not self.training_ids or set(self.networks) != {1, 2, 3, 4}:
            raise RuntimeError("boundary model not fitted")
        mask = boundary_mask(stage)
        if not np.array_equal(controls[:4], np.eye(4)[stage-1]):
            raise ValueError("stage/control mismatch")
        x = self.scaler.transform(features(state, controls)[None, :])
        delta = self.linear[stage].predict(x)[0]
        if neural:
            with torch.no_grad():
                delta += self.networks[stage](torch.from_numpy(x)).numpy()[0]*self.residual_scale[stage]
        if not np.isfinite(delta).all():
            raise RuntimeError("nonfinite boundary prediction before projection")
        result = np.asarray(state).copy()
        result[mask] += delta
        result = project_state(result)
        result[LAYOUT.u.start] = 0.  # axisymmetric radial displacement at r=0
        return result

    def save(self, path):
        payload = {"schema": "learning_boundary_state_v2", "seed": self.seed, "hidden": self.hidden,
                   "epochs": self.epochs, "training_ids": self.training_ids, "history": self.history,
                   "x_mean": torch.tensor(self.scaler.mean_), "x_scale": torch.tensor(self.scaler.scale_),
                   "linear": {s: {"coef": torch.tensor(m.coef_), "intercept": torch.tensor(m.intercept_)} for s, m in self.linear.items()},
                   "residual_scale": {s: torch.tensor(v) for s, v in self.residual_scale.items()},
                   "networks": {s: n.state_dict() for s, n in self.networks.items()}}
        with path.open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path):
        p = torch.load(path, weights_only=True)
        if p["schema"] != "learning_boundary_state_v2":
            raise ValueError("invalid checkpoint schema")
        model = cls(p["seed"], p["hidden"])
        model.scaler = StandardScaler()
        model.scaler.mean_, model.scaler.scale_ = p["x_mean"].numpy(), p["x_scale"].numpy()
        model.scaler.var_, model.scaler.n_features_in_ = model.scaler.scale_**2, 426
        model.scaler.n_samples_seen_ = len(p["training_ids"])
        with torch.random.fork_rng(devices=[]):
            for s in (1, 2, 3, 4):
                linear = Ridge(alpha=1.)
                linear.coef_, linear.intercept_ = p["linear"][s]["coef"].numpy(), p["linear"][s]["intercept"].numpy()
                linear.n_features_in_ = 426
                net = nn.Sequential(nn.Linear(426, model.hidden), nn.SiLU(), nn.Linear(model.hidden, model.hidden),
                                    nn.SiLU(), nn.Linear(model.hidden, int(boundary_mask(s).sum()))).double()
                net.load_state_dict(p["networks"][s])
                net.eval()
                model.linear[s], model.networks[s], model.residual_scale[s] = linear, net, p["residual_scale"][s].numpy()
        model.training_ids, model.epochs, model.history = p["training_ids"], p["epochs"], p["history"]
        return model


def autonomous(model, requests, case_ids, neural=True):
    """Inference accepts request dictionaries ONLY, never records or future outputs."""
    predictions = []
    for case in case_ids:
        state = request_initial_state(requests[case])
        for stage in (1, 2, 3, 4):
            state = model.predict(state, requested_controls(requests[case], stage), stage, neural)
            predictions.append(state.copy())
    return np.stack(predictions)


def run(dataset, original_protocol, out):
    if out.exists():
        raise FileExistsError("preserve previous boundary learning evidence")
    table, quality, records, requests, checks = build_boundaries(dataset)
    original = json.loads(original_protocol.read_text())
    if original["dataset_source_content_sha256"] != quality["source_content_sha256"]:
        raise ValueError("split provenance mismatch")
    train = [r for r in records if r.condition_id in original["condition_splits"]["train"]]
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"learning_only": True, "fresh_solver_calls": 0, "dataset_source_content_sha256": quality["source_content_sha256"],
                "condition_splits": original["condition_splits"], "seeds": [20260910, 20260911, 20260912], "epochs": 300,
                "implementation_sha256": sha(Path(__file__)), "initial_contract_sha256": sha(Path(__file__).parents[2]/"evaluation/inspect_learning_state_contract.py"),
                "boundary_semantics": ["declared rest", "start of hold exchange", "start of unloading after mechanical re-equilibration",
                                       "start of cooldown after side/setup transfers", "end cooldown before any unobserved next-cycle restart"],
                "initial_state": "Request-only rest and thermal defaults; no future-stage initial measurements",
                "residual_scale_policy": "RMS<=1e-10 disables neural residual on that channel; never one-unit fallback",
                "axisymmetric_center_constraint": "u(r=0)=0 after prediction for both neural and affine",
                "inference_signature": "autonomous(model,requests,case_ids): no observed state record input",
                "physical_scope": "Composite stage/interface empirical transitions with ownership and projection; not exact energy/PDE closure or fully observed Markov state",
                "data_scope": "Observed single cycles only; no trained or validated multicycle continuation",
                "test_policy": "Previously inspected exploratory split, fixed configuration"}
    write_json(out/"protocol.json", protocol)
    write_json(out/"boundary_source_checks.json", checks)
    started = time.monotonic()
    results = []
    for seed in protocol["seeds"]:
        model = BoundaryStateModel(seed).fit(train)
        dest = out/f"s{seed}"
        dest.mkdir()
        model.save(dest/"model.pt")
        loaded = BoundaryStateModel.load(dest/"model.pt")
        arrays, scores = {}, {}
        for part, conditions in original["condition_splits"].items():
            case_ids = table.loc[table.condition_id.isin(conditions), "case_id"].tolist()
            rows = [r for r in records if r.condition_id in conditions]
            target = np.stack([r.state_out for r in rows])
            predictions = {"autonomous_neural": autonomous(model, requests, case_ids),
                           "autonomous_affine": autonomous(model, requests, case_ids, False),
                           "teacher_neural": np.stack([model.predict(r.state_in, r.controls, r.stage_index) for r in rows]),
                           "teacher_affine": np.stack([model.predict(r.state_in, r.controls, r.stage_index, False) for r in rows])}
            np.testing.assert_array_equal(predictions["autonomous_neural"], autonomous(loaded, requests, case_ids))
            arrays[part+"_sample_ids"], arrays[part+"_truth"] = np.array([r.sample_id for r in rows]), target
            scores[part] = {}
            for name, prediction in predictions.items():
                arrays[part+"_"+name] = prediction
                scores[part][name] = {str(stage): {block: float(np.sqrt(np.mean((prediction[mask, slc]-target[mask, slc])**2)))
                                                for block, slc in (("sma_K", LAYOUT.sma_T), ("side_K", LAYOUT.side), ("xi", LAYOUT.xi),
                                                                   ("u_m", LAYOUT.u), ("w_m", LAYOUT.w),
                                                                   ("htf2_K", LAYOUT.htf_stage2_T), ("htf4_K", LAYOUT.htf_stage4_T))}
                                     for stage in (1, 2, 3, 4) for mask in [np.array([r.stage_index == stage for r in rows])]}
        np.savez_compressed(dest/"predictions.npz", **arrays)
        result = {"seed": seed, "history": model.history, "scores": scores, "checkpoint_roundtrip_exact": True}
        write_json(dest/"report.json", result)
        results.append(result)
    means = {name: {str(stage): {block: float(np.mean([r["scores"]["test"][name][str(stage)][block] for r in results]))
                               for block in results[0]["scores"]["test"][name][str(stage)]} for stage in (1, 2, 3, 4)}
             for name in results[0]["scores"]["test"]}
    summary = {"status": "completed_request_only_boundary_learning", "learning_only": True, "fresh_solver_calls": 0,
               "test_seed_mean_rmse": means, "elapsed_seconds": time.monotonic()-started,
               "scientific_closure_claim": False, "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    write_json(out/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.original_protocol, args.output_root), indent=2))
