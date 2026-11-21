"""Separate-profile five-step multicycle boundary learner with directional memory.

Observed first mechanical frame + declared thermal initial conditions are allowed.
No later observed states enter rollout. Composite interfaces and carried unobserved
side variables are explicit; this is not full physical Markov/energy closure.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from .learning_boundary_state import update_mechanical, update_side
from .run_learning_existing import STRICT_COLUMNS, sha, write_json
from SciML.evaluation.inspect_learning_state_contract import request_initial_state
from SciML.models.v11.state_space import LAYOUT, STAGE_NAMES

SIZE = 515
MEMORY = slice(414, 515)
BLOCKS = {"sma_K": slice(102, 153), "htf2_K": slice(204, 305), "htf4_K": slice(305, 406),
          "side_K": slice(406, 414), "xi": slice(153, 204), "u_m": slice(0, 51), "w_m": slice(51, 102), "memory_K": MEMORY}


@dataclass
class Step:
    sample_id: str
    case_id: str
    cycle: int
    stage: int
    before: np.ndarray
    after: np.ndarray


def odd(values):
    return .5*(values-values[..., ::-1])


def mask(stage):
    active = np.zeros(SIZE, bool)
    if stage not in (1, 2, 3, 4, 5):
        raise ValueError("invalid composite stage")
    active[:204] = stage != 4
    if stage == 4:
        active[102:153] = True
    if stage in (1, 2):
        active[204:305] = True
    if stage in (3, 4):
        active[305:406] = True
    if stage != 5:
        active[406:515] = True
    return active


def metadata(request):
    c, solver = request["case_spec"], request["run_options"]["solver_options"]
    return np.array([c[key] for key in ("radius_m", "thickness_m", "pressure_Pa", "flush_volume_target_ratio", "exchange_duration_s")]
                    +[c["material_id"] == "recA", c["material_id"] == "recAs", solver["T_load"], solver["T_unload"]], float)


def features(state, control, stage):
    if state.shape != (SIZE,) or control.shape != (9,) or stage not in range(1, 6):
        raise ValueError("invalid state/control/interface")
    value = np.r_[state, control, np.eye(5)[stage-1]]
    if not np.isfinite(value).all():
        raise ValueError("nonfinite macrostate")
    return value


def scale_floor():
    floor = np.full(SIZE, .5)
    floor[:51], floor[51:102], floor[153:204], floor[MEMORY] = 1e-6, 1e-4, .01, .1
    return np.r_[floor, [.003, 1.5e-5, 1e5, .1, .4, .5, .5, .01, .01], np.full(5, .5)]


def project(state):
    result = state.copy()
    result[[0, 50, 101]] = 0.
    result[153:204] = np.clip(result[153:204], 0, 1)
    result[MEMORY] = odd(result[MEMORY])
    # No temperature clipping: unstable predictions remain visible in metrics.
    return result


def split_cases(table):
    """Hold both three-cycle cases + one hash-selected two-cycle case per material."""
    test = table.loc[table.cycle_count == 3, "case_id"].tolist()
    for material in ("recA", "recAs"):
        choices = table.loc[(table.material_id == material) & (table.cycle_count == 2), "case_id"].tolist()
        if not choices:
            raise ValueError("missing two-cycle material group")
        test.append(min(choices, key=lambda c: hashlib.sha256(("historical-rollout-20260910:"+c).encode()).hexdigest()))
    if len(test) != 4:
        raise ValueError("expected fixed four-case historical test")
    return {"train": sorted(set(table.case_id)-set(test)), "test": sorted(test)}


def corpus(root):
    manifest = json.loads((root/"artifact_manifest.json").read_text())
    for item in manifest["artifacts"]:
        if sha(root/item["path"]) != item["sha256"]:
            raise ValueError("historical dataset changed")
    table = pd.read_parquet(root/"case_index.parquet").sort_values("case_id")
    if len(table) != 12 or sorted(table.cycle_count.tolist()) != [2]*10+[3]*2:
        raise ValueError("unexpected historical corpus")
    for key in STRICT_COLUMNS:
        if not table[key].map(lambda v: isinstance(v, (bool, np.bool_)) and bool(v)).all():
            raise ValueError("invalid historical source gate: "+key)
    fps = {}
    for key in ("physics_implementation_fingerprint_sha256", "pipeline_implementation_fingerprint_sha256"):
        values = table[key].unique()
        if len(values) != 1:
            raise ValueError("mixed historical implementation")
        fps[key] = str(values[0])
    records, inputs, source_checks = [], {}, []
    with h5py.File(root/"fields.h5", "r") as handle:
        for row in table.itertuples(index=False):
            source = Path(row.source_artifact)
            request_file = source.parent/"sciml_case_request.json"
            if sha(source) != row.source_artifact_sha256 or sha(request_file) != row.source_request_sha256:
                raise ValueError("historical raw source/request mismatch")
            request = json.loads(request_file.read_text())
            cycle_group = handle[row.hdf5_group+"/cycles"]
            first = cycle_group["cycle_001/stage1_loading"]
            declared = request_initial_state(request)
            initial = np.r_[update_mechanical(declared, first, 0), np.zeros(101)]
            inputs[row.case_id] = {"initial": initial, "control": metadata(request), "cycles": int(row.cycle_count)}
            previous = initial.copy()
            checks = {"case_id": row.case_id, "first_mechanics_vs_declared_rest_max": float(np.max(np.abs(initial[:204]-declared[:204]))),
                      "source_sha256": row.source_artifact_sha256, "request_sha256": row.source_request_sha256,
                      "field_reconstruction_max_K": 0., "restart_mechanics_max_changes": []}
            for cycle in range(1, int(row.cycle_count)+1):
                group = cycle_group[f"cycle_{cycle:03}"]
                s1, s2, s3, s4 = [group[name] for name in STAGE_NAMES]
                if cycle > 1:
                    restart = previous.copy()
                    restart[:414] = update_mechanical(previous[:414], s1, 0)
                    records.append(Step(f"{row.case_id}:c{cycle-1}:restart", row.case_id, cycle-1, 5, previous.copy(), restart.copy()))
                    checks["restart_mechanics_max_changes"].append({block: float(np.max(np.abs(restart[slc]-previous[slc])))
                                                                 for block, slc in BLOCKS.items() if block in ("sma_K", "xi", "u_m", "w_m")})
                    previous = restart
                b2 = previous.copy()
                b2[:414] = update_mechanical(b2[:414], s1, -1)
                b2[102:153] = np.asarray(s2["sma_T_in_51_K"])
                b2[204:305] = np.asarray(s2["htf_internal_T_K_t"])[0]
                b2[:414] = update_side(b2[:414], s2, 0, 2)
                b2[MEMORY] = odd(np.asarray(s2["sma_strip_T_K_t"])[0])
                b3 = b2.copy()
                b3[:414] = update_mechanical(b3[:414], s3, 0)
                b3[204:305] = np.asarray(s2["htf_internal_T_K_t"])[-1]
                b3[:414] = update_side(b3[:414], s2, 1, 2)
                b3[MEMORY] = odd(np.asarray(s2["sma_strip_T_K_t"])[-1])
                b4 = b3.copy()
                b4[:414] = update_mechanical(b4[:414], s3, -1)
                b4[102:153] = np.asarray(s4["sma_T_in_51_K"])
                b4[305:406] = np.asarray(s4["htf_internal_T_K_t"])[0]
                b4[:414] = update_side(b4[:414], s4, 0, 4)
                b4[MEMORY] = odd(np.asarray(s4["sma_strip_T_K_t"])[0])
                b5 = b4.copy()
                b5[102:153] = np.asarray(s4["sma_T_out_51_K"])
                b5[305:406] = np.asarray(s4["htf_internal_T_K_t"])[-1]
                b5[:414] = update_side(b5[:414], s4, 1, 4)
                b5[MEMORY] = odd(np.asarray(s4["sma_strip_T_K_t"])[-1])
                for boundary, stage_group in ((b2, s2), (b4, s4)):
                    radial = boundary[102:153]
                    reconstruction = np.r_[radial[:0:-1], radial]+boundary[MEMORY]
                    err = np.abs(reconstruction-np.asarray(stage_group["sma_strip_T_K_t"])[0]).max()
                    checks["field_reconstruction_max_K"] = max(checks["field_reconstruction_max_K"], float(err))
                    if err > 1e-9:
                        raise ValueError("historical spatial state cannot reconstruct measured entry field")
                sequence = [previous, b2, b3, b4, b5]
                for stage in range(1, 5):
                    before, after = sequence[stage-1], sequence[stage]
                    if not np.isfinite(after).all() or not np.array_equal(before[~mask(stage)], after[~mask(stage)]):
                        raise ValueError("composite ownership/finite violation")
                    records.append(Step(f"{row.case_id}:c{cycle}:stage{stage}", row.case_id, cycle, stage, before.copy(), after.copy()))
                previous = b5
            source_checks.append(checks)
    return table, records, inputs, fps, source_checks


class HistoricalModel:
    def __init__(self, seed=20260910):
        self.seed, self.training_ids = seed, []
        self.linear, self.networks, self.scales = {}, {}, {}

    def fit(self, records, controls, epochs=300):
        raw = np.stack([features(r.before, controls[r.case_id], r.stage) for r in records])
        self.mean, self.scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), scale_floor())
        x = (raw-self.mean)/self.scale
        self.history = {}
        with torch.random.fork_rng(devices=[]), threadpool_limits(limits=1):
            for stage in range(1, 6):
                selected = np.array([r.stage == stage for r in records])
                active = mask(stage)
                delta = np.stack([r.after-r.before for r in records])[selected][:, active]
                linear = Ridge(alpha=1.).fit(x[selected], delta)
                residual = delta-linear.predict(x[selected])
                scale = np.sqrt(np.mean(residual**2, axis=0))
                scale = np.where(scale > 1e-10, scale, 0.)
                normalized = np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 0)
                torch.manual_seed(self.seed+1009*stage)
                network = nn.Sequential(nn.Linear(529, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, int(active.sum()))).double()
                tx, ty = torch.from_numpy(x[selected]), torch.from_numpy(normalized)
                optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=1e-4)
                losses = []
                for step in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    loss = ((network(tx)-ty)**2).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite historical fit")
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                    if step % 50 == 0 or step == epochs-1:
                        losses.append(float(loss.detach()))
                network.eval()
                self.linear[stage], self.networks[stage], self.scales[stage] = linear, network, scale
                self.history[str(stage)] = {"training_rows": int(selected.sum()), "normalized_loss": losses}
        self.training_ids, self.epochs = [r.sample_id for r in records], epochs
        return self

    def predict(self, state, control, stage, neural=True):
        if not self.training_ids:
            raise RuntimeError("model not fitted")
        x = (features(state, control, stage)-self.mean)/self.scale
        delta = self.linear[stage].predict(x[None])[0]
        if neural:
            with torch.no_grad():
                delta += self.networks[stage](torch.from_numpy(x[None])).numpy()[0]*self.scales[stage]
        result = state.copy()
        result[mask(stage)] += delta
        if not np.isfinite(result).all():
            raise RuntimeError("nonfinite rollout; no clipping or retry")
        return project(result)

    def save(self, path):
        payload = {"schema": "historical515_five_step_v1", "seed": self.seed, "epochs": self.epochs, "training_ids": self.training_ids,
            "mean": torch.tensor(self.mean), "scale": torch.tensor(self.scale), "history": self.history,
            "linear": {s: {"coef": torch.tensor(m.coef_), "intercept": torch.tensor(m.intercept_)} for s, m in self.linear.items()},
            "networks": {s: m.state_dict() for s, m in self.networks.items()}, "scales": {s: torch.tensor(v) for s, v in self.scales.items()}}
        with path.open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path):
        p = torch.load(path, weights_only=True)
        if p["schema"] != "historical515_five_step_v1":
            raise ValueError("wrong historical state schema")
        model = cls(p["seed"])
        model.mean, model.scale = p["mean"].numpy(), p["scale"].numpy()
        model.training_ids, model.epochs, model.history = p["training_ids"], p["epochs"], p["history"]
        with torch.random.fork_rng(devices=[]):
            for stage in range(1, 6):
                linear = Ridge(alpha=1.)
                linear.coef_, linear.intercept_, linear.n_features_in_ = p["linear"][stage]["coef"].numpy(), p["linear"][stage]["intercept"].numpy(), 529
                net = nn.Sequential(nn.Linear(529, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, int(mask(stage).sum()))).double()
                net.load_state_dict(p["networks"][stage]); net.eval()
                model.linear[stage], model.networks[stage], model.scales[stage] = linear, net, p["scales"][stage].numpy()
        return model


def rollout(model, initial, control, cycles, *, neural=True, skip_restart=False, erase_memory=False):
    """Only start state, known controls and requested horizon; no target records."""
    if type(cycles) is not int or cycles < 1:
        raise ValueError("invalid requested horizon")
    state = initial.copy()
    outputs = []
    for cycle in range(1, cycles+1):
        if cycle > 1:
            if not skip_restart:
                state = model.predict(state, control, 5, neural)
            if erase_memory:
                state[MEMORY] = 0.
            outputs.append(state.copy())
        for stage in range(1, 5):
            state = model.predict(state, control, stage, neural)
            if erase_memory:
                state[MEMORY] = 0.
            outputs.append(state.copy())
    return np.stack(outputs)


def scores(prediction, target, records):
    return {f"cycle_{cycle}": {str(stage): {name: float(np.sqrt(np.mean((prediction[selected, slc]-target[selected, slc])**2)))
        for name, slc in BLOCKS.items()} for stage in range(1, 6)
        for selected in [np.array([r.cycle == cycle and r.stage == stage for r in records])] if selected.any()}
        for cycle in sorted({r.cycle for r in records})}


def run(dataset, out):
    if out.exists():
        raise FileExistsError("preserve historical learning evidence")
    table, records, inputs, fps, checks = corpus(dataset)
    groups = split_cases(table)
    train = [r for r in records if r.case_id in groups["train"]]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"protocol.json", {"learning_only": True, "fresh_solver_calls": 0, "dataset": str(dataset), "fingerprints": fps,
        "groups": groups, "state_dimension": 515, "feature_dimension": 529, "epochs": 300, "seeds": [20260910, 20260911, 20260912],
        "initial_input": "observed very first mechanical frame + declared uniform thermal defaults; no future-stage temperatures",
        "stages": ["loading+hold setup", "hold+mechanical re-equilibration", "unloading+cooldown setup", "cooldown end", "next-cycle mechanical restart"],
        "restart_side": "not directly observed; carried until next hold setup observation, not certified complete physical restart",
        "memory": "101point antisymmetric strip temperature; mirror radial plus memory reconstructs observed exchange entries",
        "constraints": "xi clipped[0,1],u center/edge andw edge zero,memory antisymmetric; NO temperature clipping",
        "test_selection": "both3cyclecases+onehashed2cyclecasepermaterial; train8whole2cyclecases, test4wholecases",
        "ablation": "skip learned restart or erase predicted memory at interfaces; inference interventions, NOT retrained ablations",
        "scope": "historical separate-profile boundary rollout, not current63 validation, fullfield trajectory or physical energy closure",
        "implementation_sha256": sha(Path(__file__))})
    write_json(out/"source_checks.json", checks)
    results = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for seed in (20260910, 20260911, 20260912):
            dest = out/f"s{seed}"
            dest.mkdir()
            model = HistoricalModel(seed).fit(train, {case: value["control"] for case, value in inputs.items()})
            model.save(dest/"model.pt")
            loaded = HistoricalModel.load(dest/"model.pt")
            arrays, report = {}, {"seed": seed, "history": model.history, "groups": {}}
            for part, case_ids in groups.items():
                selected = [r for r in records if r.case_id in case_ids]
                truth = np.stack([r.after for r in selected])
                variants = {"autonomous_neural": {}, "autonomous_affine": {"neural": False},
                            "skip_restart": {"skip_restart": True}, "erase_memory": {"erase_memory": True}}
                predictions = {name: np.concatenate([rollout(model, **inputs[case], **kwargs) for case in case_ids]) for name, kwargs in variants.items()}
                predictions["teacher_neural"] = np.stack([model.predict(r.before, inputs[r.case_id]["control"], r.stage) for r in selected])
                replay = np.concatenate([rollout(loaded, **inputs[case]) for case in case_ids])
                np.testing.assert_array_equal(replay, predictions["autonomous_neural"])
                arrays[part+"_ids"], arrays[part+"_truth"] = np.array([r.sample_id for r in selected]), truth
                report["groups"][part] = {name: scores(value, truth, selected) for name, value in predictions.items()}
                for name, value in predictions.items():
                    arrays[part+"_"+name] = value
            np.savez_compressed(dest/"predictions.npz", **arrays)
            write_json(dest/"report.json", report)
            results.append(report)
    finally:
        torch.set_num_threads(threads)
    means = {name: {cycle: {stage: {block: float(np.mean([r["groups"]["test"][name][cycle][stage][block] for r in results]))
        for block in blocks} for stage, blocks in stages.items()} for cycle, stages in cycles.items()}
        for name, cycles in results[0]["groups"]["test"].items()}
    summary = {"status": "completed_historical_multicycle_boundary_learning", "learning_only": True, "fresh_solver_calls": 0,
        "training_cases": len(groups["train"]), "test_cases": len(groups["test"]), "training_transitions": len(train),
        "test_transitions": len(records)-len(train), "test_seed_mean_rmse": means,
        "all_checkpoint_replays_exact": True, "physics_closure_claim": False,
        "artifacts": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    write_json(out/"summary.json", summary)
    return {"status": summary["status"], "training_cases": summary["training_cases"], "test_cases": summary["test_cases"],
            "test_cycle_end_sma_rmse": {name: {cycle: stages["4"]["sma_K"] for cycle, stages in value.items()} for name, value in means.items()},
            "fresh_solver_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.output_root), indent=2))
