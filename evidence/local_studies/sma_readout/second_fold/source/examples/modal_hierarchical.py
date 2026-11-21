"""Case-separated development and fixed-budget evaluation of modal HRO.

Development uses six original training cases and two inner validation cases.
All normalizers and affine priors are refitted on the six fit cases. Historical
test scores are computed only by the separately invoked formal stage.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data, write
from .train_hierarchical import training_data
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v12.learning_historical_rollout import HistoricalModel, features, scale_floor, mask
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator, joint_loss
from SciML.models.v13.modal_operator import ModalStateOperator

SEEDS = (20261121, 20261122, 20261123)
SOURCES = ['examples/modal_hierarchical.py', 'SciML/models/v13/modal_operator.py',
           'SciML/models/v13/hierarchical_operator.py', 'examples/train_hierarchical.py',
           'examples/formal_hierarchical.py', 'examples/multicycle_multistep.py']
KEYS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4)]


def fit_prior(arrays, ids, selected):
    """No stored all-case normalizer or prior enters the inner validation fold."""
    raw, deltas, stages, sample_ids = [], [], [], []
    for i in selected:
        before = torch.cat((arrays['initial'][i:i+1], arrays['states'][i, :-1])).numpy()
        after = arrays['states'][i].numpy()
        for j, (cycle, stage) in enumerate(KEYS):
            raw.append(features(before[j], arrays['control'][i].numpy(), stage))
            deltas.append(after[j]-before[j]); stages.append(stage)
            sample_ids.append(f'{ids[i]}:c{cycle}:{"restart" if stage==5 else "stage"+str(stage)}')
    raw, deltas, stages = np.stack(raw), np.stack(deltas), np.array(stages)
    base = HistoricalModel()
    base.mean = raw.mean(0); base.scale = np.maximum(raw.std(0), scale_floor())
    base.training_ids = sample_ids
    x = (raw-base.mean)/base.scale
    for stage in range(1, 6):
        use = stages == stage
        base.linear[stage] = Ridge(alpha=1.).fit(x[use], deltas[use][:, mask(stage)])
    return base


def diagnostics(model, arrays, selected):
    rows = []
    positions = torch.linspace(0, 1, 101, dtype=torch.float64)
    seen = np.linspace(0, 100, 26, dtype=int)
    unseen = np.setdiff1d(np.arange(101), seen)
    model.eval()
    with torch.no_grad():
        for i in selected:
            pred = model(arrays['initial'][i:i+1], arrays['control'][i:i+1], 2,
                         times=torch.linspace(0, 1, 50, dtype=torch.float64), positions=positions)
            truth = arrays['fields'][i:i+1]
            losses = joint_loss(pred, arrays['states'][i:i+1], model.feature_scale, field_truth=truth,
                                field_scale=model.decoder.field_scale, positions=positions)
            error = (pred['fields']-truth)[:, :, :, 1:]
            rows.append({'case_index': int(i), **{k: float(v) for k, v in losses.items()},
                         'seen_rmse_K': error[..., seen].square().mean((0, 1, 3, 4)).sqrt().tolist(),
                         'unseen_rmse_K': error[..., unseen].square().mean((0, 1, 3, 4)).sqrt().tolist(),
                         'field_rmse_K': error.square().mean((0, 1, 3, 4)).sqrt().tolist(),
                         'gradient_rmse_K_per_x': (torch.diff(error, dim=-1)/.01).square().mean((0, 1, 3, 4)).sqrt().tolist()})
    return {'per_case': rows, 'mean': {k: float(np.mean([r[k] for r in rows]))
                                     for k in ('state', 'field', 'spatial_gradient', 'total')}}


def fit(output, arrays, ids, selected, base, kind, seed, updates):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    cls = ModalStateOperator if kind == 'modal_dense' else HierarchicalStateOperator
    model = cls(base)
    delta = arrays['fields'][selected]-arrays['fields'][selected, :, :, 0:1]
    model.decoder.field_scale.copy_(delta.square().mean((0, 1, 3, 4)).sqrt().clamp_min(.1))
    frozen = {k: v.clone() for k, v in model.named_buffers()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
    def schedule(step):
        if step < 20: return (step+1)/20
        return .1+.9*.5*(1+math.cos(math.pi*(step-20)/max(1, updates-20)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(seed); history = []; started = time.perf_counter()
    model.train()
    for step in range(updates):
        chosen = np.sort(rng.choice(selected, 2, replace=False))
        ti = np.r_[0, np.sort(rng.choice(np.arange(1, 49), 6, replace=False)), 49]
        xi = np.linspace(0, 100, 26, dtype=int) if kind == 'coordinate_sparse' else np.arange(101)
        positions = torch.tensor(xi/100.)
        optimizer.zero_grad(set_to_none=True)
        pred = model(arrays['initial'][chosen], arrays['control'][chosen], 2,
                     times=torch.tensor(ti/49.), positions=positions)
        truth = arrays['fields'][chosen][:, :, :, ti][:, :, :, :, xi]
        losses = joint_loss(pred, arrays['states'][chosen], model.feature_scale, field_truth=truth,
                            field_scale=model.decoder.field_scale, positions=positions)
        losses['total'].backward()
        if step == 0:
            coverage = {name: any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0
                                 for p in module.parameters()) for name, module in model.named_children()}
            if not all(coverage.values()): raise RuntimeError(f'Missing gradients: {coverage}')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        history.append({'update': step+1, 'case_ids': [ids[i] for i in chosen], 'time_indices': ti.tolist(),
                        'space_points': len(xi), 'lr': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm),
                        **{k: float(v.detach()) for k, v in losses.items()}})
        optimizer.step(); scheduler.step()
    elapsed = time.perf_counter()-started
    for name, value in model.named_buffers():
        if not torch.equal(value, frozen[name]): raise RuntimeError('Frozen buffer changed: '+name)
    model.eval()
    model.save(output/'model.pt', metadata={'kind': kind, 'seed': seed, 'updates': updates})
    with (output/'training_state.pt').open('xb') as stream:
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'torch_rng': torch.get_rng_state(), 'numpy_rng_json': json.dumps(rng.bit_generator.state),
                    'completed_updates': updates}, stream)
    write(output/'history.json', history)
    write(output/'fit.json', {'seconds': elapsed, 'seed': seed, 'updates': updates,
                             'parameter_report': model.parameter_report(), 'fit_case_ids': [ids[i] for i in selected],
                             'gradient_coverage': {k: bool(v) for k, v in coverage.items()}})
    return model


def protocol(output, mode, ids, hashes, extra):
    output.mkdir(parents=True, exist_ok=False)
    write(output/'protocol.json', {'mode': mode, 'written_before_training': True,
        'data_sha256': hashes, 'training_pool': ids, 'source_sha256': {p: sha(ROOT/p) for p in SOURCES},
        'fresh_solver_calls': 0, 'optimizer': 'AdamW', 'lr': .0003, 'weight_decay': 1e-4,
        'clip_norm': 1., 'warmup': 20, 'cosine_final_fraction': .1, 'batch_cases': 2,
        'loss': 'state + field + 0.01 spatial gradient, training-only scales',
        'temporal_queries': 'zero, endpoint and six random interior times',
        'checkpoint_selection': 'fixed final update', 'teacher_forcing': False,
        'precision': 'CPU float64; one BLAS and torch thread', **extra})


def probe(output):
    arrays, ids, hashes = training_data()
    # Fixed sorted case per material, declared without looking at scores.
    validation = [3, 7]; selected = [i for i in range(8) if i not in validation]
    kinds = ['coordinate_sparse', 'coordinate_dense', 'modal_dense']
    protocol(output, 'inner_case_development', ids, hashes, {'fit_indices': selected, 'validation_indices': validation,
        'fit_case_ids': [ids[i] for i in selected], 'validation_case_ids': [ids[i] for i in validation],
        'variants': kinds, 'updates': 160, 'seed': 20261124, 'historical_test_evaluated': False,
        'acceptance': 'modal validation total < dense coordinate total AND state <= 1.25*dense coordinate state',
        'budget': 'three fits, 480 total updates; no parameter sweep', 'priors_and_normalizers': 'refit on six fit cases only'})
    base = fit_prior(arrays, ids, selected); rows = {}
    for kind in kinds:
        model = fit(output/kind, arrays, ids, selected, base, kind, 20261124, 160)
        rows[kind] = {'train': diagnostics(model, arrays, selected), 'validation': diagnostics(model, arrays, validation)}
        write(output/kind/'diagnostics.json', rows[kind])
    a = rows['modal_dense']['validation']['mean']; b = rows['coordinate_dense']['validation']['mean']
    result = {'status': 'completed_inner_case_comparison', 'historical_test_evaluated': False,
              'eligible_for_fixed_formal_fit': a['total'] < b['total'] and a['state'] <= 1.25*b['state'],
              'results': rows, 'fresh_solver_calls': 0,
              'artifacts': {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json', result)
    return {k: v for k, v in result.items() if k not in ('results', 'artifacts')} | {
        'validation': {k: v['validation']['mean'] for k, v in rows.items()}}


def formal(output, development):
    dev = json.loads((development/'summary.json').read_text())
    if not dev['eligible_for_fixed_formal_fit']: raise ValueError('Development acceptance failed; diagnose before any new run')
    for rel, digest in dev['artifacts'].items():
        if sha(development/rel) != digest: raise ValueError('Changed development evidence: '+rel)
    arrays, ids, hashes = training_data()
    protocol(output, 'fixed_modal_comparison', ids, hashes, {'seeds': list(SEEDS), 'updates': 240,
        'development_summary_sha256': sha(development/'summary.json'), 'historical_test_already_examined': True,
        'comparison_scope': 'exploratory historical comparison, not an untouched confirmatory test',
        'reference': 'HRO coordinate, 200 updates; additional updates and spatial labels confound architecture attribution',
        'primary_metrics': ['c3s4 SMA state', 'c3s4 SMA future field', 'c3s4 HTF future field'],
        'demo_selection': 'first sorted three-cycle case, first listed seed; no best seed selection',
        'budget': 'three fits, 720 total updates; no parameter sweep'})
    base = fit_prior(arrays, ids, list(range(8)))
    # Finish every fit before the first historical test inference.
    for seed in SEEDS:
        fit(output/f's{seed}', arrays, ids, list(range(8)), base, 'modal_dense', seed, 240)
    d, _ = data(); reports = []
    for seed in SEEDS:
        dest = output/f's{seed}'; model = ModalStateOperator.load(dest/'model.pt')
        prediction = infer(model, d)
        np.savez_compressed(dest/'test_predictions.npz', **prediction)
        report = {'seed': seed, **cohorts(prediction, d), 'training_diagnostics': diagnostics(model, arrays, list(range(8)))}
        write(dest/'evaluation.json', report); reports.append(report)
    result = {'status': 'completed_fixed_modal_comparison', 'fresh_solver_calls': 0, 'modal': aggregate(reports, SEEDS),
              'artifacts': {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json', result)
    return {'status': result['status'], 'third_cycle': {k: result['modal']['fixed_three_cycle'][k]['c3s4'] for k in ('state', 'field')}}


def verify(output):
    result = json.loads((output/'summary.json').read_text()); d, _ = data()
    for rel, digest in result['artifacts'].items():
        if sha(output/rel) != digest: raise ValueError('Changed result: '+rel)
    rows = []; maximum = 0.
    for seed in SEEDS:
        dest = output/f's{seed}'
        pred = infer(ModalStateOperator.load(dest/'model.pt'), d)
        with np.load(dest/'test_predictions.npz', allow_pickle=False) as z:
            for k in pred:
                maximum = max(maximum, float(np.max(np.abs(pred[k]-z[k]))))
                np.testing.assert_allclose(pred[k], z[k], rtol=0, atol=1e-8)
        rows.append(cohorts(pred, d))
    recomputed = aggregate(rows, SEEDS)
    # Tolerate only cross-platform arithmetic roundoff, never changed sample membership.
    def check(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: check(a[k], b[k])
        else: np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)
    check(recomputed, result['modal'])
    return {'status': 'verified_modal_comparison', 'checkpoints': 3, 'test_cases_per_checkpoint': 4,
            'maximum_replay_difference': maximum, 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['probe', 'formal', 'verify'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--development', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = probe(args.output) if args.mode == 'probe' else verify(args.output) if args.mode == 'verify' else formal(args.output, args.development)
    print(json.dumps(result, indent=2, allow_nan=False))
