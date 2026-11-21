"""Decoder refinement using cached autonomous train trajectories.

State, attention encoders and recurrent dynamics remain frozen in this phase.
The cache is produced without teacher forcing; no validation target is cached.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data, write
from .train_hierarchical import training_data
from .modal_hierarchical import diagnostics, fit, fit_prior, protocol, SEEDS, SOURCES
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v12.multistep_historical import sha


def cache(model, arrays, selected):
    latent, initial, truth = [], [], []
    state = arrays['initial'][selected]; control = arrays['control'][selected]; memory = None; exchange = 0
    with torch.no_grad():
        for stage in [1, 2, 3, 4, 5, 1, 2, 3, 4]:
            out = model.step(state, control, stage, memory)
            if stage in (2, 4):
                latent.append(out['latent']); initial.append(model.initial_fields(state, stage))
                truth.append(arrays['fields'][selected, exchange]); exchange += 1
            state, memory = out['state'], out['memory']
    return torch.cat(latent), torch.cat(initial), torch.cat(truth)


def refine(model, arrays, selected, output, seed, updates=320):
    output.mkdir(parents=True, exist_ok=False)
    frozen = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith('decoder.')}
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith('decoder.'))
    latent, initial, truth = cache(model, arrays, selected)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=.0008, weight_decay=1e-4)
    rng = np.random.default_rng(seed+501); history = []; started = time.perf_counter()
    positions = torch.linspace(0, 1, 101, dtype=torch.float64)
    scale = model.decoder.field_scale[None, :, None, None]
    for step in range(updates):
        chosen = rng.choice(len(latent), min(8, len(latent)), replace=False)
        ti = np.r_[0, np.sort(rng.choice(np.arange(1, 49), 14, replace=False)), 49]
        factor = min(1., (step+1)/20)*(.1+.9*.5*(1+math.cos(math.pi*step/updates)))
        optimizer.param_groups[0]['lr'] = .0008*factor
        optimizer.zero_grad(set_to_none=True)
        prediction = model.decoder(latent[chosen], initial[chosen], torch.tensor(ti/49.), positions)
        e = (prediction-truth[chosen][:, :, ti])/scale
        field = e[:, :, 1:].square().mean()
        gradient = (torch.diff(e[:, :, 1:], dim=-1)/.01).square().mean()
        loss = field+.01*gradient
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite decoder objective')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1., error_if_nonfinite=True)
        history.append({'update': step+1, 'cached_training_rows': chosen.tolist(), 'time_indices': ti.tolist(),
                        'field': float(field.detach()), 'gradient': float(gradient.detach()),
                        'total': float(loss.detach()), 'lr': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm)})
        optimizer.step()
    for name, value in frozen.items():
        if not torch.equal(value, model.state_dict()[name]): raise RuntimeError('State backbone changed: '+name)
    elapsed = time.perf_counter()-started
    model.save(output/'model.pt', metadata={'phase': 'cached_autonomous_decoder_refinement', 'seed': seed, 'updates': updates})
    with (output/'training_state.pt').open('xb') as stream:
        torch.save({'optimizer': optimizer.state_dict(), 'torch_rng': torch.get_rng_state(),
                    'numpy_rng_json': json.dumps(rng.bit_generator.state), 'updates': updates}, stream)
    write(output/'history.json', history)
    write(output/'fit.json', {'seconds': elapsed, 'frozen_backbone_unchanged': True,
                             'cached_rows': len(latent), 'training_case_indices': list(selected),
                             'cache_order': 'exchange-major, selected-case-major', 'updates': updates})
    return model


def run(output, development, formal=False):
    arrays, ids, hashes = training_data()
    source_summary = json.loads((development/'summary.json').read_text())
    for rel, value in source_summary['artifacts'].items():
        if sha(development/rel) != value: raise ValueError('Development evidence changed')
    selected = list(range(8)) if formal else [0, 1, 2, 4, 5, 6]
    if formal and not source_summary['eligible_for_fixed_formal_fit']:
        raise ValueError('Refinement did not pass the unchanged development gate')
    protocol(output, 'modal_two_phase_formal' if formal else 'modal_decoder_diagnosis_and_refinement', ids, hashes,
        {'source_development_summary_sha256': sha(development/'summary.json'),
         'joint_updates': 200 if formal else 'existing 160-update inner-case model', 'decoder_updates': 320,
         'decoder_lr': .0008, 'decoder_batch_exchange_segments': 8, 'decoder_temporal_queries': 16,
         'selection': 'fixed budgets, no early stopping', 'historical_test_evaluated': formal,
         'historical_test_scope': 'previously examined, exploratory only' if formal else 'not accessed for scores',
         'seeds': list(SEEDS) if formal else [20261124], 'fit_case_ids': [ids[i] for i in selected],
         'acceptance': 'unchanged: modal validation total < dense coordinate AND state <= 1.25*dense coordinate',
         'diagnosis': {'modal_basis_training_oracle_rmse_K': [.04252960, .00440854],
                       'interpretation': 'small spatial approximation floor, high training error: test decoder optimization before increasing basis size'},
         'decoder_source_sha256': sha(Path(__file__))})
    # Self-contained source snapshot for subsequent archival review.
    for rel in SOURCES+['examples/refine_modal.py']:
        p = output/'source'/rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((ROOT/rel).read_bytes())
    if not formal:
        model = ModalStateOperator.load(development/'modal_dense/model.pt')
        refine(model, arrays, selected, output/'refined', 20261124)
        results = {'train': diagnostics(model, arrays, selected), 'validation': diagnostics(model, arrays, [3, 7])}
        a = results['validation']['mean']; b = source_summary['results']['coordinate_dense']['validation']['mean']
        summary = {'status': 'completed_decoder_refinement_development', 'historical_test_evaluated': False,
                   'eligible_for_fixed_formal_fit': a['total'] < b['total'] and a['state'] <= 1.25*b['state'],
                   'results': results, 'dense_reference': b}
    else:
        base = fit_prior(arrays, ids, selected)
        for seed in SEEDS:
            model = fit(output/f's{seed}'/'joint', arrays, ids, selected, base, 'modal_dense', seed, 200)
            refine(model, arrays, selected, output/f's{seed}'/'refined', seed)
        d, _ = data(); rows = []
        for seed in SEEDS:
            dest = output/f's{seed}'; model = ModalStateOperator.load(dest/'refined/model.pt')
            prediction = infer(model, d)
            np.savez_compressed(dest/'test_predictions.npz', **prediction)
            report = {'seed': seed, **cohorts(prediction, d), 'training_diagnostics': diagnostics(model, arrays, selected)}
            write(dest/'evaluation.json', report); rows.append(report)
        summary = {'status': 'completed_two_phase_modal_comparison', 'modal': aggregate(rows, SEEDS)}
    summary['fresh_solver_calls'] = 0
    summary['artifacts'] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'summary.json', summary)
    if formal: return {'status': summary['status'], 'third_cycle': {k: summary['modal']['fixed_three_cycle'][k]['c3s4'] for k in ('state','field')}}
    return {k: v for k, v in summary.items() if k not in ('artifacts', 'results')} | {'validation': summary['results']['validation']['mean']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--development', type=Path, required=True)
    p.add_argument('--formal', action='store_true'); a = p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result = run(a.output, a.development, a.formal)
    print(json.dumps(result, indent=2, allow_nan=False))
