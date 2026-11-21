"""Diagnostic interventions, explicitly separated from autonomous evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data, write
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v12.multistep_historical import sha

SEEDS = (20261121, 20261122, 20261123)


def decompose(prediction, truth):
    """Exact MSE decomposition: entry + evolution + their signed cross term."""
    entry = prediction[:, :1]-truth[:, :1]
    error = prediction[:, 1:]-truth[:, 1:]
    evolution = error-entry
    reduce = (1, 2)
    a = np.mean(entry**2, axis=reduce); b = np.mean(evolution**2, axis=reduce)
    cross = 2*np.mean(entry*evolution, axis=reduce)
    mse = np.mean(error**2, axis=reduce)
    np.testing.assert_allclose(mse, a+b+cross, rtol=1e-12, atol=1e-12)
    return {'rmse_K': np.sqrt(mse).tolist(), 'entry_rmse_K': np.sqrt(a).tolist(),
            'evolution_rmse_K': np.sqrt(b).tolist(), 'entry_mse_K2': a.tolist(),
            'evolution_mse_K2': b.tolist(), 'cross_mse_K2': cross.tolist(),
            'bias_K': error.mean(axis=reduce).tolist(),
            'edge_bias_K': error[:, :, np.r_[0:25, 76:101]].mean(axis=reduce).tolist(),
            'center_bias_K': error[:, :, 25:76].mean(axis=reduce).tolist(),
            'early_middle_late_rmse_K': [np.sqrt(np.mean(part**2, axis=reduce)).tolist()
                                        for part in (error[:, :16], error[:, 16:33], error[:, 33:])]}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    d, provenance = data(); sources = {}; rows = []; field_records = []
    times = torch.linspace(0, 1, 50, dtype=torch.float64)
    positions = torch.linspace(0, 1, 101, dtype=torch.float64)
    fixed = np.flatnonzero((d['split']=='test') & (d['cycles']==3))
    write(output/'protocol.json', {'purpose': 'failure diagnosis, not model selection or independent confirmation',
        'models': ['coordinate', 'modal_joint', 'modal_refined'], 'seeds': list(SEEDS),
        'case_ids': d['case_ids'][fixed].tolist(), 'field_order': ['SMA','fluid'],
        'oracle_scope': 'true entry state substituted at an exchange; predicted preceding memory retained; no oracle result is a forecast',
        'decomposition': 'exact entry MSE + evolution MSE + signed cross term; initial-value substitution is diagnostic only',
        'data_sha256': provenance['snapshot_sha256'], 'source_sha256': sha(Path(__file__)), 'fresh_solver_calls': 0})
    for seed in SEEDS:
        paths = {
            'coordinate': ROOT/f'evidence/experiments/v13_formal_v1/s{seed}/model.pt',
            'modal_joint': ROOT/f'run-output-hro-modal-formal/s{seed}/joint/model.pt',
            'modal_refined': ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/refined/model.pt'}
        for name, path in paths.items():
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
            cls = HierarchicalStateOperator if name=='coordinate' else ModalStateOperator
            model = cls.load(path); physical = []
            hook = model.encoder.register_forward_hook(lambda module, inputs, value: physical.append(value.detach()))
            with torch.no_grad():
                for ci in fixed:
                    state = torch.tensor(d['initial'][ci:ci+1]); control = torch.tensor(d['control'][ci:ci+1]); memory = None
                    for cycle in range(1, 4):
                        for stage in ([5] if cycle>1 else [])+[1,2,3,4]:
                            decode = stage in (2,4); physical.clear()
                            out = model.step(state, control, stage, memory,
                                times=times if decode else None, positions=positions if decode else None)
                            context = physical[-1].clone()
                            if decode:
                                index = np.flatnonzero((d['field_case_index']==ci)&(d['field_cycle']==cycle)&(d['field_stage']==stage))[0]
                                truth = d['field_truth'][index]; prediction = out['field'][0].numpy()
                                record = {'model': name, 'seed': seed, 'case_id': str(d['case_ids'][ci]),
                                    'cycle': cycle, 'stage': stage, **decompose(prediction, truth),
                                    'physical_latent_rms': float(context.square().mean().sqrt()),
                                    'conditioned_latent_rms': float(out['latent'].square().mean().sqrt()),
                                    'memory_rms': float(out['memory'].square().mean().sqrt())}
                                before = d['before'][(d['case_index']==ci)&(d['cycle']==cycle)&(d['stage']==stage)]
                                oracle = model.step(torch.tensor(before), control, stage, memory, times=times, positions=positions)
                                record['true_entry_state_predicted_memory_rmse_K'] = np.sqrt(
                                    np.mean((oracle['field'][0].numpy()[:,1:]-truth[:,1:])**2, axis=(1,2))).tolist()
                                if name.startswith('modal'):
                                    # Input intervention only: existing decoder, no retraining or accuracy claim.
                                    no_memory_field = model.decoder(context, model.initial_fields(state,stage), times, positions)
                                    record['unconditioned_latent_intervention_rmse_K'] = np.sqrt(np.mean(
                                        (no_memory_field[0].numpy()[:,1:]-truth[:,1:])**2, axis=(1,2))).tolist()
                                rows.append(record); field_records.append(prediction)
                            state, memory = out['state'], out['memory']
            hook.remove()
    train = d['control'][d['split']=='train']
    summary = {'status':'completed_error_decomposition','rows':rows,'checkpoint_sha256':sources,
        'training_control_min':train.min(0).tolist(),'training_control_max':train.max(0).tolist(),
        'test_three_cycle_controls':{str(d['case_ids'][i]):d['control'][i].tolist() for i in fixed},
        'fresh_solver_calls':0}
    np.savez_compressed(output/'autonomous_fields.npz', field=np.stack(field_records))
    write(output/'diagnostics.json',summary)
    return {'status':summary['status'],'records':len(rows),'third_cycle_sma':[
        {k:r[k] for k in ('model','seed','case_id','rmse_K','entry_rmse_K','evolution_rmse_K','cross_mse_K2',
                          'true_entry_state_predicted_memory_rmse_K')}
        for r in rows if r['cycle']==3 and r['stage']==4]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result=run(a.output)
    print(json.dumps(result,indent=2))
