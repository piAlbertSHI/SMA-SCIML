"""Derive spatial and whole-test metrics from preserved prediction arrays."""
import json
from pathlib import Path

import numpy as np

from examples.multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha


def run():
    folder = ROOT/'evidence/experiments/hro_modal'
    with np.load(ROOT/'data/historical12_state.npz', allow_pickle=False) as d:
        truth=d['field_truth']; selected=(d['field_cycle']==3)&(d['field_stage']==4)
    results={}; sources={}; costs={}; per_case={}
    for method, root in [('coordinate',ROOT/'evidence/experiments/v13_formal_v1'),('modal',folder/'formal')]:
        rows=[]; case_rows={}
        for seed in [20261121,20261122,20261123]:
            path=root/f's{seed}/test_predictions.npz'; sources[path.relative_to(ROOT).as_posix()]=sha(path)
            with np.load(path, allow_pickle=False) as z: error=z['field'][:,:,1:]-truth[:,:,1:]
            rows.append({'seed':seed,'all_test_field_rmse_K':np.sqrt((error**2).mean((0,2,3))).tolist(),
                'c3s4_spatial_gradient_rmse_K_per_normalized_x':np.sqrt(((np.diff(error[selected],axis=-1)/.01)**2).mean((0,2,3))).tolist()})
            report=json.loads((root/f's{seed}/evaluation.json').read_text())
            for case, value in report['per_three_cycle_case'].items():
                case_rows.setdefault(case,[]).append([value['state']['c3s4']['sma_K'],value['field']['c3s4']['sma_K'],value['field']['c3s4']['htf_K']])
        results[method]={'per_seed':rows, 'mean':{k:np.mean([r[k] for r in rows],axis=0).tolist() for k in rows[0] if k!='seed'}}
        per_case[method]={c:{'sma_state_sma_field_fluid_field_mean_K':np.mean(v,axis=0).tolist()} for c,v in case_rows.items()}
    for seed in [20261121,20261122,20261123]:
        costs[str(seed)]={phase:json.loads((folder/f'formal/s{seed}/{phase}/fit.json').read_text())['seconds'] for phase in ['joint','refined']}
    result={'status':'derived_prediction_diagnostics','field_order':['SMA','fluid'],'results':results,'per_case':per_case,
            'formal_training_seconds':costs,'formal_total_training_seconds':sum(v for row in costs.values() for v in row.values()),
            'input_sha256':sources,'fresh_solver_calls':0}
    write(folder/'diagnostics.json',result)
    return result


if __name__=='__main__': print(json.dumps(run(),indent=2))
