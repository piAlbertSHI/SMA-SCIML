"""Low-cost, physical-input SMA readout study; case-separated selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data, write
from .train_hierarchical import training_data
from .adapt_sma import SELECTED, VALIDATION, SEEDS, field_metrics
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.thermal_readout import ThermalReadout, ThermalReadoutOperator

MODES=('observed','autonomous','residual')


@threadpool_limits.wrap(limits=1)
def fit_readout(source,arrays,selected,mode):
    # Freeze before forming residual targets. Freezing only when wrapping the
    # operator can change attention kernel eligibility between first/repeated
    # cache builds, even under no_grad, producing last-bit target differences.
    source.eval().requires_grad_(False)
    inputs=[];controls=[];stages=[];targets=[]
    state=arrays['initial'][selected];control=arrays['control'][selected];memory=None;exchange=0
    times=torch.linspace(0,1,50,dtype=torch.float64);positions=torch.linspace(0,1,101,dtype=torch.float64)
    with torch.no_grad():
        for stage in [1,2,3,4,5,1,2,3,4]:
            decode=stage in (2,4)
            out=source.step(state,control,stage,memory,times=times if decode else None,positions=positions if decode else None)
            if decode:
                truth=arrays['fields'][selected,exchange]
                initial=truth[:,:,0] if mode=='observed' else source.initial_fields(state,stage)
                change=truth[:,0]-out['field'][:,0] if mode=='residual' else truth[:,0]-initial[:,0,None]
                inputs.append(initial);controls.append(control);stages.append(torch.full((len(selected),),stage))
                targets.append(change);exchange+=1
            state,memory=out['state'],out['memory']
    readout=ThermalReadout(rank=12,mode=mode)
    fit=readout.fit(torch.cat(inputs),torch.cat(controls),torch.cat(stages),torch.cat(targets),alpha=1.)
    return ThermalReadoutOperator(source,readout),fit


def run(output,formal=False,development=None,cross_check=None):
    output.mkdir(parents=True,exist_ok=False);a,ids,hashes=training_data()
    source_paths=([ROOT/f'evidence/experiments/hro_modal/formal/s{s}/refined/model.pt' for s in SEEDS] if formal
                  else [ROOT/'run-output-hro-modal-refinement/refined/model.pt'])
    protocol={'purpose':'physical-input low-rank SMA correction','formal':formal,'modes':list(MODES),'rank':12,'alpha':1.,
        'fit_case_ids':ids if formal else [ids[i] for i in SELECTED],
        'validation_case_ids':[] if formal else [ids[i] for i in VALIDATION],
        'data_sha256':hashes,'checkpoint_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
        'features':'initial SMA101, fluid101, radius, thickness, flush ratio, duration, stage2/stage4; no material/cycle ID',
        'gate':'same as preceding decoder study: lower SMA mean and c2s4 mean, per-case SMA <=1.05*baseline, fluid exact',
        'historical_test_scope':'failure diagnosis already performed; no historical test candidate selection',
        'source_sha256':{p:sha(ROOT/p) for p in ['examples/fit_thermal_readout.py','SciML/models/v13/thermal_readout.py',
            'examples/adapt_sma.py','SciML/models/v13/selective_sma.py']},'fresh_solver_calls':0}
    if formal:
        if cross_check is None:raise ValueError('A passed independent training-case fold check is required')
        check=json.loads((cross_check/'summary.json').read_text())
        if not check['passed']:raise ValueError('Second training-case fold failed')
        for rel,digest in check['artifacts'].items():
            if sha(cross_check/rel)!=digest:raise ValueError('Second-fold evidence changed')
        protocol['cross_check_sha256']=sha(cross_check/'summary.json')
        dev=json.loads((development/'summary.json').read_text())
        if dev['selected_mode'] is None:raise ValueError('Development gate failed')
        for rel,digest in dev['artifacts'].items():
            if sha(development/rel)!=digest:raise ValueError('Development evidence changed')
        protocol['selected_mode']=dev['selected_mode'];protocol['development_sha256']=sha(development/'summary.json')
    write(output/'protocol.json',protocol)
    for rel in protocol['source_sha256']:
        dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    if not formal:
        source=ModalStateOperator.load(source_paths[0]);baseline=field_metrics(source,a,VALIDATION);results={};eligible=[]
        assert {s.split(':')[0] for s in source.training_ids}=={ids[i] for i in SELECTED}
        for mode in MODES:
            started=time.perf_counter();m,fit=fit_readout(source,a,SELECTED,mode)
            dest=output/mode;dest.mkdir();m.save(dest/'model.pt')
            fit['seconds']=time.perf_counter()-started;write(dest/'fit.json',fit)
            results[mode]=field_metrics(m,a,VALIDATION)
            v=results[mode];b=baseline
            passed=v['mean']['sma_rmse_K']<b['mean']['sma_rmse_K'] and v['mean']['c2s4_sma_rmse_K']<b['mean']['c2s4_sma_rmse_K']
            passed=passed and all(x['sma_rmse_K']<=1.05*y['sma_rmse_K'] and x['fluid_rmse_K']==y['fluid_rmse_K'] for x,y in zip(v['per_case'],b['per_case']))
            if passed:eligible.append(mode)
        chosen=min(eligible,key=lambda v:results[v]['mean']['sma_rmse_K']) if eligible else None
        result={'status':'completed_physical_readout_development','baseline':baseline,'results':results,'eligible':eligible,
                'selected_mode':chosen,'historical_test_scored':False}
    else:
        for seed,path in zip(SEEDS,source_paths):
            started=time.perf_counter();m,fit=fit_readout(ModalStateOperator.load(path),a,list(range(8)),protocol['selected_mode'])
            dest=output/f's{seed}';dest.mkdir();m.save(dest/'model.pt');fit['seconds']=time.perf_counter()-started;write(dest/'fit.json',fit)
        d,_=data();rows=[]
        for seed in SEEDS:
            dest=output/f's{seed}';m=ThermalReadoutOperator.load(dest/'model.pt');p=infer(m,d)
            with np.load(ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/test_predictions.npz') as old:
                np.testing.assert_array_equal(p['state'],old['state']);np.testing.assert_array_equal(p['field'][:,1],old['field'][:,1])
            np.savez_compressed(dest/'test_predictions.npz',**p);row=cohorts(p,d);rows.append(row);write(dest/'evaluation.json',row)
        result={'status':'completed_fixed_physical_readout','selected_mode':protocol['selected_mode'],
                'adapted':aggregate(rows,SEEDS),'state_and_fluid_exactly_preserved':True}
    result['artifacts']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'summary.json',result)
    if formal:return {'status':result['status'],'mode':result['selected_mode'],'c3s4_field':result['adapted']['fixed_three_cycle']['field']['c3s4']}
    return {'status':result['status'],'selected_mode':result['selected_mode'],'baseline':baseline['mean'],
            'validation':{k:v['mean'] for k,v in results.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--formal',action='store_true');p.add_argument('--development',type=Path)
    p.add_argument('--cross-check',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output,a.formal,a.development,a.cross_check)
    print(json.dumps(result,indent=2))
