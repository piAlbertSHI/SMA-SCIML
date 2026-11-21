"""Budgeted SMA-only adaptation; selection uses inner training cases only."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write, data
from .train_hierarchical import training_data
from .refine_modal import cache
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.selective_sma import SelectiveSmaOperator

SEEDS = (20261121,20261122,20261123)
VARIANTS = ('absolute','increment','paired','paired_increment')
SELECTED = [0,1,2,4,5,6]
VALIDATION = [3,7]


def observed_cache(model, arrays, selected):
    """Training-only observed entries with preceding autonomous recurrent memory."""
    latent=[]; initial=[]; targets=[]
    state=arrays['initial'][selected]; control=arrays['control'][selected]; memory=None; exchange=0
    with torch.no_grad():
        for j, stage in enumerate([1,2,3,4,5,1,2,3,4]):
            out=model.step(state,control,stage,memory)
            if stage in (2,4):
                entry=arrays['states'][selected,j-1]
                observed=model.step(entry,control,stage,memory)
                latent.append(observed['latent']); initial.append(model.initial_fields(entry,stage))
                targets.append(arrays['fields'][selected,exchange]); exchange+=1
            state,memory=out['state'],out['memory']
    result=torch.cat(latent),torch.cat(initial),torch.cat(targets)
    torch.testing.assert_close(result[1],result[2][:,:,0],rtol=0,atol=1e-10)
    return result


def field_metrics(model, arrays, selected):
    rows=[]
    with torch.no_grad():
        for i in selected:
            out=model(arrays['initial'][i:i+1],arrays['control'][i:i+1],2,
                times=torch.linspace(0,1,50,dtype=torch.float64),positions=torch.linspace(0,1,101,dtype=torch.float64))
            error=out['fields'][0,:,:,1:]-arrays['fields'][i,:,:,1:]
            rows.append({'case_index':i,'sma_rmse_K':float(error[:,0].square().mean().sqrt()),
                'fluid_rmse_K':float(error[:,1].square().mean().sqrt()),
                'c2s4_sma_rmse_K':float(error[-1,0].square().mean().sqrt()),
                'sma_gradient_rmse_K_per_x':float((torch.diff(error[:,0],dim=-1)/.01).square().mean().sqrt())})
    return {'per_case':rows,'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='case_index'}}


def train(source, arrays, ids, selected, variant, output, seed, updates=240):
    if variant not in VARIANTS: raise ValueError('Unknown training objective')
    output.mkdir(parents=True,exist_ok=False)
    model=SelectiveSmaOperator.from_model(source)
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('decoder.sma.')}
    latent,initial,truth=cache(source,arrays,selected)
    if variant.startswith('paired'):
        other=observed_cache(source,arrays,selected)
        latent,initial,truth=[torch.cat((a,b)) for a,b in zip((latent,initial,truth),other)]
    optimizer=torch.optim.AdamW(model.decoder.sma.parameters(),lr=.0003,weight_decay=1e-4)
    rng=np.random.default_rng(seed+711); history=[]; started=time.perf_counter()
    positions=torch.linspace(0,1,101,dtype=torch.float64)
    scale=model.decoder.field_scale[0]
    for step in range(updates):
        chosen=rng.choice(len(latent),8,replace=False)
        ti=np.r_[0,np.sort(rng.choice(np.arange(1,49),14,replace=False)),49]
        factor=min(1.,(step+1)/20)*(.1+.9*.5*(1+math.cos(math.pi*step/updates)))
        optimizer.param_groups[0]['lr']=.0003*factor
        optimizer.zero_grad(set_to_none=True)
        p=model.decoder.sma(latent[chosen],initial[chosen],torch.tensor(ti/49.),positions)[:,0]
        target=truth[chosen,0][:,ti]
        error=(p[:,1:]-target[:,1:])/scale
        # This increment term discourages learning field evolution as compensation
        # for state-forecast entry errors. It uses only existing fit-case labels.
        increment=((p[:,1:]-p[:,:1])-(target[:,1:]-target[:,:1]))/scale
        field=error.square().mean(); gradient=(torch.diff(error,dim=-1)/.01).square().mean()
        inc=increment.square().mean()
        loss=field+.01*gradient+(inc if 'increment' in variant else 0.)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite SMA adaptation loss')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.decoder.sma.parameters(),1.,error_if_nonfinite=True)
        history.append({'update':step+1,'rows':chosen.tolist(),'time_indices':ti.tolist(),
            'loss':float(loss.detach()),'field':float(field.detach()),'increment':float(inc.detach()),
            'gradient':float(gradient.detach()),'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr']})
        optimizer.step()
    for k,v in frozen.items():
        if not torch.equal(model.state_dict()[k],v): raise RuntimeError('Frozen state/fluid tensor changed: '+k)
    model.save(output/'model.pt',metadata={'variant':variant,'seed':seed,'updates':updates})
    with (output/'training_state.pt').open('xb') as stream:
        torch.save({'optimizer':optimizer.state_dict(),'rng_json':json.dumps(rng.bit_generator.state),'updates':updates},stream)
    write(output/'history.json',history)
    write(output/'fit.json',{'seconds':time.perf_counter()-started,'variant':variant,'seed':seed,'updates':updates,
        'fit_case_ids':[ids[i] for i in selected],'cached_rows':len(latent),'parameters':model.parameter_report(),
        'frozen_state_and_fluid_tensors_unchanged':True})
    return model


def start(output, formal=False, development=None):
    output.mkdir(parents=True,exist_ok=False)
    arrays,ids,hashes=training_data()
    settings={'purpose':'SMA-only adaptation','formal':formal,'data_sha256':hashes,'fresh_solver_calls':0,
        'updates':240,'lr':.0003,'batch_segments':8,'time_queries':16,'space_points':101,
        'fit_case_ids':ids if formal else [ids[i] for i in SELECTED],
        'validation_case_ids':[] if formal else [ids[i] for i in VALIDATION],
        'source_sha256':{p:sha(ROOT/p) for p in ['examples/adapt_sma.py','SciML/models/v13/selective_sma.py',
            'examples/refine_modal.py','SciML/models/v13/modal_operator.py','SciML/models/v13/hierarchical_operator.py']},
        'gate':'SMA mean and cycle-2/stage-4 mean both improve baseline; each validation case SMA <= 1.05*baseline; fluid exact; choose lowest SMA mean',
        'selection_scope':'six-fit/two-validation original training partition only; historical test used for failure diagnosis, not candidate scores',
        'reference':'absolute-only continuation has the identical update budget; report it even if another objective is selected',
        'no_temperature_or_error_clipping':True}
    source_paths=([ROOT/f'evidence/experiments/hro_modal/formal/s{s}/refined/model.pt' for s in SEEDS]
                  if formal else [ROOT/'run-output-hro-modal-refinement/refined/model.pt'])
    settings['checkpoint_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    if formal:
        dev=json.loads((development/'summary.json').read_text())
        if dev['selected_variant'] is None: raise ValueError('No development candidate passed')
        for rel,digest in dev['artifacts'].items():
            if sha(development/rel)!=digest: raise ValueError('Changed development artifact')
        settings['selected_variant']=dev['selected_variant'];settings['development_sha256']=sha(development/'summary.json')
    else: settings['variants']=list(VARIANTS)
    write(output/'protocol.json',settings)
    for rel in settings['source_sha256']:
        p=output/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    if not formal:
        source_path=ROOT/'run-output-hro-modal-refinement/refined/model.pt'
        source=ModalStateOperator.load(source_path)
        assert {s.split(':')[0] for s in source.training_ids}=={ids[i] for i in SELECTED}
        baseline=field_metrics(source,arrays,VALIDATION);results={};eligible=[]
        for variant in VARIANTS:
            m=train(source,arrays,ids,SELECTED,variant,output/variant,20261124)
            results[variant]=field_metrics(m,arrays,VALIDATION)
            a=results[variant];b=baseline
            ok=a['mean']['sma_rmse_K']<b['mean']['sma_rmse_K'] and a['mean']['c2s4_sma_rmse_K']<b['mean']['c2s4_sma_rmse_K']
            ok=ok and all(x['sma_rmse_K']<=1.05*y['sma_rmse_K'] and x['fluid_rmse_K']==y['fluid_rmse_K'] for x,y in zip(a['per_case'],b['per_case']))
            if ok: eligible.append(variant)
        chosen=min(eligible,key=lambda v:results[v]['mean']['sma_rmse_K']) if eligible else None
        result={'status':'completed_train_partition_development','baseline':baseline,'results':results,'eligible':eligible,
                'selected_variant':chosen,'historical_test_scored':False,'source_checkpoint_sha256':sha(source_path)}
    else:
        for seed in SEEDS:
            path=ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/refined/model.pt'
            source=ModalStateOperator.load(path)
            train(source,arrays,ids,list(range(8)),settings['selected_variant'],output/f's{seed}',seed)
        d,_=data();rows=[]
        for seed in SEEDS:
            dest=output/f's{seed}';m=SelectiveSmaOperator.load(dest/'model.pt');p=infer(m,d)
            with np.load(ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/test_predictions.npz') as z:
                np.testing.assert_array_equal(p['state'],z['state'])
                np.testing.assert_array_equal(p['field'][:,1],z['field'][:,1])
            np.savez_compressed(dest/'test_predictions.npz',**p)
            row=cohorts(p,d);write(dest/'evaluation.json',row);rows.append(row)
        result={'status':'completed_fixed_SMA_adaptation','variant':settings['selected_variant'],
                'adapted':aggregate(rows,SEEDS),'state_and_fluid_predictions_exactly_preserved':True}
    result['artifacts']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    write(output/'summary.json',result)
    if formal:return {'status':result['status'],'variant':result['variant'],
        'third_cycle_field':result['adapted']['fixed_three_cycle']['field']['c3s4']}
    return {'status':result['status'],'selected_variant':result['selected_variant'],'baseline':result['baseline']['mean'],
            'validation':{k:v['mean'] for k,v in results.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--formal',action='store_true');p.add_argument('--development',type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): result=start(a.output,a.formal,a.development)
    print(json.dumps(result,indent=2))
