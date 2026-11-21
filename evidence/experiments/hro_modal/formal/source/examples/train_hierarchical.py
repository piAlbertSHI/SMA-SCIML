"""Joint training entry point; short runs establish execution, not model accuracy."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, data, write
from SciML.models.v12.learning_historical_rollout import HistoricalModel
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.hierarchical_operator import HierarchicalStateOperator, OperatorConfig, joint_loss


def training_data():
    d,provenance=data()
    path=ROOT/'data/historical12_training_fields.npz'
    field_meta=json.loads(path.with_suffix('.json').read_text())
    if sha(path)!=field_meta['snapshot_sha256'] or field_meta['state_snapshot_sha256']!=provenance['snapshot_sha256']:
        raise ValueError('Training snapshot mismatch')
    indices=np.flatnonzero(d['split']=='train');ids=d['case_ids'][indices].tolist()
    assert ids==field_meta['train_case_ids'] and len(ids)==8 and np.all(d['cycles'][indices]==2)
    targets=np.stack([d['after'][d['case_index']==i] for i in indices])
    with np.load(path,allow_pickle=False) as fields:
        assert set(fields['case_ids'])==set(ids)
        target_fields=[]
        for case in ids:
            rows=[]
            for cycle,stage in [(1,2),(1,4),(2,2),(2,4)]:
                use=(fields['case_ids']==case)&(fields['cycle']==cycle)&(fields['stage']==stage)
                assert use.sum()==1
                rows.append(fields['truth'][use][0])
            target_fields.append(np.stack(rows))
    arrays={'initial':torch.tensor(d['initial'][indices]),'control':torch.tensor(d['control'][indices]),
            'states':torch.tensor(targets),'fields':torch.tensor(np.stack(target_fields))}
    return arrays,ids,{'state':provenance['snapshot_sha256'],'fields':field_meta['snapshot_sha256']}


def run(output,updates=200,batch_cases=8,seed=20261121,lr=.0003):
    if output.exists():raise FileExistsError('Use a new experiment directory')
    if updates<1 or not 1<=batch_cases<=8 or lr<=0:raise ValueError('Invalid training budget')
    torch.set_num_threads(1)
    arrays,ids,hashes=training_data()
    config=OperatorConfig();output.mkdir(parents=True)
    protocol={'status':'architecture_execution_experiment','not_an_accuracy_benchmark':True,'seed':seed,
        'architecture':asdict(config),'training_case_ids':ids,'training_cycles':[1,2],
        'updates':updates,'batch_cases':batch_cases,'optimizer':'AdamW','lr':lr,'weight_decay':1e-4,'clip_norm':1.,
        'schedule':'linear warmup then cosine decay to 10% of initial lr','warmup_updates':min(20,max(1,updates//10)),
        'joint_loss':{'state_blocks':'eight equally weighted normalized blocks','field_weight':1.,'spatial_gradient_weight':.01},
        'field_queries_per_stage':'time zero + endpoint + six random interior times; 26 uniformly spaced spatial indices',
        'supervision':'train-case states and existing train-case fields only; test scores not computed',
        'initial_fields':'reconstructed from predicted stage-entry state; no teacher forcing',
        'source_sha256':{p:sha(ROOT/p) for p in ['SciML/models/v13/hierarchical_operator.py','examples/train_hierarchical.py']},
        'data_sha256':hashes,'base_checkpoint_sha256':sha(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt'),
        'fresh_solver_calls':0}
    write(output/'protocol.json',protocol)
    with torch.random.fork_rng(devices=[]),threadpool_limits(limits=1):
        torch.manual_seed(seed)
        base=HistoricalModel.load(ROOT/'checkpoints/historical-rollout-r1/s20260910/model.pt')
        assert set(base.training_ids)=={f'{case}:c{c}:{s}' for case in ids for c,s in [(1,'stage1'),(1,'stage2'),(1,'stage3'),(1,'stage4'),(1,'restart'),(2,'stage1'),(2,'stage2'),(2,'stage3'),(2,'stage4')]}
        model=HierarchicalStateOperator(base,config)
        field_delta=arrays['fields']-arrays['fields'][:,:,:,0:1]
        model.decoder.field_scale.copy_(field_delta.square().mean(dim=(0,1,3,4)).sqrt().clamp_min(.1))
        frozen={k:v.clone() for k,v in model.named_buffers()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        warmup=protocol['warmup_updates']
        def factor(step):
            if step<warmup:return (step+1)/warmup
            progress=(step-warmup)/max(1,updates-warmup)
            return .1+.9*.5*(1+math.cos(math.pi*progress))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
        rng=np.random.default_rng(seed);history=[];started=time.perf_counter();model.train()
        for step in range(updates):
            chosen=np.sort(rng.choice(8,batch_cases,replace=False));ti=np.r_[0,np.sort(rng.choice(np.arange(1,49),6,replace=False)),49]
            xi=np.linspace(0,100,26,dtype=int);times=torch.tensor(ti/49.);positions=torch.tensor(xi/100.)
            optimizer.zero_grad(set_to_none=True)
            prediction=model(arrays['initial'][chosen],arrays['control'][chosen],2,times=times,positions=positions)
            fields=arrays['fields'][chosen][:,:,:,ti][:,:,:,:,xi]
            losses=joint_loss(prediction,arrays['states'][chosen],model.feature_scale,field_truth=fields,
                field_scale=model.decoder.field_scale,positions=positions)
            losses['total'].backward()
            if step==0:
                gradient_coverage={name:any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in module.parameters())
                                   for name,module in model.named_children()}
                assert all(gradient_coverage.values()),gradient_coverage
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            row={'update':step+1,'train_case_ids':[ids[i] for i in chosen],'time_indices':ti.tolist(),'position_indices':xi.tolist(),
                 'lr':float(optimizer.param_groups[0]['lr']),'gradient_norm_before_clip':float(norm),
                 **{k:float(v.detach()) for k,v in losses.items()}}
            optimizer.step();scheduler.step();history.append(row)
        elapsed=time.perf_counter()-started
        for name,value in model.named_buffers():assert torch.equal(value,frozen[name]),name
        model.eval()
        # Fixed first training case, not a selected test result: check full three-cycle execution.
        times=torch.linspace(0,1,10,dtype=torch.float64);positions=torch.linspace(0,1,51,dtype=torch.float64)
        with torch.no_grad():execution=model(arrays['initial'][:1],arrays['control'][:1],3,times=times,positions=positions)
        model.save(output/'model.pt',metadata={'status':'execution_smoke_only' if updates<100 else 'trained_not_independently_evaluated',
                    'updates':updates,'seed':seed,'data_sha256':hashes})
        loaded=HierarchicalStateOperator.load(output/'model.pt')
        with torch.no_grad():replay=loaded(arrays['initial'][:1],arrays['control'][:1],3,times=times,positions=positions)
        for key in ('states','fields','final_memory'):torch.testing.assert_close(execution[key],replay[key],rtol=0,atol=0)
        # Save optimizer/scheduler and RNG states for auditing; optional future resume must preserve protocol.
        with (output/'training_state.pt').open('xb') as f:
            torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'torch_rng':torch.get_rng_state(),
                        'numpy_rng_json':json.dumps(rng.bit_generator.state),'completed_updates':updates},f)
        np.savez_compressed(output/'execution_arrays.npz',states=execution['states'].numpy(),fields=execution['fields'].numpy(),
            initial=arrays['initial'][:1].numpy(),control=arrays['control'][:1].numpy(),times=times.numpy(),positions=positions.numpy())
        write(output/'history.json',history)
        summary={'status':'completed_architecture_execution_check','accuracy_claim':False,'fresh_solver_calls':0,
            'parameter_report':model.parameter_report(),'training_updates':updates,'training_seconds':elapsed,
            'major_module_gradient_coverage':{k:bool(v) for k,v in gradient_coverage.items()},
            'three_cycle_state_shape':list(execution['states'].shape),'three_cycle_field_shape':list(execution['fields'].shape),
            'checkpoint_replay_exact':True,'frozen_prior_and_scalers_unchanged':True,'test_scores_computed':False,
            'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
        write(output/'summary.json',summary)
    return {k:v for k,v in summary.items() if k!='artifacts'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200);p.add_argument('--batch-cases',type=int,default=8);p.add_argument('--seed',type=int,default=20261121);p.add_argument('--lr',type=float,default=.0003);a=p.parse_args()
    print(json.dumps(run(a.output,a.updates,a.batch_cases,a.seed,a.lr),indent=2))
