"""Bounded stage-two regularization study using existing training-case backbones."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write, data
from .train_hierarchical import training_data
from .fit_thermal_readout import fit_readout
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.thermal_readout import ThermalReadout, ThermalReadoutOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator

SEEDS=(20261121,20261122,20261123)
PRIOR=ROOT/'evidence/local_studies/sma_readout/formal'
FOLDS=[('run-output-hro-modal-refinement/refined/model.pt',[0,1,2,4,5,6],[3,7]),
       ('run-output-sma-physical-fold2/refined/model.pt',[0,2,3,4,6,7],[1,5])]
# Fixed before candidate evaluation; no expansion using historical test scores.
CANDIDATES=[{'name':f'shrink_{s:g}','kind':'shrink','scale':s} for s in (0.,.5,.75)] + [
    {'name':f'shared_alpha_{a:g}','kind':'shared','rank':12,'alpha':a} for a in (10.,100.)] + [
    {'name':f'separate_r{r}_alpha_{a:g}','kind':'separate','rank':r,'alpha':a} for r in (4,8) for a in (1.,10.,100.)]
SOURCES=['examples/stage_readout_study.py','SciML/models/v13/stage_thermal_readout.py',
         'examples/fit_thermal_readout.py','SciML/models/v13/thermal_readout.py']


def cache(source,arrays,selected,*,include_state=False):
    source.eval().requires_grad_(False)
    state=arrays['initial'][selected];control=arrays['control'][selected];memory=None;exchange=0
    values={k:[] for k in ('initial','control','stage','base','truth','case','cycle')}
    if include_state:values.update({'entry_state':[],'latent':[],'observed_entry_state':[]})
    times=torch.linspace(0,1,50,dtype=torch.float64);positions=torch.linspace(0,1,101,dtype=torch.float64)
    with torch.no_grad():
        for step,stage in enumerate([1,2,3,4,5,1,2,3,4]):
            decode=stage in (2,4)
            out=source.step(state,control,stage,memory,times=times if decode else None,positions=positions if decode else None)
            if decode:
                if include_state:
                    values['entry_state'].append(state);values['latent'].append(out['latent'])
                    values['observed_entry_state'].append(arrays['states'][selected,step-1])
                for k,v in {'initial':source.initial_fields(state,stage),'control':control,
                    'stage':torch.full((len(selected),),stage),'base':out['field'],
                    'truth':arrays['fields'][selected,exchange],'case':torch.tensor(selected),
                    'cycle':torch.full((len(selected),),exchange//2+1)}.items():values[k].append(v)
                exchange+=1
            state,memory=out['state'],out['memory']
    return {k:torch.cat(v) for k,v in values.items()}


def fit_head(reference,fit,candidate):
    if candidate['kind']=='shrink':return copy.deepcopy(reference.readout),{'scale':candidate['scale']}
    use=fit['stage']==2 if candidate['kind']=='separate' else torch.ones_like(fit['stage'],dtype=torch.bool)
    readout=ThermalReadout(candidate['rank'],'residual')
    delta=fit['truth'][use,0]-fit['base'][use,0]
    info=readout.fit(fit['initial'][use],fit['control'][use],fit['stage'][use],delta,alpha=candidate['alpha'])
    return readout,info


def predict_cached(head,values,scale=1.):
    with torch.no_grad():
        return values['base'][:,0]+scale*head(values['initial'],values['control'],values['stage'],
            torch.linspace(0,1,50,dtype=torch.float64),torch.linspace(0,1,101,dtype=torch.float64))


def metrics(pred,values):
    error=pred[:,1:]-values['truth'][:,0,1:]
    rows=[{'case_index':int(values['case'][i]),'cycle':int(values['cycle'][i]),'stage':int(values['stage'][i]),
           'rmse_K':float(error[i].square().mean().sqrt()),'bias_K':float(error[i].mean())} for i in range(len(error))]
    stage2=[r for r in rows if r['stage']==2]
    return {'stage2_mean_rmse_K':float(np.mean([r['rmse_K'] for r in stage2])),
            'segments':rows}


def gate(candidate,reference):
    return candidate['stage2_mean_rmse_K']<reference['stage2_mean_rmse_K'] and all(
        a['rmse_K']<=b['rmse_K'] for a,b in zip(candidate['segments'],reference['segments']) if a['stage']==2)


def protocol(output,purpose,extra):
    output.mkdir(parents=True,exist_ok=False)
    record={'purpose':purpose,'fresh_solver_calls':0,'neural_training_updates':0,
            'source_sha256':{p:sha(ROOT/p) for p in SOURCES},**extra}
    write(output/'protocol.json',record)
    for p in SOURCES:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    return record


def candidate_prediction(reference,head,values,candidate):
    prediction=predict_cached(head,values,candidate.get('scale',1.))
    reference_pred=predict_cached(reference.readout,values)
    if candidate.get('preserve_spatial_mean',False):
        delta=prediction-values['base'][:,0]
        shared=reference_pred-values['base'][:,0]
        prediction=values['base'][:,0]+delta-delta.mean(-1,keepdim=True)+shared.mean(-1,keepdim=True)
    prediction[values['stage']==4]=reference_pred[values['stage']==4]
    return prediction


def develop(output,shape_only=False):
    arrays,ids,hashes=training_data()
    candidates=[{'name':'separate_shape_r8_alpha_1','kind':'separate','rank':8,'alpha':1.,
                 'preserve_spatial_mean':True}] if shape_only else CANDIDATES
    protocol(output,'stage-two candidate selection on first internal fold, then a fixed second-fold check',{
        'candidates':candidates,'data_sha256':hashes,'folds':[
            {'checkpoint':p,'checkpoint_sha256':sha(ROOT/p),'fit_ids':[ids[i] for i in fit],
             'validation_ids':[ids[i] for i in val]} for p,fit,val in FOLDS],
        'gate':'lower stage2 mean RMSE and no validation stage2 segment regression relative to shared residual readout; stage4 unchanged by construction',
        'selection':'lowest eligible first-fold stage2 RMSE; only this candidate checked on fold2; no test scores'})
    started=time.perf_counter();records=[];chosen=None
    for fold,(path,selected,validation) in enumerate(FOLDS):
        if fold==1 and chosen is None:break
        source=ModalStateOperator.load(ROOT/path)
        assert {s.split(':')[0] for s in source.training_ids}=={ids[i] for i in selected}
        reference,_=fit_readout(source,arrays,selected,'residual')
        fit=cache(source,arrays,selected);val=cache(source,arrays,validation)
        reference_pred=predict_cached(reference.readout,val);baseline=metrics(reference_pred,val);rows=[]
        for candidate in candidates if fold==0 else [chosen]:
            head,info=fit_head(reference,fit,candidate)
            prediction=candidate_prediction(reference,head,val,candidate)
            score=metrics(prediction,val);passed=gate(score,baseline)
            rows.append({'candidate':candidate,'metrics':score,'passed':passed,'fit':info})
            dest=output/f'fold{fold+1}'/candidate['name'];dest.mkdir(parents=True)
            torch.save({'head':head.state_dict(),'candidate':candidate},dest/'readout.pt')
        record={'fold':fold+1,'reference':baseline,'candidates':rows};records.append(record)
        if fold==0:
            eligible=[r for r in rows if r['passed']]
            chosen=min(eligible,key=lambda r:r['metrics']['stage2_mean_rmse_K'])['candidate'] if eligible else None
    passed=chosen is not None and len(records)==2 and records[1]['candidates'][0]['passed']
    result={'status':'completed_stage_readout_development','selected_candidate':chosen,'passed_both_folds':passed,
            'folds':records,'seconds':time.perf_counter()-started,'historical_test_scored':False,
            'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',result)
    return {'status':result['status'],'selected_candidate':chosen,'passed_both_folds':passed,
            'seconds':result['seconds'],'folds':[{'fold':r['fold'],'reference':r['reference']['stage2_mean_rmse_K'],
              'candidates':[{'name':c['candidate']['name'],'stage2_rmse_K':c['metrics']['stage2_mean_rmse_K'],
                             'passed':c['passed']} for c in r['candidates']]} for r in records]}


def formal(output,development):
    dev=json.loads((development/'summary.json').read_text())
    if not dev['passed_both_folds']:raise ValueError('Both development folds must pass')
    for p,digest in dev['artifacts'].items():assert sha(development/p)==digest,p
    candidate=dev['selected_candidate'];arrays,ids,hashes=training_data()
    protocol(output,'fixed candidate fit on eight training cases before historical test evaluation',{
        'candidate':candidate,'development_sha256':sha(development/'summary.json'),'data_sha256':hashes,
        'fit_ids':ids,'checkpoint_sha256':{(PRIOR/f's{s}/model.pt').relative_to(ROOT).as_posix():sha(PRIOR/f's{s}/model.pt') for s in SEEDS}})
    started=time.perf_counter()
    for seed in SEEDS:
        reference=ThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        fit=cache(reference.backbone,arrays,list(range(8)));head,info=fit_head(reference,fit,candidate)
        model=StageThermalReadoutOperator(reference,head,candidate.get('scale',1.),candidate.get('preserve_spatial_mean',False))
        dest=output/f's{seed}';dest.mkdir();model.save(dest/'model.pt',metadata={'candidate':candidate})
        write(dest/'fit.json',info)
    fit_seconds=time.perf_counter()-started
    d,_=data();rows=[];comparisons=[]
    for seed in SEEDS:
        dest=output/f's{seed}';pred=infer(StageThermalReadoutOperator.load(dest/'model.pt'),d)
        with np.load(PRIOR/f's{seed}/test_predictions.npz') as old:
            for a,b in [(pred['state'],old['state']),(pred['field'][:,1],old['field'][:,1]),
                        (pred['field'][:,:,0],old['field'][:,:,0]),(pred['field'][d['field_stage']==4],old['field'][d['field_stage']==4])]:
                np.testing.assert_array_equal(a,b)
            for i in range(len(d['field_stage'])):
                comparisons.append({'seed':seed,'case_id':str(d['case_ids'][d['field_case_index'][i]]),
                    'cycle':int(d['field_cycle'][i]),'stage':int(d['field_stage'][i]),
                    'before_K':float(np.sqrt(np.mean((old['field'][i,0,1:]-d['field_truth'][i,0,1:])**2))),
                    'after_K':float(np.sqrt(np.mean((pred['field'][i,0,1:]-d['field_truth'][i,0,1:])**2)))})
        np.savez_compressed(dest/'test_predictions.npz',**pred)
        score=cohorts(pred,d);rows.append(score);write(dest/'evaluation.json',score)
    result={'status':'completed_fixed_stage_readout','candidate':candidate,'fit_seconds':fit_seconds,
        'adapted':aggregate(rows,SEEDS),'comparisons':comparisons,'stage4_state_fluid_initial_exact':True,
        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',result)
    return {'status':result['status'],'candidate':candidate,'fit_seconds':fit_seconds,
            'comparisons':comparisons,'c3s4':result['adapted']['fixed_three_cycle']['field']['c3s4']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--development',type=Path);p.add_argument('--shape-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result=formal(args.output,args.development) if args.development else develop(args.output,args.shape_only)
    print(json.dumps(result,indent=2))
