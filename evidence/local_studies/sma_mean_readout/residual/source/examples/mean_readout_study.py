"""Training-case selection for a low-cost, additive mean-temperature correction."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write, data
from .train_hierarchical import training_data
from .fit_thermal_readout import fit_readout
from .stage_readout_study import cache, fit_head, candidate_prediction, metrics, gate, FOLDS, SEEDS
from .formal_hierarchical import infer, cohorts, aggregate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.stage_thermal_readout import StageThermalReadoutOperator
from SciML.models.v13.mean_thermal_readout import MeanThermalReadout, MeanThermalReadoutOperator

SPATIAL={'kind':'separate','rank':8,'alpha':1.,'preserve_spatial_mean':True}
CANDIDATES=[{'name':f'{kind}_alpha_{alpha:g}','kind':kind,'rank':3,'alpha':alpha}
            for kind in ('compact','full') for alpha in (1.,10.)]
PRIOR=ROOT/'evidence/local_studies/sma_stage_readout/formal'
SOURCES=['SciML/models/v13/mean_thermal_readout.py','examples/mean_readout_study.py',
         'examples/stage_readout_study.py','SciML/models/v13/stage_thermal_readout.py']


def current_from_source(source,arrays,selected):
    shared,_=fit_readout(source,arrays,selected,'residual');fit=cache(source,arrays,selected)
    head,_=fit_head(shared,fit,SPATIAL)
    return StageThermalReadoutOperator(shared,head,preserve_spatial_mean=True),fit


def current_prediction(reference,values):
    return candidate_prediction(reference.reference,reference.stage2_readout,values,SPATIAL)


def fit_mean(reference,values,candidate):
    use=values['stage']==2;baseline=current_prediction(reference,values)
    target=(values['truth'][use,0]-baseline[use]).mean(-1)
    head=MeanThermalReadout(candidate['kind'],candidate['rank'])
    info=head.fit(values['initial'][use],values['control'][use],target,candidate['alpha'])
    return head,info


def mean_prediction(reference,head,values):
    pred=current_prediction(reference,values);use=values['stage']==2
    with torch.no_grad():pred[use]+=head(values['initial'][use],values['control'][use],head.time_grid)[:,:,None]
    return pred


def protocol(output,settings):
    output.mkdir(parents=True,exist_ok=False)
    record={**settings,'source_sha256':{p:sha(ROOT/p) for p in SOURCES},'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'protocol.json',record)
    for p in SOURCES:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())


def develop(output):
    arrays,ids,hashes=training_data()
    protocol(output,{'purpose':'mean-temperature correction selection on training cases only','candidates':CANDIDATES,
        'data_sha256':hashes,'folds':[{'checkpoint':p,'checkpoint_sha256':sha(ROOT/p),
        'fit_ids':[ids[i] for i in f],'validation_ids':[ids[i] for i in v]} for p,f,v in FOLDS],
        'gate':'stage2 mean RMSE decreases; every stage2 validation segment is nonregressing; preserve state/fluid/stage4/initial/spatial differences',
        'selection':'lowest eligible first-fold mean; fixed candidate on second fold; no historical test scoring'})
    started=time.perf_counter();records=[];chosen=None
    for fold,(path,selected,validation) in enumerate(FOLDS):
        if fold==1 and chosen is None:break
        source=ModalStateOperator.load(ROOT/path)
        assert {s.split(':')[0] for s in source.training_ids}=={ids[i] for i in selected}
        reference,fit=current_from_source(source,arrays,selected);val=cache(source,arrays,validation)
        before=metrics(current_prediction(reference,val),val);rows=[]
        for candidate in CANDIDATES if fold==0 else [chosen]:
            head,info=fit_mean(reference,fit,candidate);after=metrics(mean_prediction(reference,head,val),val)
            passed=gate(after,before);rows.append({'candidate':candidate,'metrics':after,'fit':info,'passed':passed})
            dest=output/f'fold{fold+1}'/candidate['name'];dest.mkdir(parents=True)
            torch.save({'candidate':candidate,'weights':head.state_dict()},dest/'readout.pt')
        records.append({'fold':fold+1,'reference':before,'candidates':rows})
        if fold==0:
            eligible=[r for r in rows if r['passed']]
            chosen=min(eligible,key=lambda r:r['metrics']['stage2_mean_rmse_K'])['candidate'] if eligible else None
    passed=chosen is not None and len(records)==2 and records[1]['candidates'][0]['passed']
    result={'status':'completed_mean_readout_development','selected_candidate':chosen,'passed_both_folds':passed,
        'folds':records,'seconds':time.perf_counter()-started,'historical_test_scored':False,
        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',result)
    return {'status':result['status'],'selected_candidate':chosen,'passed_both_folds':passed,'seconds':result['seconds'],
        'folds':[{'fold':r['fold'],'reference':r['reference']['stage2_mean_rmse_K'],
        'candidates':[{'name':c['candidate']['name'],'mean':c['metrics']['stage2_mean_rmse_K'],'passed':c['passed']} for c in r['candidates']]} for r in records]}


def formal(output,development):
    dev=json.loads((development/'summary.json').read_text())
    if not dev['passed_both_folds']:raise ValueError('Mean readout has not passed both development folds')
    for p,digest in dev['artifacts'].items():assert sha(development/p)==digest,p
    candidate=dev['selected_candidate'];arrays,ids,hashes=training_data()
    protocol(output,{'purpose':'fixed mean correction fit before historical regression evaluation','candidate':candidate,
        'development_sha256':sha(development/'summary.json'),'data_sha256':hashes,'fit_ids':ids,
        'checkpoint_sha256':{(PRIOR/f's{s}/model.pt').relative_to(ROOT).as_posix():sha(PRIOR/f's{s}/model.pt') for s in SEEDS}})
    started=time.perf_counter()
    for seed in SEEDS:
        ref=StageThermalReadoutOperator.load(PRIOR/f's{seed}/model.pt')
        fit=cache(ref.reference.backbone,arrays,list(range(8)));head,info=fit_mean(ref,fit,candidate)
        dest=output/f's{seed}';dest.mkdir();MeanThermalReadoutOperator(ref,head).save(dest/'model.pt',metadata={'candidate':candidate})
        write(dest/'fit.json',info)
    fit_seconds=time.perf_counter()-started;d,_=data();rows=[];comparison=[]
    for seed in SEEDS:
        dest=output/f's{seed}';p=infer(MeanThermalReadoutOperator.load(dest/'model.pt'),d)
        with np.load(PRIOR/f's{seed}/test_predictions.npz') as old:
            for a,b in [(p['state'],old['state']),(p['field'][:,1],old['field'][:,1]),
                (p['field'][:,:,0],old['field'][:,:,0]),(p['field'][d['field_stage']==4],old['field'][d['field_stage']==4])]:np.testing.assert_array_equal(a,b)
            np.testing.assert_allclose(np.diff(p['field'][:,0],axis=-1),np.diff(old['field'][:,0],axis=-1),rtol=0,atol=1e-12)
            for i in range(len(d['field_stage'])):
                comparison.append({'seed':seed,'case_id':str(d['case_ids'][d['field_case_index'][i]]),
                    'cycle':int(d['field_cycle'][i]),'stage':int(d['field_stage'][i]),
                    'before_K':float(np.sqrt(np.mean((old['field'][i,0,1:]-d['field_truth'][i,0,1:])**2))),
                    'after_K':float(np.sqrt(np.mean((p['field'][i,0,1:]-d['field_truth'][i,0,1:])**2)))})
        np.savez_compressed(dest/'test_predictions.npz',**p);score=cohorts(p,d);rows.append(score);write(dest/'evaluation.json',score)
    result={'status':'completed_fixed_mean_readout','candidate':candidate,'fit_seconds':fit_seconds,
        'adapted':aggregate(rows,SEEDS),'comparisons':comparison,
        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',result)
    return {'status':result['status'],'candidate':candidate,'fit_seconds':fit_seconds,'comparisons':comparison}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--development',type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=formal(a.output,a.development) if a.development else develop(a.output)
    print(json.dumps(result,indent=2))
