"""One fixed empirical-error shrinkage recipe using training cases only."""
import argparse
import json
from pathlib import Path
import time

import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT,write
from .train_hierarchical import training_data
from .mean_readout_study import cache,current_from_source,current_prediction
from .relaxation_readout_study import fit_head
from .stage_readout_study import FOLDS,metrics,gate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.risk_limited_readout import shrink_correction

SOURCES=['SciML/models/v13/risk_limited_readout.py','SciML/models/v13/relaxation_readout.py',
    'examples/risk_limited_readout_study.py','examples/relaxation_readout_study.py',
    'examples/mean_readout_study.py','examples/stage_readout_study.py']


def fit_response(values):
    central,fit=fit_head(values,'autonomous_thermal_response',True)
    cases=sorted(set(values['case'].tolist()))
    if len(cases)<3:raise ValueError('At least three training cases required for deletion analysis')
    residuals=[];records=[];members=[]
    for case in cases:
        subset={k:v[values['case']!=case] for k,v in values.items()}
        member,info=fit_head(subset,'autonomous_thermal_response',True)
        use=(values['case']==case)&(values['stage']==2)
        with torch.no_grad():
            prediction=member(values['initial'][use,0].mean(-1),values['base'][use,1].mean(-1),values['control'][use])
        error=prediction-values['truth'][use,0].mean(-1)
        residuals.append(error.square().mean(0));members.append(member)
        records.append({'deleted_case_index':case,'validation_cycles':values['cycle'][use].tolist(),
                        'fit':info,'response_mean_rmse_K':float(error[:,1:].square().mean().sqrt())})
    scale=torch.stack(residuals).mean(0).sqrt()
    return central,scale,{'central':fit,'case_deletions':records,
        'scale_aggregation':'sqrt(mean_case(mean_cycle(response_error_squared))) at each native time',
        'backbone_scope':'frozen backbone saw all fit cases; deletion applies only to response-rate fit',
        'coverage_calibrated':False},members


def predict(reference,response,error_scale,values):
    prediction=current_prediction(reference,values);use=values['stage']==2
    with torch.no_grad():
        mean=response(values['initial'][use,0].mean(-1),values['base'][use,1].mean(-1),values['control'][use])
        delta=mean-prediction[use].mean(-1);delta[:,0]=0.
        corrected=shrink_correction(delta,error_scale)
        prediction[use]+=.5*corrected[:,:,None]
    return prediction


def run(output):
    output.mkdir(parents=True,exist_ok=False);arrays,ids,hashes=training_data()
    protocol={'candidate':'case-deletion RMS shrinkage of volume-scaled response; half blend',
        'data_sha256':hashes,'mean_scale':.5,'error_scale_multiplier':1.,
        'calibration':'delete each fit case from response-rate fitting; frozen backbone unchanged; no coverage claim',
        'gate':'stage2 mean RMSE strictly decreases and every validation stage2 segment nonregresses',
        'selection':'one fixed candidate; no threshold or blend sweep; no historical test scores',
        'folds':[{'checkpoint':p,'checkpoint_sha256':sha(ROOT/p),'fit_ids':[ids[i] for i in fit],
                  'validation_ids':[ids[i] for i in val]} for p,fit,val in FOLDS],
        'source_sha256':{p:sha(ROOT/p) for p in SOURCES},'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'protocol.json',protocol)
    for p in SOURCES:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    started=time.perf_counter();records=[]
    for fold,(path,selected,validation) in enumerate(FOLDS,1):
        source=ModalStateOperator.load(ROOT/path)
        assert {s.split(':')[0] for s in source.training_ids}=={ids[i] for i in selected}
        ref,fit=current_from_source(source,arrays,selected);val=cache(source,arrays,validation)
        head,scale,info,members=fit_response(fit)
        before=metrics(current_prediction(ref,val),val);after=metrics(predict(ref,head,scale,val),val)
        passed=gate(after,before) and not any(r['fit']['rate_at_numerical_bound'] for r in info['case_deletions']) and not info['central']['rate_at_numerical_bound']
        dest=output/f'fold{fold}';dest.mkdir()
        torch.save({'response':head.state_dict(),'error_scale':scale,'members':[m.state_dict() for m in members]},dest/'readout.pt')
        write(dest/'fit.json',info)
        records.append({'fold':fold,'reference':before,'candidate':after,'passed':passed,
            'scale_min_K':float(scale.min()),'scale_max_K':float(scale.max())})
        if not passed:break
    result={'status':'completed_risk_limited_development','folds':records,
        'passed_both_folds':len(records)==2 and all(r['passed'] for r in records),
        'seconds':time.perf_counter()-started,'historical_test_scored':False,'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'summary.json',result)
    write(output/'manifest.json',{'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
    return {**{k:v for k,v in result.items() if k!='folds'},'folds':[{'fold':r['fold'],'before_K':r['reference']['stage2_mean_rmse_K'],
        'after_K':r['candidate']['stage2_mean_rmse_K'],'passed':r['passed'],
        'scale_range_K':[r['scale_min_K'],r['scale_max_K']]} for r in records]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output)
    print(json.dumps(result,indent=2))
