"""Two fixed training objectives for a causal fluid-conditioned SMA mean."""
import argparse
import json
from pathlib import Path
import time

import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from .mean_readout_study import current_from_source,current_prediction,cache
from .stage_readout_study import FOLDS,metrics,gate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.relaxation_readout import RelaxationMeanReadout, VolumeRelaxationMeanReadout

CANDIDATES=('observed_thermal_response','autonomous_thermal_response')
SOURCES=['SciML/models/v13/relaxation_readout.py','examples/relaxation_readout_study.py',
         'examples/mean_readout_study.py','examples/stage_readout_study.py']


def fit_head(values,kind,volume=False):
    use=values['stage']==2
    if kind not in CANDIDATES:raise ValueError('Unknown response objective')
    initial=values['truth'][use,0,0].mean(-1) if kind==CANDIDATES[0] else values['initial'][use,0].mean(-1)
    fluid=values['truth'][use,1].mean(-1) if kind==CANDIDATES[0] else values['base'][use,1].mean(-1)
    head=VolumeRelaxationMeanReadout() if volume else RelaxationMeanReadout()
    info=head.fit(initial,fluid,values['control'][use],values['truth'][use,0].mean(-1))
    return head,info


def predict(reference,head,values,scale=1.):
    pred=current_prediction(reference,values);use=values['stage']==2
    with torch.no_grad():
        mean=head(values['initial'][use,0].mean(-1),values['base'][use,1].mean(-1),values['control'][use])
        delta=mean-pred[use].mean(-1);delta[:,0]=0.
        pred[use]+=scale*delta[:,:,None]
    return pred


def run(output,volume=False,damped=False):
    output.mkdir(parents=True,exist_ok=False);arrays,ids,hashes=training_data()
    candidates=(CANDIDATES[1],) if damped else CANDIDATES
    scale=.5 if damped else 1.
    write(output/'protocol.json',{'candidates':list(candidates),'data_sha256':hashes,'mean_scale':scale,
        'geometry_scaling':'duration/volume' if volume else 'duration/thickness',
        'regularization':'one fixed half-strength correction after full replacement overshoot; no coefficient sweep' if damped else 'none',
        'gate':'lower stage2 mean RMSE, every stage2 validation segment nonregressing',
        'selection':'lowest eligible first-fold RMSE then only selected candidate on second fold; no historical test scores',
        'parameter_count':1,'loss':'future SMA spatial mean MSE',
        'numerical_optimization':{'rate_bounds':[.001,1000.],'max_objective_evaluations':100},
        'folds':[{'checkpoint':p,'checkpoint_sha256':sha(ROOT/p),'fit_ids':[ids[i] for i in fit],
                  'validation_ids':[ids[i] for i in val]} for p,fit,val in FOLDS],
        'source_sha256':{p:sha(ROOT/p) for p in SOURCES},'fresh_solver_calls':0,'neural_training_updates':0})
    for p in SOURCES:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    started=time.perf_counter();records=[];chosen=None
    for fold,(path,selected,validation) in enumerate(FOLDS,1):
        if fold==2 and chosen is None:break
        source=ModalStateOperator.load(ROOT/path)
        assert {v.split(':')[0] for v in source.training_ids}=={ids[i] for i in selected}
        reference,fit=current_from_source(source,arrays,selected);val=cache(source,arrays,validation)
        before=metrics(current_prediction(reference,val),val);rows=[]
        for kind in candidates if fold==1 else (chosen,):
            head,info=fit_head(fit,kind,volume);pred=predict(reference,head,val,scale);after=metrics(pred,val)
            passed=gate(after,before) and not info['rate_at_numerical_bound']
            rows.append({'candidate':kind,'metrics':after,'fit':info,'passed':passed})
            dest=output/f'fold{fold}'/kind;dest.mkdir(parents=True)
            torch.save({'weights':head.state_dict(),'candidate':kind,'volume_scaling':volume,'mean_scale':scale},dest/'readout.pt')
            torch.save({'prediction':pred,'reference':current_prediction(reference,val),'truth':val['truth'][:,0],
                **{k:val[k] for k in ('case','cycle','stage')}},dest/'validation.pt')
        records.append({'fold':fold,'reference':before,'candidates':rows})
        if fold==1:
            eligible=[r for r in rows if r['passed']]
            chosen=min(eligible,key=lambda r:r['metrics']['stage2_mean_rmse_K'])['candidate'] if eligible else None
    result={'status':'completed_relaxation_mean_development','selected_candidate':chosen,
        'passed_both_folds':len(records)==2 and records[1]['candidates'][0]['passed'],
        'folds':records,'seconds':time.perf_counter()-started,'historical_test_scored':False,
        'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'summary.json',result)
    write(output/'manifest.json',{'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
    return {**{k:v for k,v in result.items() if k!='folds'},'folds':[{'fold':r['fold'],
        'reference':r['reference']['stage2_mean_rmse_K'],'candidates':[{'name':c['candidate'],
        'rmse_K':c['metrics']['stage2_mean_rmse_K'],'passed':c['passed'],'fit':c['fit']} for c in r['candidates']]} for r in records]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--volume',action='store_true');p.add_argument('--damped',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output,a.volume,a.damped)
    print(json.dumps(result,indent=2))
