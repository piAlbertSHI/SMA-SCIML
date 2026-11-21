"""Explain a fixed candidate's errors; oracle substitutions are diagnostic only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .multicycle_multistep import ROOT,data,write
from .stage_readout_study import SEEDS
from .formal_relaxation_readout import PRIOR
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.relaxation_readout import VolumeRelaxationMeanReadout


def run(output):
    output.mkdir(parents=True,exist_ok=False);root=ROOT/'evidence/local_studies/sma_relaxation_readout/formal'
    d,_=data();rows=[];inputs={}
    use=np.flatnonzero(d['field_stage']==2);truth=d['field_truth'][use].mean(-1)
    control=torch.tensor(d['control'][d['field_case_index'][use]])
    for seed in SEEDS:
        old=PRIOR/f's{seed}/test_predictions.npz';new=root/f's{seed}/test_predictions.npz';fit=root/f's{seed}/fit.json'
        for p in (old,new,fit):inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        with np.load(old) as z:a=z['field'][use]
        with np.load(new) as z:b=z['field'][use]
        head=VolumeRelaxationMeanReadout();head.rate.fill_(json.loads(fit.read_text())['rate'])
        target=torch.tensor(truth[:,0]);initial=torch.tensor(a[:,0,0].mean(-1));fluid=torch.tensor(a[:,1].mean(-1))
        oracle=head(target[:,0],torch.tensor(truth[:,1]),control)
        entry=head(initial,torch.tensor(truth[:,1]),control)
        response=head(initial,fluid,control)
        parts=torch.stack((oracle-target,entry-oracle,response-entry),dim=-1).numpy()
        for j,i in enumerate(use):
            err=a[j,0,1:]-d['field_truth'][i,0,1:];delta=b[j,0,1:]-a[j,0,1:]
            square=float(np.mean(delta**2));cross=float(2*np.mean(err*delta))
            np.testing.assert_allclose(np.mean((err+delta)**2)-np.mean(err**2),square+cross,rtol=0,atol=1e-12)
            expected=(response[j].numpy()-a[j,0].mean(-1))*.5;expected[0]=0.
            np.testing.assert_allclose(delta.mean(-1),expected[1:],rtol=0,atol=1e-12)
            rows.append({'seed':seed,'case_id':str(d['case_ids'][d['field_case_index'][i]]),'cycle':int(d['field_cycle'][i]),
                'baseline_bias_K':float(err.mean()),'candidate_bias_K':float((err+delta).mean()),
                'correction_square_K2':square,'cross_term_K2':cross,'MSE_change_K2':square+cross,
                'response_bias_components_K':parts[j,1:].mean(0).tolist()})
    groups=[]
    for case,cycle in sorted({(r['case_id'],r['cycle']) for r in rows}):
        selected=[r for r in rows if r['case_id']==case and r['cycle']==cycle]
        groups.append({'case_id':case,'cycle':cycle,**{k:np.mean([r[k] for r in selected],axis=0).tolist() for k in
            ('baseline_bias_K','candidate_bias_K','correction_square_K2','cross_term_K2','MSE_change_K2','response_bias_components_K')}})
    result={'status':'completed_fixed_candidate_error_decomposition','rows':rows,'groups':groups,
        'response_component_order':['response_model_mismatch','entry_propagation','fluid_propagation'],
        'scope':'post-evaluation diagnosis; oracle substitutions are not inference scores or training labels',
        'input_sha256':inputs,'source_sha256':sha(Path(__file__)),'fresh_solver_calls':0,'parameters_changed':False}
    write(output/'analysis.json',result)
    return {'status':result['status'],'positive_mean_MSE_changes':[r for r in groups if r['MSE_change_K2']>0]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
