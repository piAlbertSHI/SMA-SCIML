"""Decompose response errors and audit in-fit versus withheld entry states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT,write
from .train_hierarchical import training_data
from .mean_readout_study import cache,current_from_source,current_prediction
from .stage_readout_study import FOLDS
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.relaxation_readout import RelaxationMeanReadout
from SciML.models.v13.thermal_geometry import radial_weights,strip_weights


def run(output,development):
    output.mkdir(parents=True,exist_ok=False);arrays,ids,hashes=training_data();rows=[];response=[]
    ws=torch.tensor(strip_weights());wr=torch.tensor(radial_weights())
    for fold,(path,selected,validation) in enumerate(FOLDS,1):
        source=ModalStateOperator.load(ROOT/path);reference,fit=current_from_source(source,arrays,selected)
        val=cache(source,arrays,validation)
        for partition,v in (('fit',fit),('validation',val)):
            use=v['stage']==2;indices=torch.nonzero(use).flatten();baseline=current_prediction(reference,v)
            for ix in indices.tolist():
                ci=int(v['case'][ix]);cycle=int(v['cycle'][ix]);step=(1,6)[cycle-1]
                mean_error=(baseline[ix]-v['truth'][ix,0]).mean(-1)
                entry_error=v['initial'][ix,0]-v['truth'][ix,0,0]
                pred_delta=((v['exit_state'][ix,102:153]-v['entry_state'][ix,102:153])*wr).sum()
                true_delta=((arrays['states'][ci,step,102:153]-arrays['states'][ci,step-1,102:153])*wr).sum()
                rows.append({'fold':fold,'partition':partition,'case_index':ci,'cycle':cycle,
                    'entry_arithmetic_bias_K':float(entry_error.mean()),'entry_capacity_bias_K':float((entry_error*ws).sum()),
                    'increment_error_K':float(pred_delta-true_delta),
                    'reference_mean_rmse_K':float(mean_error[1:].square().mean().sqrt()),
                    'fluid_mean_rmse_K':float((v['base'][ix,1,1:].mean(-1)-v['truth'][ix,1,1:].mean(-1)).square().mean().sqrt())})
        if fold!=1:continue
        use=val['stage']==2;initial=val['initial'][use,0].mean(-1);fluid=val['base'][use,1].mean(-1)
        trueinitial=val['truth'][use,0,0].mean(-1);truefluid=val['truth'][use,1].mean(-1)
        target=val['truth'][use,0].mean(-1);control=val['control'][use]
        for folder in sorted((development/'fold1').iterdir()):
            head=RelaxationMeanReadout();head.load_state_dict(torch.load(folder/'readout.pt',weights_only=True)['weights'])
            oracle=head(trueinitial,truefluid,control);entry_only=head(initial,truefluid,control)
            prediction=head(initial,fluid,control)
            pieces=torch.stack((oracle-target,entry_only-oracle,prediction-entry_only),dim=-1)
            torch.testing.assert_close(pieces.sum(-1),prediction-target,rtol=0,atol=1e-12)
            for ix in range(len(initial)):
                components=pieces[ix,1:];covariance=components.T@components/len(components)
                response.append({'candidate':folder.name,'case_index':int(val['case'][use][ix]),
                    'cycle':int(val['cycle'][use][ix]),'component_names':['response_mismatch','entry_propagation','fluid_propagation'],
                    'signed_bias_K':components.mean(0).tolist(),'error_second_moment_K2':covariance.tolist(),
                    'mean_rmse_K':float((prediction[ix,1:]-target[ix,1:]).square().mean().sqrt())})
    grouped=[]
    for fold in (1,2):
        for part in ('fit','validation'):
            selected=[r for r in rows if r['fold']==fold and r['partition']==part]
            grouped.append({'fold':fold,'partition':part,'segments':len(selected),**{k:{
                'rms_K':float(np.sqrt(np.mean([r[k]**2 for r in selected]))),
                'max_abs_K':float(np.max([abs(r[k]) for r in selected]))} for k in
                ('entry_arithmetic_bias_K','entry_capacity_bias_K','increment_error_K','reference_mean_rmse_K','fluid_mean_rmse_K')}})
    result={'status':'completed_relaxation_failure_diagnosis','groups':grouped,'rows':rows,'response':response,
        'data_sha256':hashes,'development_sha256':sha(development/'summary.json'),
        'fresh_solver_calls':0,'historical_test_scored':False,'claim':'diagnostic internal data; not a prediction improvement'}
    write(output/'analysis.json',result)
    write(output/'manifest.json',{'source_sha256':sha(Path(__file__)),'analysis_sha256':sha(output/'analysis.json')})
    return {'status':result['status'],'groups':grouped,'response':response}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--development',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output,a.development)
    print(json.dumps(result,indent=2))
