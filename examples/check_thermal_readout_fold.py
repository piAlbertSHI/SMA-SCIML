"""A second case fold tests the already selected correction, without reselection."""
import argparse
import json
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from .train_hierarchical import training_data
from .modal_hierarchical import fit_prior, fit
from .refine_modal import refine
from .adapt_sma import field_metrics
from .fit_thermal_readout import fit_readout
from .multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    arrays,ids,hashes=training_data();selected=[0,2,3,4,6,7];validation=[1,5]
    protocol={'purpose':'second-fold check of preselected residual correction; no method selection',
        'training_case_ids':[ids[i] for i in selected],'validation_case_ids':[ids[i] for i in validation],
        'data_sha256':hashes,'mode':'residual','rank':12,'alpha':1.,'seed':20261124,
        'joint_updates':160,'decoder_updates':320,'gate':'same: SMA mean and c2s4 mean improve; per-case SMA <=1.05*baseline; fluid exact',
        'historical_test_scored':False,'fresh_solver_calls':0,
        'source_sha256':{p:sha(ROOT/p) for p in ['examples/check_thermal_readout_fold.py','examples/fit_thermal_readout.py',
            'SciML/models/v13/thermal_readout.py','examples/modal_hierarchical.py','examples/refine_modal.py']}}
    write(output/'protocol.json',protocol)
    for rel in protocol['source_sha256']:
        p=output/'source'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
    base=fit_prior(arrays,ids,selected)
    model=fit(output/'joint',arrays,ids,selected,base,'modal_dense',20261124,160)
    refine(model,arrays,selected,output/'refined',20261124)
    baseline=field_metrics(model,arrays,validation)
    corrected,info=fit_readout(model,arrays,selected,'residual');corrected.save(output/'corrected.pt')
    after=field_metrics(corrected,arrays,validation)
    passed=after['mean']['sma_rmse_K']<baseline['mean']['sma_rmse_K'] and after['mean']['c2s4_sma_rmse_K']<baseline['mean']['c2s4_sma_rmse_K']
    passed=passed and all(a['sma_rmse_K']<=1.05*b['sma_rmse_K'] and a['fluid_rmse_K']==b['fluid_rmse_K'] for a,b in zip(after['per_case'],baseline['per_case']))
    result={'status':'completed_second_fold_check','passed':passed,'baseline':baseline,'corrected':after,'fit':info,
        'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',result)
    return {'status':result['status'],'passed':passed,'baseline':baseline['mean'],'corrected':after['mean']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output)
    print(json.dumps(result,indent=2))
