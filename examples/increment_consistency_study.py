"""One fixed, parameter-free heat-increment projection on internal folds."""
import argparse
import json
from pathlib import Path
import time

import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from .mean_readout_study import current_from_source, current_prediction, cache
from .stage_readout_study import FOLDS, metrics, gate
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.increment_consistency import project_heat_increment


def run(output):
    output.mkdir(parents=True,exist_ok=False);arrays,ids,hashes=training_data()
    sources=['examples/increment_consistency_study.py','SciML/models/v13/increment_consistency.py',
        'SciML/models/v13/thermal_geometry.py','examples/stage_readout_study.py',
        'examples/mean_readout_study.py']
    write(output/'protocol.json',{'candidate':'stage2 linear minimum-gradient heat-increment projection',
        'parameters_fitted':0,'data_sha256':hashes,
        'gate':'stage2 mean RMSE strictly decreases and every stage2 validation segment nonregresses',
        'folds':[{'fit_ids':[ids[i] for i in fit],'validation_ids':[ids[i] for i in val],
                  'checkpoint':p,'checkpoint_sha256':sha(ROOT/p)} for p,fit,val in FOLDS],
        'source_sha256':{p:sha(ROOT/p) for p in sources},
        'motivation':'training-only capacity increment identity; absolute endpoint identity is false',
        'selection':'one fixed candidate; no hyperparameter search, no historical test evaluation',
        'fresh_solver_calls':0,'neural_training_updates':0})
    for p in sources:
        dest=output/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/p).read_bytes())
    start=time.perf_counter();records=[]
    for fold,(path,selected,validation) in enumerate(FOLDS,1):
        source=ModalStateOperator.load(ROOT/path)
        assert {s.split(':')[0] for s in source.training_ids}=={ids[i] for i in selected}
        reference,_=current_from_source(source,arrays,selected);val=cache(source,arrays,validation)
        before=current_prediction(reference,val);after=before.clone();use=val['stage']==2
        corrected,mismatch=project_heat_increment(before[use],val['entry_state'][use],val['exit_state'][use])
        after[use]=corrected
        old,new=metrics(before,val),metrics(after,val)
        records.append({'fold':fold,'reference':old,'candidate':new,'passed':gate(new,old),
            'endpoint_correction_K':mismatch.tolist()})
        torch.save({'before':before,'after':after,'truth':val['truth'][:,0],
            'entry_state':val['entry_state'],'exit_state':val['exit_state'],
            'case':val['case'],'cycle':val['cycle'],'stage':val['stage']},output/f'fold{fold}.pt')
    result={'status':'completed_increment_consistency_development','folds':records,
        'passed_both_folds':all(r['passed'] for r in records),'seconds':time.perf_counter()-start,
        'historical_test_scored':False,'fresh_solver_calls':0,'neural_training_updates':0}
    write(output/'summary.json',result)
    write(output/'manifest.json',{'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.output)
    print(json.dumps(result,indent=2))
