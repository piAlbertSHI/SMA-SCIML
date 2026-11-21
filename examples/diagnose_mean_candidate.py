"""Check whether mean-branch failure is overshoot or a wrong correction direction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .multicycle_multistep import ROOT, write
from .train_hierarchical import training_data
from .stage_readout_study import FOLDS, cache
from .mean_readout_study import current_from_source, current_prediction, mean_prediction
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.modal_operator import ModalStateOperator
from SciML.models.v13.mean_thermal_readout import MeanThermalReadout


def run(development,output):
    output.mkdir(parents=True,exist_ok=False);summary=json.loads((development/'summary.json').read_text())
    candidate=summary['selected_candidate'];arrays,ids,hashes=training_data();rows=[];curves=[]
    for fold,(path,selected,validation) in enumerate(FOLDS):
        ref,_=current_from_source(ModalStateOperator.load(ROOT/path),arrays,selected)
        values=cache(ref.reference.backbone,arrays,validation)
        payload=torch.load(development/f'fold{fold+1}'/candidate['name']/'readout.pt',weights_only=True)
        assert payload['candidate']==candidate
        head=MeanThermalReadout(candidate['kind'],candidate['rank']);head.load_state_dict(payload['weights'])
        before=current_prediction(ref,values);after=mean_prediction(ref,head,values,candidate)
        for i in torch.nonzero(values['stage']==2).flatten():
            error=(before[i,1:]-values['truth'][i,0,1:]).numpy()
            delta=(after[i,1:]-before[i,1:]).numpy();power=float(np.mean(delta**2));cross=float(2*np.mean(error*delta))
            upper=max(0.,min(1.,-cross/power)) if power else 1.
            rows.append({'fold':fold+1,'case_index':int(values['case'][i]),'cycle':int(values['cycle'][i]),
                'delta_energy_K2':power,'error_delta_cross_K2':cross,'max_nonregressing_blend':upper,
                'before_mean_bias_K':float(error.mean()),'after_mean_bias_K':float((error+delta).mean()),
                'before_rmse_K':float(np.sqrt(np.mean(error**2))),'after_rmse_K':float(np.sqrt(np.mean((error+delta)**2)))})
            curves.append(np.stack((before[i].mean(-1).numpy(),after[i].mean(-1).numpy(),values['truth'][i,0].mean(-1).numpy())))
    np.savez_compressed(output/'curves.npz',mean=np.stack(curves))
    result={'status':'completed_mean_candidate_direction_diagnosis','candidate':candidate,'rows':rows,
        'common_nonregressing_blend_upper':min(r['max_nonregressing_blend'] for r in rows),
        'development_sha256':sha(development/'summary.json'),'data_sha256':hashes,'fresh_solver_calls':0}
    write(output/'diagnosis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--development',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=run(a.development,a.output)
    print(json.dumps(result,indent=2))
