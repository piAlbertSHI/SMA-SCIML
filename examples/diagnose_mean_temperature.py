"""Separate stage-entry and thermal-evolution errors in native-grid mean temperature."""
import argparse
import json
from pathlib import Path

import numpy as np

from .multicycle_multistep import ROOT, data, write
from SciML.models.v12.multistep_historical import sha


def decompose(prediction,truth):
    pred=prediction.mean(-1);target=truth.mean(-1)
    entry=pred[0]-target[0];error=pred[1:]-target[1:];evolution=error-entry
    mse=float(np.mean(error**2));parts=[float(entry**2),float(np.mean(evolution**2)),float(2*entry*np.mean(evolution))]
    np.testing.assert_allclose(mse,sum(parts),rtol=1e-10,atol=1e-12)
    return {'entry_bias_K':float(entry),'future_bias_K':float(error.mean()),
        'evolution_bias_K':float(evolution.mean()),'endpoint_bias_K':float(pred[-1]-target[-1]),
        'mean_temperature_rmse_K':float(np.sqrt(mse)),
        'entry_mse_K2':parts[0],'evolution_mse_K2':parts[1],'cross_mse_K2':parts[2]}


def run(output):
    output.mkdir(parents=True,exist_ok=False);d,provenance=data();rows=[];curves=[];hashes={}
    for seed in (20261121,20261122,20261123):
        path=ROOT/f'evidence/local_studies/sma_stage_readout/formal/s{seed}/test_predictions.npz'
        hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        with np.load(path,allow_pickle=False) as z:field=z['field']
        for i in range(len(d['field_stage'])):
            ci=int(d['field_case_index'][i]);truth=d['field_truth'][i,0]
            rows.append({'seed':seed,'case_id':str(d['case_ids'][ci]),'cycle':int(d['field_cycle'][i]),
                'stage':int(d['field_stage'][i]),**decompose(field[i,0],truth)})
            curves.append(np.stack((field[i,0].mean(-1),truth.mean(-1),field[i,1].mean(-1),d['field_truth'][i,1].mean(-1))))
    np.savez_compressed(output/'mean_curves.npz',curves=np.stack(curves))
    train=d['control'][d['split']=='train']
    report={'status':'completed_mean_temperature_diagnosis','rows':rows,'checkpoint_sha256':hashes,
        'data_sha256':provenance['snapshot_sha256'],'source_sha256':sha(Path(__file__)),
        'training_control_min':train.min(0).tolist(),'training_control_max':train.max(0).tolist(),
        'curve_channels':['predicted_SMA','reference_SMA','predicted_fluid','reference_fluid'],
        'fresh_solver_calls':0,'scope':'historical development diagnosis, not candidate selection'}
    write(output/'diagnosis.json',report)
    grouped=[]
    for case,cycle in sorted({(r['case_id'],r['cycle']) for r in rows if r['stage']==2}):
        use=[r for r in rows if r['case_id']==case and r['cycle']==cycle and r['stage']==2]
        grouped.append({'case_id':case,'cycle':cycle,**{k:float(np.mean([r[k] for r in use])) for k in
            ('entry_bias_K','evolution_bias_K','future_bias_K','mean_temperature_rmse_K')}})
    return {'status':report['status'],'stage2':grouped}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
