"""Decompose the four previously regressed segments without selecting a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from .multicycle_multistep import ROOT, data, write
from .diagnose_sma_rollout import decompose
from SciML.models.v12.multistep_historical import sha


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'assets/sma_readout_local/analysis.json'
    selected=json.loads(previous.read_text())['regressions_vs_modal']
    d,provenance=data();records=[];hashes={previous.relative_to(ROOT).as_posix():sha(previous)}
    for seed in (20261121,20261122,20261123):
        paths=[ROOT/f'evidence/experiments/hro_modal/formal/s{seed}/test_predictions.npz',
               ROOT/f'evidence/local_studies/sma_readout/formal/s{seed}/test_predictions.npz']
        values=[]
        for p in paths:
            hashes[p.relative_to(ROOT).as_posix()]=sha(p)
            with np.load(p,allow_pickle=False) as z:values.append(z['field'].copy())
        base,corrected=values
        for row in selected:
            ci=list(d['case_ids']).index(row['case_id'])
            i=np.flatnonzero((d['field_case_index']==ci)&(d['field_cycle']==row['cycle'])&(d['field_stage']==row['stage']))[0]
            error=base[i,0,1:]-d['field_truth'][i,0,1:]
            delta=corrected[i,0,1:]-base[i,0,1:]
            power=float(np.mean(delta**2));alignment=float(2*np.mean(error*delta))
            increase=float(np.mean((error+delta)**2)-np.mean(error**2))
            np.testing.assert_allclose(power+alignment,increase,rtol=1e-10,atol=1e-12)
            records.append({'case_id':row['case_id'],'cycle':row['cycle'],'stage':row['stage'],'seed':seed,
                'modal':decompose(base[i],d['field_truth'][i]),'shared':decompose(corrected[i],d['field_truth'][i]),
                'correction_energy_K2':power,'error_correction_cross_K2':alignment,'mse_change_K2':increase,
                'correction_mean_K':float(delta.mean()),
                'oracle_optimal_scalar_diagnostic_only':float(-np.mean(error*delta)/power) if power else None})
    result={'status':'diagnosed_four_stage2_segments','purpose':'historical failure diagnosis; not selection or calibration',
        'records':records,'input_sha256':hashes,'data_sha256':provenance['snapshot_sha256'],
        'source_sha256':sha(Path(__file__)),'fresh_solver_calls':0}
    write(output/'diagnosis.json',result)
    compact=[]
    for row in selected:
        use=[r for r in records if all(r[k]==row[k] for k in ('case_id','cycle','stage'))]
        compact.append({**row,**{k:float(np.mean([r[k] for r in use])) for k in
            ('correction_energy_K2','error_correction_cross_K2','mse_change_K2','correction_mean_K')}})
    return {'status':result['status'],'segments':compact}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(run(args.output),indent=2))
