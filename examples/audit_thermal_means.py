"""Audit mean and state/field semantics without training or changing a metric."""
import argparse
import json
from pathlib import Path

import numpy as np

from .multicycle_multistep import ROOT, data, write
from .train_hierarchical import training_data
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.thermal_geometry import strip_weights, radial_weights, weighted_mean


def describe(x):
    x=np.asarray(x)
    return {'max_abs_K':float(np.abs(x).max()), 'rms_K':float(np.sqrt(np.mean(x*x))),
            'signed_mean_K':float(x.mean())}


def reconstructed(state):
    r=state[...,102:153]
    return np.concatenate((r[...,1:][...,::-1],r),axis=-1)+state[...,414:]


def run(output, physics_root=None):
    output.mkdir(parents=True,exist_ok=False)
    arrays,ids,hashes=training_data();a={k:v.numpy() for k,v in arrays.items()}
    ws,wr=strip_weights(),radial_weights();rows=[]
    for i,case in enumerate(ids):
        for j,step in enumerate((1,3,6,8)):
            before=a['states'][i,step-1];after=a['states'][i,step];truth=a['fields'][i,j,0]
            entry=reconstructed(before);end=reconstructed(after)
            np.testing.assert_allclose(entry,truth[0],rtol=0,atol=1e-12)
            strip_delta=float(weighted_mean(truth[-1]-truth[0],ws))
            radial_delta=float(weighted_mean(after[102:153]-before[102:153],wr))
            rows.append({'case_id':case,'cycle':j//2+1,'stage':2 if j%2==0 else 4,
                'field_arithmetic_minus_capacity_mean':describe(truth.mean(-1)-weighted_mean(truth,ws)),
                'state_reconstructed_endpoint_error':describe(end-truth[-1]),
                'weighted_endpoint_mismatch_K':float(weighted_mean(end-truth[-1],ws)),
                'strip_delta_K':strip_delta,'radial_delta_K':radial_delta,
                'radial_minus_strip_delta_K':radial_delta-strip_delta})
    d,provenance=data();baseline=[];inputs={}
    for seed in (20261121,20261122,20261123):
        p=ROOT/f'evidence/local_studies/sma_stage_readout/formal/s{seed}/test_predictions.npz'
        inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        with np.load(p,allow_pickle=False) as z:pred=z['field'][:,0]
        for j in range(len(pred)):
            error=pred[j]-d['field_truth'][j,0];u=error.mean(-1);w=weighted_mean(error,ws)
            baseline.append({'seed':seed,'case_id':str(d['case_ids'][d['field_case_index'][j]]),
                'cycle':int(d['field_cycle'][j]),'stage':int(d['field_stage'][j]),
                'arithmetic_future_bias_K':float(u[1:].mean()),'capacity_future_bias_K':float(w[1:].mean()),
                'arithmetic_mean_rmse_K':float(np.sqrt(np.mean(u[1:]**2))),
                'capacity_mean_rmse_K':float(np.sqrt(np.mean(w[1:]**2))),
                'bias_measure_difference_K':float((w[1:]-u[1:]).mean())})
    source_bindings={}
    if physics_root:
        bound=json.loads((ROOT/'evidence/hro_legacy_physics.json').read_text())['source_bindings']
        names=['Build_Diameter_Strip_Geometry.m','Map_Radial51_To_Diameter101.m',
               'Compress_Diameter101_To_Radial51.m','Run_Simulink_Exchange_Hook.m',
               'Build_Simulink_Exchange_Params.m']
        for name in names:
            source=physics_root/name;digest=sha(source)
            instrumented=name in bound
            if instrumented:assert digest==bound[name]['sha256'],name
            dest=output/'inspected_physics_source'/name;dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(source.read_bytes())
            source_bindings[name]={'sha256':digest,'historically_instrumented':instrumented,
                'evidence_scope':'exact historical instrumentation match' if instrumented else
                    'inspected helper from recovered bundle; no per-run helper hash available'}
    group=[]
    for case,cycle in sorted({(r['case_id'],r['cycle']) for r in baseline if r['stage']==2 and '_3c' in r['case_id']}):
        use=[r for r in baseline if r['case_id']==case and r['cycle']==cycle and r['stage']==2]
        group.append({'case_id':case,'cycle':cycle,**{k:float(np.mean([r[k] for r in use])) for k in
            ('arithmetic_future_bias_K','capacity_future_bias_K','bias_measure_difference_K')}})
    summary={'status':'completed_thermal_mean_semantics_audit','training_cases':8,'training_exchanges':32,
        'training_stage_summaries':{str(s):{
            'radial_minus_strip_delta':describe([r['radial_minus_strip_delta_K'] for r in rows if r['stage']==s]),
            'weighted_endpoint_mismatch':describe([r['weighted_endpoint_mismatch_K'] for r in rows if r['stage']==s])} for s in (2,4)},
        'baseline_stage2_three_cycle_bias':group,'benchmark_metric_changed':False,'candidate_promoted':False,
        'fresh_solver_calls':0,'neural_training_updates':0,
        'scope':'weighted means diagnose the constant-cp proxy; neither heat-flux closure nor endpoint identity is assumed'}
    write(output/'analysis.json',{'summary':summary,'training_rows':rows,'baseline_rows':baseline,
        'weights':{'strip':ws.tolist(),'radial':wr.tolist()},'data_sha256':hashes,
        'state_sha256':provenance['snapshot_sha256'],'prediction_sha256':inputs,'physics_source':source_bindings})
    write(output/'manifest.json',{'artifacts':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()},
        'source_sha256':{p:sha(ROOT/p) for p in ('examples/audit_thermal_means.py','SciML/models/v13/thermal_geometry.py')}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--physics-root',type=Path);a=p.parse_args()
    print(json.dumps(run(a.output,a.physics_root),indent=2))
