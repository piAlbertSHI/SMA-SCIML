"""Source-bound error maps and complete segment comparison for stage readout."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .research_gallery import Gallery
from .multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha


def run(output):
    gallery=Gallery(output)
    with np.load(gallery.source('data/historical12_state.npz'),allow_pickle=False) as z:d={k:z[k] for k in z.files}
    selected=gallery.json('assets/sma_readout_local/analysis.json')['regressions_vs_modal']
    rows=[];overall=[];predictions={}
    for seed in (20261121,20261122,20261123):
        models={}
        for name,folder in [('modal','evidence/experiments/hro_modal/formal'),
            ('shared','evidence/local_studies/sma_readout/formal'),('staged','evidence/local_studies/sma_stage_readout/formal')]:
            with np.load(gallery.source(f'{folder}/s{seed}/test_predictions.npz'),allow_pickle=False) as z:models[name]=z['field'].copy()
            error=models[name][:,0,1:]-d['field_truth'][:,0,1:]
            overall.append({'seed':seed,'model':name,'sma_rmse_K':float(np.sqrt(np.mean(error**2)))})
        predictions[seed]=models
        for i in range(len(d['field_stage'])):
            record={'case_id':str(d['case_ids'][d['field_case_index'][i]]),'cycle':int(d['field_cycle'][i]),
                    'stage':int(d['field_stage'][i]),'seed':seed}
            for name,values in models.items():
                error=values[i,0,1:]-d['field_truth'][i,0,1:];mean=error.mean(-1,keepdims=True)
                record[name]={'rmse_K':float(np.sqrt(np.mean(error**2))),
                    'spatial_mean_mse_K2':float(np.mean(mean**2)),
                    'spatial_centered_mse_K2':float(np.mean((error-mean)**2))}
            rows.append(record)
    comparisons=[]
    for case,cycle,stage in sorted({(r['case_id'],r['cycle'],r['stage']) for r in rows}):
        values=[r for r in rows if (r['case_id'],r['cycle'],r['stage'])==(case,cycle,stage)]
        record={'case_id':case,'cycle':cycle,'stage':stage}
        for name in ('modal','shared','staged'):
            record[name+'_metrics']={k:float(np.mean([v[name][k] for v in values])) for k in values[0][name]}
        comparisons.append(record)
    targets=[r for r in comparisons if any(all(r[k]==s[k] for k in ('case_id','cycle','stage')) for s in selected)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(4,3,figsize=(12,12),layout='constrained')
    for row,target in enumerate(targets):
        ci=list(d['case_ids']).index(target['case_id'])
        index=np.flatnonzero((d['field_case_index']==ci)&(d['field_cycle']==target['cycle'])&(d['field_stage']==2))[0]
        errors=[predictions[20261121][name][index,0]-d['field_truth'][index,0] for name in ('modal','shared','staged')]
        limit=max(float(np.abs(e).max()) for e in errors)
        for col,(error,label) in enumerate(zip(errors,('Modal HRO','Shared residual readout','Stage-specific spatial readout'))):
            ax=axes[row,col];im=ax.imshow(error,origin='lower',aspect='auto',extent=(0,1,0,1),
                interpolation='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_title(label,fontsize=11);ax.set_xlabel('Position x / L')
            if col==0:ax.set_ylabel(('A' if '_06_' in target['case_id'] else 'B')+f" | cycle {target['cycle']} | stage 2\nLocal exchange time")
        fig.colorbar(im,ax=list(axes[row]),shrink=.8,label='Signed error / K')
    fig.suptitle('SMA temperature-field error | four diagnosed segments\nFixed seed 20261121; shared color scale within each row',fontsize=14)
    gallery.save(fig,'four_segments','Signed errors on native grids; first preset seed; no smoothing')
    regressions=[r for r in comparisons if r['staged_metrics']['rmse_K']>r['shared_metrics']['rmse_K']]
    result={'status':'completed_stage_readout_report','four_segments':targets,'all_segments':comparisons,
        'regressions_vs_shared':regressions,'seed_segment_regressions_vs_shared':[
            r for r in rows if r['staged']['rmse_K']>r['shared']['rmse_K']],
        'all_test_mean_rmse_K':{name:float(np.mean([r['sma_rmse_K'] for r in overall if r['model']==name])) for name in ('modal','shared','staged')},
        'all_test_by_seed':overall,'fresh_solver_calls':0}
    write(output/'analysis.json',result)
    write(output/'manifest.json',{'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        'input_sha256':gallery.sources,'generator_sha256':sha(Path(__file__))})
    return {'status':result['status'],'all_test_mean_rmse_K':result['all_test_mean_rmse_K'],
            'mean_segment_regressions':len(regressions),'seed_segment_regressions':len(result['seed_segment_regressions_vs_shared'])}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
