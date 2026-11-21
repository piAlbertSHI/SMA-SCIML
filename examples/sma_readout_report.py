"""Create local diagnostics from saved arrays; no fitting or solver invocation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multicycle_multistep import ROOT, write
from .diagnose_sma_rollout import decompose
from .research_gallery import Gallery
from SciML.models.v12.multistep_historical import sha


def run(output):
    g=Gallery(output);study='evidence/local_studies/sma_readout'
    with np.load(g.source('data/historical12_state.npz'),allow_pickle=False) as z:d={k:z[k] for k in z.files}
    forecasts={};rows=[];overall=[]
    for seed in [20261121,20261122,20261123]:
        models={}
        for name,folder in [('coordinate','evidence/experiments/v13_formal_v1'),
                            ('modal','evidence/experiments/hro_modal/formal'),('corrected',study+'/formal')]:
            with np.load(g.source(f'{folder}/s{seed}/test_predictions.npz'),allow_pickle=False) as z:models[name]=z['field'].copy()
            e=models[name][:,0,1:]-d['field_truth'][:,0,1:]
            overall.append({'model':name,'seed':seed,'all_test_sma_rmse_K':float(np.sqrt(np.mean(e*e)))})
        forecasts[seed]=models
        for index in range(len(d['field_case_index'])):
            ci=int(d['field_case_index'][index]);cycle=int(d['field_cycle'][index]);stage=int(d['field_stage'][index])
            for name,field in models.items():
                rows.append({'case_id':str(d['case_ids'][ci]),'cycle':cycle,'stage':stage,'model':name,'seed':seed,
                             **decompose(field[index],d['field_truth'][index])})
    fixed=np.flatnonzero((d['split']=='test')&(d['cycles']==3));comparisons=[];regressions=[]
    for ci in np.flatnonzero(d['split']=='test'):
        case=str(d['case_ids'][ci])
        for c,s in sorted({(r['cycle'],r['stage']) for r in rows if r['case_id']==case}):
            metrics={name:float(np.mean([r['rmse_K'][0] for r in rows if r['case_id']==case and r['cycle']==c and r['stage']==s and r['model']==name]))
                     for name in ['coordinate','modal','corrected']}
            row={'case_id':case,'cycle':c,'stage':s,**metrics};comparisons.append(row)
            if metrics['corrected']>metrics['modal']:regressions.append(row)
    selected=[]
    for ci in fixed:
        case=str(d['case_ids'][ci]);record={'case_id':case}
        for name in ['coordinate','modal','corrected']:
            selected_rows=[r for r in rows if r['case_id']==case and r['cycle']==3 and r['stage']==4 and r['model']==name]
            record[name]={k:np.mean([r[k] for r in selected_rows],axis=0).tolist() for k in selected_rows[0]
                          if k not in ('case_id','cycle','stage','model','seed')}
        selected.append(record)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#f5f8fc',
                         'axes.facecolor':'#edf2f7','savefig.facecolor':'#f5f8fc','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,5,figsize=(17,7.4),layout='constrained')
    for row,ci in enumerate(fixed):
        index=np.flatnonzero((d['field_case_index']==ci)&(d['field_cycle']==3)&(d['field_stage']==4))[0]
        truth=d['field_truth'][index,0];before=forecasts[20261121]['modal'][index,0];after=forecasts[20261121]['corrected'][index,0]
        low=min(v.min() for v in [truth,before,after]);high=max(v.max() for v in [truth,before,after])
        error_max=max(np.abs(before-truth).max(),np.abs(after-truth).max());images=[]
        for col,(values,title) in enumerate(zip([truth,before,after,np.abs(before-truth),np.abs(after-truth)],
                ['Numerical reference','Modal HRO','HRO + residual readout','Modal HRO | error','Corrected | error'])):
            ax=axes[row,col];im=ax.imshow(values,origin='lower',aspect='auto',extent=(0,1,0,1),
                cmap='magma' if col<3 else 'inferno',vmin=low if col<3 else 0,vmax=high if col<3 else error_max)
            images.append(im);ax.set_title(title,fontsize=11);ax.set_xlabel('Position x / L')
            if col==0:ax.set_ylabel(('Configuration A' if row==0 else 'Configuration B')+'\nLocal exchange time')
        fig.colorbar(images[0],ax=list(axes[row,:3]),shrink=.8,label='SMA temperature / K')
        fig.colorbar(images[3],ax=list(axes[row,3:]),shrink=.8,label='Absolute error / K')
    fig.suptitle('Third-cycle SMA temperature fields | stage 4\nFirst preset seed; shared scales within each configuration; no smoothing',fontsize=15)
    g.save(fig,'third_cycle_fields','Both three-cycle cases, seed 20261121; unsmoothed autonomous predictions')
    report={'status':'completed_local_SMA_error_report','third_cycle_cases':selected,'all_segments':comparisons,
            'regressions_vs_modal':regressions,'overall':overall,
            'overall_mean':{name:float(np.mean([r['all_test_sma_rmse_K'] for r in overall if r['model']==name])) for name in forecasts[20261121]},
            'input_sha256':g.sources,'generator_sha256':sha(Path(__file__)),'fresh_solver_calls':0}
    write(output/'analysis.json',report)
    write(output/'manifest.json',{'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
                                'generator_sha256':sha(Path(__file__)),'input_sha256':g.sources})
    return {'status':report['status'],'overall_sma_K':report['overall_mean'],'regressed_exchange_segments':len(regressions),
            'total_exchange_segments':len(comparisons),'third_cycle':[{**{'case':r['case_id']},**{m:r[m]['rmse_K'][0] for m in ['coordinate','modal','corrected']}} for r in selected]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
