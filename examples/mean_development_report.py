"""Visualize baseline mean-error decomposition and summarize rejected candidates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multicycle_multistep import write
from .research_gallery import Gallery
from SciML.models.v12.multistep_historical import sha


def run(output):
    g=Gallery(output);root='evidence/local_studies/sma_mean_readout'
    diagnosis=g.json(root+'/diagnosis/diagnosis.json');summary=g.json(root+'/summary.json')
    with np.load(g.source(root+'/diagnosis/mean_curves.npz'),allow_pickle=False) as z:curves=z['curves']
    rows=diagnosis['rows'];cases=sorted({r['case_id'] for r in rows if r['cycle']==3});tiles=[];records=[]
    for case in cases:
        fields=[]
        for cycle in (1,2,3):
            use=[i for i,r in enumerate(rows) if r['case_id']==case and r['cycle']==cycle and r['stage']==2]
            error=(curves[use,0]-curves[use,1]).mean(0);fields.append(error)
            records.append({'case_id':case,'cycle':cycle,'entry_bias_K':float(error[0]),
                            'future_bias_K':float(error[1:].mean()),'evolution_bias_K':float((error[1:]-error[0]).mean())})
        error=np.stack(fields);entry=np.repeat(error[:,:1],50,axis=1);tiles.append((entry,error-entry,error))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(12,5.6),layout='constrained');limit=max(np.abs(t).max() for case in tiles for t in case)
    for row,case in enumerate(tiles):
        for col,(values,label) in enumerate(zip(case,('Entry bias','Evolution bias','Total mean-temperature bias'))):
            ax=axes[row,col];im=ax.imshow(values,aspect='auto',origin='lower',extent=(0,1,.5,3.5),interpolation='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set(title=label,xlabel='Local exchange time',yticks=[1,2,3]);ax.set_ylabel(('A' if row==0 else 'B')+' | cycle')
    fig.colorbar(im,ax=list(axes.flat),shrink=.8,label='Signed mean-temperature error / K')
    fig.suptitle('Stage-2 mean-temperature error | retained HRO baseline\nThree-seed means; diagnostic decomposition only',fontsize=13)
    g.save(fig,'mean_error_decomposition','Retained baseline; no rejected candidate tested on historical evaluation cases')
    write(output/'analysis.json',{'status':summary['status'],'baseline_bias':records,
        'development_seconds':summary['total_recorded_development_seconds'],'candidate_promoted':False})
    write(output/'manifest.json',{'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        'input_sha256':g.sources,'generator_sha256':sha(Path(__file__))})
    return {'status':'completed_mean_development_report','candidate_promoted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
