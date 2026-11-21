"""Display fixed response gains and its failed segment gate on native fields."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .multicycle_multistep import ROOT,data,write
from .research_gallery import Gallery
from .stage_readout_study import SEEDS
from SciML.models.v12.multistep_historical import sha


def run(output):
    g=Gallery(output);root='evidence/local_studies/sma_relaxation_readout'
    summary=g.json(root+'/formal/summary.json');d,provenance=data();before=[];after=[]
    g.source('data/historical12_state.npz')
    for seed in SEEDS:
        with np.load(g.source(f'evidence/local_studies/sma_stage_readout/formal/s{seed}/test_predictions.npz')) as z:before.append(z['field'])
        with np.load(g.source(f'{root}/formal/s{seed}/test_predictions.npz')) as z:after.append(z['field'])
    a,b=np.stack(before),np.stack(after)
    case_a=next(case for case in d['case_ids'] if '_06_' in case and '_3c' in case)
    failed=[r for r in summary['groups'] if r['after']['mean']>r['before']['mean']]
    assert len(failed)==1
    selections=[(case_a,1,2),(case_a,2,2),(failed[0]['case_id'],failed[0]['cycle'],failed[0]['stage'])]
    labels=['Case A | cycle 1','Case A | cycle 2','Regressed segment | cycle 2']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,3,figsize=(12,8.1),layout='constrained')
    for row,((case,cycle,stage),label) in enumerate(zip(selections,labels)):
        index=np.flatnonzero((d['case_ids'][d['field_case_index']]==case)&(d['field_cycle']==cycle)&(d['field_stage']==stage))
        assert len(index)==1;i=int(index[0]);truth=d['field_truth'][i,0]
        errors=[np.sqrt(np.mean((p[:,i,0]-truth)**2,axis=0)) for p in (a,b)]
        im=axes[row,0].imshow(truth.T,origin='lower',aspect='auto',extent=(0,1,0,1),interpolation='nearest',cmap='viridis')
        fig.colorbar(im,ax=axes[row,0],label='Temperature / K',shrink=.9)
        limit=max(e.max() for e in errors)
        record=next(r for r in summary['groups'] if (r['case_id'],r['cycle'],r['stage'])==(case,cycle,stage))
        for col,(error,name) in enumerate(zip(errors,('before','after')),1):
            im=axes[row,col].imshow(error.T,origin='lower',aspect='auto',extent=(0,1,0,1),interpolation='nearest',cmap='magma',vmin=0,vmax=limit)
            stat=record[name]
            axes[row,col].set_title(('Retained HRO' if name=='before' else 'Thermal-response candidate')+f"\n{stat['mean']:.3f} +/- {stat['sd']:.3f} K")
        fig.colorbar(im,ax=[axes[row,1],axes[row,2]],label='Pointwise seed RMS error / K',shrink=.9)
        axes[row,0].set_title('Numerical reference')
        for ax in axes[row]:ax.set_xlabel('Local exchange time')
        axes[row,0].set_ylabel(label+'\nNormalized position')
    fig.suptitle('Fluid-conditioned mean response | stage 2\nFixed three-seed evaluation; one segment gate failed, baseline retained',fontsize=13)
    g.save(fig,'response_fields','All three seeds; paired error scales within each segment; native grid without smoothing')
    # Canonical text bytes for a newly generated vector asset; no numerical resampling.
    svg=output/'response_fields.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    write(output/'analysis.json',{'candidate_promoted':False,'overall_before_K':summary['overall_before_K'],
        'overall_after_K':summary['overall_after_K'],'failed_segment_means':failed,
        'individual_seed_regressions':summary['individual_seed_regressions'],
        'display_selection':'target case A first two cycles plus the single failed segment, disclosed after scoring',
        'data_sha256':provenance['snapshot_sha256']})
    write(output/'manifest.json',{'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        'input_sha256':g.sources,'generator_sha256':sha(Path(__file__))})
    return {'status':'completed_relaxation_diagnostic_figure','candidate_promoted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.output),indent=2))
