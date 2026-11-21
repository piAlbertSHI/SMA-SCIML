"""Native-grid field evidence and architecture for error-scaled thermal response."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

from .multicycle_multistep import ROOT,data,write
from .research_gallery import Gallery
from .stage_readout_study import SEEDS
from SciML.models.v12.multistep_historical import sha
from SciML.models.v13.risk_limited_readout import RiskLimitedRelaxationOperator


def architecture(g):
    fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,color='#eaf1f7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec='#385166',lw=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10.5,linespacing=1.55)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#385166','lw':1.6})
    box(.015,.68,.16,.19,'Initial state (515)\nControls (9)\nAutonomous rollout')
    box(.225,.68,.22,.19,'Frozen modal HRO\n3.11M parameters\nGRU memory (256)')
    box(.505,.68,.21,.19,'Stage-specific readout\nShared rank 12\nStage-2 shape rank 8')
    box(.785,.68,.2,.19,'SMA field output\nBaseline + 0.5 correction\nStage 2 only','#e6f3ed')
    for a,b in [((.175,.775),(.225,.775)),((.445,.775),(.505,.775)),((.716,.775),(.785,.775))]:arrow(a,b)
    box(.225,.29,.22,.22,'Predicted fluid forcing\nPositive response rate\nExact interval integration')
    box(.505,.29,.21,.22,'Correction shrinkage\nsign(delta) max(|delta|\n- error scale, 0)','#e6f3ed')
    box(.785,.29,.2,.22,'Train-only calibration\n1 fitted rate\n50 empirical RMS values','#fff1db')
    arrow((.335,.68),(.335,.51));arrow((.445,.4),(.505,.4));arrow((.61,.68),(.61,.51))
    arrow((.785,.4),(.716,.4));arrow((.716,.49),(.785,.68))
    ax.text(.5,.12,'Case deletion fits response rates only; the frozen backbone has seen the fit cases.\n'
        'No coverage guarantee. State recurrence, fluid fields, stage 4 and initial fields remain unchanged.',
        ha='center',va='center',fontsize=10,color='#384b5b',linespacing=1.5)
    fig.suptitle('Hierarchical Recurrent Operator | error-scaled thermal response',fontsize=15,y=.98)
    g.save(fig,'architecture','Frozen backbone and stage-specific spatial readout; train-only response calibration')


def run(output):
    g=Gallery(output);root='evidence/local_studies/sma_risk_limited_readout'
    summary=g.json(root+'/formal/summary.json');d,provenance=data();before=[];after=[];calibration=[]
    g.source('data/historical12_state.npz')
    for seed in SEEDS:
        with np.load(g.source(f'evidence/local_studies/sma_stage_readout/formal/s{seed}/test_predictions.npz')) as z:before.append(z['field'])
        with np.load(g.source(f'{root}/formal/s{seed}/test_predictions.npz')) as z:after.append(z['field'])
        model=RiskLimitedRelaxationOperator.load(g.source(f'{root}/formal/s{seed}/model.pt'))
        calibration.append({'seed':seed,'rate':float(model.response.rate),
            'error_scale_min_K':float(model.error_scale.min()),'error_scale_max_K':float(model.error_scale.max())})
    a,b=np.stack(before),np.stack(after)
    case_a=next(case for case in d['case_ids'] if '_06_' in case and '_3c' in case)
    case_b=next(case for case in d['case_ids'] if '_12_' in case and '_3c' in case)
    selections=[(case_a,1,2),(case_a,2,2),(case_a,3,2),(case_b,1,2)]
    labels=['Case A | cycle 1','Case A | cycle 2','Case A | cycle 3','Case B | cycle 1']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(4,3,figsize=(12,10.4),layout='constrained')
    for row,((case,cycle,stage),label) in enumerate(zip(selections,labels)):
        indices=np.flatnonzero((d['case_ids'][d['field_case_index']]==case)&(d['field_cycle']==cycle)&(d['field_stage']==stage))
        assert len(indices)==1;i=int(indices[0]);truth=d['field_truth'][i,0]
        errors=[np.sqrt(np.mean((p[:,i,0]-truth)**2,axis=0)) for p in (a,b)]
        im=axes[row,0].imshow(truth.T,origin='lower',aspect='auto',extent=(0,1,0,1),interpolation='nearest',cmap='viridis')
        fig.colorbar(im,ax=axes[row,0],label='Temperature / K',shrink=.9)
        limit=max(e.max() for e in errors)
        record=next(r for r in summary['groups'] if (r['case_id'],r['cycle'],r['stage'])==(case,cycle,stage))
        for col,(error,name) in enumerate(zip(errors,('before','after')),1):
            im=axes[row,col].imshow(error.T,origin='lower',aspect='auto',extent=(0,1,0,1),interpolation='nearest',cmap='magma',vmin=0,vmax=limit)
            stat=record[name]
            axes[row,col].set_title(('Stage-spatial HRO' if name=='before' else 'Error-scaled response HRO')+f"\n{stat['mean']:.3f} +/- {stat['sd']:.3f} K")
        fig.colorbar(im,ax=[axes[row,1],axes[row,2]],label='Pointwise seed RMS error / K',shrink=.9)
        axes[row,0].set_title('Numerical reference')
        for ax in axes[row]:ax.set_xlabel('Normalized exchange time')
        axes[row,0].set_ylabel(label+'\nNormalized position')
    fig.suptitle('Stage-2 SMA temperature fields | three fixed seeds\n'
        'Historical development regression; paired error scales within each segment',fontsize=13)
    g.save(fig,'four_segments','Four original target segments; native grids, no smoothing; all three seeds')
    architecture(g)
    for path in output.glob('*.svg'):
        path.write_text('\n'.join(v.rstrip() for v in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    worst_before=max(summary['groups'],key=lambda r:r['before']['mean'])
    worst_after=max(summary['groups'],key=lambda r:r['after']['mean'])
    regression_terms=[]
    for row in summary['individual_seed_regressions']:
        s=list(SEEDS).index(row['seed'])
        i=int(np.flatnonzero((d['case_ids'][d['field_case_index']]==row['case_id'])&
            (d['field_cycle']==row['cycle'])&(d['field_stage']==row['stage']))[0])
        error=a[s,i,0,1:]-d['field_truth'][i,0,1:];delta=b[s,i,0,1:]-a[s,i,0,1:]
        square=float(np.mean(delta**2));cross=float(2*np.mean(error*delta))
        regression_terms.append({**row,'correction_square_K2':square,'cross_K2':cross,'net_mse_K2':square+cross})
    write(output/'analysis.json',{'candidate_promoted':True,'overall_before_K':summary['overall_before_K'],
        'overall_after_K':summary['overall_after_K'],'calibration':calibration,
        'worst_segment_before':worst_before,'worst_segment_after':worst_after,
        'individual_seed_regressions':regression_terms,
        'display_selection':'four original target segments, all seeds; exceptions disclosed in report',
        'data_sha256':provenance['snapshot_sha256']})
    write(output/'manifest.json',{'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},
        'input_sha256':g.sources,'generator_sha256':sha(Path(__file__))})
    return {'status':'completed_risk_limited_figures','figures':len(g.figures),'calibration':calibration,
            'individual_seed_regressions':regression_terms}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(1);print(json.dumps(run(a.output),indent=2))
