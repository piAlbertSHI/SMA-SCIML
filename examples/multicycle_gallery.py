"""Plot all three fixed-budget variants; figures never train or select models."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .research_gallery import Gallery
from .multicycle_multistep import SEEDS, MODES

EXPERIMENT='evidence/experiments/multicycle_multistep_v1'
LABELS=['Archived one-step','One-step continuation','Multi-step continuation']
COLORS=['#155e91','#dd8133','#258879']
SPECS=[('state','sma_K','SMA boundary / K'),('field','sma_K','SMA future field / K'),('field','htf_K','Fluid future field / K')]


def horizons(g, summary):
    a=summary['aggregate']['fixed_three_cycle']
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    for ax,(category,metric,title) in zip(axes,SPECS):
        for mode,label,color,style in zip(MODES,LABELS,COLORS,['o-','s--','^:']):
            rows=[a[mode][category][f'c{c}s4'][metric] for c in (1,2,3)]
            y=np.array([r['mean'] for r in rows]);sd=np.array([r['sd'] for r in rows])
            ax.plot([1,2,3],y,style,color=color,label=label,markersize=5)
            ax.fill_between([1,2,3],np.maximum(0,y-sd),y+sd,color=color,alpha=.08)
        ax.axvspan(2.5,3.3,color='#9099aa',alpha=.08)
        ax.set(title=title,xticks=[1,2,3],xlim=(.85,3.15),ylim=(0,None),xlabel='Cycle (stage 4)',ylabel='Test RMSE')
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Does two-cycle training improve the unseen third cycle?\nSame two test cases at every horizon | mean ± sample SD across three seeds')
    g.save(fig,'multicycle_horizons','Fixed two-case cohort at all horizons; future fields exclude initial frame; SD over seeds, not confidence intervals')


def diagnosis(g, summary, reports):
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
    ax=axes[0]
    values=np.array([[r['modes']['multistep']['training_history'][0]['training_loss_before_update'],
        r['modes']['single_step_continue']['training_final_loss']['autonomous'],
        r['modes']['multistep']['training_final_loss']['autonomous']] for r in reports])
    ax.bar(range(3),values.mean(0),color=COLORS,alpha=.75,width=.6)
    for i,row in enumerate(values):ax.plot(np.arange(3)+.06*(i-1),row,'o--',color='#303948',alpha=.65,markersize=4)
    ax.set(yscale='log',xticks=range(3),xticklabels=['Archived','One-step\ncontinued','Multi-step\ncontinued'],
           ylabel='Normalized autonomous loss (log scale)',title='Training fit improves\nAll variants scored on the same 8 × 9 chain')
    ax.grid(axis='y',alpha=.15)
    a=summary['aggregate']['fixed_three_cycle'];items=[('state',b) for b in a[MODES[0]]['state']['c3s4'] if b!='case_count']
    items += [('field','sma_K'),('field','htf_K')]
    names=['SMA state','Hold-fluid state','Cooldown-fluid state','Side temperatures','Phase fraction','Radial displacement',
           'Transverse displacement','Directional memory','SMA future field','Fluid future field']
    changes=np.array([[100*(a[m][cat]['c3s4'][key]['mean']/a[MODES[0]][cat]['c3s4'][key]['mean']-1)
                       for m in MODES[1:]] for cat,key in items])
    ax=axes[1];limit=max(1,float(np.abs(changes).max()))
    im=ax.imshow(changes,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set(xticks=[0,1],xticklabels=['One-step continued','Multi-step continued'],yticks=range(len(items)),yticklabels=names,
           title='Third-cycle error changes are small\n% change versus archived model; negative is better')
    ax.tick_params(length=0)
    for i in range(len(items)):
        for j in range(2):ax.text(j,i,f'{changes[i,j]:+.2f}%',ha='center',va='center',fontsize=9,
                                 color='white' if abs(changes[i,j])>.65*limit else '#17202b')
    fig.colorbar(im,ax=ax,shrink=.75,label='RMSE change / %')
    fig.suptitle('Better two-cycle training fit does not establish better third-cycle generalization')
    g.save(fig,'multicycle_diagnosis','Training loss pooled over eight cases/nine transitions; test heatmap ratios of seed-mean RMSE on fixed two cases; all eight state blocks plus both fields')
    return {'training_autonomous_loss_by_seed':{str(s):dict(zip(MODES,map(float,row))) for s,row in zip(SEEDS,values)},
            'training_autonomous_loss_seed_mean':dict(zip(MODES,map(float,values.mean(0)))),
            'c3s4_percent_change_vs_archived':{cat+'_'+key:dict(zip(MODES[1:],map(float,row))) for (cat,key),row in zip(items,changes)}}


def cases(g, reports, protocol):
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    case_ids=protocol['fixed_three_cycle_cases']
    for row,case_id in enumerate(case_ids):
        material='recA' if '_reca_' in case_id else 'recAs'
        for col,(category,metric,title) in enumerate(SPECS):
            ax=axes[row,col]
            values=np.array([[r['modes'][m]['per_three_cycle_case'][case_id][category]['c3s4'][metric] for m in MODES] for r in reports])
            ax.bar(range(3),values.mean(0),color=COLORS,alpha=.72,width=.65)
            for i,v in enumerate(values):ax.plot(np.arange(3)+.05*(i-1),v,'o--',lw=.8,ms=4,color='#303948',alpha=.6)
            for i,v in enumerate(values.mean(0)):ax.text(i,values[:,i].max()+.035*values.max(),f'{v:.3f}',ha='center',fontsize=9)
            ax.set(xticks=range(3),xticklabels=['Archived','One-step\ncontinued','Multi-step\ncontinued'],
                   ylabel='RMSE / K',ylim=(0,values.max()*1.24),title=f'{material} | {title}')
            ax.grid(axis='y',alpha=.15)
    fig.suptitle('Third-cycle results for both held-out cases\nBars: three-seed mean | dashed connections: paired seeds | no best-seed selection')
    g.save(fig,'multicycle_cases','Both third-cycle test cases and all seeds; boundary RMSE and future-field RMSE are distinct metrics; each panel has its own y scale')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f7f9fc','axes.facecolor':'white','savefig.facecolor':'#f7f9fc','svg.fonttype':'none'})
    g=Gallery(a.output);summary=g.json(EXPERIMENT+'/summary.json');protocol=g.json(EXPERIMENT+'/protocol.json')
    reports=[g.json(EXPERIMENT+f'/s{s}/report.json') for s in SEEDS]
    horizons(g,summary);derived=diagnosis(g,summary,reports);cases(g,reports,protocol)
    (a.output/'derived_metrics.json').write_text(json.dumps(derived,indent=2),encoding='utf-8',newline='\n')
    result={'status':'completed_multicycle_gallery','figure_count':len(g.figures),'figures':g.figures,
        'input_sha256':g.sources,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.iterdir())},
        'fresh_solver_calls':0,'fresh_training_calls':0}
    (a.output/'gallery_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'status':result['status'],'figure_count':result['figure_count'],'derived':derived},indent=2))


if __name__=='__main__':main()
