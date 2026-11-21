"""Generate the public SMA-SCIML showcase from verified benchmark predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from .research_gallery import Gallery
from .operator_showcase import architecture
from .formal_hierarchical import DEFAULT, ROOT
from SciML.models.v12.multistep_historical import sha


def inputs(g):
    rel=DEFAULT.relative_to(ROOT).as_posix()
    protocol=g.json(rel+'/protocol.json')
    g.json(rel+'/summary.json')
    selection=protocol['demonstration_selection']
    with np.load(g.source('data/historical12_state.npz'),allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(g.source(rel+f"/s{selection['new_seed']}/test_predictions.npz"),allow_pickle=False) as z:new=z['field'].copy()
    with np.load(g.source(f"evidence/experiments/multicycle_multistep_v1/s{selection['old_seed']}/{selection['old_mode']}.npz"),allow_pickle=False) as z:old=z['field'].copy()
    cases=[]
    for ci in np.flatnonzero((d['split']=='test')&(d['cycles']==3)):
        use=d['field_case_index']==ci
        cases.append({'case':str(d['case_ids'][ci]),'truth':d['field_truth'][use],'old':old[use],'new':new[use],
                      'cycles':d['field_cycle'][use],'stages':d['field_stage'][use]})
    assert cases[0]['case']==selection['case_id']
    return cases,selection


def panels(case,atlas=False):
    fig,axes=plt.subplots(2,5,figsize=(17.5,9.2 if atlas else 7),layout='constrained')
    images=[]
    for f in range(2):
        a,b,c=[case[k][:,f] for k in ('truth','old','new')]
        lo=min(x.min() for x in (a,b,c));hi=max(x.max() for x in (a,b,c))
        bound=max(np.abs(b-a).max(),np.abs(c-a).max())
        for j,(value,title) in enumerate(zip((a,b,c,np.abs(b-a),np.abs(c-a)),
                                             ('Numerical reference','Residual cascade','HRO','Cascade | error','HRO | error'))):
            ax=axes[f,j]
            grid=value.reshape(300,101) if atlas else np.full((50,101),np.nan)
            im=ax.imshow(grid,origin='lower',aspect='auto',extent=(0,1,0,6 if atlas else 1),
                         cmap='inferno' if j>2 else 'magma',vmin=0 if j>2 else lo,vmax=bound if j>2 else hi)
            ax.set_title(title,fontsize=11)
            ax.set_xlabel('Position x / L',fontsize=10)
            if atlas:
                ax.set_yticks(np.arange(6)+.5,labels=['C1 / S2','C1 / S4','C2 / S2','C2 / S4','C3 / S2','C3 / S4'] if j==0 else ['']*6)
                for y in range(1,6):ax.axhline(y,color='white',alpha=.5,lw=.7)
            if j==0:ax.set_ylabel(('SMA' if f==0 else 'Fluid')+' | '+('Exchange segment' if atlas else 'Local stage time'))
            images.append(im)
        fig.colorbar(images[f*5],ax=list(axes[f,:3]),shrink=.8,pad=.02,label='Temperature / K')
        fig.colorbar(images[f*5+3],ax=list(axes[f,3:]),shrink=.8,pad=.02,label='Absolute error / K')
    return fig,images


def atlas(g,case,index):
    fig,_=panels(case,True)
    fig.suptitle('Autonomous three-cycle prediction | '+('Configuration A' if index==0 else 'Configuration B')+' test case\n'
                 'Residual cascade: Residual MLP + DeepONet | HRO: Hierarchical Recurrent Operator',fontsize=15)
    g.save(fig,f'field_atlas_{index+1}',f"{case['case']}; six exchange segments with local time; first preset seeds, not selected by error; shared limits for both model errors")


def movie(g,case):
    fig,images=panels(case)
    title=fig.suptitle('',fontsize=14)
    def update(frame):
        s,t=frame
        for f in range(2):
            a,b,c=[case[k][s,f] for k in ('truth','old','new')]
            for j,value in enumerate((a,b,c,np.abs(b-a),np.abs(c-a))):
                shown=value.copy();shown[t+1:]=np.nan;images[f*5+j].set_data(shown)
        title.set_text('Autonomous thermal prediction | Residual MLP + DeepONet vs HRO\n'
                       f"Cycle {case['cycles'][s]} / stage {case['stages'][s]} · local time {t/49:.2f} | Configuration A · fixed checkpoints")
        return images+[title]
    frames=[(s,t) for s in range(6) for t in list(range(0,50,6))+[49]]
    animation=FuncAnimation(fig,update,frames=frames,interval=150,blit=False)
    animation.save(g.out/'multicycle_comparison.gif',writer=PillowWriter(fps=7),dpi=90)
    plt.close(fig)


def explorer(g,case,selection):
    template=g.source('examples/templates/operator_explorer.html').read_text(encoding='utf-8')
    payload={'truth':np.round(case['truth'],4).tolist(),'prediction':np.round(case['new'],4).tolist(),
             'cycles':case['cycles'].tolist(),'stages':case['stages'].tolist(),'case':'Configuration A | three-cycle test trajectory',
             'seed':selection['new_seed'],'model':'Hierarchical Recurrent Operator (HRO) | autonomous state and temperature-field prediction'}
    (g.out/'explorer.html').write_text(template.replace('__FIELD_DATA__',json.dumps(payload,separators=(',',':'))),encoding='utf-8',newline='\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-animation',action='store_true');a=p.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f5f8fc','axes.facecolor':'#e7edf4','savefig.facecolor':'#f5f8fc','svg.fonttype':'none'})
    g=Gallery(a.output);cases,selection=inputs(g)
    g.source('examples/operator_showcase.py');g.source('examples/research_gallery.py')
    architecture(g,'Three-seed benchmark · autonomous state and field prediction')
    for i,case in enumerate(cases):atlas(g,case,i)
    explorer(g,cases[0],selection)
    if not a.no_animation:movie(g,cases[0])
    report={'status':'completed_operator_showcase','selection':selection,'figures':g.figures,
        'input_sha256':g.sources,'generator_sha256':sha(Path(__file__)),
        'artifacts':{p.name:sha(p) for p in sorted(a.output.iterdir())},'fresh_training_calls':0,'fresh_solver_calls':0}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'artifacts':list(report['artifacts']),'fresh_solver_calls':0},indent=2))


if __name__=='__main__':main()
