"""Space-time comparisons for coordinate and modal HRO, from stored forecasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
import numpy as np

from .research_gallery import Gallery
from .multicycle_multistep import ROOT
from SciML.models.v12.multistep_historical import sha


def architecture(g):
    fig, ax = plt.subplots(figsize=(15, 6.6)); ax.set(xlim=(0, 15), ylim=(0, 6.6)); ax.axis('off')
    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.13',facecolor=color,edgecolor='#c0c9d4'))
        ax.text(x+w/2,y+h-.25,title,ha='center',va='top',fontsize=12,weight='bold',color='#16324f')
        ax.text(x+w/2,y+h-.67,body,ha='center',va='top',fontsize=10,linespacing=1.6,color='#263c50')
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#527c9f','lw':1.7})
    box(.2,3.2,2.7,2.1,'Physical state + control','515 state variables\n9 known controls\n5 stage identities','#e8f0fa')
    box(3.4,3.2,3.,2.1,'Hierarchical encoders','Radial + axial residual CNNs\n16 latent tokens, width 128\n3 attention blocks / 8 heads','#e8f0fa')
    box(6.9,3.2,3.1,2.1,'Recurrent transition','GRU memory: 256 variables\nFiLM + five stage heads\nFrozen affine prior + correction','#e8f0fa')
    box(10.6,3.2,4.,2.1,'State rollout','Explicit next state + memory\nAutonomous stage chaining\nTwo-cycle train / three-cycle test','#e8f0fa')
    arrow((2.95,4.3),(3.2,4.3)); arrow((6.5,4.3),(6.75,4.3)); arrow((10.1,4.3),(10.45,4.3))
    box(3.4,.55,6.6,1.95,'Modal field decoder','Time-query attention -> 2 x 24 coefficients\nSpatial basis: constant + linear + cosine modes\nT(x,t) = T(entry,x) + [1 - exp(-4t)] x scaled residual','#dff3ef')
    box(10.6,.55,4.,1.95,'Two-phase optimization','200 joint trajectory updates\n320 cached decoder updates\nAll 101 spatial points supervised','#fff0d7')
    arrow((8.3,3.05),(8.3,2.67)); arrow((10.1,1.5),(10.45,1.5))
    ax.text(.2,6.2,'HRO | Structured dynamics and modal temperature fields',fontsize=19,weight='bold',color='#16324f')
    ax.text(.2,5.7,'Causal inference from initial conditions and controls; exact field initial values',fontsize=11,color='#526578')
    ax.text(.2,.12,'Decoded fields do not overwrite the state. Stage-end identity and physical energy closure are not enforced.',fontsize=10,color='#526578')
    g.save(fig,'architecture','HRO modal decoder and two-phase training; implemented modules, no claim of PDE conservation')


def panels(case, atlas):
    fig, axes = plt.subplots(2, 5, figsize=(17.5, 8.8 if atlas else 6.8), layout='constrained')
    images=[]
    for channel in range(2):
        a,b,c=[case[k][:,channel] for k in ('truth','coordinate','modal')]
        low=min(v.min() for v in (a,b,c)); high=max(v.max() for v in (a,b,c))
        bound=max(np.abs(b-a).max(),np.abs(c-a).max())
        for j,(v,title) in enumerate(zip((a,b,c,np.abs(b-a),np.abs(c-a)),
                ('Numerical reference','HRO coordinate','HRO modal','Coordinate | error','Modal | error'))):
            ax=axes[channel,j]; grid=v.reshape(300,101) if atlas else np.full((50,101),np.nan)
            im=ax.imshow(grid,origin='lower',aspect='auto',extent=(0,1,0,6 if atlas else 1),
                         cmap='inferno' if j>2 else 'magma',vmin=0 if j>2 else low,vmax=bound if j>2 else high)
            images.append(im); ax.set_title(title,fontsize=11); ax.set_xlabel('Position x / L')
            if atlas:
                ax.set_yticks(np.arange(6)+.5,labels=['C1 / S2','C1 / S4','C2 / S2','C2 / S4','C3 / S2','C3 / S4'] if j==0 else ['']*6)
                for y in range(1,6): ax.axhline(y,color='white',lw=.6,alpha=.5)
            if j==0: ax.set_ylabel(('SMA' if channel==0 else 'Fluid')+' | '+('Exchange segment' if atlas else 'Local stage time'))
        fig.colorbar(images[channel*5],ax=list(axes[channel,:3]),shrink=.8,label='Temperature / K')
        fig.colorbar(images[channel*5+3],ax=list(axes[channel,3:]),shrink=.8,label='Absolute error / K')
    return fig,images


def run(output, no_animation=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#f5f8fc',
                         'axes.facecolor':'#e7edf4','savefig.facecolor':'#f5f8fc','svg.fonttype':'none'})
    g=Gallery(output); experiment='evidence/experiments/hro_modal/formal'
    g.json(experiment+'/protocol.json'); g.json(experiment+'/summary.json')
    with np.load(g.source('data/historical12_state.npz'),allow_pickle=False) as z: d={k:z[k] for k in z.files}
    with np.load(g.source('evidence/experiments/v13_formal_v1/s20261121/test_predictions.npz')) as z: coordinate=z['field']
    with np.load(g.source(experiment+'/s20261121/test_predictions.npz')) as z: modal=z['field']
    cases=[]
    for index,ci in enumerate(np.flatnonzero((d['split']=='test')&(d['cycles']==3))):
        use=d['field_case_index']==ci
        case={'truth':d['field_truth'][use],'coordinate':coordinate[use],'modal':modal[use],
              'cycles':d['field_cycle'][use],'stages':d['field_stage'][use]}; cases.append(case)
        fig,_=panels(case,True)
        fig.suptitle('Autonomous three-cycle thermal fields | Configuration '+('A' if index==0 else 'B')+
                     '\nCoordinate HRO and two-phase modal HRO | fixed first seed, shared scales',fontsize=15)
        g.save(fig,f'field_atlas_{index+1}','Stored predictions on both three-cycle cases; no smoothing; local exchange time')
    architecture(g)
    if not no_animation:
        case=cases[0]; fig,images=panels(case,False); title=fig.suptitle('',fontsize=14)
        def update(frame):
            segment,t=frame
            for channel in range(2):
                a,b,c=[case[k][segment,channel] for k in ('truth','coordinate','modal')]
                for j,v in enumerate((a,b,c,np.abs(b-a),np.abs(c-a))):
                    shown=v.copy(); shown[t+1:]=np.nan; images[channel*5+j].set_data(shown)
            title.set_text(f'HRO | Autonomous temperature fields and errors\nCycle {case["cycles"][segment]} / stage {case["stages"][segment]} | local time {t/49:.2f} | Configuration A')
            return images+[title]
        animation=FuncAnimation(fig,update,frames=[(s,t) for s in range(6) for t in [0,7,14,21,28,35,42,49]],interval=160)
        animation.save(g.out/'multicycle_comparison.gif',writer=PillowWriter(fps=6),dpi=90); plt.close(fig)
    report={'status':'completed_modal_showcase','first_seed':20261121,'selection':'both three-cycle cases; animation first sorted case',
            'input_sha256':g.sources,'figures':g.figures,'generator_sha256':sha(Path(__file__)),
            'artifacts':{p.name:sha(p) for p in output.iterdir() if p.is_file()},'fresh_solver_calls':0}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return {'status':report['status'],'artifacts':list(report['artifacts'])}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-animation',action='store_true'); a=p.parse_args()
    print(json.dumps(run(a.output,a.no_animation),indent=2))
