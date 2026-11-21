"""Architecture figure and numerical field demos with explicit model provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np

from .research_gallery import Gallery


def architecture(g):
    summary=g.json('evidence/architecture/v13_execution_v1/summary.json')
    fig,ax=plt.subplots(figsize=(18,10.7));fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,18),ylim=(0,10.7));ax.axis('off')
    ink='#132c46';blue='#176da0';teal='#087c78';orange='#bc641c';muted='#56687b'
    ax.text(.55,10.1,'SMA Neural State-Space Operator',fontsize=25,weight='bold',color=ink)
    ax.text(.55,9.65,'Spatial representation  /  latent temporal memory  /  continuous field queries',fontsize=13,color=muted)
    ax.text(.55,9.15,f"IMPLEMENTED  ·  {summary['parameter_report']['trainable']:,} trainable parameters  ·  Accuracy evaluation pending",fontsize=11,color=teal)
    for x,label in [(.55,'01  PHYSICAL INPUTS'),(4.3,'02  SPATIAL ENCODING'),(8.5,'03  RECURRENT FUSION'),(13.,'04  JOINT OUTPUTS')]:
        ax.text(x,8.65,label,fontsize=11,weight='bold',color=muted)
    def box(x,y,w,h,title,body,color=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',facecolor='white',edgecolor=color,lw=1.4))
        ax.text(x+.17,y+h-.34,title,fontsize=13,weight='bold',color=color,va='center')
        ax.text(x+.17,y+h-.64,body,fontsize=11,color=ink,va='top',linespacing=1.55)
    def arrow(a,b,color=blue,rad=0):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.5,color=color,connectionstyle=f'arc3,rad={rad}'))
    box(.55,6.4,3.1,1.7,'Mechanical state','4 channels × 51 radial nodes\nu, w, SMA temperature, phase')
    box(.55,4.35,3.1,1.7,'Transport + memory','3 channels × 101 axial nodes\nTwo fluid paths + odd T memory')
    box(.55,2.55,3.1,1.35,'Stage context','9 controls + 8 side values\n+ 5 stage indicators',teal)
    box(4.3,5.65,3.45,2.45,'Two spatial encoders','Width 128 · residual Conv1D\nDilation 1 / 2 / 4\nFourier position embeddings\n51 + 101 = 152 spatial tokens')
    box(4.3,2.55,3.45,1.35,'Condition encoder','22 → 128 → 128\nShared stage / control context',teal)
    box(8.5,5.65,3.75,2.45,'Latent cross-attention','16 learned latent tokens\n3 Transformer blocks\n8 heads · width 128\nSpatial information → latent state')
    box(8.5,2.55,3.75,1.75,'Temporal memory','GRU hidden state: 256\nPersistent across stages / cycles\nFiLM conditioning of latent tokens',teal)
    box(13.,6.35,4.45,1.75,'Five transition heads','FiLM residual + frozen affine prior\nActive-channel update + projection\n515-state output → next transition',orange)
    box(13.,3.65,4.45,2.05,'Coordinate field decoder','Query-to-latent cross-attention\nSMA + fluid temperature fields\nT(x,t) = T₀(x) + g(t) · s · N(x,t)\nHard initial values; no PDE claim',orange)
    box(13.,1.85,4.45,1.2,'Space / time queries','Fourier features of (x, t)\nNonuniform queries + chunking',teal)
    arrow((3.65,7.2),(4.3,7.2));arrow((3.65,5.2),(4.3,6.1))
    arrow((3.65,3.2),(4.3,3.2),teal);arrow((7.75,6.9),(8.5,6.9))
    arrow((7.75,3.2),(8.5,3.2),teal);arrow((6.1,3.9),(9.,5.65),teal,rad=-.12)
    arrow((10.35,5.65),(10.35,4.3),teal)
    arrow((11.75,4.3),(11.75,5.65),teal)
    ax.text(11.83,4.9,'FiLM',fontsize=10,color=teal)
    arrow((12.25,7.2),(13.,7.2),orange);arrow((12.25,6.),(13.,5.3),orange)
    arrow((12.25,3.3),(13.,4.),teal);arrow((15.2,3.05),(15.2,3.65),teal)
    arrow((9.1,2.55),(11.6,2.55),teal,rad=.65)
    ax.text(10.35,1.25,'hⱼ → hⱼ₊₁',ha='center',fontsize=12,color=teal)
    ax.text(.55,1.55,'JOINT TRAINING',fontsize=10,weight='bold',color=muted)
    ax.text(.55,1.08,'Multi-step state loss + field loss + supervised spatial-gradient loss',fontsize=12,color=ink)
    ax.text(.55,.48,'Causal rollout: predicted states and hidden memory only  •  Full backpropagation through the restart  •  No new physical simulations',fontsize=10.5,color=muted)
    g.save(fig,'architecture','Implemented v13 architecture, 3-update execution validation only; not an accuracy claim; parameters read from execution summary')


def archived_fields(g):
    with np.load(g.source('data/historical12_state.npz'),allow_pickle=False) as z:
        ci=int(np.flatnonzero((z['split']=='test')&(z['cycles']==3))[0])
        use=z['field_case_index']==ci
        truth=z['field_truth'][use].copy();cycles=z['field_cycle'][use];stages=z['field_stage'][use]
        case=str(z['case_ids'][ci])
    source='evidence/experiments/multicycle_multistep_v1/s20260910/archived_one_step.npz'
    with np.load(g.source(source),allow_pickle=False) as z:prediction=z['field'][use].copy()
    assert len(truth)==6
    return truth,prediction,cycles,stages,case


def atlas(g,truth,prediction):
    fig,axes=plt.subplots(2,3,figsize=(13.8,8.8),layout='constrained')
    for field in range(2):
        lo=min(truth[:,field].min(),prediction[:,field].min());hi=max(truth[:,field].max(),prediction[:,field].max())
        error=prediction[:,field]-truth[:,field];bound=float(np.abs(error).max())
        for col,(value,label) in enumerate([(truth[:,field],'Numerical reference'),(prediction[:,field],'Autonomous prediction'),(error,'Prediction − reference')]):
            grid=value.reshape(300,101)
            ax=axes[field,col]
            image=ax.imshow(grid,origin='lower',extent=(0,1,0,6),aspect='auto',cmap='RdBu_r' if col==2 else 'magma',
                            vmin=-bound if col==2 else lo,vmax=bound if col==2 else hi)
            ax.set(title=f"{['SMA','Fluid'][field]}  |  {label}",xlabel='Normalized axial position',
                   yticks=np.arange(6)+.5,yticklabels=['C1 / S2','C1 / S4','C2 / S2','C2 / S4','C3 / S2','C3 / S4'])
            ax.set_ylabel('Exchange segment · local time runs upward')
            for y in range(1,6):ax.axhline(y,color='white',alpha=.45,lw=.7)
            fig.colorbar(image,ax=ax,shrink=.87,label='K')
    fig.suptitle('Three-cycle thermal evolution | trained reference model\nFixed first three-cycle test case · seed 20260910 · separate historical profile',fontsize=14)
    g.save(fig,'multicycle_field_atlas','v12 archived state model and frozen historical DeepONet; first three-cycle test case; six exchange segments, not continuous wall-clock time; no v13 prediction')


def movie(g,truth,prediction,cycles,stages):
    fig,axes=plt.subplots(2,3,figsize=(12,6.6));fig.subplots_adjust(left=.06,right=.94,bottom=.09,top=.83,hspace=.5,wspace=.38)
    images=[]
    for field in range(2):
        lo=min(truth[:,field].min(),prediction[:,field].min());hi=max(truth[:,field].max(),prediction[:,field].max())
        bound=float(np.abs(prediction[:,field]-truth[:,field]).max())
        for col,label in enumerate(['Numerical reference','Autonomous prediction','Absolute error']):
            ax=axes[field,col]
            im=ax.imshow(np.full((50,101),np.nan),origin='lower',aspect='auto',extent=(0,1,0,1),
                         cmap='inferno' if col==2 else 'magma',vmin=0 if col==2 else lo,vmax=bound if col==2 else hi)
            ax.set(title=f"{['SMA','Fluid'][field]} | {label}",xlabel='Position',ylabel='Local stage time' if col==0 else '')
            fig.colorbar(im,ax=ax,pad=.03,label='K',fraction=.04);images.append(im)
    title=fig.suptitle('',fontsize=13)
    frames=[(segment,step) for segment in range(6) for step in list(range(0,50,4))+[49]]
    def update(frame):
        segment,step=frame
        for f in range(2):
            values=[truth[segment,f],prediction[segment,f],np.abs(prediction[segment,f]-truth[segment,f])]
            for j,v in enumerate(values):
                shown=v.copy();shown[step+1:]=np.nan;images[3*f+j].set_data(shown)
        title.set_text(f'Trained reference model · autonomous three-cycle field replay\nCycle {cycles[segment]} / stage {stages[segment]} · local time {step/49:.2f} · v13 accuracy is not yet evaluated')
        return images+[title]
    ani=FuncAnimation(fig,update,frames=frames,interval=130,blit=False)
    ani.save(g.out/'multicycle_fields.gif',writer=PillowWriter(fps=8),dpi=90)
    plt.close(fig)


def explorer(g,truth,prediction,cycles,stages,case):
    template=g.source('examples/templates/field_explorer.html').read_text(encoding='utf-8')
    payload={'truth':np.round(truth,4).tolist(),'prediction':np.round(prediction,4).tolist(),
        'cycles':cycles.tolist(),'stages':stages.tolist(),'case':case,'seed':20260910,
        'model':'Archived v12 state model + historical DeepONet; not the new v13 architecture'}
    (g.out/'explorer.html').write_text(template.replace('__FIELD_DATA__',json.dumps(payload,separators=(',',':'))),encoding='utf-8',newline='\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    options=p.add_mutually_exclusive_group();options.add_argument('--no-animation',action='store_true');options.add_argument('--animation-only',action='store_true');a=p.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f5f8fc','axes.facecolor':'#e7edf4','savefig.facecolor':'#f5f8fc','svg.fonttype':'none'})
    g=Gallery(a.output)
    if not a.animation_only:architecture(g)
    truth,prediction,cycles,stages,case=archived_fields(g)
    if not a.animation_only:atlas(g,truth,prediction);explorer(g,truth,prediction,cycles,stages,case)
    if not a.no_animation:movie(g,truth,prediction,cycles,stages)
    report={'status':'completed_architecture_and_field_showcase','architecture':'v13 implemented, execution-checked, accuracy pending',
        'numerical_demos':'archived v12 autonomous model and historical field decoder; never relabelled as v13 results',
        'case_id':case,'seed':20260910,'figures':g.figures,'input_sha256':g.sources,
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.iterdir())},
        'fresh_training_calls':0,'fresh_solver_calls':0}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'status':report['status'],'artifacts':list(report['artifacts']),'case_id':case},indent=2))


if __name__=='__main__':main()
