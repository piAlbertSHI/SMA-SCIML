"""Render archived numerical results and replay trusted local checkpoints. No solver/training."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from SciML.models.v12.learning_coordinate_operator import CoordinateOperator
from SciML.models.v12.learning_ensemble import LearningEnsemble

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#125b8e','#e38433','#258879','#8c658b']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=Path('demo-output'));ap.add_argument('--no-animation',action='store_true');a=ap.parse_args()
    if a.output.exists(): raise FileExistsError('Use a new output directory; preserve earlier demos')
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f9fc','axes.facecolor':'white','savefig.facecolor':'#f7f9fc'})
    m=json.loads((ROOT/'data/metrics.json').read_text(encoding='utf-8'))
    def save(fig,name): fig.savefig(a.output/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    # Figures never treat seed SD as predictive uncertainty.
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    models=['ridge','mlp','ensemble5']; names=['Ridge','MLP','Ensemble (5)']
    for ax,j,unit in zip(axes,[0,2,3],['Cooling / J','COP','SCP / W kg$^{-1}$']):
        values=[m['scalar']['comparison'][k]['seed_mean_test_rmse'][j] for k in models]
        sd=[m['scalar']['comparison'][k]['seed_std_test_rmse'][j] for k in models]
        ax.bar(names,values,yerr=sd,color=COLORS[:3],capsize=4);ax.set_title(unit);ax.set_ylabel('Test RMSE');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Scalar prediction | 37 train / 13 test conditions · mean ± seed SD (3 seeds)')
    save(fig,'scalar_benchmarks')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    base=m['operator_baselines']['test_comparison']['coarse_pod_ridge']
    for i,field in enumerate(['sma','htf']):
        for j,kind in enumerate(['deeponet','fourier']):
            rows=[next(v for v in m['operator'] if v['kind']==kind and v['stage']==s and v['field']==field) for s in [2,4]]
            axes[i].errorbar([2,4],[v['mean_future_rmse_K'] for v in rows],yerr=[v['sample_std_future_rmse_K'] for v in rows],marker='o',capsize=4,label=kind,color=COLORS[j])
        axes[i].plot([2,4],[base[str(s)][field]['future_rmse_K'] for s in [2,4]],'s--',color=COLORS[2],label='Matched POD/Ridge')
        axes[i].set(xticks=[2,4],xlabel='Exchange stage',ylabel='Future-field RMSE / K',title=field.upper());axes[i].legend();axes[i].grid(alpha=.15)
    fig.suptitle('Operator comparison | 13 held-out conditions · first frame excluded')
    save(fig,'operator_comparison')
    fig,ax=plt.subplots(figsize=(8.5,4.8),layout='constrained')
    old=m['operator_baselines']['test_comparison']['request_boundary_to_deeponet']['4'];new=m['memory']['scores']['test']['4']
    x=np.arange(2)
    ax.bar(x-.18,[old[k]['future_rmse_K'] for k in ['sma','htf']],.36,color=COLORS[1],label='Radial mirror only')
    ax.bar(x+.18,[new[k]['future_rmse_K'] for k in ['sma','htf']],.36,color=COLORS[0],label='Carry predicted odd component')
    ax.set(xticks=x,xticklabels=['SMA','HTF'],ylabel='Stage 4 field RMSE / K',title='Fixing state information loss | fixed seed 20260910');ax.legend();ax.grid(axis='y',alpha=.15)
    save(fig,'memory_repair')
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,j,title in zip(axes,[0,2,3],['Cooling / J','COP','SCP / W kg$^{-1}$']):
        for k,color in zip(['random','uncertainty','composite'],COLORS):
            cur=m['active']['learning_curves'][k];mean=np.asarray(cur['seed_mean_rmse'])[:,j];sd=np.asarray(cur['seed_std_rmse'])[:,j]
            ax.plot([12,17,22],mean,'o-',label=k,color=color);ax.fill_between([12,17,22],np.maximum(0,mean-sd),mean+sd,color=color,alpha=.12)
        ax.set(xticks=[12,17,22],xlabel='Revealed labels',ylabel='Test RMSE',title=title);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('Offline active learning | equal budgets · 3 seeds · no fresh simulations')
    save(fig,'active_learning')
    # Genuine frozen checkpoint replay, not a reconstruction from a chart.
    d=np.load(ROOT/'data/field_demo.npz',allow_pickle=False)
    model=CoordinateOperator.load(ROOT/'checkpoints/wp4-deeponet-r1/model.pt')
    pred=model.predict(d['branch'],d['initial'],d['times'],d['positions'])
    replay_error=float(np.max(np.abs(pred-d['saved_prediction'])))
    if replay_error>1e-10: raise ValueError('Field replay mismatch beyond explicit cross-batch float64 tolerance')
    truth=d['truth'];rmse=np.sqrt(np.mean((pred[:,:,1:]-truth[:,:,1:])**2,axis=(0,2,3)))
    fig,axes=plt.subplots(2,3,figsize=(12,6.6),layout='constrained')
    for j,label in enumerate(['SMA','HTF']):
        lo=min(truth[0,j].min(),pred[0,j].min());hi=max(truth[0,j].max(),pred[0,j].max())
        for i,(values,title) in enumerate([(truth[0,j],'Archived solver'),(pred[0,j],'DeepONet'),(pred[0,j]-truth[0,j],'Prediction − truth')]):
            kw={'vmin':lo,'vmax':hi,'cmap':'viridis'} if i<2 else {'vmin':-np.max(abs(values)),'vmax':np.max(abs(values)),'cmap':'coolwarm'}
            im=axes[j,i].imshow(values,origin='lower',aspect='auto',extent=[0,1,0,1],**kw)
            axes[j,i].set(title=f'{label} | {title}',xlabel='Normalized position',ylabel='Normalized stage time');fig.colorbar(im,ax=axes[j,i],label='K',shrink=.8)
    fig.suptitle('First test stage 4 | observed entry fields · fixed case, not best-case selection')
    save(fig,'field_replay')
    scalar=np.load(ROOT/'data/scalar63.npz',allow_pickle=False)
    ens=LearningEnsemble.load(ROOT/'checkpoints/wp6-uq-r1/ensemble.pt'); members=ens.predict_members(scalar['x'])
    saved=np.load(ROOT/'data/uq_predictions.npz',allow_pickle=False)
    np.testing.assert_allclose(members,saved['member_predictions'],rtol=0,atol=1e-10)
    mask=scalar['split']=='test';rmse_uq=np.sqrt(np.mean((members.mean(0)[mask]-scalar['y'][mask])**2,axis=0))
    fig,ax=plt.subplots(figsize=(8.7,5),layout='constrained');observations=m['design_candidates']['observations']
    for i,o in enumerate(observations):
        ax.scatter(o['targets'][2],o['targets'][3],s=180,color=COLORS[i]);ax.annotate(o['case_id'].split('_')[2],(o['targets'][2],o['targets'][3]),xytext=(10,7),textcoords='offset points')
    ax.set(xlabel='Archived COP label',ylabel='SCP / W kg$^{-1}$',title='Three selected candidates | numerical archive check, safety unknown');ax.margins(.2);ax.grid(alpha=.15)
    save(fig,'design_tradeoff')
    if not a.no_animation:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');lines=[]
        for j,ax in enumerate(axes):
            lines.append((ax.plot(d['positions'],truth[0,j,0],color=COLORS[2],label='Archived solver')[0],ax.plot(d['positions'],pred[0,j,0],'--',color=COLORS[0],label='DeepONet')[0]))
            lo=min(truth[0,j].min(),pred[0,j].min());hi=max(truth[0,j].max(),pred[0,j].max());ax.set(ylim=(lo-.3,hi+.3),xlabel='Normalized position',ylabel='Temperature / K',title=['SMA','HTF'][j]);ax.legend()
        title=fig.suptitle('Observed-entry field prediction | fixed held-out stage 4')
        def frame(i):
            for j,(l1,l2) in enumerate(lines): l1.set_ydata(truth[0,j,i]);l2.set_ydata(pred[0,j,i])
            title.set_text(f'Observed-entry prediction | normalized stage time {d["times"][i]:.2f}')
            return [title]+[line for pair in lines for line in pair]
        FuncAnimation(fig,frame,frames=range(0,50,2),interval=120).save(a.output/'temperature_evolution.gif',writer=PillowWriter(fps=8),dpi=95);plt.close(fig)
    summary={'status':'passed_portable_checkpoint_demo','sample_id':str(d['sample_id']),'field_input':'observed stage4 entry, not request-only cascade','field_replay_max_abs_K':replay_error,'field_case_rmse_K':rmse.tolist(),'uq_test_rmse':rmse_uq.tolist(),'fresh_training_calls':0,'fresh_solver_calls':0,'historical_aggregate_metrics':'from archived JSON, not refit in this demo','safety_certified':False}
    (a.output/'demo_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
