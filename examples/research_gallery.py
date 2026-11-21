"""Build source-bound research figures from archived branches and the diversity experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#155e91', '#dd8133', '#258879', '#8a5da8']


class Gallery:
    def __init__(self, out):
        if out.exists():
            raise FileExistsError('Use a fresh gallery directory')
        out.mkdir(parents=True)
        self.out, self.sources, self.figures = out, {}, []

    def source(self, name):
        path = ROOT / name
        self.sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def json(self, name):
        return json.loads(self.source(name).read_text(encoding='utf-8'))

    def save(self, fig, name, scope):
        for suffix in ('png', 'svg'):
            kwargs = {'metadata': {'Date': None}} if suffix == 'svg' else {}
            fig.savefig(self.out / f'{name}.{suffix}', dpi=150, bbox_inches='tight', **kwargs)
        plt.close(fig)
        self.figures.append({'name': name, 'scope': scope})


def state_plot(g, m):
    b = m['boundary']['test_seed_mean_rmse']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout='constrained')
    for ax, key, unit in zip(axes, ['sma_K', 'xi'], ['SMA-state RMSE / K', 'Phase-fraction RMSE']):
        for kind, label, color in [('autonomous_affine', 'Affine', COLORS[1]), ('autonomous_neural', 'Ridge + neural residual', COLORS[0])]:
            ax.plot(range(1, 5), [b[kind][str(s)][key] for s in range(1, 5)], 'o-', color=color, label=label)
        ax.set(xticks=range(1, 5), xlabel='Cycle stage', ylabel=unit, ylim=(0, None)); ax.grid(alpha=.18)
    axes[0].legend(fontsize=9)
    fig.suptitle('Autonomous state propagation\n13 test cases | three-seed means | no future stage-state inputs')
    g.save(fig, 'state_dynamics', 'Current63 autonomous state errors; state temperature is distinct from field RMSE')


def operator_ablation(g, m):
    kinds = ['deeponet', 'fourier', 'deeponet_no_stage', 'deeponet_no_ic']
    labels = ['DeepONet', 'Fourier', 'No stage input', 'No hard initial condition']
    rows = [[next(r for r in m['operator'] if r['kind'] == k and r['stage'] == s and r['field'] == f)
             for s, f in [(2, 'sma'), (2, 'htf'), (4, 'sma'), (4, 'htf')]] for k in kinds]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, key, title in zip(axes, ['mean_future_rmse_K', 'worst_test_initial_max_K'],
                              ['Future-field RMSE / K\nMean over 3 seeds', 'Initial-frame max error / K\nWorst across test cases and 3 seeds']):
        a = np.array([[r[key] for r in row] for row in rows])
        im = ax.imshow(a, cmap='YlOrRd', vmin=0, aspect='auto')
        ax.set(xticks=range(4), xticklabels=['S2 SMA', 'S2 fluid', 'S4 SMA', 'S4 fluid'],
               yticks=range(4), yticklabels=labels, title=title)
        ax.tick_params(axis='both', length=0)
        for i in range(4):
            for j in range(4): ax.text(j, i, f'{a[i,j]:.3f}', ha='center', va='center', color='white' if a[i,j] > .65*a.max() else '#222')
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Architecture ablations | accuracy and interface consistency are separate metrics')
    g.save(fig, 'operator_ablations', 'Current63; future RMSE versus initial-condition maximum; metric scales differ')


def pinn_plot(g):
    with np.load(g.source('data/pinn_predictions.npz'), allow_pickle=False) as d:
        coords, truth = d['grid'], d['truth'].ravel()
        data, pinn = d['s410_data_only_prediction'].ravel(), d['s410_pinn_prediction'].ravel()
    summary = g.json('evidence/archive/wp5-pinn-r1/summary.json')
    x, ix = np.unique(coords[:, 0], return_inverse=True); t, it = np.unique(coords[:, 1], return_inverse=True)
    def field(v):
        a = np.empty((len(t), len(x))); a[it, ix] = v; return a
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    extent = [float(x.min()), float(x.max()), float(t.min()), float(t.max())]
    for ax, v, name in zip(axes[0], [truth, data, pinn], ['Manufactured reference', 'Data-only network', 'Inverse PINN']):
        im = ax.imshow(field(v), origin='lower', aspect='auto', extent=extent, cmap='viridis', vmin=truth.min(), vmax=truth.max())
        ax.set(title=name, xlabel='Position x', ylabel='Time t'); fig.colorbar(im, ax=ax, shrink=.82, label='Temperature (nondimensional)')
    limit = max(np.max(np.abs(data-truth)), np.max(np.abs(pinn-truth)))
    for ax, v, name in zip(axes[1, :2], [data-truth, pinn-truth], ['Data-only error', 'PINN error']):
        im = ax.imshow(field(v), origin='lower', aspect='auto', extent=extent, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title=name+' (same color scale)', xlabel='Position x', ylabel='Time t'); fig.colorbar(im, ax=ax, shrink=.82)
    runs = [r for r in summary['runs'] if r['tag'].endswith('_pinn')]
    ax = axes[1, 2]; values = [100*r['relative_parameter_error'] for r in runs]
    bars = ax.bar(['410', '411', '412'], values, color=COLORS[0]); ax.bar_label(bars, fmt='%.4f%%', padding=3, fontsize=9)
    ax.set(xlabel='Training seed', ylabel='Coefficient relative error / %', title='Diffusivity identification', ylim=(0, max(values)*1.3)); ax.grid(axis='y', alpha=.18)
    fig.suptitle('PINN inverse problem | 12 observations | fields: fixed seed 410\nManufactured heat equation, separate from the SMA simulator')
    g.save(fig, 'pinn_inverse', 'Manufactured diffusion problem; seed410 field plots and all3 coefficient errors')


def local_plot(g):
    rows = g.json('evidence/archive/wp5-local-thermal-r2/summary.json')['results']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained'); x=np.arange(len(rows))
    for off, key, label, color in [(-.18, 'effective_rate_per_s', 'Local ODE PINN', COLORS[0]), (.18, 'classical_rate_per_s', 'Classical ODE fit', COLORS[1])]:
        axes[0].bar(x+off, [r[key] for r in rows], .36, label=label, color=color)
    for off, key, color in [(-.18, 'held_tail_rmse_K', COLORS[0]), (.18, 'classical_held_tail_rmse_K', COLORS[1])]:
        axes[1].bar(x+off, [r[key] for r in rows], .36, color=color)
    labels=[r['tag'].replace('case','C').replace('_stage',' / S') for r in rows]
    for ax, title, ylabel in zip(axes, ['Effective response rate', 'Withheld time-tail prediction'], ['Rate / s$^{-1}$', 'Tail RMSE / K']):
        ax.set(xticks=x, xticklabels=labels, title=title, ylabel=ylabel); ax.tick_params(axis='x', labelrotation=25); ax.grid(axis='y', alpha=.18)
    axes[0].legend(fontsize=9)
    fig.suptitle('Local thermal inversion | 3 cases × 2 exchange stages\nEffective rates, not general SMA material constants')
    g.save(fig, 'local_thermal', 'Six local fits; classical comparison and withheld time-tail error')


def uncertainty_plot(g, m):
    with np.load(g.source('data/scalar63.npz'), allow_pickle=False) as d: ids, test=d['case_ids'],d['split']=='test'
    with np.load(g.source('data/uq_predictions.npz'), allow_pickle=False) as d:
        np.testing.assert_array_equal(ids, d['case_ids'])
        truth=d['targets'][test]; mean=d['member_predictions'].mean(0)[test]; width=d['width_0.9']
    assert width.shape==truth.shape
    np.testing.assert_allclose(np.mean(2*width,axis=0),m['uq']['intervals'][0]['mean_full_width'],rtol=0,atol=1e-10)
    coverage=float(np.mean(np.all(np.abs(truth-mean)<=width,axis=1))); assert coverage==1.
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    for ax,j,title in zip(axes,[0,2,3],['Cooling energy / J','COP','Specific cooling power / W kg$^{-1}$']):
        order=np.argsort(truth[:,j]); x=np.arange(1,len(order)+1)
        ax.errorbar(x,mean[order,j],yerr=width[order,j],fmt='o',color=COLORS[0],capsize=3,label='Mean + calibrated interval',markersize=4)
        ax.scatter(x,truth[order,j],color=COLORS[1],s=22,zorder=3,label='Reference')
        ax.set(title=title,xlabel='Test case (sorted by reference)',xticks=[1,4,7,10,13]);ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle('Uncertainty calibration | nominal 90% simultaneous intervals\n13/13 test cases covered across all 6 outputs; intervals are broad')
    g.save(fig,'uncertainty_intervals','Five-member six-output ensemble; half-widths from saved90% simultaneous calibration; not seed SD')


def ood_plot(g):
    rows=g.json('evidence/archive/ood-audit-r1/aggregate.json')
    selected=[r for scheme in ['geometry','material','protocol'] for r in rows if r['scheme']==scheme]
    fig,axes=plt.subplots(1,3,figsize=(12.5,5.2),layout='constrained');x=np.arange(len(selected))
    labels=[r['scheme'].title()+'\n'+('ID' if r['group']=='id_test' else 'OOD')+f" (n={r['case_count']})" for r in selected]
    values=[r['scalar_models']['ensemble3']['mean_rmse'][2] for r in selected]
    sd=[r['scalar_models']['ensemble3']['std_rmse'][2] for r in selected]
    color=[COLORS[0] if r['group']=='id_test' else COLORS[1] for r in selected]
    axes[0].bar(x,values,yerr=sd,capsize=3,color=color); axes[0].set(title='Performance ensemble',ylabel='COP RMSE')
    axes[1].bar(x,[r['deeponet_mean_rmse_K'][2] for r in selected],yerr=[r['deeponet_std_rmse_K'][2] for r in selected],capsize=3,color=color)
    axes[1].set(title='DeepONet stage-4 SMA',ylabel='Future-field RMSE / K')
    for i,r in enumerate(selected):
        c=np.asarray(r['simultaneous90_coverage_by_seed']);axes[2].bar(i,c.mean(),color=color[i]);axes[2].scatter([i]*len(c),c,s=22,color='#222',zorder=3)
    axes[2].axhline(.9,ls='--',color=COLORS[2],label='Nominal 90%');axes[2].set(title='Simultaneous coverage',ylabel='Fraction of cases',ylim=(0,1.12));axes[2].legend(fontsize=8)
    for ax in axes:ax.set(xticks=x,xticklabels=labels);ax.tick_params(axis='x',labelrotation=35,labelsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Structured distribution shifts | separately trained splits\nBars: three-seed means; error bars: seed SD; coverage dots: individual seeds')
    g.save(fig,'ood_generalization','Geometry/material/protocol experiments use separate splits and retrained models; no cross-task pooling')


def physics_plot(g):
    d=g.json('evidence/archive/staged-physics-r1/summary.json')['aggregate']
    kinds=['data','pde','pde_interfaces','full'];labels=['Data','+ PDE','+ Interfaces','+ Energy']
    metrics=[('unobserved_field_rmse_K','Unobserved field RMSE / K'),('pde_rms_K_per_s','PDE residual RMS / K s$^{-1}$'),
             ('cycle_jump_rms_K','Cycle-interface jump RMS / K'),('energy_rate_residual_rms_W','Energy-rate residual RMS / W')]
    fig,axes=plt.subplots(1,4,figsize=(13,4.5),layout='constrained')
    for ax,(key,title) in zip(axes,metrics):
        ax.bar(labels,[d[k][key]['mean'] for k in kinds],yerr=[d[k][key]['std'] for k in kinds],color=COLORS,capsize=3)
        ax.set(title=title);ax.tick_params(axis='x',labelrotation=30,labelsize=9);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Physics-loss ablations | manufactured four-stage heat equation\nThree seeds per variant; equal training budget; both cycles sparsely observed')
    g.save(fig,'physics_ablation','Manufactured heat-equation ablation; not actual SMA energy closure or horizon extrapolation')


def history_plot(g):
    d=g.json('evidence/archive/historical-fields-r1/summary.json')
    keys=[f'cycle{c}_stage{s}' for c in [1,2,3] for s in [2,4]];x=np.arange(6)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,field in zip(axes,['sma','htf']):
        for mode,label,color in [('observed_start','DeepONet: observed entry',COLORS[1]),('autonomous_boundary','DeepONet: autonomous entry',COLORS[0])]:
            ax.plot(x,[d['fixed_three_cycle_seed_mean_rmse_K'][mode][k][field] for k in keys],'o-',color=color,label=label)
        ax.plot(x,[d['fixed_three_cycle_pod_scores'][k][field]['future_rmse_K'] for k in keys],'s--',color=COLORS[2],label='POD: observed entry')
        ax.set(title=field.upper(),ylabel='Future-field RMSE / K',xticks=x,xticklabels=['C1/S2','C1/S4','C2/S2','C2/S4','C3/S2','C3/S4'],ylim=(0,None));ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle('Historical multicycle field prediction | fixed 2-case three-cycle test cohort\n8 training / 4 test cases overall; training cycles 1–2; independent solver profile')
    g.save(fig,'historical_rollout','Independent historical profile; same two3cycle test cases at all horizons; observed and autonomous entry are distinct input privileges')


def acquisition_plot(g):
    d=g.json('evidence/experiments/acquisition_diversity_v1/summary.json')
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    labels=['Cooling / J','Pump work / J','COP','Specific cooling power / W kg$^{-1}$','Temperature span / K','COP-denominator work / J']
    for j,(ax,title) in enumerate(zip(axes.flat,labels)):
        for (kind,color,label) in zip(d['learning_curves'],COLORS,['Random','Uncertainty','Composite','Composite + diversity']):
            c=d['learning_curves'][kind];mean=np.asarray(c['seed_mean_rmse'])[:,j];sd=np.asarray(c['seed_std_rmse'])[:,j]
            ax.plot(d['budgets'],mean,'o-',color=color,label=label,ms=4)
            ax.fill_between(d['budgets'],np.maximum(0,mean-sd),mean+sd,color=color,alpha=.08)
        ax.set(title=title,xticks=d['budgets'],xlabel='Revealed labels',ylabel='Test RMSE');ax.grid(alpha=.15)
        if j==1:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('New experiment: diversity-aware acquisition | all six targets\nFour strategies × three seeds; equal budgets; no coefficient tuning after results')
    g.save(fig,'diversity_acquisition','New fixed0.5 composite/diversity strategy; all6 outputs and3seed SD; archived-label reveals, not fresh simulations')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'#f7f9fc','axes.facecolor':'white','savefig.facecolor':'#f7f9fc','svg.fonttype':'none'})
    g=Gallery(args.output);m=g.json('data/metrics.json')
    state_plot(g,m);operator_ablation(g,m);pinn_plot(g);local_plot(g);uncertainty_plot(g,m)
    ood_plot(g);physics_plot(g);history_plot(g);acquisition_plot(g)
    summary={'status':'completed_research_gallery','figure_count':len(g.figures),'formats':['png','svg'],
             'figures':g.figures,'input_sha256':g.sources,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(g.out.iterdir())},
             'fresh_solver_calls':0,'fresh_training_calls_in_plotting':0}
    (g.out/'gallery_manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({'status':summary['status'],'figures':[f['name'] for f in g.figures],'formats':summary['formats']},indent=2))


if __name__=='__main__':main()
