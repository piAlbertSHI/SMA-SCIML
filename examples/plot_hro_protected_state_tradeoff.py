"""Create a compact trade-off and stage-error figure from saved HRO audits."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .multicycle_multistep import ROOT
from SciML.models.v12.multistep_historical import sha


DEFAULT_OUTPUT = ROOT / 'assets/hro_protected_state'


def run(output=DEFAULT_OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / 'evidence/experiments/hro_protected_state_ablation_20261123/summary.json').read_text())
    stage = json.loads((ROOT / 'evidence/local_studies/hro_protected_state_ablation_20261123_stage_metrics.json').read_text())
    names = list(summary['aggregate'])
    short = {
        'coordinate_reference': 'Coordinate',
        'jitter_fluid2_stateguard': 'State guard',
        'jitter_fluid2_stageguard': 'Stage guard',
        'jitter_fluid2_pareto': 'Pareto balance',
        'jitter_fluid2_strong_stateguard': 'Strong guard',
    }
    colors = plt.get_cmap('viridis')(np.linspace(.12, .88, len(names)))
    offsets = {
        'coordinate_reference': (6, 6),
        'jitter_fluid2_stateguard': (-72, 12),
        'jitter_fluid2_stageguard': (-72, -16),
        'jitter_fluid2_pareto': (8, 8),
        'jitter_fluid2_strong_stateguard': (8, -20),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    for name, color in zip(names, colors):
        gate = summary['gates']['candidates'][name]
        x = 100 * gate['max_state_regression_fraction']
        y = summary['aggregate'][name]['field']['combined_future_K']['mean']
        ax.scatter(x, y, s=90, color=color, edgecolor='white', linewidth=.8, label=short[name], zorder=3)
        ax.annotate(short[name], (x, y), xytext=offsets[name], textcoords='offset points', fontsize=8)
    ax.axvline(5, color='#9b2c2c', linestyle='--', linewidth=1, label='5% state gate')
    ax.set_xlabel('Maximum state-block regression (%)')
    ax.set_ylabel('Combined future-field RMSE (K)')
    ax.set_title('State–field trade-off')
    ax.grid(alpha=.22)
    ax.set_xlim(left=-.2)
    ax.legend(frameon=False, fontsize=8, loc='upper right')

    stages = ('c1s2', 'c1s4', 'c2s2', 'c2s4')
    matrix = np.asarray([[stage['aggregate'][name]['field_by_stage'][key]['combined_future_K']['mean'] for key in stages] for name in names])
    im = axes[1].imshow(matrix, aspect='auto', cmap='magma_r')
    axes[1].set_xticks(range(len(stages)), stages)
    axes[1].set_yticks(range(len(names)), [short[name] for name in names])
    axes[1].set_title('Stage-resolved future-field RMSE')
    axes[1].set_xlabel('Exchange stage')
    for i in range(len(names)):
        for j in range(len(stages)):
            axes[1].text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white' if matrix[i, j] > np.median(matrix) else '#2d1e2f', fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=.046, pad=.04, label='RMSE (K)')
    fig.suptitle('HRO protected-state screening', fontsize=14, fontweight='bold')
    svg = output / 'protected_state_tradeoff.svg'
    png = output / 'protected_state_tradeoff.png'
    fig.savefig(svg, format='svg')
    fig.savefig(png, dpi=180)
    plt.close(fig)
    manifest = {
        'status': 'generated_hro_protected_state_tradeoff_figure',
        'generator_sha256': sha(ROOT / 'examples/plot_hro_protected_state_tradeoff.py'),
        'source_summary_sha256': sha(ROOT / 'evidence/experiments/hro_protected_state_ablation_20261123/summary.json'),
        'source_stage_metrics_sha256': sha(ROOT / 'evidence/local_studies/hro_protected_state_ablation_20261123_stage_metrics.json'),
        'artifacts': {path.name: sha(path) for path in (svg, png)},
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
