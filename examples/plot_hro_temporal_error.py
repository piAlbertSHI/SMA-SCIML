"""Plot post-selection third-cycle temporal error by material and channel."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .multicycle_multistep import ROOT
from SciML.models.v12.multistep_historical import sha


DEFAULT_INPUT = ROOT / 'evidence/local_studies/hro_protected_state_confirmation_20261123/temporal_error.json'
DEFAULT_OUTPUT = ROOT / 'assets/hro_protected_state'


def run(source=DEFAULT_INPUT, output=DEFAULT_OUTPUT):
    source = Path(source)
    output = Path(output)
    data = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    tau = np.asarray(data['time_fraction'])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True, constrained_layout=True)
    for row, material in enumerate(('recA', 'recAS')):
        for col, channel in enumerate(('sma_K', 'htf_K')):
            ax = axes[row, col]
            for model, color, label in (('protected_state', '#238b8d', 'Protected state'), ('formal_reference', '#4c2a85', 'Formal reference')):
                mean = np.asarray(data['aggregate'][model][material][channel]['mean'])
                ax.plot(tau, mean, color=color, linewidth=2, label=label)
            ax.set_title(f'{material} · {channel.replace("_K", "").upper()}')
            ax.set_ylabel('RMSE (K)')
            ax.grid(alpha=.22)
            if row == 1:
                ax.set_xlabel('Normalized stage-4 time')
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Third-cycle stage-4 temporal error', fontsize=14, fontweight='bold')
    svg = output / 'third_cycle_temporal_error.svg'
    png = output / 'third_cycle_temporal_error.png'
    fig.savefig(svg, format='svg')
    fig.savefig(png, dpi=180)
    plt.close(fig)
    manifest = {
        'status': 'generated_third_cycle_temporal_error_figure',
        'source_sha256': sha(source),
        'generator_sha256': sha(ROOT / 'examples/plot_hro_temporal_error.py'),
        'artifacts': {path.name: sha(path) for path in (svg, png)},
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
    }
    (output / 'temporal_error_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8', newline='\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
