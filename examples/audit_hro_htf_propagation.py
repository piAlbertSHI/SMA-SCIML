"""Diagnose descriptive links between HTF error, memory and late states."""
import argparse
import json
from pathlib import Path

import numpy as np

from .multicycle_multistep import ROOT, write
from SciML.models.v12.multistep_historical import sha


DEFAULT_INPUT = ROOT / 'evidence/local_studies/hro_temporal_channel_confirmation_20261123/test_audit.json'
DEFAULT_OUTPUT = ROOT / 'evidence/local_studies/hro_temporal_channel_confirmation_20261123/htf_propagation_diagnostic.json'


def _corr(rows, key, target='htf_field_K'):
    x = np.asarray([row[key] for row in rows], dtype=float)
    y = np.asarray([row[target] for row in rows], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def run(source=DEFAULT_INPUT, output=DEFAULT_OUTPUT):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new HTF propagation diagnostic path')
    audit = json.loads(source.read_text())
    rows = []
    for report in audit['reports']:
        for case_id, score in report['per_three_cycle_case'].items():
            if 'c3s4' not in score['state'] or 'c3s4' not in score['field']:
                continue
            rows.append({
                'seed': report['seed'],
                'case_id': case_id,
                'material': 'recAS' if '_recas_' in case_id.lower() else 'recA',
                'memory_K': score['state']['c3s4']['memory_K'],
                'htf2_state_K': score['state']['c3s4']['htf2_K'],
                'htf4_state_K': score['state']['c3s4']['htf4_K'],
                'sma_state_K': score['state']['c3s4']['sma_K'],
                'htf_field_K': score['field']['c3s4']['htf_K'],
                'sma_field_K': score['field']['c3s4']['sma_K'],
            })
    correlations = {
        group: {
            key: _corr(selected, key)
            for key in ('memory_K', 'htf2_state_K', 'htf4_state_K', 'sma_state_K')
        }
        for group, selected in {
            'all': rows,
            'recA': [row for row in rows if row['material'] == 'recA'],
            'recAS': [row for row in rows if row['material'] == 'recAS'],
        }.items()
    }
    result = {
        'status': 'completed_htf_memory_stage_diagnostic',
        'source': str(source.relative_to(ROOT)).replace('\\', '/'),
        'source_sha256': sha(source),
        'rows': rows,
        'correlation_with_c3s4_htf_field': correlations,
        'interpretation': 'Correlations are descriptive over six seed-material rows. They diagnose co-variation and do not establish causal attribution or a physical energy balance.',
        'fresh_training_calls': 0,
        'fresh_solver_calls': 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    return {'status': result['status'], 'row_count': len(rows), 'fresh_solver_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))
