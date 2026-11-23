"""Refined material-conditioned HTF screen with smaller channel weights."""
import argparse
import copy
import json
from pathlib import Path

from .hro_material_conditioned_ablation import BASE, run, verify


REFINED_CANDIDATES = {
    'global_pareto_control': copy.deepcopy(BASE),
    'recAS_htf_light': {
        **copy.deepcopy(BASE),
        'material_field_channel_weights': {'recA': [1., 1.05], 'recAS': [1., 1.10]},
    },
    'recAS_htf_moderate': {
        **copy.deepcopy(BASE),
        'material_field_channel_weights': {'recA': [1., 1.10], 'recAS': [1., 1.25]},
    },
}
DEFAULT = Path(__file__).resolve().parents[1] / 'evidence/experiments/hro_material_conditioned_refined_20261123'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--batch-cases', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output)
    else:
        result = run(args.output, args.updates, args.batch_cases, args.lr,
                     candidate_settings=REFINED_CANDIDATES,
                     experiment_name='hro_material_conditioned_refined_20261123')
    print(json.dumps(result, indent=2))
