"""Refined low-weight SMA-only late-time screen."""
import argparse
import copy
import json
from pathlib import Path

from .hro_temporal_profile_ablation import BASE, run, verify


CANDIDATES = {
    'protected_state_control': {**copy.deepcopy(BASE), 'late_time_weight': 0.0},
    'protected_state_sma_late_micro': {
        **copy.deepcopy(BASE), 'late_time_weight': 0.10, 'late_time_channel_weights': [1., 0.]
    },
    'protected_state_sma_late_small': {
        **copy.deepcopy(BASE), 'late_time_weight': 0.15, 'late_time_channel_weights': [1., 0.]
    },
}
DEFAULT = Path(__file__).resolve().parents[1] / 'evidence/experiments/hro_temporal_channel_refined_20261123'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--batch-cases', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    result = verify(args.output) if args.verify_only else run(
        args.output, args.updates, args.batch_cases, args.lr,
        candidate_settings=CANDIDATES,
        experiment_name='hro_temporal_channel_refined_20261123')
    print(json.dumps(result, indent=2))
