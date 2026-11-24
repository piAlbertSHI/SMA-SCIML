"""Screen late-time profiles applied to the SMA field channel only."""
import argparse
import copy
import json
from pathlib import Path

from .hro_temporal_profile_ablation import BASE, run, verify


CANDIDATES = {
    'protected_state_control': {**copy.deepcopy(BASE), 'late_time_weight': 0.0},
    'protected_state_sma_late_light': {
        **copy.deepcopy(BASE), 'late_time_weight': 0.25, 'late_time_channel_weights': [1., 0.]
    },
    'protected_state_sma_late_moderate': {
        **copy.deepcopy(BASE), 'late_time_weight': 0.5, 'late_time_channel_weights': [1., 0.]
    },
}
DEFAULT = Path(__file__).resolve().parents[1] / 'evidence/experiments/hro_temporal_channel_profile_ablation_20261123'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--batch-cases', type=int, default=2)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output)
    else:
        result = run(args.output, args.updates, args.batch_cases, args.lr,
                     candidate_settings=CANDIDATES,
                     experiment_name='hro_temporal_channel_profile_ablation_20261123')
    print(json.dumps(result, indent=2))
