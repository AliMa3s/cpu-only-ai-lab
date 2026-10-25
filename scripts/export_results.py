"""Export shareable measurements and synthetic data, excluding model weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, default=Path('artifacts/train-run'))
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    status = json.loads((args.training_dir / 'run_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise RuntimeError('Training must finish before exporting final results.')
    metrics = json.loads((args.training_dir / 'metrics.json').read_text(encoding='utf-8'))
    if not metrics['dataset']['source'].startswith('deterministic synthetic'):
        raise RuntimeError('Custom data may appear in raw predictions. Review and export those results manually.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'metrics.json': 'training.json',
        'training_curve.json': 'training-curve.json',
        'resolved_config.json': 'training-config.json',
    }
    for source, destination in files.items():
        shutil.copyfile(args.training_dir / source, args.output_dir / destination)
    # Export generated synthetic data only. Explicit user datasets are not published.
    if metrics['dataset']['source'].startswith('deterministic synthetic'):
        data_output = args.output_dir / 'data'
        data_output.mkdir(exist_ok=True)
        for split in ('train', 'validation', 'test'):
            shutil.copyfile(args.training_dir / 'data' / f'{split}.jsonl', data_output / f'{split}.jsonl')
    print(f'Exported training measurements to {args.output_dir}')


if __name__ == '__main__':
    main()
