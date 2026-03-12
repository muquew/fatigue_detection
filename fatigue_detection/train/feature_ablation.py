from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import project_relative, write_json
from train_mlp import build_splitter, set_random_seed, split_iterator, train_one_split

from src.config import default_config


FEATURE_SETS = {
    'raw_frame': [
        'ear_left',
        'ear_right',
        'ear_avg',
        'mar',
        'pitch',
        'yaw',
        'roll',
    ],
    'eye_window': [
        'ear_left',
        'ear_right',
        'ear_avg',
        'ear_mean',
        'ear_min',
        'low_ear_streak',
        'close_eye_ratio',
    ],
    'eye_mouth_window': [
        'ear_left',
        'ear_right',
        'ear_avg',
        'ear_mean',
        'ear_min',
        'low_ear_streak',
        'close_eye_ratio',
        'mar',
        'mar_mean',
        'mar_max',
        'high_mar_streak',
        'yawn_ratio',
    ],
}


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description='Run grouped feature-ablation experiments.')
    parser.add_argument('--dataset', default=str(config.dataset_path))
    parser.add_argument('--output', default=str(config.ablation_report_path))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def summarize(values: list[float]) -> dict[str, float]:
    return {
        'mean': float(np.mean(values)) if values else 0.0,
        'std': float(np.std(values)) if values else 0.0,
    }


def main() -> int:
    args = build_parser().parse_args()
    dataset = np.load(args.dataset, allow_pickle=True)
    X = dataset['X'].astype(np.float32)
    y = dataset['y'].astype(np.int64)
    feature_names = [str(item) for item in dataset['feature_names'].tolist()]
    groups = dataset['groups'].astype(str) if 'groups' in dataset else np.asarray(['default'] * len(X))
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    feature_sets = dict(FEATURE_SETS)
    feature_sets['full'] = feature_names

    splitter, split_name = build_splitter(groups, y)
    experiments: list[dict] = []

    for experiment_index, (experiment_name, selected_names) in enumerate(feature_sets.items(), start=1):
        indices = [feature_index[name] for name in selected_names]
        selected_X = X[:, indices]
        fold_reports: list[dict] = []
        print(f'Running ablation={experiment_name} features={selected_names}')
        for fold_index, (train_idx, val_idx) in enumerate(split_iterator(splitter, selected_X, y, groups), start=1):
            set_random_seed(args.seed + experiment_index * 100 + fold_index)
            _, train_metrics, val_metrics = train_one_split(
                X_train=selected_X[train_idx],
                y_train=y[train_idx],
                X_val=selected_X[val_idx],
                y_val=y[val_idx],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            fold_reports.append(
                {
                    'fold': fold_index,
                    'held_out_groups': sorted(set(groups[val_idx].tolist())),
                    'train': train_metrics,
                    'val': val_metrics,
                }
            )

        experiments.append(
            {
                'name': experiment_name,
                'features': selected_names,
                'num_features': len(selected_names),
                'split_strategy': split_name,
                'seed': args.seed,
                'val_accuracy': summarize([item['val']['accuracy'] for item in fold_reports]),
                'val_precision': summarize([item['val']['precision'] for item in fold_reports]),
                'val_recall': summarize([item['val']['recall'] for item in fold_reports]),
                'val_f1': summarize([item['val']['f1'] for item in fold_reports]),
                'folds': fold_reports,
            }
        )

    best_experiment = max(experiments, key=lambda item: item['val_f1']['mean']) if experiments else None
    report = {
        'dataset': project_relative(args.dataset),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'experiments': experiments,
        'best_by_val_f1': best_experiment['name'] if best_experiment else None,
    }
    write_json(Path(args.output), report)
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
