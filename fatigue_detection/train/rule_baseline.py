from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import binary_metrics, project_relative, write_json

from src.config import default_config


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description='Evaluate a hand-crafted rule baseline.')
    parser.add_argument('--dataset', default=str(config.dataset_path))
    parser.add_argument('--output', default=str(config.rule_baseline_report_path))
    return parser


def build_feature_index(feature_names: np.ndarray) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(feature_names.tolist())}


def predict_rules(X: np.ndarray, feature_index: dict[str, int]) -> np.ndarray:
    close_eye_ratio = X[:, feature_index['close_eye_ratio']]
    low_ear_streak = X[:, feature_index['low_ear_streak']]
    yawn_ratio = X[:, feature_index['yawn_ratio']]
    mar_max = X[:, feature_index['mar_max']]
    ear_min = X[:, feature_index['ear_min']]

    predictions = (
        (close_eye_ratio >= 0.25)
        | (low_ear_streak >= 4.0)
        | (yawn_ratio >= 0.10)
        | (mar_max >= 0.18)
        | (ear_min <= 0.22)
    )
    return predictions.astype(np.int64)


def main() -> int:
    args = build_parser().parse_args()
    dataset = np.load(args.dataset, allow_pickle=True)
    X = dataset['X'].astype(np.float32)
    y = dataset['y'].astype(np.int64)
    feature_index = build_feature_index(dataset['feature_names'])
    groups = dataset['groups'].astype(str) if 'groups' in dataset else np.asarray(['default'] * len(X))

    predictions = predict_rules(X, feature_index)
    overall = binary_metrics(y.tolist(), predictions.tolist())

    group_reports: list[dict] = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        metrics = binary_metrics(y[mask].tolist(), predictions[mask].tolist())
        metrics['group'] = group
        metrics['num_samples'] = int(mask.sum())
        group_reports.append(metrics)

    macro = {
        metric: float(np.mean([report[metric] for report in group_reports]))
        for metric in ('accuracy', 'precision', 'recall', 'f1')
    }

    report = {
        'dataset': project_relative(args.dataset),
        'thresholds': {
            'close_eye_ratio_min': 0.25,
            'low_ear_streak_min': 4.0,
            'yawn_ratio_min': 0.10,
            'mar_max_min': 0.18,
            'ear_min_max': 0.22,
        },
        'overall': overall,
        'macro_by_group': macro,
        'groups': group_reports,
        'num_samples': int(len(y)),
    }
    output_path = Path(args.output)
    write_json(output_path, report)
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
