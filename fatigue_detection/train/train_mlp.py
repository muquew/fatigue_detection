from __future__ import annotations

import argparse
import pickle
import random
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import binary_metrics, write_json

from src.classifier import MLPConfig, SimpleMLPFactory
from src.config import default_config


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description='Train the fatigue MLP classifier.')
    parser.add_argument('--dataset', default=str(config.dataset_path))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int):
    indices = np.random.permutation(len(X))
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]


def evaluate_numpy(model, X: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return binary_metrics(y.tolist(), predictions.tolist()), predictions


def build_splitter(groups: np.ndarray, y: np.ndarray):
    unique_groups = np.unique(groups)
    if len(unique_groups) > 1:
        if len(unique_groups) <= 5:
            return LeaveOneGroupOut(), 'leave_one_group_out'
        return GroupKFold(n_splits=min(5, len(unique_groups))), 'group_kfold'
    return StratifiedKFold(
        n_splits=min(5, max(2, len(np.unique(y)))),
        shuffle=True,
        random_state=42,
    ), 'stratified_kfold'


def split_iterator(splitter, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    if isinstance(splitter, (LeaveOneGroupOut, GroupKFold)):
        yield from splitter.split(X, y, groups=groups)
    else:
        indices = np.arange(len(X))
        yield from splitter.split(indices, y)


def train_one_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[dict, dict, dict]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    model_config = MLPConfig(input_dim=X_train_scaled.shape[1])
    model = SimpleMLPFactory.build_torch_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_state_dict = None
    best_train_metrics = None
    best_val_metrics = None
    best_val_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch_X, batch_y in iterate_minibatches(X_train_scaled, y_train, batch_size):
            optimizer.zero_grad()
            logits = model(torch.from_numpy(batch_X).float())
            loss = criterion(logits, torch.from_numpy(batch_y).long())
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        train_metrics, _ = evaluate_numpy(model, X_train_scaled, y_train)
        val_metrics, _ = evaluate_numpy(model, X_val_scaled, y_val)
        print(
            f'epoch={epoch:03d} loss={np.mean(epoch_losses):.4f} '
            f'train_f1={train_metrics["f1"]:.4f} val_f1={val_metrics["f1"]:.4f}'
        )

        if val_metrics['f1'] >= best_val_f1:
            best_val_f1 = float(val_metrics['f1'])
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

    if best_state_dict is None or best_train_metrics is None or best_val_metrics is None:
        raise RuntimeError('Training did not produce a valid checkpoint.')

    return best_state_dict, best_train_metrics, best_val_metrics


def summarize_fold_metrics(folds: Iterable[dict]) -> dict:
    folds = list(folds)
    metric_names = ('accuracy', 'precision', 'recall', 'f1')
    summary: dict[str, dict[str, float]] = {}
    for scope in ('train', 'val'):
        summary[scope] = {}
        for metric_name in metric_names:
            values = [float(item[scope][metric_name]) for item in folds]
            summary[scope][f'{metric_name}_mean'] = float(np.mean(values))
            summary[scope][f'{metric_name}_std'] = float(np.std(values))
    return summary


def fit_full_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[torch.nn.Module, StandardScaler, MLPConfig, dict]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    model_config = MLPConfig(input_dim=X_scaled.shape[1])
    model = SimpleMLPFactory.build_torch_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in iterate_minibatches(X_scaled, y, batch_size):
            optimizer.zero_grad()
            logits = model(torch.from_numpy(batch_X).float())
            loss = criterion(logits, torch.from_numpy(batch_y).long())
            loss.backward()
            optimizer.step()

    model.eval()
    metrics, _ = evaluate_numpy(model, X_scaled, y)
    return model, scaler, model_config, metrics


def main() -> int:
    args = build_parser().parse_args()
    config = default_config()
    dataset = np.load(args.dataset, allow_pickle=True)
    X = dataset['X'].astype(np.float32)
    y = dataset['y'].astype(np.int64)
    groups = dataset['groups'].astype(str) if 'groups' in dataset else np.asarray(['default'] * len(X))

    splitter, split_name = build_splitter(groups, y)
    fold_reports: list[dict] = []
    best_fold_index = -1
    best_fold_f1 = -1.0

    for fold_index, (train_idx, val_idx) in enumerate(split_iterator(splitter, X, y, groups), start=1):
        held_out_groups = sorted(set(groups[val_idx].tolist()))
        print(f'fold={fold_index:02d} held_out_groups={held_out_groups}')
        set_random_seed(args.seed + fold_index)
        _, train_metrics, val_metrics = train_one_split(
            X_train=X[train_idx],
            y_train=y[train_idx],
            X_val=X[val_idx],
            y_val=y[val_idx],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        fold_report = {
            'fold': fold_index,
            'train': train_metrics,
            'val': val_metrics,
            'train_size': int(len(train_idx)),
            'val_size': int(len(val_idx)),
            'held_out_groups': held_out_groups,
        }
        fold_reports.append(fold_report)
        if float(val_metrics['f1']) >= best_fold_f1:
            best_fold_f1 = float(val_metrics['f1'])
            best_fold_index = fold_index

    if not fold_reports:
        print('No folds were generated for training.')
        return 1

    summary = summarize_fold_metrics(fold_reports)

    set_random_seed(args.seed)
    final_model, final_scaler, final_model_config, full_train_metrics = fit_full_model(
        X=X,
        y=y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    config.scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with config.scaler_path.open('wb') as handle:
        pickle.dump(final_scaler, handle)

    SimpleMLPFactory.save_checkpoint(final_model, config.torch_model_path, final_model_config)

    training_report = {
        'cross_validation': {
            'split_strategy': split_name,
            'num_groups': int(len(np.unique(groups))),
            'num_folds': int(len(fold_reports)),
            'best_fold': best_fold_index,
            'folds': fold_reports,
            'summary': summary,
        },
        'final_model': {
            'trained_on_full_dataset': True,
            'train_metrics': full_train_metrics,
        },
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'feature_dim': int(X.shape[1]),
        'num_samples': int(len(X)),
        'class_distribution': {
            'normal': int((y == 0).sum()),
            'fatigue': int((y == 1).sum()),
        },
    }
    write_json(config.training_report_path, training_report)
    print(f'Saved model to {config.torch_model_path}')
    print(f'Saved scaler to {config.scaler_path}')
    print(f'Saved metrics to {config.training_report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
