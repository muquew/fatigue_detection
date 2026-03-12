from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch

from common import binary_metrics, write_json

from src.classifier import SimpleMLPFactory
from src.config import default_config


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description="Evaluate the trained fatigue model.")
    parser.add_argument("--dataset", default=str(config.dataset_path))
    parser.add_argument("--checkpoint", default=str(config.torch_model_path))
    parser.add_argument("--scaler", default=str(config.scaler_path))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = default_config()
    dataset = np.load(args.dataset, allow_pickle=True)
    X = dataset["X"].astype(np.float32)
    y = dataset["y"].astype(np.int64)

    with open(args.scaler, "rb") as handle:
        scaler = pickle.load(handle)
    X_scaled = scaler.transform(X).astype(np.float32)

    model, _, _ = SimpleMLPFactory.load_checkpoint(args.checkpoint)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled).float())
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    metrics = binary_metrics(y.tolist(), predictions.tolist())
    metrics["num_samples"] = int(len(y))
    write_json(config.evaluation_report_path, metrics)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
