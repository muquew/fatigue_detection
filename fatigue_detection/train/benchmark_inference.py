from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch

from common import project_relative, write_json

from src.classifier import SimpleMLPFactory
from src.config import default_config
from src.infer_onnx import OnnxFatigueInferencer


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description='Benchmark PyTorch and ONNX inference latency.')
    parser.add_argument('--dataset', default=str(config.dataset_path))
    parser.add_argument('--checkpoint', default=str(config.torch_model_path))
    parser.add_argument('--scaler', default=str(config.scaler_path))
    parser.add_argument('--onnx', default=str(config.onnx_model_path))
    parser.add_argument('--output', default=str(config.benchmark_report_path))
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=32)
    return parser


def load_scaled_subset(dataset_path: Path, scaler_path: Path, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = np.load(dataset_path, allow_pickle=True)
    X = dataset['X'].astype(np.float32)
    with scaler_path.open('rb') as handle:
        scaler = pickle.load(handle)
    X_scaled = scaler.transform(X).astype(np.float32)
    count = min(len(X_scaled), max(1, num_samples))
    return X[:count], X_scaled[:count]


def benchmark_torch_forward(model, X_scaled: np.ndarray, warmup: int) -> float:
    tensors = [torch.from_numpy(row.reshape(1, -1)).float() for row in X_scaled]
    with torch.no_grad():
        for index in range(min(warmup, len(tensors))):
            model(tensors[index])
        start = perf_counter()
        for tensor in tensors:
            model(tensor)
        elapsed = perf_counter() - start
    return (elapsed / max(len(tensors), 1)) * 1000.0


def benchmark_onnx_forward(session, input_name: str, X_scaled: np.ndarray, warmup: int) -> float:
    for index in range(min(warmup, len(X_scaled))):
        session.run(None, {input_name: X_scaled[index : index + 1]})
    start = perf_counter()
    for row in X_scaled:
        session.run(None, {input_name: row.reshape(1, -1)})
    elapsed = perf_counter() - start
    return (elapsed / max(len(X_scaled), 1)) * 1000.0


def benchmark_predict_api(inferencer: OnnxFatigueInferencer, X_raw: np.ndarray, warmup: int) -> float:
    for index in range(min(warmup, len(X_raw))):
        inferencer.predict(X_raw[index].tolist())
    start = perf_counter()
    for row in X_raw:
        inferencer.predict(row.tolist())
    elapsed = perf_counter() - start
    return (elapsed / max(len(X_raw), 1)) * 1000.0


def torch_predictions(model, X_scaled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.from_numpy(X_scaled).float())
        return torch.argmax(logits, dim=1).cpu().numpy()


def onnx_predictions(session, input_name: str, X_scaled: np.ndarray) -> np.ndarray:
    logits = session.run(None, {input_name: X_scaled})[0]
    return np.argmax(np.asarray(logits), axis=1)


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    scaler_path = Path(args.scaler)
    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)

    X_raw, X_scaled = load_scaled_subset(dataset_path, scaler_path, args.num_samples)
    model, _, _ = SimpleMLPFactory.load_checkpoint(checkpoint_path)
    model.eval()

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    inferencer = OnnxFatigueInferencer(
        model_path=onnx_path,
        scaler_path=scaler_path,
        confidence_threshold=default_config().fatigue_confidence_threshold,
    )
    inferencer.load()

    torch_ms = benchmark_torch_forward(model, X_scaled, args.warmup)
    onnx_ms = benchmark_onnx_forward(session, input_name, X_scaled, args.warmup)
    end_to_end_onnx_ms = benchmark_predict_api(inferencer, X_raw, args.warmup)

    torch_pred = torch_predictions(model, X_scaled)
    onnx_pred = onnx_predictions(session, input_name, X_scaled)
    agreement = float((torch_pred == onnx_pred).mean()) if len(torch_pred) else 0.0

    report = {
        'dataset': project_relative(dataset_path),
        'num_samples': int(len(X_scaled)),
        'torch_forward_ms_per_sample': torch_ms,
        'onnx_forward_ms_per_sample': onnx_ms,
        'onnx_end_to_end_ms_per_sample': end_to_end_onnx_ms,
        'forward_speedup_vs_torch': float(torch_ms / max(onnx_ms, 1e-6)),
        'prediction_agreement': agreement,
    }
    write_json(output_path, report)
    print(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
