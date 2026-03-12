from __future__ import annotations

import argparse

import torch

from common import ROOT_DIR  # noqa: F401

from src.classifier import SimpleMLPFactory
from src.config import default_config


def build_parser() -> argparse.ArgumentParser:
    config = default_config()
    parser = argparse.ArgumentParser(description="Export trained torch model to ONNX.")
    parser.add_argument("--checkpoint", default=str(config.torch_model_path))
    parser.add_argument("--output", default=str(config.onnx_model_path))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model, model_config, _ = SimpleMLPFactory.load_checkpoint(args.checkpoint)
    dummy_input = torch.randn(1, model_config.input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        dynamo=False,
    )
    print(f"Exported ONNX model to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
