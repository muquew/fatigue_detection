from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dims: tuple[int, int] = (32, 16)
    num_classes: int = 2


class SimpleMLPFactory:
    @staticmethod
    def build_torch_model(config: MLPConfig):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed.") from exc

        return nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.num_classes),
        )

    @staticmethod
    def save_checkpoint(model, path: Path, config: MLPConfig) -> None:
        import torch

        checkpoint = {
            "state_dict": model.state_dict(),
            "input_dim": config.input_dim,
            "hidden_dims": config.hidden_dims,
            "num_classes": config.num_classes,
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path: Path, device: str = "cpu"):
        import torch

        payload = torch.load(path, map_location=device)
        config = MLPConfig(
            input_dim=int(payload["input_dim"]),
            hidden_dims=tuple(payload.get("hidden_dims", (32, 16))),
            num_classes=int(payload.get("num_classes", 2)),
        )
        model = SimpleMLPFactory.build_torch_model(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, config, payload
