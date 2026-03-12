from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class InferenceResult:
    label_id: int
    confidence: float
    source: str
    raw_label_id: int | None = None
    fatigue_probability: float | None = None


class OnnxFatigueInferencer:
    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        confidence_threshold: float,
    ) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.confidence_threshold = confidence_threshold
        self._session = None
        self._input_name = None
        self._scaler = None

    def load(self) -> bool:
        if self.scaler_path.exists():
            with self.scaler_path.open("rb") as handle:
                self._scaler = pickle.load(handle)

        if not self.model_path.exists():
            return False

        try:
            import onnxruntime as ort
        except ImportError:
            return False

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        return True

    def _transform(self, features: list[float]) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if self._scaler is None:
            return array
        if hasattr(self._scaler, "transform"):
            return self._scaler.transform(array).astype(np.float32)
        if isinstance(self._scaler, dict) and {"mean_", "scale_"} <= set(self._scaler):
            return ((array - self._scaler["mean_"]) / self._scaler["scale_"]).astype(
                np.float32
            )
        return array

    def predict(self, features: list[float]) -> InferenceResult:
        if self._session is not None and self._input_name is not None:
            transformed = self._transform(features)
            outputs = self._session.run(None, {self._input_name: transformed})
            logits = np.asarray(outputs[0], dtype=np.float32)
            logits = logits[0] if logits.ndim > 1 else logits
            shifted = logits - np.max(logits)
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
            raw_label_id = int(np.argmax(probabilities))
            fatigue_probability = float(probabilities[1]) if probabilities.size > 1 else float(probabilities[0])
            label_id = 1 if fatigue_probability >= self.confidence_threshold else 0
            confidence = fatigue_probability if label_id == 1 else 1.0 - fatigue_probability
            return InferenceResult(
                label_id=label_id,
                confidence=confidence,
                source="onnx",
                raw_label_id=raw_label_id,
                fatigue_probability=fatigue_probability,
            )

        ear_component = max(0.0, (0.30 - features[2]) * 3.0)
        mar_component = max(0.0, (features[3] - 0.45) * 1.8)
        fatigue_score = max(0.0, min(1.0, ear_component + mar_component))
        label_id = 1 if fatigue_score >= self.confidence_threshold else 0
        return InferenceResult(
            label_id=label_id,
            confidence=fatigue_score if label_id == 1 else 1.0 - fatigue_score,
            source="heuristic",
            raw_label_id=label_id,
            fatigue_probability=fatigue_score,
        )
