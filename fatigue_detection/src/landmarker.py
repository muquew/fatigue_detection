from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Any


@dataclass
class LandmarkResult:
    face_detected: bool
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)
    backend: str = "none"
    image_size: tuple[int, int] = (0, 0)
    score: float | None = None

    def point(self, index: int) -> tuple[float, float]:
        x, y, _ = self.landmarks[index]
        return x, y

    def sampled_points(self, step: int = 8) -> list[tuple[int, int]]:
        return [
            (int(x), int(y))
            for idx, (x, y, _) in enumerate(self.landmarks)
            if idx % step == 0
        ]


class FaceLandmarker:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._detector = None
        self._mp = None
        self.backend = "none"
        self.last_error = ""

    def _prepare_model_path(self) -> Path:
        if all(ord(char) < 128 for char in str(self.model_path)):
            return self.model_path

        target_path = Path(tempfile.gettempdir()) / "face_landmarker.task"
        if (
            not target_path.exists()
            or target_path.stat().st_size != self.model_path.stat().st_size
        ):
            shutil.copyfile(self.model_path, target_path)
        return target_path

    def load(self) -> bool:
        try:
            import mediapipe as mp
        except ImportError:
            self.last_error = "mediapipe is not installed"
            return False

        self._mp = mp

        if self.model_path.exists():
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import (
                    FaceLandmarker as MpFaceLandmarker,
                    FaceLandmarkerOptions,
                    RunningMode,
                )

                model_path = self._prepare_model_path()
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path)),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True,
                )
                self._detector = MpFaceLandmarker.create_from_options(options)
                self.backend = "mediapipe_tasks"
                self.last_error = ""
                return True
            except Exception as exc:
                self._detector = None
                self.last_error = f"MediaPipe Tasks load failed: {exc}"
        else:
            self.last_error = f"Model file not found: {self.model_path}"

        self._detector = None
        self.backend = "none"
        return False

    def detect(self, frame: Any) -> LandmarkResult:
        if frame is None or self._detector is None:
            return LandmarkResult(face_detected=False, backend=self.backend)

        try:
            import cv2
        except ImportError:
            return LandmarkResult(face_detected=False, backend=self.backend)

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "mediapipe_tasks":
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            result = self._detector.detect(mp_image)
            if not result.face_landmarks:
                return LandmarkResult(
                    face_detected=False,
                    backend=self.backend,
                    image_size=(width, height),
                )
            raw_landmarks = result.face_landmarks[0]
        else:
            result = self._detector.process(rgb_frame)
            if not result.multi_face_landmarks:
                return LandmarkResult(
                    face_detected=False,
                    backend=self.backend,
                    image_size=(width, height),
                )
            raw_landmarks = result.multi_face_landmarks[0].landmark

        landmarks = [
            (point.x * width, point.y * height, point.z * width)
            for point in raw_landmarks
        ]
        return LandmarkResult(
            face_detected=True,
            landmarks=landmarks,
            backend=self.backend,
            image_size=(width, height),
        )

    def close(self) -> None:
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
        self._detector = None
