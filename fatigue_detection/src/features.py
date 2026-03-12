from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, sqrt

import numpy as np

try:
    from .config import AppConfig
    from .landmarker import LandmarkResult
except ImportError:
    from config import AppConfig
    from landmarker import LandmarkResult


FRAME_FEATURE_NAMES = (
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "pitch",
    "yaw",
    "roll",
)

LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
MOUTH_HORIZONTAL_INDICES = (61, 291)
MOUTH_VERTICAL_PAIRS = ((81, 178), (13, 14), (311, 402))
POSE_INDICES = (1, 152, 33, 263, 61, 291)
POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass
class FrameFeatures:
    ear_left: float
    ear_right: float
    ear_avg: float
    mar: float
    pitch: float
    yaw: float
    roll: float

    def as_list(self) -> list[float]:
        return [
            self.ear_left,
            self.ear_right,
            self.ear_avg,
            self.mar,
            self.pitch,
            self.yaw,
            self.roll,
        ]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FRAME_FEATURE_NAMES, self.as_list()))


class FeatureExtractor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _eye_aspect_ratio(
        self,
        landmarks: LandmarkResult,
        indices: tuple[int, int, int, int, int, int],
    ) -> float:
        p1, p2, p3, p4, p5, p6 = [landmarks.point(index) for index in indices]
        horizontal = self.euclidean(p1, p4)
        if horizontal <= 1e-6:
            return 0.0
        vertical = self.euclidean(p2, p6) + self.euclidean(p3, p5)
        return vertical / (2.0 * horizontal)

    def _mouth_aspect_ratio(self, landmarks: LandmarkResult) -> float:
        left = landmarks.point(MOUTH_HORIZONTAL_INDICES[0])
        right = landmarks.point(MOUTH_HORIZONTAL_INDICES[1])
        horizontal = self.euclidean(left, right)
        if horizontal <= 1e-6:
            return 0.0
        vertical = sum(
            self.euclidean(landmarks.point(start), landmarks.point(end))
            for start, end in MOUTH_VERTICAL_PAIRS
        )
        return vertical / (3.0 * horizontal)

    def _estimate_pose_fallback(
        self, landmarks: LandmarkResult
    ) -> tuple[float, float, float]:
        left_eye = np.mean(
            [landmarks.point(LEFT_EYE_INDICES[0]), landmarks.point(LEFT_EYE_INDICES[3])],
            axis=0,
        )
        right_eye = np.mean(
            [
                landmarks.point(RIGHT_EYE_INDICES[0]),
                landmarks.point(RIGHT_EYE_INDICES[3]),
            ],
            axis=0,
        )
        nose = np.array(landmarks.point(1))
        mouth = np.mean([landmarks.point(61), landmarks.point(291)], axis=0)

        eye_vector = right_eye - left_eye
        inter_eye = max(float(np.linalg.norm(eye_vector)), 1e-6)
        roll = degrees(atan2(float(eye_vector[1]), float(eye_vector[0])))

        eye_mid = (left_eye + right_eye) / 2.0
        vertical_scale = max(float(abs(mouth[1] - eye_mid[1])), 1e-6)
        yaw = float((nose[0] - eye_mid[0]) / inter_eye) * 45.0
        pitch = float((nose[1] - ((eye_mid[1] + mouth[1]) / 2.0)) / vertical_scale) * 45.0
        return pitch, yaw, roll

    def _rotation_matrix_to_euler(
        self, rotation_matrix: np.ndarray
    ) -> tuple[float, float, float]:
        sy = sqrt(
            rotation_matrix[0, 0] * rotation_matrix[0, 0]
            + rotation_matrix[1, 0] * rotation_matrix[1, 0]
        )
        singular = sy < 1e-6
        if not singular:
            pitch = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = atan2(-rotation_matrix[2, 0], sy)
            roll = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = atan2(-rotation_matrix[2, 0], sy)
            roll = 0.0
        return degrees(pitch), degrees(yaw), degrees(roll)

    def _estimate_pose(self, landmarks: LandmarkResult) -> tuple[float, float, float]:
        try:
            import cv2
        except ImportError:
            return self._estimate_pose_fallback(landmarks)

        width, height = landmarks.image_size
        if width <= 0 or height <= 0:
            return self._estimate_pose_fallback(landmarks)

        image_points = np.array(
            [landmarks.point(index) for index in POSE_INDICES],
            dtype=np.float64,
        )
        focal_length = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            POSE_MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return self._estimate_pose_fallback(landmarks)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        return self._rotation_matrix_to_euler(rotation_matrix)

    def extract(self, landmarks: LandmarkResult) -> FrameFeatures | None:
        if not landmarks.face_detected:
            return None

        try:
            ear_left = self._eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
            ear_right = self._eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
            mar = self._mouth_aspect_ratio(landmarks)
            pitch, yaw, roll = self._estimate_pose(landmarks)
        except (IndexError, ZeroDivisionError, ValueError):
            return None

        return FrameFeatures(
            ear_left=float(ear_left),
            ear_right=float(ear_right),
            ear_avg=float((ear_left + ear_right) / 2.0),
            mar=float(mar),
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
        )
