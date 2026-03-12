from __future__ import annotations

from collections import deque

try:
    from .config import AppConfig
    from .features import FRAME_FEATURE_NAMES, FrameFeatures
except ImportError:
    from config import AppConfig
    from features import FRAME_FEATURE_NAMES, FrameFeatures


WINDOW_STAT_FEATURE_NAMES = (
    "ear_mean",
    "ear_min",
    "mar_mean",
    "mar_max",
    "low_ear_streak",
    "high_mar_streak",
    "close_eye_ratio",
    "yawn_ratio",
)
WINDOW_FEATURE_NAMES = FRAME_FEATURE_NAMES + WINDOW_STAT_FEATURE_NAMES


def _tail_streak(values: list[float], predicate) -> int:
    count = 0
    for value in reversed(values):
        if not predicate(value):
            break
        count += 1
    return count


def build_window_feature_vector(
    frames: list[FrameFeatures], config: AppConfig
) -> list[float] | None:
    if len(frames) < config.window_size:
        return None

    window = frames[-config.window_size :]
    ear_values = [item.ear_avg for item in window]
    mar_values = [item.mar for item in window]
    latest = window[-1]

    low_ear_streak = _tail_streak(
        ear_values, lambda value: value < config.low_ear_threshold
    )
    high_mar_streak = _tail_streak(
        mar_values, lambda value: value > config.high_mar_threshold
    )
    close_eye_ratio = sum(
        1 for value in ear_values if value < config.low_ear_threshold
    ) / len(ear_values)
    yawn_ratio = sum(
        1 for value in mar_values if value > config.high_mar_threshold
    ) / len(mar_values)

    return latest.as_list() + [
        sum(ear_values) / len(ear_values),
        min(ear_values),
        sum(mar_values) / len(mar_values),
        max(mar_values),
        float(low_ear_streak),
        float(high_mar_streak),
        close_eye_ratio,
        yawn_ratio,
    ]


class FeatureWindowBuffer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._frames: deque[FrameFeatures] = deque(maxlen=config.window_size)

    def add(self, features: FrameFeatures) -> None:
        self._frames.append(features)

    def ready(self) -> bool:
        return len(self._frames) == self.config.window_size

    def clear(self) -> None:
        self._frames.clear()

    def build_feature_vector(self) -> list[float] | None:
        return build_window_feature_vector(list(self._frames), self.config)
