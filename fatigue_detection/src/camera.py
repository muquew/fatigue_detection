from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass
class FramePacket:
    frame: Any
    timestamp: float


class CameraStream:
    def __init__(self, source: int | str, width: int, height: int) -> None:
        self.source = source
        self.width = width
        self.height = height
        self._capture = None

    def open(self) -> bool:
        try:
            import cv2
        except ImportError:
            return False

        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self._capture = None
            return False

        if isinstance(self.source, int):
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def read(self) -> FramePacket | None:
        if self._capture is None:
            return None

        success, frame = self._capture.read()
        if not success:
            return None

        return FramePacket(frame=frame, timestamp=perf_counter())

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
