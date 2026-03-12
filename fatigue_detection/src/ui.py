from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .landmarker import LandmarkResult
except ImportError:
    from landmarker import LandmarkResult


LANGUAGE_PACK = {
    "en": {
        "state": "State",
        "confidence": "Confidence",
        "fps": "FPS",
        "raw_state": "Raw",
        "vote_ratio": "Vote",
        "fatigue_prob": "Fatigue Prob",
        "ear": "EAR",
        "mar": "MAR",
        "pose": "Pose",
        "backend": "Backend",
        "controls_compact": "L: lang  I: details  Q/ESC: exit",
        "controls_detail": "L: language  I: details  Q/ESC: exit",
        "alert": "ALERT: sustained fatigue risk",
    },
    "zh": {
        "state": "状态",
        "confidence": "置信度",
        "fps": "帧率",
        "raw_state": "原始判断",
        "vote_ratio": "投票比例",
        "fatigue_prob": "疲劳概率",
        "ear": "眼部EAR",
        "mar": "嘴部MAR",
        "pose": "头姿",
        "backend": "推理后端",
        "controls_compact": "L切语言  I看详情  Q/ESC退出",
        "controls_detail": "L切换语言  I展开详情  Q/ESC退出",
        "alert": "告警：持续疲劳风险",
    },
}

STATE_TRANSLATIONS = {
    "en": {
        "Normal": "Normal",
        "Fatigue Risk": "Fatigue Risk",
        "No Face": "No Face",
        "Warmup": "Warmup",
        "Invalid Features": "Invalid Features",
    },
    "zh": {
        "Normal": "正常",
        "Fatigue Risk": "疲劳风险",
        "No Face": "未检测到人脸",
        "Warmup": "预热中",
        "Invalid Features": "特征异常",
    },
}


@dataclass
class StatusOverlay:
    state_label: str
    confidence: float
    fps: float
    alert: bool
    ear: float | None = None
    mar: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    backend: str = ""
    message: str = ""
    raw_state_label: str = ""
    vote_ratio: float | None = None
    fatigue_probability: float | None = None


class UIOverlay:
    def __init__(self, language: str = "zh") -> None:
        self.language = language if language in LANGUAGE_PACK else "en"
        self.show_details = False
        self._font_path = self._resolve_font_path()

    def toggle_language(self) -> str:
        self.language = "en" if self.language == "zh" else "zh"
        return self.language

    def toggle_details(self) -> bool:
        self.show_details = not self.show_details
        return self.show_details

    def _resolve_font_path(self) -> str | None:
        windows_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        if not windows_root:
            return None
        windows_dir = Path(windows_root)
        fonts_dir = windows_dir / "Fonts"
        candidates = [
            fonts_dir / "msyh.ttc",
            fonts_dir / "msyhbd.ttc",
            fonts_dir / "simhei.ttf",
            fonts_dir / "simsun.ttc",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _text(self, key: str) -> str:
        return LANGUAGE_PACK[self.language].get(key, key)

    def _state_text(self, state_label: str) -> str:
        return STATE_TRANSLATIONS.get(self.language, {}).get(state_label, state_label)

    def _panel_lines(self, overlay: StatusOverlay) -> list[tuple[str, tuple[int, int, int], int]]:
        state_color = (40, 220, 80) if not overlay.alert else (40, 80, 255)
        metric_color = (255, 240, 160)
        lines = [
            (f"{self._text('state')}: {self._state_text(overlay.state_label)}", state_color, 26),
            (f"{self._text('confidence')}: {overlay.confidence:.2f}", metric_color, 22),
            (f"{self._text('fps')}: {overlay.fps:.2f}", metric_color, 22),
        ]

        if self.show_details:
            if overlay.raw_state_label:
                lines.append(
                    (
                        f"{self._text('raw_state')}: {self._state_text(overlay.raw_state_label)}",
                        metric_color,
                        20,
                    )
                )
            if overlay.vote_ratio is not None:
                lines.append((f"{self._text('vote_ratio')}: {overlay.vote_ratio:.2f}", metric_color, 20))
            if overlay.fatigue_probability is not None:
                lines.append(
                    (f"{self._text('fatigue_prob')}: {overlay.fatigue_probability:.2f}", metric_color, 20)
                )
            if overlay.ear is not None:
                lines.append((f"{self._text('ear')}: {overlay.ear:.3f}", metric_color, 20))
            if overlay.mar is not None:
                lines.append((f"{self._text('mar')}: {overlay.mar:.3f}", metric_color, 20))
            if overlay.pitch is not None and overlay.yaw is not None and overlay.roll is not None:
                lines.append(
                    (
                        f"{self._text('pose')}: {overlay.pitch:.1f}/{overlay.yaw:.1f}/{overlay.roll:.1f}",
                        metric_color,
                        20,
                    )
                )
            if overlay.backend:
                lines.append((f"{self._text('backend')}: {overlay.backend}", metric_color, 20))
            if overlay.message:
                lines.append((overlay.message, (190, 255, 255), 20))

        control_key = "controls_detail" if self.show_details else "controls_compact"
        lines.append((self._text(control_key), (180, 220, 255), 18))
        return lines

    def _draw_text_with_pil(
        self,
        frame: Any,
        lines: list[tuple[str, tuple[int, int, int], int]],
        x: int,
        y_start: int,
        line_height: int,
    ) -> Any:
        from PIL import Image, ImageDraw, ImageFont

        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)

        for index, (line, color, font_size) in enumerate(lines):
            font = ImageFont.truetype(self._font_path, font_size) if self._font_path else ImageFont.load_default()
            y = y_start + index * line_height
            draw.text((x, y), line, font=font, fill=(color[2], color[1], color[0]))

        return np.asarray(image)[:, :, ::-1].copy()

    def draw(
        self,
        frame: Any,
        overlay: StatusOverlay,
        landmarks: LandmarkResult | None = None,
    ) -> Any:
        try:
            import cv2
        except ImportError:
            return frame

        if frame is None:
            return None

        if landmarks is not None and landmarks.face_detected:
            for x, y in landmarks.sampled_points(step=10):
                cv2.circle(frame, (x, y), 1, (255, 190, 60), -1)

        height, width = frame.shape[:2]
        lines = self._panel_lines(overlay)
        panel_width = 280 if not self.show_details else 360
        panel_width = min(panel_width, max(240, width - 24))
        line_height = 28 if not self.show_details else 30
        padding = 14
        panel_height = padding * 2 + line_height * len(lines)

        panel_x1 = 12
        panel_y1 = height - panel_height - 12
        panel_x2 = panel_x1 + panel_width
        panel_y2 = height - 12

        canvas = frame.copy()
        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (18, 24, 38), -1)
        cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (65, 88, 120), 2)
        cv2.addWeighted(canvas, 0.58, frame, 0.42, 0.0, frame)

        text_x = panel_x1 + padding
        text_y = panel_y1 + padding

        if self.language == "zh":
            frame = self._draw_text_with_pil(frame, lines, text_x, text_y, line_height)
        else:
            for index, (line, color, font_size) in enumerate(lines):
                y = text_y + (index + 1) * line_height - 8
                scale = 0.62 if font_size <= 20 else 0.72 if font_size <= 22 else 0.84
                cv2.putText(
                    frame,
                    line,
                    (text_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if overlay.alert:
            banner = self._text("alert")
            banner_width = min(width - 24, 360)
            cv2.rectangle(frame, (width - banner_width - 12, 12), (width - 12, 56), (20, 35, 180), -1)
            if self.language == "zh":
                frame = self._draw_text_with_pil(
                    frame,
                    [(banner, (255, 255, 255), 22)],
                    width - banner_width,
                    20,
                    28,
                )
            else:
                cv2.putText(
                    frame,
                    banner,
                    (width - banner_width, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame
