from __future__ import annotations

from time import monotonic


class AlertManager:
    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        consecutive_threshold: int = 3,
        confidence_threshold: float = 0.8,
        vote_ratio_threshold: float = 0.75,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_threshold = consecutive_threshold
        self.confidence_threshold = confidence_threshold
        self.vote_ratio_threshold = vote_ratio_threshold
        self._last_alert_time = 0.0
        self._positive_streak = 0

    def should_alert(self, label_id: int, confidence: float, vote_ratio: float) -> bool:
        if (
            label_id != 1
            or confidence < self.confidence_threshold
            or vote_ratio < self.vote_ratio_threshold
        ):
            self._positive_streak = 0
            return False

        self._positive_streak += 1
        if self._positive_streak < self.consecutive_threshold:
            return False

        now = monotonic()
        if now - self._last_alert_time < self.cooldown_seconds:
            return False

        self._last_alert_time = now
        return True
