from __future__ import annotations
import math
from typing import Any
from ..base import GestureRecognizer, GestureEvent

_PINCH_RATIO = 0.30   # pinch_dist / hand_size to count as pinch
_RELEASE_RATIO = 0.45


def _dist(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class PinchRecognizer(GestureRecognizer):
    gesture_id = "pinch"
    name = "Pinch"

    def __init__(self) -> None:
        self._active = False

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        hand_size = _dist(landmarks[0], landmarks[9]) or 0.001
        pinch_dist = _dist(landmarks[4], landmarks[8])
        ratio = pinch_dist / hand_size

        if ratio < _PINCH_RATIO:
            confidence = min(1.0, 1.0 - ratio / _PINCH_RATIO * 0.3)
            phase = "started" if not self._active else "updated"
            self._active = True
            return GestureEvent(
                gesture_id=self.gesture_id,
                confidence=confidence,
                phase=phase,
                timestamp=frame_time,
            )

        if self._active and ratio > _RELEASE_RATIO:
            self._active = False
            return GestureEvent(
                gesture_id=self.gesture_id,
                confidence=0.7,
                phase="ended",
                timestamp=frame_time,
            )

        return None

    def reset(self) -> None:
        self._active = False
