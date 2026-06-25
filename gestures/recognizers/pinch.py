from __future__ import annotations
import math
from typing import Any, Literal
from ..base import GestureRecognizer, GestureEvent

_PINCH_RATIO = 0.30   # pinch_dist / hand_size to count as pinch
_RELEASE_RATIO = 0.45

# After cv2.flip: user's left → MediaPipe "Right", user's right → MediaPipe "Left"
_MP_HAND: dict[str, str] = {"left": "Right", "right": "Left"}


def _dist(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class PinchRecognizer(GestureRecognizer):
    gesture_id = "pinch"
    name = "Pinch"

    def __init__(self, hand: Literal["any", "left", "right"] = "any") -> None:
        self._mp_hand: str | None = _MP_HAND.get(hand)
        self._active = False
        if hand != "any":
            self.gesture_id = f"pinch_{hand}"
            self.name = f"Pinch ({hand.capitalize()})"

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        if self._mp_hand is not None and getattr(landmarks, "handedness", None) != self._mp_hand:
            return None
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
