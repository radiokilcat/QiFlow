from __future__ import annotations
from typing import Any, Literal
from ..base import GestureRecognizer, GestureEvent

# After cv2.flip: user's left → MediaPipe "Right", user's right → MediaPipe "Left"
_MP_HAND: dict[str, str] = {"left": "Right", "right": "Left"}


class PointRecognizer(GestureRecognizer):
    """Index finger extended, middle/ring/pinky curled."""

    gesture_id = "point"
    name = "Point"

    def __init__(self, hand: Literal["any", "left", "right"] = "any") -> None:
        self._mp_hand: str | None = _MP_HAND.get(hand)
        self._active = False
        if hand != "any":
            self.gesture_id = f"point_{hand}"
            self.name = f"Point ({hand.capitalize()})"

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        if self._mp_hand is not None and getattr(landmarks, "handedness", None) != self._mp_hand:
            return None

        # Index extended: tip (8) above PIP (6)
        index_extended = landmarks[8].y < landmarks[6].y

        # Middle / ring / pinky curled: tip below its MCP
        curled = sum([
            landmarks[12].y > landmarks[9].y,   # middle
            landmarks[16].y > landmarks[13].y,  # ring
            landmarks[20].y > landmarks[17].y,  # pinky
        ])

        if index_extended and curled >= 2:
            confidence = min(1.0, 0.70 + 0.08 * curled)
            phase = "started" if not self._active else "updated"
            self._active = True
            return GestureEvent(
                gesture_id=self.gesture_id,
                confidence=confidence,
                phase=phase,
                timestamp=frame_time,
            )

        if self._active:
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
