from __future__ import annotations
from typing import Any, Literal
from ..base import GestureRecognizer, GestureEvent

_TIPS = [8, 12, 16, 20]
_MCPS = [5,  9, 13, 17]

# After cv2.flip: user's left → MediaPipe "Right", user's right → MediaPipe "Left"
_MP_HAND: dict[str, str] = {"left": "Right", "right": "Left"}


class FistRecognizer(GestureRecognizer):
    gesture_id = "fist"
    name = "Fist"

    def __init__(self, hand: Literal["any", "left", "right"] = "any") -> None:
        self._mp_hand: str | None = _MP_HAND.get(hand)
        self._active = False
        if hand != "any":
            self.gesture_id = f"fist_{hand}"
            self.name = f"Fist ({hand.capitalize()})"

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        if self._mp_hand is not None and getattr(landmarks, "handedness", None) != self._mp_hand:
            return None
        curled = sum(1 for t, m in zip(_TIPS, _MCPS) if landmarks[t].y > landmarks[m].y)

        if curled >= 3:
            confidence = min(1.0, 0.7 + 0.075 * curled)
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
