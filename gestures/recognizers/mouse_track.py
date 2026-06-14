from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent

_FIST_TIPS = [8, 12, 16, 20]
_FIST_MCPS = [5,  9, 13, 17]
_SMOOTH = 0.35


def _is_fist(hand: Any) -> bool:
    return sum(1 for t, m in zip(_FIST_TIPS, _FIST_MCPS) if hand[t].y > hand[m].y) >= 3


class MouseTrackRecognizer(GestureRecognizer):
    """
    One hand makes a fist (activator), the other hand's index finger controls the cursor.
    Does not rely on handedness labels — detects roles by shape.
    """

    gesture_id = "mouse_track"
    name = "Mouse Track"
    is_multi_hand = True

    def __init__(self) -> None:
        self._active = False
        self._sx = 0.5
        self._sy = 0.5

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def process_all(self, hands: list[Any], frame_time: float) -> list[GestureEvent]:
        if len(hands) < 2:
            return self._deactivate(frame_time)

        # After cv2.flip the frame is mirrored, so MediaPipe swaps handedness labels:
        # user's left hand → "Right", user's right hand → "Left"
        left = next((h for h in hands if h.handedness == "Right"), None)
        right = next((h for h in hands if h.handedness == "Left"), None)

        fist = left if (left is not None and _is_fist(left)) else None
        pointer = right

        if fist is None or pointer is None or left is None or right is None:
            return self._deactivate(frame_time)

        tip = pointer[8]  # index finger tip of the non-fist hand
        self._sx = _SMOOTH * tip.x + (1 - _SMOOTH) * self._sx
        self._sy = _SMOOTH * tip.y + (1 - _SMOOTH) * self._sy

        phase = "started" if not self._active else "updated"
        self._active = True
        return [GestureEvent(
            gesture_id=self.gesture_id,
            confidence=0.95,
            phase=phase,
            timestamp=frame_time,
            payload={"x": self._sx, "y": self._sy},
        )]

    def _deactivate(self, frame_time: float) -> list[GestureEvent]:
        if self._active:
            self._active = False
            return [GestureEvent(
                gesture_id=self.gesture_id,
                confidence=0.9,
                phase="ended",
                timestamp=frame_time,
            )]
        return []

    def reset(self) -> None:
        self._active = False
