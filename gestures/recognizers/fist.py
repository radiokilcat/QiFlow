from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent

_TIPS = [8, 12, 16, 20]
_MCPS = [5,  9, 13, 17]


class FistRecognizer(GestureRecognizer):
    gesture_id = "fist"
    name = "Fist"

    def __init__(self) -> None:
        self._active = False

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
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
