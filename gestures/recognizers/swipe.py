from __future__ import annotations
from collections import deque
from typing import Any
from ..base import GestureRecognizer, GestureEvent

_WINDOW_S = 0.5      # seconds of history to keep
_MIN_DELTA = 0.25    # min wrist travel (fraction of frame width) to count as swipe
_MIN_SAMPLES = 4     # need at least this many frames in the window


class _SwipeRecognizer(GestureRecognizer):
    def __init__(self, direction: int) -> None:
        # direction: -1 = left (x decreases), +1 = right (x increases)
        self._dir = direction
        self._active = False
        self._history: deque[tuple[float, float]] = deque()  # (time, x)

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        x = landmarks[0].x  # wrist x, normalized 0..1
        self._history.append((frame_time, x))

        cutoff = frame_time - _WINDOW_S
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < _MIN_SAMPLES:
            return None

        delta = (self._history[-1][1] - self._history[0][1]) * self._dir

        if delta > _MIN_DELTA and not self._active:
            self._active = True
            confidence = min(1.0, 0.6 + delta)
            return GestureEvent(
                gesture_id=self.gesture_id,
                confidence=confidence,
                phase="ended",
                timestamp=frame_time,
            )

        if delta < 0.05:
            self._active = False

        return None

    def reset(self) -> None:
        self._active = False
        self._history.clear()


class SwipeLeftRecognizer(_SwipeRecognizer):
    gesture_id = "swipe_left"
    name = "Swipe Left"

    def __init__(self) -> None:
        super().__init__(direction=-1)


class SwipeRightRecognizer(_SwipeRecognizer):
    gesture_id = "swipe_right"
    name = "Swipe Right"

    def __init__(self) -> None:
        super().__init__(direction=+1)
