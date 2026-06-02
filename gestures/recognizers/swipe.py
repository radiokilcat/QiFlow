from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent


class SwipeLeftRecognizer(GestureRecognizer):
    gesture_id = "swipe_left"
    name = "Swipe Left"

    def __init__(self) -> None:
        self._active = False

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def reset(self) -> None:
        self._active = False


class SwipeRightRecognizer(GestureRecognizer):
    gesture_id = "swipe_right"
    name = "Swipe Right"

    def __init__(self) -> None:
        self._active = False

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def reset(self) -> None:
        self._active = False
