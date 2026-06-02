from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent


class PinchRecognizer(GestureRecognizer):
    gesture_id = "pinch"
    name = "Pinch"

    def __init__(self) -> None:
        self._active = False

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        return None

    def reset(self) -> None:
        self._active = False
