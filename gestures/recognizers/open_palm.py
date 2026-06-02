from __future__ import annotations
from typing import Any
from ..base import GestureRecognizer, GestureEvent


class OpenPalmRecognizer(GestureRecognizer):
    gesture_id = "open_palm"
    name = "Open Palm"

    def __init__(self) -> None:
        self._active = False

    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        # Stub: real implementation would analyze finger extension from landmarks
        return None

    def reset(self) -> None:
        self._active = False
