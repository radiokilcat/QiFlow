from __future__ import annotations
from typing import Iterator
from .base import GestureRecognizer


class GestureRegistry:
    def __init__(self) -> None:
        self._recognizers: dict[str, GestureRecognizer] = {}

    def register(self, recognizer: GestureRecognizer) -> None:
        if recognizer.gesture_id in self._recognizers:
            raise ValueError(f"Gesture '{recognizer.gesture_id}' already registered")
        self._recognizers[recognizer.gesture_id] = recognizer

    def get(self, gesture_id: str) -> GestureRecognizer:
        if gesture_id not in self._recognizers:
            raise KeyError(f"Gesture '{gesture_id}' not found")
        return self._recognizers[gesture_id]

    def all(self) -> Iterator[GestureRecognizer]:
        return iter(self._recognizers.values())

    def ids(self) -> list[str]:
        return list(self._recognizers.keys())
