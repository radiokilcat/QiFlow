from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


GesturePhase = Literal["started", "updated", "ended"]


@dataclass
class GestureEvent:
    gesture_id: str
    confidence: float
    phase: GesturePhase
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


class GestureRecognizer(ABC):
    gesture_id: str
    name: str

    @abstractmethod
    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        """Process landmarks and return a GestureEvent if gesture is detected."""
        ...

    def reset(self) -> None:
        """Reset internal state. Override if stateful."""
        pass
