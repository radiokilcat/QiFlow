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
    is_multi_hand: bool = False  # if True, process_all() is called instead of process()

    @abstractmethod
    def process(self, landmarks: Any, frame_time: float) -> GestureEvent | None:
        """Process a single hand's landmarks. Not called when is_multi_hand=True."""
        ...

    def process_all(self, hands: list[Any], frame_time: float) -> list[GestureEvent]:
        """Process all detected hands at once. Override when is_multi_hand=True."""
        return []

    def reset(self) -> None:
        """Reset internal state. Override if stateful."""
        pass
