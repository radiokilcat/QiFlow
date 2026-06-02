from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any


@dataclass
class OverlayMessage:
    kind: Literal["message", "confirmation", "error"]
    text: str
    title: str = ""
    instruction: str = ""


@dataclass
class FramePacket:
    """Payload sent from CameraWorker to the main thread each frame."""
    frame: Any          # np.ndarray BGR uint8
    timestamp_ms: int
    gesture_id: str | None = None
    confidence: float = 0.0
    has_hands: bool = False


@dataclass
class ActionLogEntry:
    timestamp: float
    action_preview: str
    gesture_id: str = ""
