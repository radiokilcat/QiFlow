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
    texture_data: Any   # np.ndarray float32 RGBA flat, ready for dpg.set_value
    timestamp_ms: int
    gesture_id: str | None = None
    confidence: float = 0.0
    has_hands: bool = False
    hand_side: str | None = None  # "left" | "right" | None for multi-hand / unknown


@dataclass
class ActionLogEntry:
    timestamp: float
    action_preview: str
    gesture_id: str = ""
