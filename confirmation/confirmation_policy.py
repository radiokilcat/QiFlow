from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


ConfirmationType = Literal["none", "hold", "repeat_gesture", "second_gesture", "hotkey"]


class ConfirmationPolicy(BaseModel):
    type: ConfirmationType = "none"
    timeout_ms: int = 1500
    hold_ms: int = 700
    second_gesture_id: str | None = None
