from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, field_validator, model_validator
from confirmation.confirmation_policy import ConfirmationPolicy


TriggerType = Literal["on_start", "on_hold", "on_release"]


class Binding(BaseModel):
    id: str
    gesture_id: str
    action_id: str
    action_params: dict[str, Any] = {}
    trigger: TriggerType = "on_start"
    min_confidence: float = 0.7
    stable_for_ms: int = 0
    cooldown_ms: int = 500
    confirmation: ConfirmationPolicy = ConfirmationPolicy()
    enabled: bool = True
    exclusive: bool = False  # while active, all other bindings are suppressed

    @field_validator("min_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        return v
