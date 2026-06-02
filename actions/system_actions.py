from __future__ import annotations
from typing import Type
from .base import Action, ActionParams, ActionContext, RiskLevel


class VolumeChangeParams(ActionParams):
    delta: int


class VolumeChangeAction(Action):
    action_id = "system.volume_change"
    name = "Change Volume"
    description = "Increase or decrease system volume"
    risk_level: RiskLevel = "low"
    params_model: Type[VolumeChangeParams] = VolumeChangeParams

    def preview(self, params: VolumeChangeParams) -> str:
        direction = "up" if params.delta > 0 else "down"
        return f"Volume {direction} by {abs(params.delta)}"

    def execute(self, params: VolumeChangeParams, context: ActionContext) -> None:
        context.os_adapter.change_volume(params.delta)
