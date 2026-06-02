from __future__ import annotations
from typing import Literal, Type
from .base import Action, ActionParams, ActionContext, RiskLevel


class MouseClickParams(ActionParams):
    button: Literal["left", "right", "middle"] = "left"


class MouseClickAction(Action):
    action_id = "mouse.click"
    name = "Mouse Click"
    description = "Click a mouse button"
    risk_level: RiskLevel = "low"
    params_model: Type[MouseClickParams] = MouseClickParams

    def preview(self, params: MouseClickParams) -> str:
        return f"Click {params.button} mouse button"

    def execute(self, params: MouseClickParams, context: ActionContext) -> None:
        context.os_adapter.click_mouse(params.button)


class MouseScrollParams(ActionParams):
    delta: int


class MouseScrollAction(Action):
    action_id = "mouse.scroll"
    name = "Mouse Scroll"
    description = "Scroll the mouse wheel"
    risk_level: RiskLevel = "low"
    params_model: Type[MouseScrollParams] = MouseScrollParams

    def preview(self, params: MouseScrollParams) -> str:
        direction = "up" if params.delta > 0 else "down"
        return f"Scroll {direction} by {abs(params.delta)}"

    def execute(self, params: MouseScrollParams, context: ActionContext) -> None:
        context.os_adapter.scroll(params.delta)
