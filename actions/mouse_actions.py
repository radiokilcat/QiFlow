from __future__ import annotations
import ctypes
import functools
from typing import Any, Literal, Type
from .base import Action, ActionParams, ActionContext, RiskLevel


@functools.lru_cache(maxsize=1)
def _screen_size() -> tuple[int, int]:
    return (
        ctypes.windll.user32.GetSystemMetrics(0),
        ctypes.windll.user32.GetSystemMetrics(1),
    )


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

    def execute(self, params: MouseClickParams, context: ActionContext, event_payload: dict = {}) -> None:
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

    def execute(self, params: MouseScrollParams, context: ActionContext, event_payload: dict = {}) -> None:
        context.os_adapter.scroll(params.delta)


class MouseMoveParams(ActionParams):
    pass  # coordinates come from event_payload at runtime


class MouseMoveAction(Action):
    action_id = "mouse.move"
    name = "Mouse Move"
    description = "Move mouse cursor to index finger position"
    risk_level: RiskLevel = "low"
    params_model: Type[MouseMoveParams] = MouseMoveParams

    def preview(self, params: MouseMoveParams) -> str:
        return "Move mouse to right index finger tip"

    def execute(self, params: MouseMoveParams, context: ActionContext, event_payload: dict = {}) -> None:
        x_norm = float(event_payload.get("x", 0.5))
        y_norm = float(event_payload.get("y", 0.5))
        w, h = _screen_size()
        context.os_adapter.move_mouse(int(x_norm * w), int(y_norm * h))
